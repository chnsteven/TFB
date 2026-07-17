import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PsychFactor(nn.Module):
    """
    Daily dynamic psych factor Ψ.

    Inputs:
        E: (B, 1, T, H, W)  daily event seq
        M: (B, 3, T, H, W)  3 weather fields (temp, humidity, wind)
    Output:
        Ψ: (B, T, H, W)  daily psych factor field
        top_k_cycles: (B, H, W, top_k)  dominant period lengths in days
    """

    def __init__(
        self,
        gamma=0.2,
        top_k: int = 2,
        method="fft",
        option="",
        eps: float = 1e-8,
        lambda_=None,
    ):
        super().__init__()
        if option == "eval":
            torch.manual_seed(111)

        top_k = int(top_k)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # M has 3 ch; linear 3 -> 1
        self.phi_base_linear = nn.Linear(in_features=3, out_features=1, bias=True)
        self.lambda_ = None if lambda_ is None else float(lambda_)
        # LayerNorm: normalized_shape created dynamically in forward from (T,H,W)

        self.gamma = gamma
        self.top_k = top_k
        self.method = method.lower()
        self.eps = eps
        self.core_cycles = [7, 30]  # cycles: 7d (week), 30d (month)

    # ------------------------------------------------------------------
    # Input dim validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_input(M: torch.Tensor):
        """Validate the meteorological tensor dimensions."""
        if M.dim() != 5:
            raise ValueError(f"M must be 5-D (B, 3, T, H, W), got {M.dim()}-D")
        if M.shape[1] != 3:
            raise ValueError(f"M channel must be 3, got {M.shape[1]}")

    # ------------------------------------------------------------------
    # Psych factor (Ψ)
    # ------------------------------------------------------------------
    def forward(self, M: torch.Tensor):
        self._validate_input(M)
        return self.compute_psych_factor(M)

    def compute_psych_factor(self, M: torch.Tensor):
        """
        Input:
            M: (B, 3, T, H, W)
        Output:
            psi: (B, T, H, W)
            top_k_cycles: (B, H, W, top_k) period lengths in days
        """
        phi_base = self.compute_phi_base(M)  # (B, T, H, W)
        phi_cycle, top_k_cycles = self.compute_phi_cycle(M, top_k=self.top_k)  # (B, T, H, W)

        # Per-sample LayerNorm over (T,H,W)
        ln = nn.LayerNorm(
            phi_base.shape[1:],
            eps=self.eps,
            device=phi_base.device,
            dtype=phi_base.dtype,
        )
        phi_base_norm = ln(phi_base)
        phi_cycle_norm = ln(phi_cycle)

        psi = torch.sigmoid((phi_base_norm + phi_cycle_norm) / 2.0)
        return psi, top_k_cycles

    # ------------------------------------------------------------------
    # Φ_base: linear-weighted weather field
    # ------------------------------------------------------------------
    def compute_phi_base(self, M: torch.Tensor) -> torch.Tensor:
        """
        Input: M (B, 3, T, H, W)
        Output: phi_base (B, T, H, W)
        """
        assert M.dim() == 5, f"Expected M dim 5, got {M.dim()}"
        # (B, 3, T, H, W) -> (B, T, H, W, 3)
        M_perm = M.permute(0, 2, 3, 4, 1)
        # Linear 3 -> 1 -> (B, T, H, W, 1)
        phi_base = self.phi_base_linear(M_perm)
        return phi_base.squeeze(-1)  # (B, T, H, W)

    # ------------------------------------------------------------------
    # Morlet CWT
    # ------------------------------------------------------------------
    def morlet_cwt(
        self,
        x: torch.Tensor,
        periods,
        wavelet_name: str = "cmor1.5-1.0",
        sampling_period: float = 1.0,
    ) -> torch.Tensor:
        """
        Morlet CWT via FFT

        Args:
            x: (..., T) tensor; last dim is time
            periods: target period list or tensor (sample steps)

        Returns:
            mag: (..., n_scales, T) per-scale magnitude (no mean; keep scale dim)
        """
        parts = wavelet_name.replace("cmor", "").split("-")
        Fb = float(parts[0])
        Fc = float(parts[1])

        *batch, T = x.shape
        device, dtype = x.device, x.dtype

        freqs = torch.fft.rfftfreq(T, d=sampling_period, device=device)  # (T//2+1,)

        if not isinstance(periods, torch.Tensor):
            periods = torch.tensor(
                [float(p) for p in periods], device=device, dtype=torch.float64
            )
        else:
            periods = periods.to(device=device, dtype=torch.float64)

        scales = Fc * periods / sampling_period  # (n_scales,)
        freqs_d = freqs.double()
        center = (Fc / scales).unsqueeze(1)  # (n_scales, 1)
        psi_hat = torch.exp(
            -2
            * (torch.pi**2)
            * Fb
            * scales.unsqueeze(1) ** 2
            * (freqs_d.unsqueeze(0) - center) ** 2
        )  # (n_scales, T//2+1)

        X_f = torch.fft.rfft(x.double(), n=T, dim=-1)  # (..., T//2+1)
        X_f_exp = X_f.unsqueeze(-2)  # (..., 1, T//2+1)
        conv = X_f_exp * psi_hat  # (..., n_scales, T//2+1)
        coefs = torch.fft.irfft(conv, n=T, dim=-1)  # (..., n_scales, T)

        return coefs.abs().to(dtype)  # (..., n_scales, T)

    # ------------------------------------------------------------------
    # Φ_cycle: per-ch wavelet then aggregate
    # ------------------------------------------------------------------
    def compute_phi_cycle(
        self,
        M: torch.Tensor,
        n_periods: int = 32,
        top_k: int = 2,
        period_min: float = 3.0,
        period_max: float = 60.0,
    ):
        """
        Full-band Morlet CWT per weather ch in M;
        pick top_k energy periods (shared across channels); aggregate magnitudes.

        Input: M (B, 3, T, H, W)
        Output:
            phi_cycle: (B, T, H, W)
            top_k_cycles: (B, H, W, top_k) selected period lengths in days
        """
        assert M.dim() == 5, f"Expected M dim 5, got {M.dim()}"
        B, C, T, H, W = M.shape
        device, dtype = M.device, M.dtype
        top_k = min(int(top_k), n_periods)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Candidate periods (log-uniform, period_min ~ period_max)
        periods_all = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(period_min)),
                torch.log(torch.tensor(period_max)),
                n_periods,
            )
        ).to(device)  # (n_periods,)

        energy_norm_stack = []
        mag_stack = []

        for c in range(C):
            x = M[:, c].permute(0, 2, 3, 1)  # (B, H, W, T)
            mag = self.morlet_cwt(x, periods_all)  # (B, H, W, n_periods, T)
            energy = mag.mean(dim=-1)  # (B, H, W, n_periods)
            energy_max = energy.amax(dim=-1, keepdim=True).clamp(min=self.eps)
            energy_norm_stack.append(energy / energy_max)
            mag_stack.append(mag)

        # Shared top_k across channels: mean energy per (B, H, W, n_periods)
        energy_norm_mean = torch.stack(energy_norm_stack, dim=0).mean(dim=0)
        top_idx = torch.topk(energy_norm_mean, k=top_k, dim=-1).indices  # (B,H,W,top_k)
        top_k_cycles = periods_all[top_idx]  # (B, H, W, top_k)

        phi_cycle = torch.zeros(B, T, H, W, device=device, dtype=dtype)
        top_idx_exp = top_idx.unsqueeze(-1).expand(*top_idx.shape, T)

        for c in range(C):
            mag = mag_stack[c]
            energy_norm = energy_norm_stack[c]
            mag_top = mag.gather(dim=-2, index=top_idx_exp)  # (B, H, W, top_k, T)
            weights = energy_norm.gather(dim=-1, index=top_idx)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            agg = (mag_top * weights.unsqueeze(-1)).sum(dim=-2)
            phi_cycle = phi_cycle + agg.permute(0, 3, 1, 2)

        return phi_cycle / C, top_k_cycles
