import torch
from . import utils
from .psych_factor import PsychFactor

# Module-level singleton; avoid re-init + .to(device) each forward
_psych_factor_cache: dict = {}


def _get_psych_factor(device, cycle_gamma=0.2, psych_top_k=2):
    cache_key = f"{device}_{cycle_gamma}_{psych_top_k}"
    if cache_key not in _psych_factor_cache:
        _psych_factor_cache[cache_key] = PsychFactor(
            gamma=cycle_gamma, top_k=psych_top_k
        ).to(device)
    return _psych_factor_cache[cache_key]


def forecast_full_masking(x, T, his_t_patches, option="", seed=None):
    """
    Forecasting eval mask: history tokens fully visible, entire future fully masked.

    Aligns generative eval with direct forecasting — score all future positions
    (with eval_scope=forecast) while keeping the full history in the encoder.

    Args:
        x: [N, L, D] token tensor after patch embedding.
        T: number of temporal patch tokens (L must be divisible by T).
        his_t_patches: count of history time patches (pred tail starts here).

    Returns:
        x_masked, mask, ids_restore, ids_keep, mask_info
    """
    del option, seed  # deterministic; no randomness

    N, L, D = x.shape
    device = x.device
    S = L // T

    if T <= 0 or L % T != 0:
        raise ValueError(f"Incompatible T={T} for token length L={L}")
    if his_t_patches < 0 or his_t_patches > T:
        raise ValueError(
            f"his_t_patches={his_t_patches} out of range for T={T} temporal patches"
        )

    t_idx = torch.arange(T, device=device)
    temporal_mask = t_idx >= his_t_patches
    mask = (
        temporal_mask.unsqueeze(1)
        .expand(T, S)
        .reshape(L)
        .unsqueeze(0)
        .expand(N, L)
        .float()
    )

    mask_info = {
        "strategy": "forecast_full",
        "t_mask_rate": temporal_mask.float().mean().item(),
        "s_mask_rate": 0.0,
        "union_rate": mask.float().mean().item(),
    }

    ids_shuffle = torch.argsort(mask, dim=1, stable=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    len_keep = int((mask[0] == 0).sum().item())
    if len_keep <= 0:
        raise ValueError(
            "forecast_full: no history tokens to keep "
            f"(his_t_patches={his_t_patches}, S={S})"
        )

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )
    return x_masked, mask, ids_restore, ids_keep, mask_info


def random_spatiotemporal_masking(x, T, spatial_ratio=0.15, temporal_ratio=0.15, option="", seed=None):
    """
    Randomly mask tokens with joint spatial and temporal constraints.
    A token is masked if its time step is masked OR its spatial position is masked.

    Args:
        x: [N, L, D] token tensor after patch embedding.
        T: number of temporal tokens (L must be divisible by T).
        spatial_ratio: fraction of spatial positions to mask (default 0.15).
        temporal_ratio: fraction of time steps to mask (default 0.15).
        option: when set to "eval" uses a fixed random seed for determinism.

    Returns:
        x_masked, mask, ids_restore, ids_keep
    """

    def _mask():
        N, L, D = x.shape
        device = x.device
        S = L // T

        if T <= 0 or L % T != 0:
            raise ValueError(f"Incompatible T={T} for token length L={L}")

        num_t_mask = max(1, int(round(T * temporal_ratio)))
        t_noise = torch.rand(N, T, device=device)
        t_ids = torch.argsort(t_noise, dim=1)
        temporal_mask = torch.zeros(N, T, dtype=torch.bool, device=device)
        temporal_mask.scatter_(1, t_ids[:, :num_t_mask], True)

        num_s_mask = max(1, int(round(S * spatial_ratio)))
        s_noise = torch.rand(N, S, device=device)
        s_ids = torch.argsort(s_noise, dim=1)
        spatial_mask = torch.zeros(N, S, dtype=torch.bool, device=device)
        spatial_mask.scatter_(1, s_ids[:, :num_s_mask], True)

        mask = (temporal_mask.unsqueeze(2) | spatial_mask.unsqueeze(1)).reshape(N, L).float()

        mask_info = {
            "strategy": "random_spatiotemporal",
            "t_mask_rate": temporal_mask.float().mean().item(),
            "s_mask_rate": spatial_mask.float().mean().item(),
            "union_rate": mask.float().mean().item(),
        }

        ids_shuffle = torch.argsort(mask, dim=1, stable=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = max(1, int(round(L * (1.0 - temporal_ratio) * (1.0 - spatial_ratio))))
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_masked, mask, ids_restore, ids_keep, mask_info

    if option == "eval":
        with torch.random.fork_rng():
            torch.manual_seed(111 if seed is None else seed)
            return _mask()
    return _mask()


def _mask_prob_capped(values: torch.Tensor, cap: float) -> torch.Tensor:
    """Truncate mask probabilities to ``[0, cap]`` (``cycle_gamma`` upper bound)."""
    cap_t = values.new_tensor(float(cap))
    return values.clamp(min=0.0, max=cap_t)


def psych_gradient_masking(
    x_tokens,
    x_raw,
    patch_size,
    t_patch_size,
    option="",
    seed=None,
    cycle_gamma=0.2,
    psych_top_k=2,
    component="union",
):
    """
    Content-aware mask from psych factor and/or spatial gradient.

    Args:
        x_tokens: [B, L, embed_dim]
        x_raw: [B, 4, T_raw, H_raw, W_raw]
        patch_size: spatial patch size
        t_patch_size: temporal patch size
        option: "eval" fixes random seed
        component: which mask to apply —
            ``union`` (default): cycle-psych temporal | spatial gradient;
            ``psych``: cycle-psych temporal only (ablation);
            ``spatial``: spatial gradient only (ablation).
        cycle_gamma: shared upper cap for both branch mask probabilities;
            psych uses ``clamp(psi_patch * cycle_gamma, 0, cycle_gamma)``,
            spatial uses ``clamp(grad_patch, 0, cycle_gamma)``.
        psych_top_k: number of dominant CWT periods per spatial location (PsychFactor).

    Returns:
        x_masked, mask, ids_restore, ids_keep, mask_info
    """
    if component not in ("union", "psych", "spatial"):
        raise ValueError(
            "component must be 'union', 'psych', or 'spatial', got {!r}".format(
                component
            )
        )

    def _mask():
        B, L, D = x_tokens.shape
        device = x_tokens.device

        B_raw, C_raw, T_raw, H_raw, W_raw = x_raw.shape
        T_patch = T_raw // t_patch_size
        H_patch = H_raw // patch_size
        W_patch = W_raw // patch_size

        expected_L = T_patch * H_patch * W_patch
        if expected_L != L:
            raise ValueError(
                f"Dimension mismatch: L={L}, but expected "
                f"T_patch({T_patch}) × H_patch({H_patch}) × W_patch({W_patch}) = {expected_L}"
            )

        E = x_raw[:, 0:1]
        M = x_raw[:, 1:4]

        psych = _get_psych_factor(
            device, cycle_gamma=cycle_gamma, psych_top_k=psych_top_k
        )
        psi, top_k_cycles = psych.compute_psych_factor(E, M)

        tau_cycle = utils.build_tau_cycle(psi, top_k_cycles)
        psi_patch = utils.downsample_to_patch_resolution(psi, T_patch, H_patch, W_patch)
        tau_patch = utils.map_tau_cycle_to_patch(tau_cycle, t_patch_size, patch_size)
        cap = float(cycle_gamma)
        p = _mask_prob_capped(psi_patch * cap, cap)
        t_mask_full = tau_patch & torch.bernoulli(p).bool()

        grad = utils.compute_central_spatio_gradient(M)
        grad_mean = grad.mean(dim=1)
        grad_patch = utils.downsample_to_patch_resolution(grad_mean, T_patch, H_patch, W_patch)
        s_prob = _mask_prob_capped(grad_patch, cap)
        s_mask_full = torch.bernoulli(s_prob).bool()

        if component == "psych":
            union_mask = t_mask_full
            strategy_name = "psych_gradient"
        elif component == "spatial":
            union_mask = s_mask_full
            strategy_name = "spatio_gradient"
        else:
            union_mask = t_mask_full | s_mask_full
            strategy_name = "psych_gradient"

        mask_flat = union_mask.reshape(B, L).float()

        mask_info = {
            "strategy": strategy_name,
            "component": component,
            "cycle_gamma": cap,
            "psych_top_k": int(psych_top_k),
            "t_mask_rate": t_mask_full.float().mean().item(),
            "s_mask_rate": s_mask_full.float().mean().item(),
            "union_rate": mask_flat.float().mean().item(),
        }

        ids_shuffle = torch.argsort(mask_flat, dim=1, stable=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep_per_sample = (mask_flat == 0).sum(dim=1)
        max_keep = max(int(len_keep_per_sample.max().item()), 1)

        ids_keep = ids_shuffle[:, :max_keep]
        x_masked = torch.gather(
            x_tokens, dim=1,
            index=ids_keep.unsqueeze(-1).expand(B, max_keep, D),
        )
        return x_masked, mask_flat, ids_restore, ids_keep, mask_info

    if option == "eval":
        with torch.random.fork_rng():
            torch.manual_seed(111 if seed is None else seed)
            return _mask()
    return _mask()


def spatiotemporal_restore(x, ids_restore, N, T, H, W, C, mask_token):
    """Restore masked tokens for random_spatiotemporal / psych_gradient."""
    L_restore = ids_restore.shape[1]
    L_keep = x.shape[1]
    num_mask = L_restore - L_keep
    if num_mask < 0:
        raise ValueError(f"ids_restore({L_restore}) smaller than x tokens({L_keep})")

    if num_mask > 0:
        mask_tokens = mask_token.expand(N, num_mask, C)
        x_ = torch.cat([x, mask_tokens], dim=1)
    else:
        x_ = x

    x_restored = torch.gather(
        x_, dim=1,
        index=ids_restore.unsqueeze(-1).expand(N, L_restore, C),
    )
    return x_restored.view(N, T * H * W, C)
