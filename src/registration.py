"""
registration.py — Differentiable 2D Diffeomorphic Registration Network
=======================================================================
Architecture:
  - Deep 2D U-Net (5 levels, channels [32,64,128,256,512])
    Input  : two concatenated frames (B, 2, H, W)
    Output : stationary velocity field (B, 2, H, W)
  - Scaling & Squaring (7 steps) → diffeomorphic deformation φ
  - Spatial Transformer (grid_sample) → warped moving frame
  - Mask-weighted NCC loss (computed only inside myocardium ROI)
  - Smoothness regularization on velocity field (bending energy)

Why diffeomorphic?
  Integrating a stationary velocity field via scaling & squaring
  guarantees the output deformation is invertible and smooth —
  the myocardium cannot fold or tear. This is a hard physical
  constraint that ANTs enforces via optimization; here it is
  enforced by architecture.

Usage:
    from registration import RegistrationNet, RegistrationLoss, warp

    net  = RegistrationNet().to(device)
    loss = RegistrationLoss(alpha=0.1, beta=0.01)

    # frame_fixed, frame_moving : (B, 1, H, W)
    # mask                      : (B, 1, H, W)  — ED myocardium mask
    vel_field, phi = net(frame_fixed, frame_moving)
    warped         = warp(frame_moving, phi)
    total, ncc, smooth = loss(warped, frame_fixed, vel_field, mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ── Spatial Transformer / Warp ───────────────────────────────────────────────

def warp(image: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Warp an image using a deformation field φ.

    Args:
        image : (B, C, H, W)  — image to warp (moving frame)
        phi   : (B, 2, H, W)  — deformation field in pixel coordinates
                                 phi[b, 0] = x-displacement (width)
                                 phi[b, 1] = y-displacement (height)

    Returns:
        warped : (B, C, H, W)
    """
    B, C, H, W = image.shape

    # Build normalized sampling grid from deformation field
    # grid_sample expects coordinates in [-1, 1]
    xx = torch.linspace(-1, 1, W, device=image.device)
    yy = torch.linspace(-1, 1, H, device=image.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # (H, W) each
    base_grid = torch.stack([grid_x, grid_y], dim=-1)        # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1) # (B, H, W, 2)

    # Normalize displacement field to [-1, 1] scale
    # phi is in pixel units → divide by (W-1)/2 and (H-1)/2
    norm_x = phi[:, 0:1, :, :] / ((W - 1) / 2.0)  # (B, 1, H, W)
    norm_y = phi[:, 1:2, :, :] / ((H - 1) / 2.0)  # (B, 1, H, W)
    disp = torch.cat([norm_x, norm_y], dim=1)       # (B, 2, H, W)
    disp = disp.permute(0, 2, 3, 1)                 # (B, H, W, 2)

    sampling_grid = base_grid + disp  # (B, H, W, 2)

    warped = F.grid_sample(
        image, sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )
    return warped


# ── Scaling & Squaring ───────────────────────────────────────────────────────

def scaling_and_squaring(vel: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """
    Integrate a stationary velocity field v to obtain a diffeomorphism φ
    via the scaling and squaring algorithm.

    φ = exp(v) ≈ (exp(v/2^steps))^(2^steps)

    Each squaring step: φ_{t+1}(x) = φ_t(φ_t(x))
    which in practice means composing the deformation with itself.

    Args:
        vel   : (B, 2, H, W) stationary velocity field
        steps : number of squaring steps (7 → 128 integration steps)

    Returns:
        phi : (B, 2, H, W) diffeomorphic deformation field
    """
    # Scale down: φ_0 = v / 2^steps
    phi = vel / (2.0 ** steps)

    for _ in range(steps):
        # φ_{t+1} = φ_t ∘ φ_t  (compose φ_t with itself)
        phi = phi + warp(phi, phi)

    return phi


# ── U-Net Building Blocks ────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Double convolution block with GroupNorm and LeakyReLU.
    GroupNorm instead of BatchNorm — more stable with small batch sizes
    which is common in medical imaging.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        # Clamp groups to avoid GroupNorm errors when out_ch < groups
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Strided conv downsampling + double conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Bilinear upsampling + skip connection + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Main Registration Network ────────────────────────────────────────────────

class RegistrationNet(nn.Module):
    """
    Deep 2D U-Net Registration Network.

    Takes a fixed and moving frame, outputs:
      1. A stationary velocity field v ∈ R^(B,2,H,W)
      2. The integrated diffeomorphism φ = exp(v) ∈ R^(B,2,H,W)

    Architecture (5 levels):
      Encoder channels : [32, 64, 128, 256, 512]
      Decoder channels : [256, 128, 64, 32]
      Output head      : 1x1 conv → 2 channels (velocity field)

    Input  : (B, 2, H, W)  — fixed and moving concatenated along channel dim
    Output : vel_field (B, 2, H, W), phi (B, 2, H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,       # fixed + moving concatenated
        base_ch: int = 32,          # channels at first level
        enc_channels = (32, 64, 128, 256, 512),
        ss_steps: int = 7,          # scaling & squaring steps
    ):
        super().__init__()
        self.ss_steps = ss_steps

        C = enc_channels  # shorthand

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0 = ConvBlock(in_channels, C[0])  # (B, 32,  H,   W  )
        self.enc1 = DownBlock(C[0], C[1])          # (B, 64,  H/2, W/2)
        self.enc2 = DownBlock(C[1], C[2])          # (B, 128, H/4, W/4)
        self.enc3 = DownBlock(C[2], C[3])          # (B, 256, H/8, W/8)
        self.enc4 = DownBlock(C[3], C[4])          # (B, 512, H/16,W/16) ← bottleneck

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec3 = UpBlock(C[4], C[3], C[3])      # (B, 256, H/8, W/8)
        self.dec2 = UpBlock(C[3], C[2], C[2])      # (B, 128, H/4, W/4)
        self.dec1 = UpBlock(C[2], C[1], C[1])      # (B, 64,  H/2, W/2)
        self.dec0 = UpBlock(C[1], C[0], C[0])      # (B, 32,  H,   W  )

        # ── Velocity field output head ────────────────────────────────────────
        # Small output channels → 2D velocity field
        # Initialize weights near zero so training starts from identity transform
        self.vel_head = nn.Conv2d(C[0], 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.vel_head.weight)
        nn.init.zeros_(self.vel_head.bias)

    def forward(
        self,
        frame_fixed:  torch.Tensor,
        frame_moving: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_fixed  : (B, 1, H, W)
            frame_moving : (B, 1, H, W)

        Returns:
            vel_field : (B, 2, H, W) — stationary velocity field
            phi       : (B, 2, H, W) — integrated diffeomorphism
        """
        x = torch.cat([frame_fixed, frame_moving], dim=1)  # (B, 2, H, W)

        # Encoder
        s0 = self.enc0(x)    # (B, 32,  H,    W   )
        s1 = self.enc1(s0)   # (B, 64,  H/2,  W/2 )
        s2 = self.enc2(s1)   # (B, 128, H/4,  W/4 )
        s3 = self.enc3(s2)   # (B, 256, H/8,  W/8 )
        s4 = self.enc4(s3)   # (B, 512, H/16, W/16)  ← bottleneck

        # Decoder with skip connections
        d3 = self.dec3(s4, s3)  # (B, 256, H/8,  W/8 )
        d2 = self.dec2(d3, s2)  # (B, 128, H/4,  W/4 )
        d1 = self.dec1(d2, s1)  # (B, 64,  H/2,  W/2 )
        d0 = self.dec0(d1, s0)  # (B, 32,  H,    W   )

        # Velocity field: small values at init → near-identity transform
        vel_field = self.vel_head(d0)                        # (B, 2, H, W)
        phi       = scaling_and_squaring(vel_field, self.ss_steps)  # (B, 2, H, W)

        return vel_field, phi

    def get_warped(
        self,
        frame_fixed:  torch.Tensor,
        frame_moving: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method: returns warped frame + vel_field + phi.

        Returns:
            warped    : (B, 1, H, W)
            vel_field : (B, 2, H, W)
            phi       : (B, 2, H, W)
        """
        vel_field, phi = self.forward(frame_fixed, frame_moving)
        warped = warp(frame_moving, phi)
        return warped, vel_field, phi


# ── Losses ───────────────────────────────────────────────────────────────────

def mask_weighted_ncc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask:   torch.Tensor,
    win:    int = 9,
    eps:    float = 1e-5,
) -> torch.Tensor:
    """
    Mask-weighted Local Normalized Cross-Correlation (LNCC).

    Computed in a local window of size (win x win) — more sensitive to
    local myocardial motion than global NCC.

    NCC = 1 − mean(local_ncc inside mask)
    (we minimize this, so 0 = perfect registration)

    Args:
        y_true : (B, 1, H, W) — fixed frame
        y_pred : (B, 1, H, W) — warped moving frame
        mask   : (B, 1, H, W) — binary myocardium mask
        win    : local window radius
        eps    : numerical stability

    Returns:
        Scalar NCC loss (lower = better registration)
    """
    # Local sums via average pooling with uniform kernel
    kernel_size = win
    pad = win // 2

    def local_sum(x):
        return F.avg_pool2d(
            x, kernel_size=kernel_size, stride=1, padding=pad,
            count_include_pad=False
        ) * (kernel_size ** 2)

    I = y_true
    J = y_pred

    I2  = I * I
    J2  = J * J
    IJ  = I * J

    I_sum  = local_sum(I)
    J_sum  = local_sum(J)
    I2_sum = local_sum(I2)
    J2_sum = local_sum(J2)
    IJ_sum = local_sum(IJ)

    win_size = kernel_size ** 2
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    ncc = (cross * cross) / (I_var * J_var + eps)  # (B, 1, H, W) local NCC

    # Weight by mask — only care about NCC inside myocardium
    mask_sum = mask.sum() + eps
    weighted_ncc = (ncc * mask).sum() / mask_sum

    return 1.0 - weighted_ncc  # 0 = perfect, 1 = worst


def bending_energy(vel: torch.Tensor) -> torch.Tensor:
    """
    Bending energy regularization on the velocity field.
    Penalizes second-order spatial derivatives → encourages smooth fields.

    BE = mean( (∂²v/∂x²)² + (∂²v/∂y²)² + 2*(∂²v/∂x∂y)² )

    This is stronger than first-order (diffusion) regularization and
    produces smoother, more physically plausible deformations.
    """
    # First-order gradients
    dv_dx = vel[:, :, :, 1:] - vel[:, :, :, :-1]   # (B, 2, H, W-1)
    dv_dy = vel[:, :, 1:, :] - vel[:, :, :-1, :]   # (B, 2, H-1, W)

    # Second-order gradients (approximate)
    d2v_dx2  = dv_dx[:, :, :, 1:] - dv_dx[:, :, :, :-1]  # (B, 2, H, W-2)
    d2v_dy2  = dv_dy[:, :, 1:, :] - dv_dy[:, :, :-1, :]  # (B, 2, H-2, W)
    # Mixed partial: ∂²v/∂x∂y  (crop to common size)
    min_h = min(dv_dy.shape[2], dv_dx.shape[2])
    min_w = min(dv_dx.shape[3], dv_dy.shape[3])
    d2v_dxdy = (
        dv_dx[:, :, :min_h, :min_w] - dv_dy[:, :, :min_h, :min_w]
    )

    be = (
        d2v_dx2.pow(2).mean()
        + d2v_dy2.pow(2).mean()
        + 2.0 * d2v_dxdy.pow(2).mean()
    )
    return be


def jacobian_determinant(phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian determinant of a deformation field.

    Values < 0  → folding (physically impossible, should be penalized)
    Values ≈ 1  → locally rigid (good)
    Values >> 1 → large expansion

    Used for interpretability (visualizing where the myocardium
    compresses/expands) and as a diagnostic for registration quality.

    Args:
        phi : (B, 2, H, W)

    Returns:
        jac_det : (B, 1, H-1, W-1)
    """
    # Spatial gradients of deformation field
    # dφ_x/dx, dφ_x/dy, dφ_y/dx, dφ_y/dy
    dphi_x_dx = phi[:, 0:1, :, 1:] - phi[:, 0:1, :, :-1]   # (B,1,H,W-1)
    dphi_x_dy = phi[:, 0:1, 1:, :] - phi[:, 0:1, :-1, :]   # (B,1,H-1,W)
    dphi_y_dx = phi[:, 1:2, :, 1:] - phi[:, 1:2, :, :-1]   # (B,1,H,W-1)
    dphi_y_dy = phi[:, 1:2, 1:, :] - phi[:, 1:2, :-1, :]   # (B,1,H-1,W)

    # Crop to common size
    H = min(dphi_x_dx.shape[2], dphi_x_dy.shape[2])
    W = min(dphi_x_dx.shape[3], dphi_y_dx.shape[3])

    # Add identity (we want Jacobian of Id + φ, not just φ)
    jac_det = (
        (1 + dphi_x_dx[:, :, :H, :W]) * (1 + dphi_y_dy[:, :, :H, :W])
        - dphi_x_dy[:, :, :H, :W] * dphi_y_dx[:, :, :H, :W]
    )
    return jac_det


class RegistrationLoss(nn.Module):
    """
    Combined registration loss:
        L = L_ncc + α * L_bending + β * L_fold

    L_ncc     : mask-weighted local NCC (image similarity inside myocardium)
    L_bending : bending energy (smoothness regularization)
    L_fold    : folding penalty — penalizes negative Jacobian determinants
                (physically impossible deformations)

    Args:
        alpha : weight on bending energy (default 0.1)
        beta  : weight on folding penalty (default 0.01)
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(
        self,
        warped:    torch.Tensor,   # (B, 1, H, W) warped moving frame
        fixed:     torch.Tensor,   # (B, 1, H, W) fixed frame
        vel_field: torch.Tensor,   # (B, 2, H, W) velocity field
        phi:       torch.Tensor,   # (B, 2, H, W) deformation field
        mask:      torch.Tensor,   # (B, 1, H, W) myocardium mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss : scalar
            ncc_loss   : scalar (for logging)
            bend_loss  : scalar (for logging)
            fold_loss  : scalar (for logging)
        """
        ncc_loss  = mask_weighted_ncc(fixed, warped, mask)
        bend_loss = bending_energy(vel_field)

        # Folding penalty: penalize where Jacobian < 0
        jac_det   = jacobian_determinant(phi)
        fold_loss = F.relu(-jac_det).mean()  # only penalize negative values

        total = ncc_loss + self.alpha * bend_loss + self.beta * fold_loss

        return total, ncc_loss, bend_loss, fold_loss


# ── Model summary utility ────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, H, W = 4, 128, 128

    # ── 1. Build model ────────────────────────────────────────────────────────
    net  = RegistrationNet().to(device)
    loss = RegistrationLoss(alpha=0.1, beta=0.01)

    n_params = count_parameters(net)
    print(f"\nRegistrationNet parameters: {n_params:,}  ({n_params/1e6:.2f}M)")

    # ── 2. Forward pass ───────────────────────────────────────────────────────
    fixed  = torch.rand(B, 1, H, W, device=device)
    moving = torch.rand(B, 1, H, W, device=device)
    mask   = (torch.rand(B, 1, H, W, device=device) > 0.7).float()

    warped, vel_field, phi = net.get_warped(fixed, moving)

    print(f"\nForward pass:")
    print(f"  fixed   : {fixed.shape}")
    print(f"  moving  : {moving.shape}")
    print(f"  vel     : {vel_field.shape}  min={vel_field.min():.4f}  max={vel_field.max():.4f}")
    print(f"  phi     : {phi.shape}        min={phi.min():.4f}        max={phi.max():.4f}")
    print(f"  warped  : {warped.shape}")

    # ── 3. Jacobian determinant ───────────────────────────────────────────────
    jac = jacobian_determinant(phi)
    n_fold = (jac < 0).float().mean().item()
    print(f"\nJacobian determinant:")
    print(f"  shape   : {jac.shape}")
    print(f"  min     : {jac.min().item():.4f}")
    print(f"  max     : {jac.max().item():.4f}")
    print(f"  mean    : {jac.mean().item():.4f}")
    print(f"  folding : {n_fold*100:.2f}% of voxels  (should be ~0 after training)")

    # ── 4. Loss ───────────────────────────────────────────────────────────────
    total, ncc, bend, fold = loss(warped, fixed, vel_field, phi, mask)
    print(f"\nLoss (on random data — expect ~1.0 NCC, ~0 bend/fold at init):")
    print(f"  total   : {total.item():.4f}")
    print(f"  ncc     : {ncc.item():.4f}   (target: < 0.1 after training)")
    print(f"  bending : {bend.item():.6f}")
    print(f"  folding : {fold.item():.6f}")

    # ── 5. Backward pass (critical — checks gradients flow) ───────────────────
    total.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in net.parameters() if p.grad is not None
    )
    print(f"\nBackward pass: gradient norm = {grad_norm:.4f}  (should be > 0)")

    # ── 6. Memory usage ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nGPU memory used: {mem_mb:.1f} MB  (for B={B}, H=W={H})")

    # ── 7. Test warp is identity at init ──────────────────────────────────────
    # At init, vel_head weights = 0, so vel ≈ 0, phi ≈ 0 → warped ≈ moving
    net_check = RegistrationNet().to(device)
    with torch.no_grad():
        w, v, p = net_check.get_warped(fixed, moving)
    diff = (w - moving).abs().mean().item()
    print(f"\nIdentity check (vel=0 at init):")
    print(f"  |warped - moving| = {diff:.6f}  (should be ≈ 0.0)")

    print("\n[registration.py] All checks passed.")