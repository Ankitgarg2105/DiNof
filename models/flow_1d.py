"""
1D Normalizing Flow for time-series DiNof.
Accepts [B, C, 1, L] tensors (4D, height=1).

REDESIGN: Instead of flattening the full C*L=3264 dimensions (which requires
huge MLPs), we operate feature-wise: treat each time step as a C-dimensional
observation and use a C-dim flow. This is both faster and converges much better.

Architecture: RealNVP on the C=51 feature dimension, applied independently
across the L=64 time steps, with a shared flow (parameter efficient).

Interface compatible with MultiscaleFlow from models/normflows/core.py:
  - forward_kld(x)  → (z, nll_loss_scalar)   [used in losses.py]
  - sample(n)       → (samples, log_q)        [used in sampling.py]
  - log_prob(x)     → (log_prob, z)
"""
import torch
import torch.nn as nn
import numpy as np


class MaskedAffineCoupling(nn.Module):
    """
    Affine coupling layer with binary mask. Operates on [*, D] tensors.
    """
    def __init__(self, dim, mask, hidden_dim=128, num_layers=3):
        super().__init__()
        self.register_buffer('mask', mask)   # [D]

        layers = []
        in_d = dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.GELU()]
            in_d = hidden_dim
        layers += [nn.Linear(in_d, dim * 2)]
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        self.net = nn.Sequential(*layers)
        self.scale_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: [..., D] → (z, log_det) where log_det: [...]"""
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * torch.exp(self.scale_factor) * (1 - self.mask)
        t = t * (1 - self.mask)
        z = x * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """z: [..., D] → (x, log_det)"""
        z_masked = z * self.mask
        st = self.net(z_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * torch.exp(self.scale_factor) * (1 - self.mask)
        t = t * (1 - self.mask)
        x = (z - t) * torch.exp(-s)
        return x, -s.sum(dim=-1)


class ActNormFlat(nn.Module):
    """Data-dependent ActNorm for [..., D] tensors."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False))

    def initialize(self, x):
        with torch.no_grad():
            flat = x.reshape(-1, x.shape[-1])
            mean = flat.mean(0)
            std  = flat.std(0).clamp(min=1e-6)
            self.bias.data  = -mean / std
            self.scale.data = -torch.log(std)
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        z = x * torch.exp(self.scale) + self.bias
        log_det = self.scale.sum().expand(x.shape[:-1])
        return z, log_det

    def inverse(self, z):
        x = (z - self.bias) * torch.exp(-self.scale)
        log_det = -self.scale.sum().expand(z.shape[:-1])
        return x, log_det


class RealNVP1D(nn.Module):
    """
    Feature-wise RealNVP normalizing flow for time-series DiNof.

    Operates on each time-step's feature vector (dim=C=51) independently,
    sharing flow parameters across the L=64 time steps.

    This is much more efficient than flattening [B, C, 1, L] → [B, C*L]:
      - C=51 dim instead of C*L=3264 dim → tiny MLPs, fast convergence
      - Each time step gets its own NLL, averaged over L → better signal

    Interface matches MultiscaleFlow:
      - forward_kld(x) → (z, nll_loss)
      - sample(n)      → (samples_4d, log_q)
      - log_prob(x)    → (log_prob_per_sample, z)
    """

    def __init__(self, num_channels, window_len, num_coupling=8,
                 hidden_dim=128, num_layers=3):
        super().__init__()
        self.C = num_channels   # 51
        self.L = window_len     # 64
        D = num_channels        # flow operates in C-dim space

        self.flows    = nn.ModuleList()
        self.actnorms = nn.ModuleList()

        for i in range(num_coupling):
            mask = (torch.arange(D) % 2 == (i % 2)).float()
            self.actnorms.append(ActNormFlat(D))
            self.flows.append(MaskedAffineCoupling(
                D, mask, hidden_dim=hidden_dim, num_layers=num_layers
            ))

        self.register_buffer('prior_mean', torch.zeros(D))
        self.register_buffer('prior_std',  torch.ones(D))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _to_steps(self, x):
        """[B, C, 1, L] → [B*L, C]  (treat each time step as a sample)"""
        B, C, _, L = x.shape
        return x.squeeze(2).permute(0, 2, 1).reshape(B * L, C)

    def _from_steps(self, z, B):
        """[B*L, C] → [B, C, 1, L]"""
        BL, C = z.shape
        L = BL // B
        return z.reshape(B, L, C).permute(0, 2, 1).unsqueeze(2)

    def _prior_log_prob(self, z):
        """[..., C] → [...] standard Gaussian log prob"""
        C = z.shape[-1]
        return -0.5 * (C * np.log(2 * np.pi) + (z ** 2).sum(dim=-1))

    # ── flow forward / inverse ───────────────────────────────────────────────

    def _forward_flat(self, x_flat):
        """x_flat: [N, C] → (z, log_det) where log_det: [N]"""
        z = x_flat
        log_det = torch.zeros(x_flat.shape[0], device=x_flat.device)
        for actnorm, coupling in zip(self.actnorms, self.flows):
            z, ld1 = actnorm(z)
            log_det = log_det + ld1
            z, ld2 = coupling(z)
            log_det = log_det + ld2
        return z, log_det

    def _inverse_flat(self, z_flat):
        """z_flat: [N, C] → (x, log_det) where log_det: [N]"""
        x = z_flat
        log_det = torch.zeros(z_flat.shape[0], device=z_flat.device)
        for actnorm, coupling in zip(
                reversed(list(self.actnorms)), reversed(list(self.flows))):
            x, ld = coupling.inverse(x)
            log_det = log_det + ld
            x, ld2 = actnorm.inverse(x)
            log_det = log_det + ld2
        return x, log_det

    # ── public interface ─────────────────────────────────────────────────────

    def log_prob(self, x, y=None):
        """
        x: [B, C, 1, L]
        Returns: (log_prob [B], z [B*L, C])
        log_prob is averaged over the L time steps.
        """
        B = x.shape[0]
        x_flat = self._to_steps(x)          # [B*L, C]
        z, log_det = self._forward_flat(x_flat)
        log_pz = self._prior_log_prob(z)    # [B*L]
        log_px_per_step = log_pz + log_det  # [B*L]
        # Average over time steps to get per-sample log prob
        log_px = log_px_per_step.reshape(B, self.L).mean(dim=1)  # [B]
        return log_px, z

    def forward_kld(self, x, y=None):
        """
        Returns (z, nll_loss) where nll_loss is a scalar.
        Called as: _, loss_flow = model[1].forward_kld(perturbed_data_flow)
        """
        log_prob, z = self.log_prob(x)
        nll = -log_prob.mean()
        return z, nll

    def sample(self, num_samples, y=None, temperature=1.0):
        """
        Returns (samples_4d [num_samples, C, 1, L], log_q [num_samples])
        """
        with torch.no_grad():
            # Sample L time steps per sample → num_samples * L latents
            N = num_samples * self.L
            z = torch.randn(N, self.C,
                            device=self.prior_mean.device) * temperature
            x_flat, log_det = self._inverse_flat(z)
            log_pz = self._prior_log_prob(z)
            log_px_per_step = log_pz + log_det             # [N]
            log_px = log_px_per_step.reshape(num_samples, self.L).mean(1)  # [B]
            samples = self._from_steps(x_flat, num_samples)
        return samples, log_px
