"""
1D Score Model (U-Net) for time-series DiNof.
Input/Output: [B, num_channels, 1, window_len]  (4D, height=1 for SDE compat.)
Time embedding: sinusoidal + 2-layer MLP → added to each residual block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Register with models.utils registry (import guard avoids circular import at test time)
try:
    from models.utils import register_model as _register_model
    _REGISTER = _register_model
except ImportError:
    def _REGISTER(cls=None, *, name=None):
        return cls if cls is not None else (lambda c: c)


# ─── Time Embedding ───────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps, embedding_dim: int):
    """Sinusoidal time embedding. timesteps: [B], returns [B, embedding_dim]"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        emb = get_timestep_embedding(t, self.dim)
        return self.mlp(emb)  # [B, dim*4]


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    1D Residual block with time conditioning.
    Operates on [B, C, L] tensors.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        # GroupNorm: use min(8, out_ch) groups so it always divides out_ch
        g1 = self._safe_groups(in_ch)
        g2 = self._safe_groups(out_ch)
        self.norm1    = nn.GroupNorm(g1, in_ch)
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.time_proj= nn.Linear(time_emb_dim, out_ch)
        self.norm2    = nn.GroupNorm(g2, out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.dropout  = nn.Dropout(dropout)
        self.act      = nn.SiLU()
        self.skip     = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _safe_groups(c):
        for g in [8, 4, 2, 1]:
            if c % g == 0:
                return g
        return 1

    def forward(self, x, t_emb):
        """x: [B, C, L], t_emb: [B, time_emb_dim]"""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.act(self.time_proj(t_emb))[:, :, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttnBlock1D(nn.Module):
    """Multi-head self-attention over the time dimension."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        g = ResBlock1D._safe_groups(channels)
        self.norm = nn.GroupNorm(g, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        """x: [B, C, L]"""
        h = self.norm(x).permute(0, 2, 1)   # [B, L, C]
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)       # [B, C, L]


# ─── Full U-Net Score Model ────────────────────────────────────────────────────

@_REGISTER(name='ScoreModel1D')
class ScoreModel1D(nn.Module):
    """
    1D U-Net score model for time-series DiNof.

    Input/Output: [B, num_channels, 1, window_len]
    Internally squeezes height=1 → [B, num_channels, window_len].

    Architecture (3 levels):
      Input proj → Encoder (Down×2) → Bottleneck → Decoder (Up×2) → Output proj
    """

    def __init__(self, config):
        super().__init__()

        C    = config.data.num_channels   # 51 for SWaT
        nf   = config.model.nf            # 64
        nr   = config.model.num_res_blocks # 2
        mults= list(config.model.ch_mult)  # [1,2,2]
        drop = config.model.dropout
        tdim = nf * 4

        self.time_embedding = TimeEmbedding(nf)

        # ── Input projection ────────────────────────────────────────────────────
        self.in_proj = nn.Conv1d(C, nf, 3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────────
        # Level 0: nf → nf*mults[0],  no downsample
        # Level 1: nf*mults[0] → nf*mults[1], Downsample
        # Level 2: nf*mults[1] → nf*mults[2], Downsample
        enc_chs = [nf]  # channel after each encoder level
        for m in mults:
            enc_chs.append(nf * m)
        # enc_chs = [64, 64, 128, 128]  for mults=[1,2,2]

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = nf
        for i, m in enumerate(mults):
            ch_out = nf * m
            blks = nn.ModuleList()
            for _ in range(nr):
                blks.append(ResBlock1D(ch, ch_out, tdim, dropout=drop))
                ch = ch_out
            self.enc.append(blks)
            if i < len(mults) - 1:
                self.down.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down.append(nn.Identity())

        # ── Bottleneck ──────────────────────────────────────────────────────────
        self.bot = nn.ModuleList([
            ResBlock1D(ch, ch, tdim, dropout=drop),
            AttnBlock1D(ch),
            ResBlock1D(ch, ch, tdim, dropout=drop),
        ])

        # ── Decoder ─────────────────────────────────────────────────────────────
        # Mirror of encoder.  Skip connections concat enc output of same level.
        # ch_in at decoder level i = ch_after_bottleneck + skip_ch from enc level (nlvl-1-i)
        self.dec   = nn.ModuleList()
        self.up    = nn.ModuleList()
        self.skips_ch = list(reversed(enc_chs[1:]))  # skip channels per dec level

        for i, (m, skip_ch) in enumerate(zip(reversed(mults), self.skips_ch)):
            ch_out_dec = nf * m
            # upsample if not the first decoder level (which corresponds to deepest enc level)
            if i > 0:
                self.up.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(ch, ch, 3, padding=1),
                ))
            else:
                self.up.append(nn.Identity())

            blks = nn.ModuleList()
            ch_in_first = ch + skip_ch   # after concat
            for j in range(nr + 1):
                in_c = ch_in_first if j == 0 else ch_out_dec
                blks.append(ResBlock1D(in_c, ch_out_dec, tdim, dropout=drop))
            ch = ch_out_dec
            self.dec.append(blks)

        # Also skip from input projection (level 0 → decoder last level)
        self.dec_skip0_ch = enc_chs[0]  # nf

        # Final extra level to get back to nf using skip from in_proj
        ch_in_last = ch + self.dec_skip0_ch
        self.final_res = ResBlock1D(ch_in_last, nf, tdim, dropout=drop)

        # ── Output projection ────────────────────────────────────────────────────
        g = ResBlock1D._safe_groups(nf)
        self.out_norm = nn.GroupNorm(g, nf)
        self.out_proj = nn.Conv1d(nf, C, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)

    def forward(self, x, time_cond, *args, **kwargs):
        """
        x:         [B, C, 1, L]
        time_cond: [B]
        Returns:   [B, C, 1, L]
        """
        B, C, H, L = x.shape
        assert H == 1, f'Expected H=1, got {H}'
        x1d = x.squeeze(2)           # [B, C, L]

        t = self.time_embedding(time_cond)  # [B, nf*4]

        # Input proj
        h = self.in_proj(x1d)        # [B, nf, L]
        h0 = h                       # save for final skip

        # Encoder
        enc_outs = []
        for blks, ds in zip(self.enc, self.down):
            for blk in blks:
                h = blk(h, t)
            enc_outs.append(h)       # save before downsampling
            h = ds(h)

        # Bottleneck
        for blk in self.bot:
            if isinstance(blk, ResBlock1D):
                h = blk(h, t)
            else:
                h = blk(h)

        # Decoder
        for i, (blks, us) in enumerate(zip(self.dec, self.up)):
            h = us(h)
            skip = enc_outs[-(i+1)]
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            for j, blk in enumerate(blks):
                h = blk(h, t)

        # Final level with in_proj skip
        if h.shape[-1] != h0.shape[-1]:
            h = F.interpolate(h, size=h0.shape[-1], mode='nearest')
        h = torch.cat([h, h0], dim=1)
        h = self.final_res(h, t)

        h = self.out_proj(F.silu(self.out_norm(h)))  # [B, C, L]
        return h.unsqueeze(2)                         # [B, C, 1, L]
