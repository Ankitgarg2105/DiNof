"""
losses_ts.py — Training losses for DiNof time-series adaptation.

Reuses the existing losses.py logic (which already handles 4D [B,C,1,L] tensors
via std[:, None, None, None] broadcasting), but provides a clean wrapper for
the TS runner that imports only what's needed.

The core loss function is taken directly from losses.get_sde_loss_fn():
  L_total = w_score * L_score + w_flow * L_flow

where:
  - L_score = score-matching loss (denoising score matching, DSM)
  - L_flow  = -log p_flow(x_Tm)  (NLL of perturbed data under NF at t=Tm)
  - Tm = start_step * T          (the DiNof mixing timestep)
"""

import torch
import numpy as np
from models import utils as mutils
from sde_lib import VPSDE, VESDE


# ── Reuse existing optimizer utilities ─────────────────────────────────────────
from losses import get_optimizer, optimization_manager   # noqa: F401


def get_step_fn_ts(sde, train, optimize_fn, reduce_mean=True,
                   continuous=True, start=0.6, w=None):
    """
    Create a one-step train/eval function for time-series DiNof.

    Args:
        sde:          An sde_lib.SDE object (VPSDE recommended).
        train:        True for training mode, False for eval.
        optimize_fn:  Returned by optimization_manager(config).
        reduce_mean:  Average loss over dimensions (True) or sum (False).
        continuous:   Continuous-time SDE (True).
        start:        Tm fraction — flow is applied at t = start * T.
                      Paper default: 0.6
        w:            [w_score, w_flow] loss weights. Default [1.0, 1.0].

    Returns:
        step_fn(state, batch) → (score_loss, flow_loss)  both scalars
    """
    if w is None:
        w = [1.0, 1.0]
    w_score, w_flow = w

    reduce_op = (torch.mean if reduce_mean
                 else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs))

    eps = 1e-5  # smallest time step

    def step_fn(state, batch):
        """
        One training / evaluation step.

        Args:
            state: dict with keys 'model' ([score_model, flow]),
                   'optimizer', 'ema' ([ema_score, ema_flow]), 'step'
            batch: [B, C, 1, L] float tensor on device

        Returns:
            (score_loss, flow_loss) — both Python floats via .item()
        """
        model_list = state['model']   # [score_model, flow]
        optimizer  = state['optimizer']
        ema_list   = state['ema']     # [ema_score, ema_flow]
        step       = state['step']

        # ── Forward pass ────────────────────────────────────────────────────────
        if train:
            for m in model_list:
                m.train()
            optimizer.zero_grad()
        else:
            for m in model_list:
                m.eval()

        score_model, flow = model_list[0], model_list[1]

        with torch.set_grad_enabled(train):
            # ── Score loss (DSM over [eps, Tm * T]) ─────────────────────────────
            score_fn = mutils.get_score_fn(sde, score_model,
                                           train=train,
                                           continuous=continuous)

            B = batch.shape[0]
            z = torch.randn_like(batch)

            # Sample t uniformly in [eps, Tm*T]
            t = torch.rand(B, device=batch.device) * (start * sde.T - eps) + eps

            mean, std = sde.marginal_prob(batch, t)
            perturbed = mean + std[:, None, None, None] * z
            score = score_fn(perturbed, t)

            if not False:  # likelihood_weighting=False → use standard weighting
                losses = torch.square(score * std[:, None, None, None] + z)
                losses = reduce_op(losses.reshape(B, -1), dim=-1)
            score_loss = torch.mean(losses)

            # ── Flow loss (NLL at t = Tm) ────────────────────────────────────────
            t_flow = (start * torch.ones(B, device=batch.device)) * (sde.T - eps) + eps
            mean_f, std_f = sde.marginal_prob(batch, t_flow)
            x_tm = mean_f + std_f[:, None, None, None] * torch.randn_like(batch)
            _, flow_loss = flow.forward_kld(x_tm)

            # ── Combined loss ────────────────────────────────────────────────────
            total_loss = w_score * score_loss + w_flow * flow_loss

        # ── Backward + step ──────────────────────────────────────────────────────
        if train:
            total_loss.backward()
            optimize_fn(optimizer,
                        list(score_model.parameters()) + list(flow.parameters()),
                        step=step)
            # Update EMA
            ema_list[0].update(score_model.parameters())
            ema_list[1].update(flow.parameters())
            state['step'] += 1

        return score_loss.item(), flow_loss.item()

    return step_fn
