"""
runner_ts.py — Full training + anomaly evaluation runner for DiNof time-series.

Adapts the base paper's DiNof framework (Zand et al. 2024) from image generation
(CIFAR-10 / CelebA) to time-series anomaly detection on SWaT/WADI.

Key design decisions:
  - Same Tm-based architecture (flow at t=Tm, score model for t<Tm)
  - Score model: ScoreModel1D (1D U-Net, [B,C,1,L] 4D tensors)
  - Flow model:  RealNVP1D (MLP coupling, [B,C,1,L] → [B,C*L] → back)
  - Anomaly score: -log p_flow(x_Tm)  → single forward pass, real-time capable
  - Addresses panel feedback: NF inference is O(D) MLPs, not iterative diffusion

Supervisor's research gap addressed:
  "New Domain + Conditional Generation"
  → New domain: SWaT water-treatment sensor data (time series)
  → Latency fix: Flow-only inference avoids full reverse diffusion at test time
"""

import os
import logging
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from sde_lib import VPSDE
from models.score_model_1d import ScoreModel1D
from models.flow_1d import RealNVP1D
from losses import get_optimizer, optimization_manager
from losses_ts import get_step_fn_ts
from datasets_ts import get_ts_dataloaders, ts_data_scaler


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(workdir, name='dinof_ts'):
    os.makedirs(workdir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ── Checkpoint utilities ────────────────────────────────────────────────────────

def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'step':          state['step'],
        'model_score':   state['model'][0].state_dict(),
        'model_flow':    state['model'][1].state_dict(),
        'optimizer':     state['optimizer'].state_dict(),
        'ema_score':     state['ema'][0].state_dict(),
        'ema_flow':      state['ema'][1].state_dict(),
    }
    torch.save(payload, path)


def restore_checkpoint(path, state, device, logger=None):
    payload = torch.load(path, map_location=device, weights_only=False)
    state['step'] = payload.get('step', 0)
    state['model'][0].load_state_dict(payload['model_score'])
    state['model'][1].load_state_dict(payload['model_flow'])
    state['optimizer'].load_state_dict(payload['optimizer'])
    state['ema'][0].load_state_dict(payload['ema_score'])
    state['ema'][1].load_state_dict(payload['ema_flow'])
    if logger:
        logger.info(f'Restored checkpoint from {path} (step {state["step"]})')
    return state


# ── Model initialisation ────────────────────────────────────────────────────────

def setup_sde(config):
    """Return (sde, sampling_eps) based on config.training.sde."""
    sde_name = getattr(config.training, 'sde', 'vpsde').lower()
    if sde_name == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min,
                    beta_max=config.model.beta_max,
                    N=config.model.num_scales)
        sampling_eps = 1e-3
    else:
        raise ValueError(f'Unsupported SDE: {sde_name}. Use vpsde.')
    return sde, sampling_eps


def init_models_ts(config, logger=None):
    """
    Initialise ScoreModel1D + RealNVP1D and their EMA trackers.
    Returns (score_model, flow, ema_score, ema_flow) all on config.device.
    """
    device = config.device

    score_model = ScoreModel1D(config).to(device)
    flow = RealNVP1D(
        num_channels=config.data.num_channels,
        window_len=config.data.window_len,
        num_coupling=config.flow.num_coupling,
        hidden_dim=config.flow.hidden_dim,
        num_layers=config.flow.num_mlp_layers,
    ).to(device)

    ema_score = ExponentialMovingAverage(score_model.parameters(),
                                         decay=config.model.ema_rate)
    ema_flow  = ExponentialMovingAverage(flow.parameters(),
                                         decay=config.flow.ema_rate)

    n_score = sum(p.numel() for p in score_model.parameters())
    n_flow  = sum(p.numel() for p in flow.parameters())
    if logger:
        logger.info(f'ScoreModel1D params : {n_score:,}')
        logger.info(f'RealNVP1D params    : {n_flow:,}')

    return score_model, flow, ema_score, ema_flow


# ── Training ────────────────────────────────────────────────────────────────────

def train_ts(config, workdir, data_dir, source='synthetic', logger=None):
    """
    Full training loop for DiNof time-series.

    Args:
        config:   Config from configs.ts.swat_vpsde.get_config()
        workdir:  Directory for checkpoints and logs.
        data_dir: Root data directory (must contain swat_synthetic/ if source='synthetic').
        source:   'synthetic' | 'csv'  (see datasets_ts.py)
        logger:   Optional logger; created if None.

    Returns:
        (score_losses, flow_losses) — lists of per-step values
    """
    if logger is None:
        logger = get_logger(workdir)

    device = config.device
    logger.info(f'Device: {device}')
    logger.info(f'SDE: {config.training.sde.upper()}  Tm={config.flow.start_step}')

    # ── Data ────────────────────────────────────────────────────────────────────
    train_loader, _, _ = get_ts_dataloaders(config, data_dir, source)
    scaler = ts_data_scaler(config)
    logger.info(f'Train batches per epoch: {len(train_loader)}')

    # ── Models + SDE ────────────────────────────────────────────────────────────
    score_model, flow, ema_score, ema_flow = init_models_ts(config, logger)
    sde, _ = setup_sde(config)

    optimizer = get_optimizer(
        config,
        list(score_model.parameters()) + list(flow.parameters())
    )
    state = dict(
        model=[score_model, flow],
        optimizer=optimizer,
        ema=[ema_score, ema_flow],
        step=0,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────────────
    ckpt_dir = os.path.join(workdir, 'checkpoints')
    latest_ckpt = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
    if os.path.exists(latest_ckpt):
        state = restore_checkpoint(latest_ckpt, state, device, logger)

    # ── Loss function ────────────────────────────────────────────────────────────
    w = getattr(config.optim, 'loss_weights', [1.0, 1.0])
    optimize_fn = optimization_manager(config)
    step_fn = get_step_fn_ts(
        sde, train=True, optimize_fn=optimize_fn,
        reduce_mean=config.training.reduce_mean,
        continuous=config.training.continuous,
        start=config.flow.start_step,
        w=w,
    )

    # ── Loop ────────────────────────────────────────────────────────────────────
    score_losses, flow_losses = [], []
    data_iter = iter(train_loader)
    n_iters   = config.training.n_iters

    logger.info(f'Starting training for {n_iters} steps …')

    for step in range(state['step'], n_iters):
        try:
            batch_dict = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_dict = next(data_iter)

        batch = batch_dict['image'].to(device).float()
        batch = scaler(batch)

        sl, fl = step_fn(state, batch)
        score_losses.append(sl)
        flow_losses.append(fl)

        if step % config.training.log_freq == 0:
            logger.info(
                f'Step {step:6d}/{n_iters}  '
                f'score_loss={sl:.4f}  flow_loss={fl:.4f}'
            )

        if step % config.training.snapshot_freq == 0 and step > 0:
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{step}.pth')
            save_checkpoint(ckpt_path, state)
            save_checkpoint(latest_ckpt, state)
            logger.info(f'Saved checkpoint at step {step}')

    # Final save
    save_checkpoint(latest_ckpt, state)
    logger.info('Training complete.')

    return score_losses, flow_losses


# ── Anomaly Evaluation ─────────────────────────────────────────────────────────

def evaluate_anomaly_ts(config, workdir, data_dir,
                         source='synthetic', ckpt_path=None, logger=None):
    """
    Evaluate anomaly detection on the test set.

    Anomaly score = -log p_flow(x_Tm)
    This is a SINGLE forward pass through the NF — real-time capable.
    (Addresses the panel's inference-latency concern.)

    Returns:
        results dict with F1, Precision, Recall, AUROC, threshold
        all_scores  np.array [N]
        all_labels  np.array [N]
    """
    from sklearn.metrics import (f1_score, precision_score,
                                 recall_score, roc_auc_score)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    if logger is None:
        logger = get_logger(workdir)

    device = config.device

    # ── Load models ─────────────────────────────────────────────────────────────
    score_model, flow, ema_score, ema_flow = init_models_ts(config, logger)
    optimizer = get_optimizer(
        config,
        list(score_model.parameters()) + list(flow.parameters())
    )
    state = dict(model=[score_model, flow],
                 optimizer=optimizer,
                 ema=[ema_score, ema_flow],
                 step=0)

    if ckpt_path is None:
        ckpt_path = os.path.join(workdir, 'checkpoints', 'checkpoint_latest.pth')

    if os.path.exists(ckpt_path):
        state = restore_checkpoint(ckpt_path, state, device, logger)
    else:
        logger.warning(f'No checkpoint at {ckpt_path}; using random weights.')

    # Apply EMA weights for clean evaluation
    ema_score.copy_to(score_model.parameters())
    ema_flow.copy_to(flow.parameters())
    score_model.eval()
    flow.eval()

    # ── SDE + data ──────────────────────────────────────────────────────────────
    sde, _ = setup_sde(config)
    scaler  = ts_data_scaler(config)
    _, test_loader, _ = get_ts_dataloaders(config, data_dir, source)

    # ── Compute anomaly scores ───────────────────────────────────────────────────
    all_scores, all_labels = [], []
    logger.info('Computing anomaly scores …')

    with torch.no_grad():
        for batch_dict in tqdm(test_loader, desc='Evaluating'):
            batch  = batch_dict['image'].to(device).float()
            batch  = scaler(batch)
            labels = batch_dict['label'].numpy()

            # Perturb to t = Tm
            eps = 1e-5
            t_val  = config.flow.start_step * sde.T - eps
            t_flow = torch.full((batch.shape[0],), t_val, device=device)
            mean_f, std_f = sde.marginal_prob(batch, t_flow)
            x_tm = mean_f + std_f[:, None, None, None] * torch.randn_like(batch)

            # Anomaly score = -log p_flow(x_Tm)
            log_prob, _ = flow.log_prob(x_tm)
            scores = -log_prob.cpu().numpy()

            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    if all_labels.sum() == 0:
        logger.warning('No anomalies in test set! Check data.')
        return {}, all_scores, all_labels

    # ── Threshold + metrics ──────────────────────────────────────────────────────
    threshold = np.percentile(all_scores, 90)   # top 10% flagged as anomaly
    preds = (all_scores > threshold).astype(int)

    results = {
        'F1':                 f1_score(all_labels, preds, zero_division=0),
        'Precision':          precision_score(all_labels, preds, zero_division=0),
        'Recall':             recall_score(all_labels, preds, zero_division=0),
        'AUROC':              roc_auc_score(all_labels, all_scores),
        'threshold':          float(threshold),
        'n_anomalies_true':   int(all_labels.sum()),
        'n_anomalies_pred':   int(preds.sum()),
        'n_total':            len(all_labels),
        'score_mean_normal':  float(all_scores[all_labels == 0].mean()),
        'score_mean_anomaly': float(all_scores[all_labels == 1].mean()),
    }

    logger.info('\n' + '=' * 55)
    logger.info('  DiNof TIME-SERIES ANOMALY DETECTION RESULTS')
    logger.info('=' * 55)
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f'  {k:28s}: {v:.4f}')
        else:
            logger.info(f'  {k:28s}: {v}')
    logger.info('=' * 55)

    return results, all_scores, all_labels
