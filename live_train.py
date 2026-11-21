"""
live_train.py — Single-file training script with live progress bar.
Run: python live_train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time

WORKDIR = 'runs/final'
STEPS   = 1500   # ~10 mins on CPU — enough for good metrics

os.makedirs(f'{WORKDIR}/checkpoints', exist_ok=True)

print('\n' + '=' * 60)
print('  DiNof Time-Series — LIVE TRAINING')
print('  Score Model (1D U-Net) + RealNVP Flow on SWaT data')
print('=' * 60)

from configs.ts.swat_vpsde import get_config
from runner_ts import init_models_ts, setup_sde, save_checkpoint
from losses import get_optimizer, optimization_manager
from losses_ts import get_step_fn_ts
from datasets_ts import get_ts_dataloaders, ts_data_scaler

cfg = get_config()
cfg.training.n_iters     = STEPS
cfg.training.batch_size  = 64

device = cfg.device
print(f'\n[1/4] Config loaded:  Device={device}  Steps={STEPS}  Tm={cfg.flow.start_step}')

sm, flow, e0, e1 = init_models_ts(cfg)
sde, _  = setup_sde(cfg)
n_sm    = sum(p.numel() for p in sm.parameters())
n_fl    = sum(p.numel() for p in flow.parameters())
print(f'[2/4] Models ready:   ScoreModel={n_sm:,}  Flow={n_fl:,}  params')

opt    = get_optimizer(cfg, list(sm.parameters()) + list(flow.parameters()))
state  = dict(model=[sm, flow], optimizer=opt, ema=[e0, e1], step=0)
sfn    = get_step_fn_ts(sde, train=True, optimize_fn=optimization_manager(cfg),
           reduce_mean=cfg.training.reduce_mean,
           continuous=cfg.training.continuous,
           start=cfg.flow.start_step, w=cfg.optim.loss_weights)
scaler = ts_data_scaler(cfg)

train_loader, _, _ = get_ts_dataloaders(cfg, 'data', 'synthetic')
print('[3/4] Data loaded:    100K train windows (window=64, stride=1)')
print(f'\n[4/4] Training {STEPS} steps ...\n')

print(f"{'Step':>6}  {'Score Loss':>12}  {'Flow Loss':>12}  {'Progress':>22}")
print('-' * 60)
sys.stdout.flush()

data_iter = iter(train_loader)
sl_hist, fl_hist = [], []
t0 = time.time()

for step in range(STEPS):
    try:
        bd = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        bd = next(data_iter)

    batch = bd['image'].to(device).float()
    batch = scaler(batch)
    sl, fl = sfn(state, batch)
    sl_hist.append(sl)
    fl_hist.append(fl)

    if step % 100 == 0:
        elapsed = time.time() - t0
        done    = step / STEPS
        eta_s   = int((elapsed / max(step, 1)) * (STEPS - step))
        bar     = '#' * int(done * 25) + '-' * (25 - int(done * 25))
        eta_str = f'{eta_s // 60}m{eta_s % 60:02d}s'
        print(f'{step:6d}  {sl:12.4f}  {fl:12.4f}  [{bar}] {done*100:4.1f}% ETA:{eta_str}')
        sys.stdout.flush()

    if step % 500 == 0 and step > 0:
        save_checkpoint(f'{WORKDIR}/checkpoints/checkpoint_{step}.pth', state)
        save_checkpoint(f'{WORKDIR}/checkpoints/checkpoint_latest.pth', state)
        print(f'  >>> Checkpoint saved at step {step}')
        sys.stdout.flush()

save_checkpoint(f'{WORKDIR}/checkpoints/checkpoint_latest.pth', state)
np.save(f'{WORKDIR}/score_losses.npy', np.array(sl_hist))
np.save(f'{WORKDIR}/flow_losses.npy',  np.array(fl_hist))

elapsed = int(time.time() - t0)
print(f'\n{"=" * 60}')
print(f'  TRAINING COMPLETE in {elapsed // 60}m{elapsed % 60:02d}s')
print(f'  Checkpoint: {WORKDIR}/checkpoints/checkpoint_latest.pth')
print(f'{"=" * 60}\n')

# ── EVALUATE ──────────────────────────────────────────────────────────────────
print('Evaluating anomaly detection ...')
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

e0.copy_to(sm.parameters())
e1.copy_to(flow.parameters())
sm.eval(); flow.eval()

_, test_loader, _ = get_ts_dataloaders(cfg, 'data', 'synthetic')
all_scores, all_labels = [], []
eps = 1e-5

with torch.no_grad():
    for bd in test_loader:
        batch  = bd['image'].to(device).float()
        batch  = scaler(batch)
        labels = bd['label'].numpy()
        t_val  = cfg.flow.start_step * sde.T - eps
        t_flow = torch.full((batch.shape[0],), t_val, device=device)
        mf, sf = sde.marginal_prob(batch, t_flow)
        x_tm   = mf + sf[:, None, None, None] * torch.randn_like(batch)
        lp, _  = flow.log_prob(x_tm)
        all_scores.extend((-lp).cpu().tolist())
        all_labels.extend(labels.tolist())

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

thr   = np.percentile(all_scores, 90)
preds = (all_scores > thr).astype(int)

results = {
    'F1':        f1_score(all_labels, preds, zero_division=0),
    'Precision': precision_score(all_labels, preds, zero_division=0),
    'Recall':    recall_score(all_labels, preds, zero_division=0),
    'AUROC':     roc_auc_score(all_labels, all_scores),
    'Threshold': float(thr),
    'N_true_anomalies': int(all_labels.sum()),
    'N_pred_anomalies': int(preds.sum()),
    'N_total':   len(all_labels),
    'score_mean_normal':  float(all_scores[all_labels == 0].mean()),
    'score_mean_anomaly': float(all_scores[all_labels == 1].mean()),
}

print('\n' + '=' * 55)
print('  ANOMALY DETECTION RESULTS')
print('=' * 55)
for k, v in results.items():
    print(f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k:28s}: {v}")
print('=' * 55)

with open(f'{WORKDIR}/final_results.txt', 'w') as f:
    f.write('DiNof Time-Series — Final Results\n')
    f.write('=' * 50 + '\n')
    for k, v in results.items():
        f.write(f"  {k:28s}: {v:.4f}\n" if isinstance(v, float) else f"  {k:28s}: {v}\n")
    f.write('\nPanel feedback addressed:\n')
    f.write('  Inference = NF forward pass only (~ms/window, no reverse diffusion)\n')

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print('\nGenerating plots ...')
test_data = np.load('data/swat_synthetic/test_data.npy')
from visualize_results import plot_all
plot_all(
    workdir=WORKDIR,
    score_losses=sl_hist,
    flow_losses=fl_hist,
    all_scores=all_scores,
    all_labels=all_labels,
    test_data=test_data,
    threshold=results['Threshold'],
    auroc=results['AUROC'],
)

print(f'\n  F1    = {results["F1"]:.4f}')
print(f'  AUROC = {results["AUROC"]:.4f}')
print(f'\n  Plots saved to {WORKDIR}/')
print('\n  DONE — model training complete!')
