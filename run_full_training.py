"""
run_full_training.py
====================
Standalone script — runs the COMPLETE DiNof pipeline:
  1. Generate synthetic SWaT data
  2. Train 5000 steps (resumes from checkpoint if exists)
  3. Evaluate anomaly detection
  4. Save all 4 plots
  5. Write final_results.txt

Run from project root:
    python run_full_training.py

This script is self-contained and will keep running even if the chat
session is closed. It logs everything to runs/final/train.log.
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKDIR  = "runs/final"
DATA_DIR = "data"
STEPS    = 5000

os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(os.path.join(WORKDIR, "checkpoints"), exist_ok=True)

import logging
log = logging.getLogger("dinof")
log.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(WORKDIR, "train.log"), mode='a')
ch = logging.StreamHandler()
for h in [fh, ch]:
    h.setFormatter(logging.Formatter('%(asctime)s  %(message)s', '%H:%M:%S'))
    log.addHandler(h)

log.info("=" * 60)
log.info("  DiNof Time-Series — Full Training Run")
log.info("=" * 60)

# ── 1. Generate data ──────────────────────────────────────────────────────────
swat_dir = os.path.join(DATA_DIR, "swat_synthetic")
if not os.path.exists(os.path.join(swat_dir, "train_data.npy")):
    log.info("Generating synthetic SWaT data ...")
    from data.synthetic_swat import generate_swat_data
    generate_swat_data(output_dir=swat_dir)
    log.info("Data generated.")
else:
    log.info("Synthetic data already exists.")

# ── 2. Config ─────────────────────────────────────────────────────────────────
from configs.ts.swat_vpsde import get_config
config = get_config()
config.training.n_iters      = STEPS
config.training.batch_size   = 64
config.training.log_freq     = 100
config.training.snapshot_freq= 1000

import torch, numpy as np
device = config.device
log.info(f"Device: {device}  |  Steps: {STEPS}")

# ── 3. Models ─────────────────────────────────────────────────────────────────
from runner_ts import init_models_ts, setup_sde, save_checkpoint, restore_checkpoint
from losses import get_optimizer, optimization_manager
from losses_ts import get_step_fn_ts
from datasets_ts import get_ts_dataloaders, ts_data_scaler

score_model, flow, ema_score, ema_flow = init_models_ts(config, log)
sde, _ = setup_sde(config)
optimizer = get_optimizer(config, list(score_model.parameters()) + list(flow.parameters()))
state = dict(model=[score_model, flow], optimizer=optimizer,
             ema=[ema_score, ema_flow], step=0)

ckpt_latest = os.path.join(WORKDIR, "checkpoints", "checkpoint_latest.pth")
if os.path.exists(ckpt_latest):
    state = restore_checkpoint(ckpt_latest, state, device, log)

# ── 4. Training loop ──────────────────────────────────────────────────────────
w = config.optim.loss_weights
opt_fn   = optimization_manager(config)
step_fn  = get_step_fn_ts(sde, train=True, optimize_fn=opt_fn,
               reduce_mean=config.training.reduce_mean,
               continuous=config.training.continuous,
               start=config.flow.start_step, w=w)
scaler   = ts_data_scaler(config)
train_loader, _, _ = get_ts_dataloaders(config, DATA_DIR, 'synthetic')

score_losses, flow_losses = [], []
data_iter = iter(train_loader)
start_step = state['step']

log.info(f"Training from step {start_step} to {STEPS} ...")

for step in range(start_step, STEPS):
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
        log.info(f"Step {step:5d}/{STEPS}  score={sl:.4f}  flow={fl:.4f}")

    if step % config.training.snapshot_freq == 0 and step > 0:
        p = os.path.join(WORKDIR, "checkpoints", f"checkpoint_{step}.pth")
        save_checkpoint(p, state)
        save_checkpoint(ckpt_latest, state)
        log.info(f"Checkpoint saved at step {step}")

save_checkpoint(ckpt_latest, state)
log.info("Training complete!")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
log.info("Evaluating anomaly detection ...")

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ema_score.copy_to(score_model.parameters())
ema_flow.copy_to(flow.parameters())
score_model.eval(); flow.eval()

_, test_loader, _ = get_ts_dataloaders(config, DATA_DIR, 'synthetic')
eps = 1e-5
all_scores, all_labels = [], []

with torch.no_grad():
    for bd in test_loader:
        batch  = bd['image'].to(device).float()
        batch  = scaler(batch)
        labels = bd['label'].numpy()
        t_val  = config.flow.start_step * sde.T - eps
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
    'score_mean_normal':  float(all_scores[all_labels==0].mean()),
    'score_mean_anomaly': float(all_scores[all_labels==1].mean()),
}

log.info("=" * 55)
log.info("  ANOMALY DETECTION RESULTS")
log.info("=" * 55)
for k, v in results.items():
    log.info(f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k:28s}: {v}")
log.info("=" * 55)

# Save to text file
with open(os.path.join(WORKDIR, "final_results.txt"), "w") as f:
    f.write("DiNof Time-Series Anomaly Detection — Final Results\n")
    f.write("=" * 55 + "\n")
    for k, v in results.items():
        f.write(f"  {k:28s}: {v:.4f}\n" if isinstance(v, float) else f"  {k:28s}: {v}\n")
    f.write("\nKey insight (panel feedback addressed):\n")
    f.write("  Anomaly scoring = single NF forward pass (~ms/window)\n")
    f.write("  No iterative reverse diffusion needed at test time.\n")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
log.info("Generating plots ...")
try:
    test_data = np.load(os.path.join(swat_dir, "test_data.npy"))
except Exception:
    test_data = None

from visualize_results import plot_all
plot_all(
    workdir=WORKDIR,
    score_losses=score_losses,
    flow_losses=flow_losses,
    all_scores=all_scores,
    all_labels=all_labels,
    test_data=test_data,
    threshold=results['Threshold'],
    auroc=results['AUROC'],
)

log.info("=" * 55)
log.info(f"  F1 = {results['F1']:.4f}")
log.info(f"  AUROC = {results['AUROC']:.4f}")
log.info(f"  All plots saved to {WORKDIR}/")
log.info("DONE — open runs/final/ to see all results.")
log.info("=" * 55)
