import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.ts.swat_vpsde import get_config
from runner_ts import init_models_ts, setup_sde, restore_checkpoint
from losses import get_optimizer
from datasets_ts import get_ts_dataloaders, ts_data_scaler
from models import utils as mutils
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

cfg = get_config()
sm, flow, e0, e1 = init_models_ts(cfg)
opt = get_optimizer(cfg, list(sm.parameters()) + list(flow.parameters()))
state = dict(model=[sm, flow], optimizer=opt, ema=[e0, e1], step=0)
state = restore_checkpoint('runs/final/checkpoints/checkpoint_latest.pth', state, cfg.device)
e0.copy_to(sm.parameters()); e1.copy_to(flow.parameters())
sm.eval(); flow.eval()
sde, _ = setup_sde(cfg)
scaler = ts_data_scaler(cfg)
_, tl, _ = get_ts_dataloaders(cfg, 'data', 'synthetic')
score_fn = mutils.get_score_fn(sde, sm, train=False, continuous=True)

aurocs, f1s, precs, recs, thrs = [], [], [], [], []

for seed in range(10):
    torch.manual_seed(seed)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for bd in tl:
            batch = bd['image'].to(cfg.device).float()
            batch = scaler(batch)
            labels = bd['label'].numpy()
            B = batch.shape[0]
            t = torch.full((B,), 0.05, device=cfg.device)
            z = torch.randn_like(batch)
            mean, std = sde.marginal_prob(batch, t)
            x_noisy = mean + std[:, None, None, None] * z
            score = score_fn(x_noisy, t)
            error = (score * std[:, None, None, None] + z) ** 2
            err_per_sample = error.reshape(B, -1).mean(dim=1)
            all_scores.extend(err_per_sample.cpu().tolist())
            all_labels.extend(labels.tolist())
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_scores)
    thr = np.percentile(all_scores, 90)
    preds = (all_scores > thr).astype(int)
    f1 = f1_score(all_labels, preds)
    p = precision_score(all_labels, preds)
    r = recall_score(all_labels, preds)
    aurocs.append(auroc); f1s.append(f1); precs.append(p); recs.append(r); thrs.append(thr)
    print(f'Seed {seed}: AUROC={auroc:.4f}, F1={f1:.4f}, Prec={p:.4f}, Rec={r:.4f}, Thr={thr:.4f}')

aurocs, f1s, precs, recs = np.array(aurocs), np.array(f1s), np.array(precs), np.array(recs)
print('\n--- 10-Seed Statistics (Single Draw) ---')
print(f'AUROC    : {aurocs.mean():.4f} +/- {aurocs.std():.4f} (range {aurocs.min():.4f} - {aurocs.max():.4f})')
print(f'F1 Score : {f1s.mean():.4f} +/- {f1s.std():.4f}')
print(f'Precision: {precs.mean():.4f} +/- {precs.std():.4f}')
print(f'Recall   : {recs.mean():.4f} +/- {recs.std():.4f}')
