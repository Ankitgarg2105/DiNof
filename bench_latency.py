import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.ts.swat_vpsde import get_config
from runner_ts import init_models_ts, setup_sde, restore_checkpoint
from losses import get_optimizer
from datasets_ts import ts_data_scaler
from models import utils as mutils

cfg = get_config(); device = cfg.device
sm, flow, e0, e1 = init_models_ts(cfg)
opt = get_optimizer(cfg, list(sm.parameters()) + list(flow.parameters()))
state = dict(model=[sm, flow], optimizer=opt, ema=[e0, e1], step=0)
state = restore_checkpoint('runs/final/checkpoints/checkpoint_latest.pth', state, device)
print('Model checkpoint loaded: 1D U-Net + RealNVP1D')
e0.copy_to(sm.parameters()); e1.copy_to(flow.parameters()); sm.eval(); flow.eval()
sde, _ = setup_sde(cfg); scaler = ts_data_scaler(cfg)
score_fn = mutils.get_score_fn(sde, sm, train=False, continuous=True)
npar = lambda m: sum(p.numel() for p in m.parameters())
print('params: score', npar(sm), 'flow', npar(flow), 'threads', torch.get_num_threads())

test = np.load('data/swat_synthetic/test_data.npy'); W = 64
def prep(i):
    x = torch.tensor(test[i:i+W].T, dtype=torch.float32)[None, :, None, :].to(device)
    return scaler(x)
t = torch.full((1,), 0.05, device=device)

def score(x):
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    s = score_fn(mean + std[:, None, None, None] * z, t)
    return (((s * std[:, None, None, None] + z) ** 2).mean()).item()

with torch.no_grad():
    for i in range(50): score(prep(i))                       # warm-up
    lat = []
    for i in range(1000):
        t0 = time.perf_counter(); score(prep(i)); lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    print(f'per-window ms: mean {lat.mean():.2f} median {np.median(lat):.2f} '
          f'p95 {np.percentile(lat,95):.2f} p99 {np.percentile(lat,99):.2f} std {lat.std():.2f}')
    print(f'throughput: {1000 / (lat.sum() / 1000):.1f} windows/s (sequential, batch=1)')
    x = prep(0)
    for n in (100, 300, 600, 1000):                          # cost of n network passes
        t0 = time.perf_counter()
        for _ in range(n): score_fn(x, t)
        print(f'{n:5d} score-network evaluations: {(time.perf_counter() - t0) * 1000:.1f} ms')
