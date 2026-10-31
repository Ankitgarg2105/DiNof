"""
SWaT VPSDE config for DiNof time-series anomaly detection.
Applies the same Tm-based flow-guided diffusion from the base paper
to SWaT water-treatment sensor data (51 features, window=64).
"""
import ml_collections
from configs.default_ts import get_default_configs


def get_config():
    config = get_default_configs()

    # ── SDE ─────────────────────────────────────────────────────────────────────
    config.training.sde = 'vpsde'
    config.training.continuous = True
    config.training.reduce_mean = True
    config.training.likelihood_weighting = False

    # ── Data overrides ───────────────────────────────────────────────────────────
    config.data.dataset = 'SWaT'
    config.data.num_channels = 51
    config.data.window_len = 64
    config.data.image_size = 64   # kept for SDE compat

    # ── Score model ──────────────────────────────────────────────────────────────
    config.model.name = 'ScoreModel1D'
    config.model.nf = 32               # smaller = 2x faster on CPU
    config.model.num_res_blocks = 1    # 1 res block per level (was 2)
    config.model.attn_resolutions = [1]
    config.model.ch_mult = (1, 2, 2)
    config.model.dropout = 0.1
    config.model.beta_min = 0.1
    config.model.beta_max = 20.
    config.model.ema_rate = 0.999
    config.model.scale_by_sigma = False

    # ── Flow ─────────────────────────────────────────────────────────────────────
    # Tm = 0.6  — the key DiNof hyperparameter (best on CIFAR: 0.6)
    config.flow.start_step = 0.6
    config.flow.num_coupling = 8
    config.flow.hidden_dim = 128   # feature-wise flow: C=51 dim, 128 hidden is plenty
    config.flow.num_mlp_layers = 3
    config.flow.ema_rate = 0.999

    # ── Training schedule ────────────────────────────────────────────────────────
    config.training.batch_size = 64
    config.training.n_iters = 20001
    config.training.log_freq = 200
    config.training.snapshot_freq = 5000
    config.training.eval_freq = 2000

    # ── Optimization ────────────────────────────────────────────────────────────
    config.optim.lr = 2e-4
    config.optim.warmup = 1000
    config.optim.grad_clip = 1.
    config.optim.loss_weights = [1.0, 1.0]  # [w_score, w_flow]

    return config
