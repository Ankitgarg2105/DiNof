"""
Default config for Time-Series DiNof (SWaT/WADI anomaly detection).
"""
import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()

    # ── Training ───────────────────────────────────────────────────────────────
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 64
    training.n_iters = 20001
    training.snapshot_freq = 5000
    training.log_freq = 200
    training.eval_freq = 1000
    training.snapshot_sampling = False
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True
    training.sde = 'vpsde'

    # ── Sampling ────────────────────────────────────────────────────────────────
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # ── Evaluation ──────────────────────────────────────────────────────────────
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 128
    evaluate.enable_sampling = False
    evaluate.num_samples = 1000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'
    evaluate.fid_dir = ''

    # ── Data ────────────────────────────────────────────────────────────────────
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'SWaT'
    data.num_channels = 51      # SWaT has 51 sensor features
    data.window_len = 64        # sliding window length
    data.stride = 1
    data.centered = True        # z-score normalize
    data.uniform_dequantization = False
    data.random_flip = False
    data.image_size = 64        # alias for window_len (kept for SDE compat)

    # ── Model ───────────────────────────────────────────────────────────────────
    config.model = model = ml_collections.ConfigDict()
    model.name = 'ScoreModel1D'
    model.nf = 64               # base channel width
    model.num_res_blocks = 2    # res blocks per U-Net level
    model.attn_resolutions = [1]  # add attention at level index 1
    model.ch_mult = (1, 2, 2)   # channel multipliers for 3 U-Net levels
    model.dropout = 0.1
    model.scale_by_sigma = False
    model.sigma_min = 0.01
    model.sigma_max = 50.
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.embedding_type = 'fourier'
    model.ema_rate = 0.999

    # ── Flow ────────────────────────────────────────────────────────────────────
    config.flow = flow = ml_collections.ConfigDict()
    flow.start_step = 0.6       # Tm = 0.6T  (key DiNof hyperparameter)
    flow.num_coupling = 8       # number of coupling layers
    flow.hidden_dim = 256       # MLP hidden dim per coupling layer
    flow.num_mlp_layers = 3     # MLP depth per coupling layer
    flow.ema_rate = 0.999

    # ── Optimization ────────────────────────────────────────────────────────────
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.
    optim.loss_weights = [1.0, 1.0]  # [w_score, w_flow]

    config.seed = 42
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return config
