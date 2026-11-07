"""
visualize_results.py — Generate presentation-quality plots for DiNof TS results.

Produces:
  1. loss_curves.png        — Score loss + Flow loss vs training step
  2. anomaly_scores.png     — Score distribution: normal vs anomalous windows
  3. roc_curve.png          — ROC curve with AUROC annotation
  4. ts_with_anomalies.png  — Raw sensor time series with anomaly regions shaded

Usage:
    from visualize_results import plot_all
    plot_all(workdir, score_losses, flow_losses, all_scores, all_labels, test_data)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc


# ── Style ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})
BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'
RED    = '#d62728'
GREEN  = '#2ca02c'


def plot_loss_curves(workdir, score_losses, flow_losses, smooth=50):
    """
    Plot score loss and flow loss vs training step (dual y-axis).
    """
    def smooth_arr(arr, w):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='valid')

    steps_s = np.arange(len(score_losses))
    steps_f = np.arange(len(flow_losses))

    s_sm = smooth_arr(np.array(score_losses), smooth)
    f_sm = smooth_arr(np.array(flow_losses),  smooth)
    st_s = np.arange(len(s_sm)) + smooth // 2
    st_f = np.arange(len(f_sm)) + smooth // 2

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(steps_s, score_losses, alpha=0.2, color=BLUE)
    ax1.plot(st_s, s_sm, color=BLUE, linewidth=2, label='Score Loss (DSM)')
    ax2.plot(steps_f, flow_losses, alpha=0.2, color=ORANGE)
    ax2.plot(st_f, f_sm, color=ORANGE, linewidth=2, label='Flow Loss (NLL)')

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Score Loss', color=BLUE)
    ax2.set_ylabel('Flow Loss (NLL)', color=ORANGE)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax1.set_title('DiNof Training: Score + Flow Loss Curves')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    path = os.path.join(workdir, 'loss_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def plot_anomaly_scores(workdir, all_scores, all_labels, threshold=None):
    """
    Histogram of anomaly scores: normal (blue) vs anomalous (red).
    """
    scores_normal  = all_scores[all_labels == 0]
    scores_anomaly = all_scores[all_labels == 1]

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(all_scores.min(), np.percentile(all_scores, 99), 60)
    ax.hist(scores_normal,  bins=bins, alpha=0.6, color=BLUE,
            label=f'Normal (n={len(scores_normal):,})',   density=True)
    ax.hist(scores_anomaly, bins=bins, alpha=0.6, color=RED,
            label=f'Anomaly (n={len(scores_anomaly):,})', density=True)

    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold:.2f}')

    ax.set_xlabel('Anomaly Score  $(-\\log p_{\\mathrm{flow}}(x_{T_m}))$')
    ax.set_ylabel('Density')
    ax.set_title('DiNof: Anomaly Score Distribution (SWaT)')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(workdir, 'anomaly_scores.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def plot_roc_curve(workdir, all_scores, all_labels, auroc=None):
    """
    ROC curve with AUROC score.
    """
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    _auroc = auc(fpr, tpr)
    if auroc is None:
        auroc = _auroc

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2,
            label=f'DiNof (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — SWaT Anomaly Detection')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    path = os.path.join(workdir, 'roc_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def plot_ts_with_anomalies(workdir, test_data, all_labels,
                            all_scores=None, feature_idx=0, n_points=2000):
    """
    Raw sensor time-series with anomaly ground truth shaded in red.
    Optionally overlays anomaly score on a twin axis.

    test_data: np.array [T, F]  (original un-windowed test data)
    all_labels: np.array [N_windows] — window-level labels
    """
    T_show = min(n_points, len(test_data))
    t = np.arange(T_show)
    signal = test_data[:T_show, feature_idx]

    # Map window labels back to approx time indices (stride=1 assumed)
    # For visualisation we just show the raw labels aligned to window centres
    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(t, signal, color=BLUE, linewidth=0.8, label=f'Sensor #{feature_idx+1}')
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel(f'Sensor {feature_idx+1} (normalised)', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)

    # Shade anomaly windows
    in_anomaly = False
    anom_start = None
    win_len    = 64
    win_labels = all_labels[:T_show] if len(all_labels) >= T_show else all_labels

    for i, lbl in enumerate(win_labels):
        pos = i  # stride=1: window i starts at sample i
        if pos >= T_show:
            break
        if lbl == 1 and not in_anomaly:
            anom_start = pos
            in_anomaly = True
        elif lbl == 0 and in_anomaly:
            ax1.axvspan(anom_start, pos, alpha=0.25, color=RED)
            in_anomaly = False
    if in_anomaly:
        ax1.axvspan(anom_start, T_show, alpha=0.25, color=RED)

    if all_scores is not None and len(all_scores) >= T_show:
        ax2 = ax1.twinx()
        sc  = all_scores[:T_show]
        ax2.plot(np.arange(T_show), sc, color=ORANGE, linewidth=0.6,
                 alpha=0.7, label='Anomaly Score')
        ax2.set_ylabel('Anomaly Score', color=ORANGE)
        ax2.tick_params(axis='y', labelcolor=ORANGE)

    red_patch  = mpatches.Patch(color=RED, alpha=0.3, label='Anomaly (ground truth)')
    blue_line  = mpatches.Patch(color=BLUE, label=f'Sensor #{feature_idx+1}')
    ax1.legend(handles=[blue_line, red_patch], loc='upper right')
    ax1.set_title('SWaT: Sensor Signal with Anomaly Regions Highlighted')

    plt.tight_layout()
    path = os.path.join(workdir, 'ts_with_anomalies.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def plot_all(workdir, score_losses, flow_losses,
             all_scores, all_labels, test_data,
             threshold=None, auroc=None):
    """
    Generate all 4 plots and return a list of paths.
    """
    os.makedirs(workdir, exist_ok=True)
    paths = []
    if score_losses and flow_losses:
        paths.append(plot_loss_curves(workdir, score_losses, flow_losses))
    paths.append(plot_anomaly_scores(workdir, all_scores, all_labels, threshold))
    paths.append(plot_roc_curve(workdir, all_scores, all_labels, auroc))
    if test_data is not None:
        paths.append(plot_ts_with_anomalies(workdir, test_data, all_labels, all_scores))
    return paths
