# show_dataset.py : quick dataset walkthrough for the panel  ->  python show_dataset.py
import numpy as np, matplotlib
import matplotlib.pyplot as plt
D = 'data/swat_synthetic/'
tr = np.load(D + 'train_data.npy'); te = np.load(D + 'test_data.npy'); y = np.load(D + 'test_labels.npy')
print('TRAIN :', tr.shape, '-> 100000 sec x 51 sensors, all normal')
print('TEST  :', te.shape, '-> 20000 sec x 51 sensors, attacks injected')
print(f'Attack points: {int(y.sum())} of {len(y)} ({100*y.mean():.1f}%)')
# attack blocks = contiguous runs of label 1
d = np.diff(np.r_[0, y, 0]); starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
print('\nAttack blocks (start sec -> end sec, length):')
for s, e in zip(starts, ends): print(f'  {s:6d} -> {e:6d}   ({e-s} s)')
W = 64; wl = [int(y[i:i+W].max()) for i in range(0, len(y) - W + 1, W)]
print(f'\nTest windows (W=64, stride=64): {len(wl)} total, {sum(wl)} attack, {len(wl)-sum(wl)} normal')
print('Groups: P1 s0-8 | P2 s9-17 | P3 s18-26 | P4 s27-35 | P5 s36-43 | P6 s44-50')

fig, ax = plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(tr[:600, [0, 12, 30, 45]]); ax[0].set_title('TRAIN (normal): sensors 1, 13, 31, 46, first 600 s')
s0, e0 = starts[0], ends[0]; a, b = max(0, s0 - 200), min(len(te), e0 + 200)
ax[1].plot(range(a, b), te[a:b, [0, 12, 30, 45]]); ax[1].axvspan(s0, e0, color='red', alpha=.2, label='attack')
ax[1].set_title(f'TEST around first attack block ({s0}-{e0} s)'); ax[1].legend()
ax[2].imshow(te[s0:s0 + 64].T, aspect='auto', cmap='viridis'); ax[2].set_title('One 64 s window = 51 sensors x 64 time steps (the model input)')
ax[2].set_xlabel('time step'); ax[2].set_ylabel('sensor')
plt.tight_layout(); plt.savefig('dataset_overview.png', dpi=130); print('\nSaved dataset_overview.png')
plt.show()
