"""
export_csv.py — Export SWaT time-series dataset from .npy to .csv format.

Usage:
    python export_csv.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'swat_synthetic')
OUT_DIR = os.path.join(ROOT, 'data')

def export():
    print("Loading NumPy binary files from data/swat_synthetic/...")
    te = np.load(os.path.join(DATA_DIR, 'test_data.npy'))
    y = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    tr = np.load(os.path.join(DATA_DIR, 'train_data.npy'))

    cols = [f'Sensor_{i+1:02d}' for i in range(51)]

    # Test CSV with ground truth labels
    test_path = os.path.join(OUT_DIR, 'swat_test.csv')
    df_test = pd.DataFrame(te, columns=cols)
    df_test.insert(0, 'Timestamp_sec', np.arange(len(te)))
    df_test['Normal_Attack'] = np.where(y == 1, 'Attack', 'Normal')
    df_test.to_csv(test_path, index=False)
    print(f"Exported: {test_path} ({df_test.shape[0]} rows, {df_test.shape[1]} cols)")

    # Train CSV (Normal operational baseline)
    train_path = os.path.join(OUT_DIR, 'swat_train.csv')
    df_train = pd.DataFrame(tr, columns=cols)
    df_train.insert(0, 'Timestamp_sec', np.arange(len(tr)))
    df_train['Normal_Attack'] = 'Normal'
    df_train.to_csv(train_path, index=False)
    print(f"Exported: {train_path} ({df_train.shape[0]} rows, {df_train.shape[1]} cols)")

    print("\nDataset ready in CSV format!")

if __name__ == '__main__':
    export()
