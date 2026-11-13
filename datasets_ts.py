"""
Time-Series Dataset for DiNof.
Supports synthetic SWaT data and real SWaT/WADI CSV files.
Data shape: [B, num_features, 1, window_len]  (4D, compatible with existing losses)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    Sliding-window time series dataset.
    
    Returns windows of shape [num_features, 1, window_len] (4D compatible with SDE losses).
    
    Args:
        data: numpy array [N, num_features]
        labels: optional numpy array [N] with 0=normal, 1=anomaly
        window_len: sliding window length
        stride: step between windows (1 = fully overlapping)
        train: if True, only return normal windows (for training)
    """
    def __init__(self, data, labels=None, window_len=64, stride=1, train=True):
        self.window_len = window_len
        self.stride = stride
        self.train = train
        self.data = data  # [N, num_features]
        self.labels = labels  # [N] or None
        
        # Build valid window indices
        self.indices = []
        n = len(data)
        for i in range(0, n - window_len + 1, stride):
            end = i + window_len
            if labels is not None and train:
                # During training: only use windows with all-normal labels
                if labels[i:end].max() == 0:
                    self.indices.append(i)
            else:
                self.indices.append(i)
        
        print(f"  Dataset: {len(self.indices)} windows "
              f"(window={window_len}, stride={stride}, "
              f"{'train' if train else 'test'})")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        window = self.data[i:i + self.window_len]  # [window_len, num_features]
        window = window.T  # [num_features, window_len]
        
        # Reshape to [num_features, 1, window_len] for 4D SDE compatibility
        window_4d = window[:, np.newaxis, :]  # [num_features, 1, window_len]
        
        label = 0
        if self.labels is not None:
            # Window label = 1 if any point in window is anomalous
            label = int(self.labels[i:i + self.window_len].max())
        
        return {
            'image': torch.tensor(window_4d, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_swat_data(data_dir, source='synthetic'):
    """
    Load SWaT data from directory.
    
    Args:
        data_dir: directory containing data files
        source: 'synthetic' (use generated data) or 'csv' (use real SWaT CSV)
    
    Returns:
        train_data, test_data, test_labels (all numpy arrays [N, features])
    """
    if source == 'synthetic':
        if os.path.exists(os.path.join(data_dir, 'swat_synthetic', 'train_data.npy')):
            data_dir = os.path.join(data_dir, 'swat_synthetic')
        train_path = os.path.join(data_dir, 'train_data.npy')
        test_path  = os.path.join(data_dir, 'test_data.npy')
        label_path = os.path.join(data_dir, 'test_labels.npy')
        
        if not os.path.exists(train_path):
            # Generate data if not already done
            print("Generating synthetic SWaT data...")
            from data.synthetic_swat import generate_synthetic_swat
            generate_synthetic_swat(save_dir=data_dir)
        
        train_data  = np.load(train_path)
        test_data   = np.load(test_path)
        test_labels = np.load(label_path)
        return train_data, test_data, test_labels
    
    elif source == 'csv':
        # Real SWaT CSV format (requires formal request from iTrust SUTD)
        import pandas as pd
        
        train_csv = os.path.join(data_dir, 'SWaT_Dataset_Normal_v1.csv')
        test_csv  = os.path.join(data_dir, 'SWaT_Dataset_Attack_v0.csv')
        
        # Load train (normal operation)
        train_df = pd.read_csv(train_csv, dayfirst=True)
        train_df = train_df.drop(columns=['Timestamp', 'Normal/Attack'], errors='ignore')
        train_data = train_df.values.astype(np.float32)
        
        # Load test (with attacks)
        test_df = pd.read_csv(test_csv, dayfirst=True)
        test_labels = (test_df['Normal/Attack'].values == 'Attack').astype(np.int32)
        test_df = test_df.drop(columns=['Timestamp', 'Normal/Attack'], errors='ignore')
        test_data = test_df.values.astype(np.float32)
        
        # Normalize using training statistics
        mean = train_data.mean(axis=0)
        std  = train_data.std(axis=0) + 1e-8
        train_data = (train_data - mean) / std
        test_data  = (test_data - mean) / std
        
        return train_data, test_data, test_labels
    
    else:
        raise ValueError(f"Unknown source: {source}. Use 'synthetic' or 'csv'.")


def get_ts_dataloaders(config, data_dir, source='synthetic'):
    """
    Create PyTorch DataLoaders for time-series training/evaluation.
    
    Returns:
        train_loader, test_loader
    """
    train_data, test_data, test_labels = load_swat_data(data_dir, source)
    
    window_len  = config.data.image_size   # = window length
    batch_size  = config.training.batch_size
    
    print(f"Creating datasets with window_len={window_len}")
    print(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
    print(f"Anomaly rate: {test_labels.mean():.3f}")
    
    train_ds = TimeSeriesDataset(
        train_data, labels=None, 
        window_len=window_len, stride=1, train=True
    )
    test_ds = TimeSeriesDataset(
        test_data, labels=test_labels, 
        window_len=window_len, stride=window_len, train=False
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=True, num_workers=0, drop_last=True,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader, test_labels


def ts_data_scaler(config):
    """Data normalizer: scales to [-1, 1] if centered else identity."""
    if config.data.centered:
        return lambda x: x * 2. - 1.
    return lambda x: x


def ts_data_inverse_scaler(config):
    """Inverse normalizer."""
    if config.data.centered:
        return lambda x: (x + 1.) / 2.
    return lambda x: x
