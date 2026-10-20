import numpy as np

def generate_synthetic_swat(n_train=100000, n_test=20000, n_features=51, 
                             anomaly_ratio=0.12, seed=42, save_dir=None, output_dir=None):
    """
    Generate synthetic SWaT-like time series data.
    
    Returns:
        train_data: [n_train, n_features] normal operation data
        test_data:  [n_test, n_features]  mix of normal + anomalous
        test_labels:[n_test] 0=normal, 1=anomaly
    """
    if output_dir is not None:
        save_dir = output_dir
    rng = np.random.RandomState(seed)
    
    # Create correlated normal data
    # 51 sensors grouped into 6 sub-processes (like actual SWaT)
    # Group 1 (P1): sensors 0-8   - water intake
    # Group 2 (P2): sensors 9-17  - chemical dosing
    # Group 3 (P3): sensors 18-26 - filtration
    # Group 4 (P4): sensors 27-35 - reverse osmosis
    # Group 5 (P5): sensors 36-43 - backwash
    # Group 6 (P6): sensors 44-50 - distribution
    
    groups = [range(0, 9), range(9, 18), range(18, 27), 
              range(27, 36), range(36, 44), range(44, 51)]
    
    # Base frequencies for each group (Hz, at 1-second sampling)
    freqs = [0.01, 0.015, 0.02, 0.008, 0.012, 0.018]
    amplitudes = rng.uniform(0.5, 2.0, n_features)
    phases = rng.uniform(0, 2*np.pi, n_features)
    offsets = rng.uniform(-1, 1, n_features)
    
    def generate_normal(n_samples, t_offset=0):
        t = np.arange(n_samples) + t_offset
        data = np.zeros((n_samples, n_features))
        for gi, group in enumerate(groups):
            # Base signal for this group
            base = amplitudes[list(group)[0]] * np.sin(2*np.pi * freqs[gi] * t[:, None])
            for fi, sensor_idx in enumerate(group):
                # Individual sensor = group base + small individual variation
                ind_freq = freqs[gi] * (1 + 0.1*fi)
                data[:, sensor_idx] = (
                    amplitudes[sensor_idx] * np.sin(2*np.pi * ind_freq * t + phases[sensor_idx])
                    + 0.3 * base[:, 0]  # group correlation
                    + offsets[sensor_idx]
                    + rng.normal(0, 0.05, n_samples)  # noise
                )
        return data
    
    # Generate training data (all normal)
    train_data = generate_normal(n_train, t_offset=0)
    
    # Normalize to [0,1] range
    data_min = train_data.min(axis=0)
    data_max = train_data.max(axis=0)
    data_range = data_max - data_min + 1e-8
    
    train_data = (train_data - data_min) / data_range
    
    # Generate test data with anomalies
    test_raw = generate_normal(n_test, t_offset=n_train)
    test_data = (test_raw - data_min) / data_range
    test_labels = np.zeros(n_test, dtype=np.int32)
    
    # Inject anomalies in blocks (like real cyber attacks)
    n_anomalies = int(n_test * anomaly_ratio)
    n_attack_blocks = 8  # multiple attack periods
    
    attack_positions = []
    i = 0
    while i < n_attack_blocks and len(attack_positions) < n_anomalies:
        # Random attack start
        start = rng.randint(100, n_test - 500)
        duration = rng.randint(100, 400)
        end = min(start + duration, n_test)
        
        # Choose attack type
        attack_type = rng.randint(0, 4)
        affected_sensors = rng.choice(n_features, size=rng.randint(2, 8), replace=False)
        
        if attack_type == 0:  # Sudden spike
            test_data[start:end, affected_sensors] += rng.uniform(0.5, 2.0, len(affected_sensors))
        elif attack_type == 1:  # Mean shift
            test_data[start:end, affected_sensors] += rng.uniform(0.3, 0.8, len(affected_sensors))
        elif attack_type == 2:  # Sensor flatline (stuck-at)
            flat_val = test_data[start, affected_sensors]
            test_data[start:end, affected_sensors] = flat_val
        elif attack_type == 3:  # Gradual drift
            drift = np.linspace(0, rng.uniform(0.4, 1.0), end-start)
            test_data[start:end, affected_sensors] += drift[:, None]
        
        test_labels[start:end] = 1
        attack_positions.append((start, end, attack_type))
        i += 1
    
    test_data = np.clip(test_data, -0.5, 1.5)  # clip outliers
    
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'train_data.npy'), train_data.astype(np.float32))
        np.save(os.path.join(save_dir, 'test_data.npy'), test_data.astype(np.float32))
        np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)
        np.save(os.path.join(save_dir, 'normalization.npy'), 
                np.array([data_min, data_max]))
        print(f"Saved synthetic SWaT data to {save_dir}")
        print(f"Train: {train_data.shape}, Test: {test_data.shape}")
        print(f"Anomaly ratio: {test_labels.mean():.3f}")
        print(f"Attack blocks: {attack_positions}")
    
    return train_data.astype(np.float32), test_data.astype(np.float32), test_labels


# Alias for compatibility
generate_swat_data = generate_synthetic_swat


if __name__ == '__main__':
    import os
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'swat_synthetic')
    train, test, labels = generate_synthetic_swat(save_dir=save_dir)
