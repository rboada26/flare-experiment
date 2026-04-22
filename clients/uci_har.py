"""
UCI HAR (Human Activity Recognition) dataset loader.
Raw inertial signals: 128 timesteps x 9 channels per sample.
6 activity classes: Walking, Upstairs, Downstairs, Sitting, Standing, Laying.
Shared by GRU, LSTM, and BiLSTM clients.
"""
import os, urllib.request, zipfile
import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_DIR = './data/uci_har'
URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
       '/00240/UCI%20HAR%20Dataset.zip')

SIGNALS = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
]


def _load_signals(base, split):
    arrays = []
    for sig in SIGNALS:
        path = os.path.join(base, 'UCI HAR Dataset', split,
                            'Inertial Signals', f'{sig}_{split}.txt')
        arrays.append(np.loadtxt(path))          # (n_samples, 128)
    return np.stack(arrays, axis=2).astype(np.float32)  # (n, 128, 9)


def _load_labels(base, split):
    path = os.path.join(base, 'UCI HAR Dataset', split, f'y_{split}.txt')
    return np.loadtxt(path, dtype=np.int64) - 1  # zero-indexed 0–5


def load_uci_har(train_limit=50, test_limit=20):
    # Defaults are intentionally small for quick smoke-tests.
    # For full replication, pass train_limit=7352, test_limit=2947 (full UCI HAR split).
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path  = os.path.join(DATA_DIR, 'uci_har.zip')
    extracted = os.path.join(DATA_DIR, 'UCI HAR Dataset')

    if not os.path.exists(extracted):
        if not os.path.exists(zip_path):
            print('[uci_har] Downloading UCI HAR dataset...', flush=True)
            urllib.request.urlretrieve(URL, zip_path)
            print('[uci_har] Download complete.', flush=True)
        print('[uci_har] Extracting...', flush=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)

    X_train = _load_signals(DATA_DIR, 'train')[:train_limit]
    y_train = _load_labels(DATA_DIR,  'train')[:train_limit]
    X_test  = _load_signals(DATA_DIR, 'test')[:test_limit]
    y_test  = _load_labels(DATA_DIR,  'test')[:test_limit]

    trainset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    testset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    print(f'[uci_har] train={len(trainset)}, test={len(testset)}, '
          f'input shape=(128, 9), classes=6', flush=True)
    return trainset, testset


def apply_jitter():
    import subprocess, random, os
    if os.environ.get('NETWORK_JITTER', 'false').lower() != 'true':
        return
    delay  = random.randint(10, 50)
    jitter = random.randint(5, 20)
    result = subprocess.run(
        ['tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'netem',
         'delay', f'{delay}ms', f'{jitter}ms'],
        capture_output=True
    )
    if result.returncode == 0:
        print(f'[jitter] eth0: delay={delay}ms ±{jitter}ms', flush=True)
    else:
        print(f'[jitter] tc skipped: {result.stderr.decode().strip()}', flush=True)


if __name__ == '__main__':
    load_uci_har()
