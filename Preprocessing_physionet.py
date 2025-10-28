import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import tfr_array_morlet
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_spectrogram
import os
from torch.utils.data import Dataset, DataLoader,Subset
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, cheby2
from scipy.signal import savgol_filter, stft
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import random
import scipy
import torch.nn.functional as F
from Preprocessing import EEGSpectrogramDataset, channel_normalization, compute_morlet_spectrogram, read_data_physionet
from sklearn.preprocessing import StandardScaler
import numpy as np

class ChannelNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, features):
        """
        Calcola mean e std SOLO sui dati di train.
        features: shape (n_trials, n_channels, n_times)
        """
        n_trials, n_channels, n_times = features.shape
        reshaped = features.transpose(1, 0, 2).reshape(n_channels, -1).T  # (n_trials*n_times, n_channels)
        self.scaler.fit(reshaped)
        return self

    def transform(self, features):
        """
        Applica la normalizzazione calcolata sul train a un dataset (train/val/test).
        """
        n_trials, n_channels, n_times = features.shape
        reshaped = features.transpose(1, 0, 2).reshape(n_channels, -1).T
        normalized = self.scaler.transform(reshaped)
        features_norm = normalized.T.reshape(n_channels, n_trials, n_times).transpose(1, 0, 2)
        return features_norm

    def fit_transform(self, features):
        """
        Utility per usare sul train direttamente.
        """
        self.fit(features)
        return self.transform(features)

def preprocess_physionet(subject_id='A09', augment=False, filter="Butter", batch_size=32):
    if augment:
        x, y, is_real = read_data_physionet(
            "./BciPhysionet", subject=subject_id, preload=True, augment=augment, filter=filter
        )
    else:
        x, y = read_data_physionet(
            "./BciPhysionet", subject=subject_id, preload=True, augment=augment, filter=filter
        )
        is_real = np.ones(len(x))  # se non c'è augment, considero tutto "real"

    # Split indici
    if augment:
        real_indices = np.where(is_real == 1)[0]
        aug_indices  = np.where(is_real == 0)[0]

        indices = np.random.permutation(len(real_indices))
        train_len = int(0.8 * len(indices))
        val_len   = int(0.1 * len(indices))

        train_real = real_indices[:train_len]
        val_real   = real_indices[train_len:train_len + val_len]
        test_real  = real_indices[train_len + val_len:]

        train_indices = np.concatenate([train_real, aug_indices])
        val_indices   = val_real
        test_indices  = test_real
    else:
        indices = np.random.permutation(len(x))
        train_len = int(0.8 * len(indices))
        val_len   = int(0.1 * len(indices))

        train_indices = indices[:train_len]
        val_indices   = indices[train_len:train_len + val_len]
        test_indices  = indices[train_len + val_len:]

    # Creo split in NumPy
    X_train, y_train = x[train_indices], y[train_indices]
    X_val, y_val     = x[val_indices], y[val_indices]
    X_test, y_test   = x[test_indices], y[test_indices]

    # Normalizzazione corretta: fit solo sul train
    normalizer = ChannelNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val   = normalizer.transform(X_val)
    X_test  = normalizer.transform(X_test)

    # Morlet con standardizzazione coerente
    X_train, mean, std = compute_morlet_spectrogram(X_train, sfreq=160, freqs=np.linspace(4, 50, 32))
    X_val, _, _  = compute_morlet_spectrogram(X_val,  sfreq=160, freqs=np.linspace(4, 50, 32), mean=mean, std=std)
    X_test, _, _ = compute_morlet_spectrogram(X_test, sfreq=160, freqs=np.linspace(4, 50, 32), mean=mean, std=std)

    # Creo dataset PyTorch
    train_dataset = EEGSpectrogramDataset(X_train, y_train)
    val_dataset   = EEGSpectrogramDataset(X_val, y_val)
    test_dataset  = EEGSpectrogramDataset(X_test, y_test)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader









