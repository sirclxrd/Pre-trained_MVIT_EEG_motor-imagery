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
from Preprocessing import EEGSpectrogramDataset, channel_normalization, compute_morlet_spectrogram, read_data



from sklearn.preprocessing import StandardScaler
import numpy as np

class ChannelNormalizer:
    """
    Normalizzazione per canale usando mean/std (o median/IQR se robust=True).
    Fit sui dati di train, transform anche su val/test usando parametri calcolati sul train.
    """

    def __init__(self, robust=False):
        self.robust = robust
        self.params = None  # dict con mean/std o median/IQR

    def fit(self, features):
        # features: (n_trials, n_channels, n_times)
        if self.robust:
            median = np.median(features, axis=(0,2), keepdims=True)
            q75, q25 = np.percentile(features, [75, 25], axis=(0,2), keepdims=True)
            iqr = q75 - q25 + 1e-8
            self.params = {"center": median, "scale": iqr}
        else:
            mean = features.mean(axis=(0,2), keepdims=True)
            std  = features.std(axis=(0,2), keepdims=True) + 1e-8
            self.params = {"center": mean, "scale": std}
        return self

    def transform(self, features):
        return (features - self.params["center"]) / self.params["scale"]

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)


def preprocess_2a(subject_id ,tmin=2, tmax=6.028, augment = True, filter = "Butter", batch_size=32):
    # Carico dati (qui assumo che read_data_physionet ritorni anche is_real se augment=True)
    root='./BciCompetitionIv2a/Train'
    path_train = os.path.join(root, f'{subject_id}T.gdf')
    path_test = os.path.join(root.replace('Train', 'Test'), f'{subject_id}E.gdf')
    if augment:
        x_train, y_train, is_real = read_data(path_train, tmin=2, tmax=6.028, is_test=False, augment = True, filter = filter)
        x_test, y_test = read_data(path_test, tmin=2, tmax=6.028, is_test=True, augment = False, filter = filter)
    else:
        x_train, y_train = read_data(path_train, tmin=2, tmax=6.028, is_test=False, augment = augment, filter = filter)
        x_test, y_test = read_data(path_test, tmin=2, tmax=6.028, is_test=True, augment = augment, filter = filter)
        
    # Split indici
    if augment:
        real_indices = np.where(is_real == 1)[0]
        aug_indices  = np.where(is_real == 0)[0]

        # Splitto solo i reali
        indices = np.random.permutation(len(real_indices))
        train_len = int(0.8 * len(indices))
        train_real = real_indices[:train_len]
        val_real   = real_indices[train_len:]

        # Aggiungo i dati augmentati solo al training
        train_indices = np.concatenate([train_real, aug_indices])
        val_indices = val_real

    else:
        indices = np.random.permutation(len(x_train))
        train_len = int(0.8 * len(indices))
        train_indices = indices[:train_len]
        val_indices   = indices[train_len:]




    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("len indices train:", len(np.random.permutation(len(x_train))))

    # Creo split in NumPy
    X, y = x_train[train_indices], y_train[train_indices]
    X_val, y_val     = x_train[val_indices], y_train[val_indices]
    X_test, y_test   = x_test, y_test



    # Normalizzazione corretta: fit solo sul train
    normalizer = ChannelNormalizer()
    X_train = normalizer.fit_transform(X)
    X_val   = normalizer.transform(X_val)
    X_test  = normalizer.transform(X_test)

    # Morlet con standardizzazione coerente
    X_train, mean, std = compute_morlet_spectrogram(X_train, sfreq=250, freqs=np.linspace(8, 30, 32))
    X_val, _, _  = compute_morlet_spectrogram(X_val,  sfreq=250, freqs=np.linspace(8, 30, 32), mean=mean, std=std)
    X_test, _, _ = compute_morlet_spectrogram(X_test, sfreq=250, freqs=np.linspace(8, 30, 32), mean=mean, std=std)

    # Creo dataset PyTorch
    train_dataset = EEGSpectrogramDataset(X_train, y)
    val_dataset   = EEGSpectrogramDataset(X_val, y_val)
    test_dataset  = EEGSpectrogramDataset(X_test, y_test)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader









