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
from pathlib import Path

'''
In ogni paziente (000,001) ho aaaaaaaa, aaaaaaab.
Dentro queste ho s001_2015/
01_tcp_ar/
aaaaaaaa_s001_t000.edf/

montaggio elettroencefalografico utilizzato:
(LE) Linked Ears
(AR) Average Reference

es: 
000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf
000/aaaaaaab/s001_2002 or s002_2002 or s003_2002/02_tcp_le/aaaaaaab_s001_t000.edf  aaaaaaab_s001_t001.edf  aaaaaaab_s001_t003.edf  aaaaaaab_s001_t004.edf  aaaaaaab_s001_t005.edf
ultimo paziente: 150/aaaaawgc
Struttura finale: [Paziente, Sessione, Canale, Tempo]
'''

dataset_path = Path("../../../mnt/Datasets/TUH_EEG/v2.0.1/edf") #da qui trovo le cartelle 000, 001, 002 dei soggetti
patient_id = "000"
patient_path = dataset_path / patient_id

# Cerca tutti i file .edf ricorsivamente
edf_files = list(patient_path.glob("**/*.edf"))

# Filtra per montaggio
montaggio_ar = []
montaggio_le = []

for f in edf_files:
    path_str = str(f).lower()  # minuscolo per sicurezza
    if "tcp_ar" in path_str:
        montaggio_ar.append(f)
    elif "tcp_le" in path_str:
        montaggio_le.append(f)

print(f"Trovati {len(edf_files)} file totali")
print(f"File TCP_AR: {len(montaggio_ar)}")
print(f"File TCP_LE: {len(montaggio_le)}")

import pyedflib
import numpy as np

edf_file = str(montaggio_ar[0])  # esempio
print(edf_file)
f = pyedflib.EdfReader(edf_file)
n_channels = f.signals_in_file
print(n_channels)
signal_labels = f.getSignalLabels()
sfreq = f.getSampleFrequency(0)  # assume stessa fs per tutti i canali
print(sfreq)
data = np.zeros((n_channels, f.getNSamples()[0]))
for i in range(n_channels):
    data[i, :] = f.readSignal(i)
f._close()
del f
print(f"Shape dei dati: {data.shape}, Canali: {signal_labels}, Fs: {sfreq}")










