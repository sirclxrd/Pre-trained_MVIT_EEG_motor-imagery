import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import tfr_array_morlet
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_spectrogram
import os
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, cheby2
from scipy.signal import savgol_filter, stft
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import random
import scipy
import numpy as np
from scipy.linalg import fractional_matrix_power
import torch.nn.functional as F
from scipy.stats import zscore






LOW_FREQ = 4
HIGH_FREQ = 38
N_FREQ = 224

class EEGSpectrogramDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)  # [N, 22, 32, 1008]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def euclidean_alignment(eeg_data: np.ndarray, R_mean_inv_sqrt= None) -> np.ndarray:
    """
    Applica la Euclidean Alignment (EA) ai dati EEG.
    Allinea i dati di test e train in uno stesso spazio euclideo
    ovvero rende la matrice di covarianza unitaria dopo al trasformazione.
    Devo farmi restituire la matrice R_mean dai dati di train
    poi successivamente usarla per quelli di test
    """
    n_trials, n_channels, n_samples = eeg_data.shape
    
    if R_mean_inv_sqrt is None:
        # Calcolo matrice media delle covarianze
        R_mean = np.zeros((n_channels, n_channels), dtype=np.float64)
        for i in range(n_trials):
            X = eeg_data[i]
            R = X @ X.T / n_samples
            R_mean += R
        R_mean /= n_trials
        R_mean_inv_sqrt = fractional_matrix_power(R_mean, -0.5)
    
    # Trasformo i dati (train o test)
    aligned_data = np.zeros_like(eeg_data, dtype=np.float32)
    for i in range(n_trials):
        aligned_data[i] = (R_mean_inv_sqrt @ eeg_data[i]).astype(np.float32)

    return aligned_data, R_mean_inv_sqrt


def trial_based_normalization(eeg_data: np.ndarray) -> np.ndarray:
    """
    Normalizza i dati EEG su base trial nell'intervallo [-1, 1].
    In questo modo mantengo la proporzione interna tra canali
    I canali sono scalati in modo uniforme e quindi restano confrontabili.
    """
    normalized_data = np.zeros_like(eeg_data, dtype=np.float32)

    for i in range(eeg_data.shape[0]):  # per ogni trial
        trial = eeg_data[i]
        max_val = np.max(np.abs(trial))  # massimo in valore assoluto
        if max_val == 0:
            normalized_data[i] = trial  # evito divisione per zero
        else:
            normalized_data[i] = trial / max_val

    return normalized_data


def segment_and_rec_total_augmentation(features, labels, dataset="2a"):
    """
    Augmentation per segmentazione e ricostruzione.
    Divide ogni epoca in 8 segmenti e ricombina segmenti della stessa classe.
    Ritorna features e labels aumentati (concat con quelli originali).
    In questo modo creo un nuovo sample completamente nuovo dato dalla concatenzatione con segmenti casuali
    della stessa classe. Come EEG-Conformer
    """
    #rng = np.random.default_rng(2025)
    if dataset == "2a":
        segment_length = 1008 // 8  # 1008 / 8
    else:
        segment_length = 1136 // 8
    num_segments = 8
    C = features.shape[1]

    synthetic_samples = []
    synthetic_labels = []

    for cls in np.unique(labels):
        idx_cls = np.where(labels == cls)[0]
        feats_cls = features[idx_cls]

        for _ in range(len(idx_cls)) :
            segments = []
            for i in range(num_segments):
                rand_feat = feats_cls[np.random.randint(0, len(feats_cls))]
                #rand_feat = feats_cls[rng.integers(0, len(feats_cls))]
                seg = rand_feat[:, i*segment_length:(i+1)*segment_length]
                segments.append(seg)
            new_sample = np.concatenate(segments, axis=1)  # (C, 1008)
            synthetic_samples.append(new_sample)
            synthetic_labels.append(cls)

    part = np.array(synthetic_samples)
    labels_part = np.array(synthetic_labels).reshape(-1, 1)
    features_aug = np.concatenate([features, part], axis=0)
    labels_aug = np.concatenate([labels, labels_part], axis=0)

    is_real = np.concatenate([
    np.ones(len(features), dtype=int),
    np.zeros(len(part), dtype=int)
    ])

    return features_aug, labels_aug, is_real



#mano sinistra, destra, piedi, lingua
#cheby è [4,40]
#butter è [8,30]
#tmin=2, tmax=6.028
def read_data(path, tmin=2, tmax=6.028, is_test=False, augment = False, filter = "Butter"):
    raw=mne.io.read_raw_gdf(path,preload=True,
                            eog=['EOG-left', 'EOG-central', 'EOG-right'])
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    if filter == "Butter":
        raw.filter(l_freq=8, h_freq=30, method='iir', iir_params=dict(order=5, ftype='butter'), phase='zero') # filtro Butterworth [8,30]Hz ordine 5
        LOW_FREQ = 8
        HIGH_FREQ = 30
    elif filter == "Cheby":
        raw.filter(
            l_freq=4,
            h_freq=40,
            method='iir',
            iir_params=dict(order=6, ftype='cheby2', rs=40),
            phase = 'zero'
        )
        LOW_FREQ = 4
        HIGH_FREQ = 40
    else:
        LOW_FREQ = 0.5
        HIGH_FREQ = 100
        


    events=mne.events_from_annotations(raw)
    if is_test:
        subj = os.path.basename(path)[1:3]  #A02E.gdf → '02'
        epochs = mne.Epochs(raw, events[0], event_id = [6],
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)

        # Carico le etichette vere dal file .mat
        labels_path = f'../Python/BciCompetitionIv2a/true_labels/A{subj}E.mat'
        true = loadmat(labels_path)
        labels = true['classlabel'] - 1
        features=epochs.get_data()
        print("ciao")
        features = trial_based_normalization(features)
        print("ciao2")
        print("Test Features shape: ",features.shape)
        print("Test Labels shape: ",labels.shape)
        print("Test dataset done")
    else:
        subj = os.path.basename(path)[1:3]
        if subj == "04":
            epochs = mne.Epochs(raw, events[0], event_id = [4],
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)
        else:
            epochs = mne.Epochs(raw, events[0], event_id = [6],
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)
        labels_path = f'../Python/BciCompetitionIv2a/true_labels/A{subj}T.mat'
        true = loadmat(labels_path) 
        labels = true['classlabel'] - 1
        features=epochs.get_data()
        if augment == True:
            print("Augmentation...")
            features, labels, is_real = segment_and_rec_total_augmentation(features, labels)
        features = trial_based_normalization(features)
        print("Train Features shape: ",features.shape)
        print("Train Labels shape: ",labels.shape)
        print("Train dataset done")
        if augment == True:
            return features, labels, is_real


    return features,labels

def read_data_2b(subject_id, base_path, tmin=0, tmax=4, augment=False, filter="Butter", is_test = False):
    all_features = []
    all_labels = []
    subject_id = subject_id[1:3]

    if is_test:
        for session in ['4', '5']:
            file_name = f"B{subject_id}0{session}E.gdf"
            file_path = os.path.join(base_path, "Test" ,file_name)

            # Leggi segnale
            raw = mne.io.read_raw_gdf(file_path, preload=True)

            raw.filter(
            l_freq=4, h_freq=38,
            fir_design='firwin',
            fir_window='blackman'    
            )

            events, event_dict = mne.events_from_annotations(raw)
            event_id = {'Unknown': event_dict['783']}
            selected_events = events[np.isin(events[:, 2], list(event_id.values()))]              
            raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                            exclude='bads')
            epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=tmin,tmax=tmax,preload=True,baseline=None)
            features = epochs.get_data()
            mat_path = os.path.join(base_path, "true_labels", f"B{subject_id}0{session}E.mat")
            true = loadmat(mat_path)
            labels = true['classlabel'] - 1
            features = trial_based_normalization(features)
            all_features.append(features)
            all_labels.append(labels)
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print("Test Features shape: ",features.shape)
        print("Test Labels shape: ",labels.shape)
        print("Test dataset done")    
    else:
        for session in ['1', '2', '3']:
            print("subject_id:", subject_id)
            file_name = f"B{subject_id}0{session}T.gdf"
            file_path = os.path.join(base_path, "Train" ,file_name)

            raw = mne.io.read_raw_gdf(file_path, preload=True)

            raw.filter(
            l_freq=4, h_freq=38,
            fir_design='firwin',
            fir_window='blackman'    
            )

            events, event_dict = mne.events_from_annotations(raw) 
            event_id = {'Left': event_dict['769'], 'Right': event_dict['770']}
            selected_events = events[np.isin(events[:, 2], list(event_id.values()))]              
            raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                            exclude='bads')
            epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=tmin,tmax=tmax,preload=True,baseline=None)
            features = epochs.get_data()

            mat_path = os.path.join(base_path, "true_labels", f"B{subject_id}0{session}T.mat")
            true = loadmat(mat_path)
            labels = true['classlabel'] - 1  # [1, 2] → [0, 1]

            features = trial_based_normalization(features)
            all_features.append(features)
            all_labels.append(labels)
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print("Train Features shape: ",features.shape)
        print("Train Labels shape: ",labels.shape)
        print("Train dataset done")
            
    return features, labels


def compute_morlet_spectrogram(features, sfreq, freqs=np.linspace(LOW_FREQ, HIGH_FREQ, N_FREQ), n_cycles=7, mean = None, std = None):
    """
    features: ndarray (n_epochs, n_channels, n_times)
    sfreq: frequenza di campionamento (Hz), per BCIC IV 2a è 250 Hz
    freqs: array di frequenze su cui è centrata la wavelet
    n_cycles: indica quanto è lunga l'onda wavelet, compromesso tempo frequenza. 
    Ogni onda wavelet ha una determinata frequenza, n_cycles definisce il numero di cicli e quindi quanto è larga
    Piu' è larga più è precisa in frequenza ma meno in tempo, io la faccio diventare più larga man mano che crescono le frequenze
    """
    wvlts = tfr_array_morlet(features, sfreq=sfreq, freqs=freqs,
                             n_cycles=7, output='power', n_jobs=1)

    if mean is None or std is None:
        mean = np.mean(wvlts, axis=(0,1,2), keepdims=True)
        std  = np.std(wvlts, axis=(0,1,2), keepdims=True)
    wvlts = (wvlts - mean) / (std + 1e-10)
    
    wvlts = torch.tensor(wvlts, dtype=torch.float32)

    wvlts = F.interpolate(
        wvlts, size=(224, 224), mode='bilinear', align_corners=False
    )
    return wvlts.numpy(), mean, std


def prepare_dataloaders(subject_id='A09', root='./BciCompetitionIv2a/Train', onlytest = False, augment = False, filter = "Butter", BCI = "2a", root_2b = './BciCompetitionIv2b'):
    train_path = os.path.join(root, f'{subject_id}T.gdf')
    test_path = os.path.join(root.replace('Train', 'Test'), f'{subject_id}E.gdf')
    print(f"You are using the {BCI} dataset.")
    if onlytest == False:
        # TRAIN
        if BCI == "2a":
            if augment == True:
                x_train, y_train, is_real = read_data(train_path, is_test=False, augment = augment, filter = filter)
            else:
                x_train, y_train = read_data(train_path, is_test=False, augment = augment, filter = filter)
        else:
            root_train = root_2b
            if augment == True:
                x_train, y_train, is_real = read_data_2b(subject_id, root_train, augment=augment, filter=filter, is_test = False)
            else:
                x_train, y_train = read_data_2b(subject_id, root_train, augment=augment, filter=filter, is_test = False)
                x_train,R = euclidean_alignment(x_train)

        x_train, mean, std = compute_morlet_spectrogram(x_train, sfreq=250)
        print(x_train.shape)

        # TEST
        if BCI == "2a":
            x_test, y_test = read_data(test_path, is_test=True, filter = filter)
        else:
            root_test = root_2b
            x_test, y_test = read_data_2b(subject_id, root_test, augment=augment, filter=filter, is_test = True)
            x_test, R = euclidean_alignment(x_test, R)
        x_test, mean, std = compute_morlet_spectrogram(x_test, sfreq=250, mean=mean, std=std)

        # DATASET
        train_dataset = EEGSpectrogramDataset(x_train, y_train)
        test_dataset = EEGSpectrogramDataset(x_test, y_test)

        if augment:
            return train_dataset, test_dataset, is_real
        else:
            return train_dataset, test_dataset
    else:
        if BCI == "2a":
            x_test, y_test = read_data(test_path, is_test=True, filter = filter)
        else:
            root_test = root_2b
            x_test, y_test = read_data_2b(subject_id, root_test, augment=augment, filter=filter, is_test = True)
        x_test = compute_morlet_spectrogram(x_test, sfreq=250, mean=mean, std=std)

        # DATASET
        test_dataset = EEGSpectrogramDataset(x_test, y_test)
        return test_dataset







