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


#Normalizzazione per tutti i dati
#Con flatten i dati vengono concatenati
def total_normalization(features):
    #all_values = features.flatten()
    all_values = features
    mean = all_values.mean()
    std = all_values.std()
    features_norm = (features - mean) / std
    return features_norm

from scipy.stats import zscore
#Test per vedere se normalizzando gli spettrogrammi migliora 
def normalize_spectrogram(spectrogram):
    """Log + z-score sullo spettrogramma per canale e frequenza"""
    # Log-compression
    spectrogram = np.log1p(spectrogram)  # oppure np.log10(spectrogram + eps)
    # z-score per ogni (canale, frequenza) su tutte le epoche e tempi
    # reshape per applicare su asse (0, 3) → (trial, time)
    n_trials, n_channels, n_freqs, n_times = spectrogram.shape
    reshaped = spectrogram.transpose(1, 2, 0, 3).reshape(n_channels * n_freqs, -1)
    normalized = zscore(reshaped, axis=1)
    spectrogram_norm = normalized.reshape(n_channels, n_freqs, n_trials, n_times).transpose(2, 0, 1, 3)
 
    return spectrogram_norm  # shape: (n_trials, n_channels, n_freqs, n_times)

def segment_and_rec_1fold_augmentation(features, labels, num_aug_per_sample=1, num_segments=8):
    """
    Ogni epoca viene usata `num_aug_per_sample` volte per creare nuove epoche,
    ciascuna con un solo segmento sostituito da un segmento di un'altra epoca della stessa classe.
    Questo è S&R con 1 fold, come MSCFormer
    
    features: shape (N, C, T)
    labels: shape (N,)
    """
    segment_length = features.shape[2] // num_segments

    augmented_features = []
    augmented_labels = []

    for idx in range(features.shape[0]):
        base = features[idx]
        label = labels[idx]
        
        # Indici di epoche con la stessa classe (diverse dalla corrente)
        same_class_idx = np.where((labels == label) & (np.arange(len(labels)) != idx))[0]
        if len(same_class_idx) == 0:
            continue  # skip se non ci sono altri esempi della stessa classe

        for _ in range(num_aug_per_sample):
            # Copia dell'epoca originale
            new_epoch = np.copy(base)
            # Segmento da sostituire
            seg_idx = np.random.randint(0, num_segments)
            start = seg_idx * segment_length
            end = (seg_idx + 1) * segment_length
            # elemento casuale della stessa classe da cui estrarre il segmento
            same_class_segment_idx = np.random.choice(same_class_idx)
            same_class_segment = features[same_class_segment_idx]
            new_epoch[:, start:end] = same_class_segment[:, start:end]

            augmented_features.append(new_epoch)
            augmented_labels.append(label)

    # Concatenazione dei dati originali + augmented
    features_aug = np.concatenate([features, np.array(augmented_features)], axis=0)
    labels_aug = np.concatenate([labels, np.array(augmented_labels)], axis=0)

    return features_aug, labels_aug


def segment_and_rec_total_augmentation(features, labels, dataset="2a"):
    """
    Augmentation per segmentazione e ricostruzione.
    Divide ogni epoca in 8 segmenti e ricombina segmenti della stessa classe.
    Ritorna features e labels aumentati (concat con quelli originali).
    In questo modo creo un nuovo sample completamente nuovo dato dalla concatenzatione con segmenti casuali
    della stessa classe. Come EEG-Conformer
    """

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
                # Prendi una epoca casuale della stessa classe
                rand_feat = feats_cls[np.random.randint(0, len(feats_cls))]
                seg = rand_feat[:, i*segment_length:(i+1)*segment_length]
                segments.append(seg)
            # Ricompone epoca
            new_sample = np.concatenate(segments, axis=1)  # (C, 1008)
            synthetic_samples.append(new_sample)
            synthetic_labels.append(cls)

    part = np.array(synthetic_samples)
    labels_part = np.array(synthetic_labels).reshape(-1, 1)
    features_aug = np.concatenate([features, part], axis=0)
    labels_aug = np.concatenate([labels, labels_part], axis=0)

    # Augementation, per non confondere train pure con augmented
    is_real = np.concatenate([
    np.ones(len(features), dtype=int),
    np.zeros(len(part), dtype=int)
    ])

    return features_aug, labels_aug, is_real

def segment_and_rec_spectrogram_batch_augmentation(inputs, labels, n_segments=8):
    """
    Data augmentation per batch di spettrogrammi.
    Combina segmenti temporali da epoche della stessa classe.

    inputs: tensor (B, C, F, T)
    labels: tensor (B,)
    returns: augmented_inputs (B, C, F, T), augmented_labels (B,)
    """
    B, C, F, T = inputs.shape
    segment_len = T // n_segments
    device = inputs.device
    unique_labels = labels.unique()

    aug_inputs = []
    aug_labels = []

    for cls in unique_labels:
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        cls_inputs = inputs[cls_idx]  # (N_cls, C, F, T)

        for _ in range(len(cls_idx)):
            segments = []
            for seg_id in range(n_segments):
                rand_idx = torch.randint(0, len(cls_inputs), (1,))
                seg = cls_inputs[rand_idx][:, :, :, seg_id * segment_len : (seg_id + 1) * segment_len]
                segments.append(seg.squeeze(0))  # (C, F, segment_len)

            new_sample = torch.cat(segments, dim=-1)  # concat lungo tempo → (C, F, T)
            aug_inputs.append(new_sample.unsqueeze(0))
            aug_labels.append(cls.item())

    aug_inputs = torch.cat(aug_inputs, dim=0)  # (B, C, F, T)
    aug_labels = torch.tensor(aug_labels, dtype=torch.long, device=device)

    return aug_inputs.to(device), aug_labels.to(device)




def three_augmentation(features, labels):
    """
    Divide ogni trial in 3 segmenti uguali lungo l'asse temporale.
    Restituisce un dataset triplicato con le stesse etichette.
    Non divido per 2 perchè poi non è divisibile per 16.
    Questo è l'augmentation usato nel paper MVIT.
    """
    n_trials, n_channels, n_times = features.shape
    assert n_times % 3 == 0, "Il numero di timepoint deve essere divisibile per 3"
    third = n_times // 3
    part1 = features[:, :, :third]
    part2 = features[:, :, third:2*third]
    part3 = features[:, :, 2*third:]
    features_aug = np.concatenate([part1, part2, part3], axis=0)
    labels_aug = np.concatenate([labels, labels, labels], axis=0)
    return features_aug, labels_aug

def bandpass_filter_raw(raw, l_freq=LOW_FREQ, h_freq=HIGH_FREQ, order=5, rs=40):
    """
    Questo tipo di filtro non aggiunge sfasamenti e quindi ritardi,
    è lo stesso di prima col parametro "phase = 0"
    """
    fs = raw.info['sfreq']
    b, a = butter(order, [l_freq, h_freq], btype='bandpass', fs=fs)
    #b, a = cheby2(order, rs=rs, Wn=[l_freq, h_freq], btype='bandpass', fs=fs)

    
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_data = raw.get_data(picks=eeg_picks)

    # Applica filtfilt canale per canale
    filtered_data = np.array([
        filtfilt(b, a, channel)
        for channel in eeg_data
    ])

    # Sovrascrive i dati EEG filtrati nel Raw
    raw._data[eeg_picks] = filtered_data

    return raw

def channel_cluster_swapping(features, labels, n_clusters=3, augment_per_sample=1):
    """
    features: ndarray (N, C, T)
    labels: ndarray (N,)
    n_clusters: numero di cluster di canali (K-means sui canali)
    augment_per_sample: quante versioni augmentate per epoca
    """
    N, C, T = features.shape

    # 1. Calcola la matrice di correlazione media tra i canali (shape C x C)
    corr_matrix = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i != j:
                all_corr = [pearsonr(features[n, i], features[n, j])[0] for n in range(N)]
                corr_matrix[i, j] = np.mean(all_corr)
            else:
                corr_matrix[i, j] = 1.0

    # 2. Applica KMeans sulla correlazione per clusterizzare i canali
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(corr_matrix)

    # 3. Crea nuovi sample scambiando i cluster tra trial della stessa classe
    synthetic_features = []
    synthetic_labels = []

    for idx in range(N):
        label = labels[idx]
        x = features[idx]

        # Trova altri sample con la stessa etichetta
        same_class_idx = np.where((labels == label) & (np.arange(N) != idx))[0]
        if len(same_class_idx) == 0:
            continue

        for _ in range(augment_per_sample):
            new_x = np.copy(x)

            # Scegli un altro trial casuale della stessa classe
            other_idx = np.random.choice(same_class_idx)
            other_x = features[other_idx]

            # Per ogni cluster, scambia i canali
            for c in range(n_clusters):
                cluster_channels = np.where(cluster_labels == c)[0]
                new_x[cluster_channels] = other_x[cluster_channels]

            synthetic_features.append(new_x)
            synthetic_labels.append(label)

    # Concatenazione finale
    synthetic_features = np.array(synthetic_features)
    synthetic_labels = np.array(synthetic_labels)
    
    features_aug = np.concatenate([features, synthetic_features], axis=0)
    labels_aug = np.concatenate([labels, synthetic_labels], axis=0)

    return features_aug, labels_aug

def flip_augmentation(features, labels):
    # features: [N, C, T]
    # Applica flip verticale: x'' = max(x) - x per ogni canale
    flipped = features.max(axis=2, keepdims=True) - features  # [N, C, T]

    # Concatena i dati originali e flippati
    features_aug = np.concatenate([features, flipped], axis=0)
    labels_aug = np.concatenate([labels, labels], axis=0)

    return features_aug, labels_aug

def split_raw_segments(features, labels, num_segments=4):
    """
    features: np.array (B, C, T)
    labels: np.array (B,) oppure (B,1)
    """
    B, C, T = features.shape
    seg_len = T // num_segments  # lunghezza di ogni segmento

    # Divido in segmenti
    features_split = np.concatenate(
        np.split(features, num_segments, axis=2), axis=0
    )  # (B*num_segments, C, T/num_segments)

    # Replico le label
    labels_split = np.repeat(labels, num_segments, axis=0)

    return features_split, labels_split


#mano sinistra, destra, piedi, lingua
#total_normalization perchè molti, tra cui il conformer, fanno così
#cheby è [4,40]
#butter è [8,30]
#tmin=2, tmax=6.028
def read_data(path, tmin=2, tmax=6.028, is_test=False, augment = False, filter = "Butter"):
    raw=mne.io.read_raw_gdf(path,preload=True,
                            eog=['EOG-left', 'EOG-central', 'EOG-right'])
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    #event_id = dict(left=769, right=770, feet=771, tongue=772)
    if filter == "Butter":
        raw.filter(l_freq=8, h_freq=30, method='iir', iir_params=dict(order=5, ftype='butter'), phase='zero') # filtro Butterworth [8,30]Hz ordine 5
        #bandpass_filter_raw(raw)
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
        

    #raw=bandpass_filter_raw(raw)

    raw.set_eeg_reference()
    events=mne.events_from_annotations(raw)
    if is_test:
        # Carico tutte le epoche disponibili (senza specificare event_id)
        subj = os.path.basename(path)[1:3]  #A02E.gdf → '02'
        epochs = mne.Epochs(raw, events[0], event_id = [6],
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)

        # Carico le etichette vere dal file .mat
        labels_path = f'../Python/BciCompetitionIv2a/true_labels/A{subj}E.mat'
        true = loadmat(labels_path) # La cross-entropy vuole che partano da 0 le labels
        labels = true['classlabel'] - 1
        features=epochs.get_data()
        print("ciao")
        #features = channel_normalization(features) #normalizzo
        features = trial_based_normalization(features)
        print("ciao2")
        #features = savgol_filter(features, window_length=11, polyorder=3)

        #features = total_normalization(features)
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
        #labels=epochs.events[:,-1] - 7 # converto da [7,8,9,10] a [0,1,2,3]
        labels_path = f'../Python/BciCompetitionIv2a/true_labels/A{subj}T.mat'
        true = loadmat(labels_path) # La cross-entropy vuole che partano da 0 le labels
        labels = true['classlabel'] - 1
        features=epochs.get_data()
        if augment == True:
            print("Augmentation...")
            #features, labels = three_augmentation(features, labels)
            #features, labels = segment_and_rec_1fold_augmentation(features, labels)
            features, labels, is_real = segment_and_rec_total_augmentation(features, labels)
            #features, labels = channel_cluster_swapping(features, labels)
            #features, labels = flip_augmentation(features, labels)
        #features = channel_normalization(features) #normalizzo
        #features = savgol_filter(features, window_length=11, polyorder=3)
        features = trial_based_normalization(features)
        #features = total_normalization(features)
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
    #mne.set_log_level('ERROR')


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

            #raw.set_eeg_reference() ?????????????????????????????
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
            labels = true['classlabel'] - 1  # [1, 2] → [0, 1]
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

            # Leggi segnale
            raw = mne.io.read_raw_gdf(file_path, preload=True)
            #raw.drop_channels(['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])

            raw.filter(
            l_freq=4, h_freq=38,
            fir_design='firwin',
            fir_window='blackman'    
            )

            #events, _ = mne.events_from_annotations(raw)

            events, event_dict = mne.events_from_annotations(raw) 
            event_id = {'Left': event_dict['769'], 'Right': event_dict['770']}
            selected_events = events[np.isin(events[:, 2], list(event_id.values()))]              
            raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                            exclude='bads')
            epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=tmin,tmax=tmax,preload=True,baseline=None)
            features = epochs.get_data()

            # Leggi etichette dal file .mat corrispondente
            mat_path = os.path.join(base_path, "true_labels", f"B{subject_id}0{session}T.mat")
            true = loadmat(mat_path)
            labels = true['classlabel'] - 1  # [1, 2] → [0, 1]
            # print(labels)

            features = trial_based_normalization(features)
            all_features.append(features)
            all_labels.append(labels)# Concatena tutte le sessioni
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print("Train Features shape: ",features.shape)
        print("Train Labels shape: ",labels.shape)
        print("Train dataset done")
            
    return features, labels

def read_mat_data(dir_path, dataset_type, n_sub, mode='train', augment=False):

    if mode=='train':
        mode_s = 'T'
    else:
        mode_s = 'E'
    data_mat = scipy.io.loadmat(dir_path + '{}{:02d}{}.mat'.format(dataset_type, n_sub, mode_s))
    data = data_mat['data']  # (288, 22, 1000)
    labels =data_mat['label']
    features = channel_normalization(data) #normalizzo
    if augment == True:
        print("Augmentation...")
        features, labels = three_augmentation(features, labels)
    return features, labels

def block_reduce_sum(arr, block_size_freq, block_size_time):
    n_epochs, n_channels, n_freqs, n_times = arr.shape
    f_blocks = n_freqs // block_size_freq
    t_blocks = n_times // block_size_time
    
    arr = arr[:, :, :f_blocks*block_size_freq, :t_blocks*block_size_time]
    arr = arr.reshape(n_epochs, n_channels, f_blocks, block_size_freq,
                        t_blocks, block_size_time)
    return arr.sum(axis=(3, 5))  # somma su blocchi freq e tempo


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
        mean = np.mean(wvlts, axis=(0, 1, 2), keepdims=True)
        std  = np.std(wvlts, axis=(0, 1, 2), keepdims=True)
    wvlts = (wvlts - mean) / (std + 1e-10)
    
    wvlts = torch.tensor(wvlts, dtype=torch.float32)

    # Ridimensiona a 224x224 (bilinear interpolation)
    wvlts = F.interpolate(
        wvlts, size=(224, 224), mode='bilinear', align_corners=False
    )
    # wvlts = np.log1p(wvlts)
    
    # mean = np.mean(wvlts, axis=(0), keepdims=True)
    # std = np.std(wvlts, axis=(0), keepdims=True)
    
    # wvlts = (wvlts - mean) / (std + 1e-10)
    return wvlts.numpy(), mean, std

def debug_data_stats(x_train, y_train, x_test, y_test, R_mean_inv_sqrt=None):
    import numpy as np

    # Converti in numpy se sono tensori
    if not isinstance(x_train, np.ndarray):
        x_train = x_train.numpy()
    if not isinstance(x_test, np.ndarray):
        x_test = x_test.numpy()
    
    print("=== SHAPES ===")
    print("Train:", x_train.shape, "Test:", x_test.shape)
    print("Train labels:", y_train.shape, "Test labels:", y_test.shape)

    print("\n=== VALUE RANGE ===")
    print("Train min/max:", x_train.min(), x_train.max())
    print("Test min/max:", x_test.min(), x_test.max())
    print("Train mean/std:", x_train.mean(), x_train.std())
    print("Test mean/std:", x_test.mean(), x_test.std())

    # Per canale
    n_channels = x_train.shape[1]
    for ch in range(n_channels):
        print(f"Channel {ch}: TRAIN mean={x_train[:,ch].mean():.4f}, TEST mean={x_test[:,ch].mean():.4f}, "
              f"std={x_train[:,ch].std():.4f}/{x_test[:,ch].std():.4f}")

    # Controlla Euclidean Alignment
    if R_mean_inv_sqrt is not None:
        print("\n=== EUCLIDEAN ALIGNMENT ===")
        # Covarianza media primo trial
        cov_train = np.cov(x_train[0].reshape(n_channels, -1))
        cov_test  = np.cov(x_test[0].reshape(n_channels, -1))
        print("Covariance (first train trial):\n", cov_train)
        print("Covariance (first test trial):\n", cov_test)
        print("R_mean_inv_sqrt used:\n", R_mean_inv_sqrt)

    # Dummy prediction
    from sklearn.metrics import accuracy_score
    dummy_pred_train = (x_train.mean(axis=(1,2,3)) > 0).astype(int)
    dummy_pred_test  = (x_test.mean(axis=(1,2,3)) > 0).astype(int)
    print("\n=== DUMMY ACCURACY ===")
    print("Train dummy acc:", accuracy_score(y_train, dummy_pred_train))
    print("Test dummy acc :", accuracy_score(y_test, dummy_pred_test))


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
            #x_train, y_train = read_mat_data("mymat_raw/", "A", int(subject_id[2:3]), mode='train', augment=False)
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
            #x_test, y_test = read_mat_data("mymat_raw/", "A", int(subject_id[2:3]), mode='test', augment=False)
        else:
            root_test = root_2b
            x_test, y_test = read_data_2b(subject_id, root_test, augment=augment, filter=filter, is_test = True)
            x_test, R = euclidean_alignment(x_test, R)
        x_test, mean, std = compute_morlet_spectrogram(x_test, sfreq=250, mean=mean, std=std)

        debug_data_stats(x_train, y_train, x_test, y_test, R_mean_inv_sqrt=R)   
        #print(x_test.shape)

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
        #print(x_test.shape)

        # DATASET
        test_dataset = EEGSpectrogramDataset(x_test, y_test)
        return test_dataset



# batch_size = 8
# train_dataset, test_dataset = prepare_dataloaders()
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# print(len(train_loader))







