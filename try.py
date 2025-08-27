import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import tfr_array_morlet
import numpy as np

def total_normalization(features):
    all_values = features.flatten()
    mean = all_values.mean()
    std = all_values.std()
    features_norm = (features - mean) / std
    return features_norm

def channel_normalization(features):
    n_trials, n_channels, n_times = features.shape
    reshaped = features.transpose(1, 0, 2).reshape(n_channels, -1).T  # shape: (n_trials * n_times, n_channels)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(reshaped)

    # Ricomponi i dati normalizzati nella forma originale
    features_norm = normalized.T.reshape(n_channels, n_trials, n_times).transpose(1, 0, 2)
    return features_norm

path = "./BciCompetitionIv2a/Train/A01T.gdf"
def read_data(path):
    raw=mne.io.read_raw_gdf(path,preload=True,
                            eog=['EOG-left', 'EOG-central', 'EOG-right'])
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.filter(l_freq=8, h_freq=30, method='iir', iir_params=dict(order=5, ftype='butter'))

    # raw.filter(
    #     l_freq=4,
    #     h_freq=40,
    #     method='iir',
    #     iir_params=dict(order=6, ftype='cheby2', rs=40)
    # )
    
    raw.set_eeg_reference()
    events=mne.events_from_annotations(raw)
    print(events[0][9])        
    # events[0][7] è l'inizio, 500 campioni ovvero 2 sec dopo ho il primo task. Se vado prima di events[0][7] ovvero events[0][6] ho gli eventi tipo muovere gli occhi descritti nel dataset.
    epochs = mne.Epochs(raw, events[0], event_id=[6,7,8,9,10], tmin = 0, tmax = 7.5, baseline = None)
    labels=epochs.events[:,-1]
    features=epochs.get_data()
    #features = channel_normalization(features) #normalizzo
    return features,labels

features, labels = read_data(path)

def compute_morlet_spectrogram(features, sfreq, freqs=np.linspace(8, 30, 32), n_cycles=7):
    """
    features: ndarray (n_epochs, n_channels, n_times)
    sfreq: frequenza di campionamento (Hz), per BCIC IV 2a è 250 Hz
    freqs: array di frequenze su cui viene calcolata la wavelet, il mio segnale è [4,40]Hz quindi ne faccio 32 per coprirlo tutto
    n_cycles: indica quanto è lunga l'onda, compromesso tempo frequenza
    """
    n_cycles = freqs/2
    wvlts = tfr_array_morlet(features, sfreq=sfreq, freqs=freqs,
                             n_cycles=n_cycles, output='power', decim=1, n_jobs=4)
    return wvlts

import matplotlib.pyplot as plt

def plot_morlet_trials_by_class(features, labels, target_class, channel=20, n_trials=25, sfreq=250):
    selected_idx = np.where(labels == target_class)[0][:n_trials]
    if len(selected_idx) == 0:
        print(f"Nessuna epoca trovata per la classe {target_class}.")
        return

    # Estrai le epoche selezionate
    features_sel = features[selected_idx]  # shape (n_trials, n_channels, n_times)

    # Calcola spettrogramma
    spectrograms = compute_morlet_spectrogram(features_sel, sfreq)
    freqs = np.linspace(8, 30, 32)
    times = np.linspace(0, features.shape[-1] / sfreq, spectrograms.shape[-1])

    # Plot
    fig, axes = plt.subplots(3, 5, figsize=(18, 6))
    axes = axes.flatten()

    for i in range(min(n_trials, len(axes))):
        ax = axes[i]
        power = spectrograms[i, channel]  # shape (n_freqs, n_times)
        im = ax.imshow(np.log1p(power), aspect='auto', origin='lower',
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       cmap='viridis')
        ax.set_title(f"Trial {i+1} - Classe {target_class}")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequenza (Hz)")

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02)
    plt.suptitle(f"Spettrogramma Morlet - Canale {channel} - Classe {target_class}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_morlet_trials_by_class(features,labels, 8,channel=2)


# features = compute_morlet_spectrogram(features, sfreq = 250)
# print(features.shape) #n_channel, n_freq, tempo
# plot_spectrogram(features, epoch = 76, channel = 0)
    
#epochs = mne.Epochs(raw, events[0], event_id=[7,8,9,10], tmin = 0, tmax = 4, baseline = None)
#è uguale a mne.Epochs(raw, events[0], event_id=[6,7,8,9,10], tmin = 2, tmax = 6, baseline = None)
#98512 - 96509 = 2003/500 = 8 ovvero la durata di un singolo try
#98512 = events[0][9] inizio try successivo
#96509 = events[0][8] inizio try iniziale



    
