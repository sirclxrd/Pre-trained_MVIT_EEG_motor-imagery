import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from mne.time_frequency import tfr_array_morlet  # libreria MNE

class EEGFileLoader:
    """
    Loader EEG sequenziale: può caricare i segnali grezzi o spettrogrammi già salvati.
    Supporta split train/validation a livello di file.
    """

    def __init__(self, paths, dataset=None, znorm=True, batch_size=32, 
                 shuffle=True, device="cpu", mode="train", split_ratio=0.8,
                 sfreq=250, freqs=np.linspace(4, 80, 32), n_cycles=7, 
                 load_spectrograms=False, random_seed=2025):
        """
        paths: lista di cartelle
        dataset: None o '2a'/'2b'
        znorm: applica z-normalizzazione (solo se si leggono i segnali grezzi)
        batch_size: dimensione dei mini-batch
        shuffle: shuffle dei batch
        device: 'cpu' o 'cuda'
        mode: 'train' o 'val'
        split_ratio: percentuale di file per train
        sfreq: frequenza di campionamento
        freqs: array di frequenze per spettrogramma
        n_cycles: numero di cicli della wavelet
        load_spectrograms: se True, carica spettrogrammi già pronti invece dei segnali grezzi
        """
        self.paths = paths
        self.dataset = dataset
        self.znorm = znorm
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.mode = mode
        self.split_ratio = split_ratio
        self.sfreq = sfreq
        self.freqs = freqs
        self.n_cycles = n_cycles
        self.load_spectrograms = load_spectrograms

        # Lista completa dei file
        all_files = []
        for folder in paths:
            for fname in sorted(os.listdir(folder)):
                if self.load_spectrograms:
                    if fname.startswith("spectro_") and fname.endswith(".h5"):
                        all_files.append(os.path.join(folder, fname))
                else:
                    if fname.endswith(".h5"):
                        all_files.append(os.path.join(folder, fname))

        rng = np.random.default_rng(random_seed)
        rng.shuffle(all_files)

        # split train/val
        split_idx = int(split_ratio * len(all_files))
        if mode == "train":
            self.files = all_files[:split_idx]
        elif mode == "val":
            self.files = all_files[split_idx:]
        else:
            raise ValueError("mode deve essere 'train' o 'val'")

    def morlet_wavelet_fft(self,signals, freqs, n_cycles, sfreq, device='cuda'):
        """
        signals: [S, C, T] tensore torch
        freqs: array di frequenze
        n_cycles: numero di cicli della wavelet
        sfreq: frequenza di campionamento
        ritorna: spettrogramma [S, C, F, T] (potenza)
        """
        S, C, T = signals.shape
        F = len(freqs)
        signals = signals.to(device)
        specs = []

        # FFT del segnale una volta sola
        sig_f = torch.fft.fft(signals, n=T, dim=-1)

        for f in freqs:
            sigma = n_cycles / (2 * np.pi * f)
            t = torch.arange(T, device=device) / sfreq
            wavelet = torch.exp(2j * np.pi * f * t) * torch.exp(-t**2 / (2*sigma**2))
            wavelet = wavelet / torch.sqrt((wavelet.abs()**2).sum())  # normalizzazione energia
            wav_f = torch.fft.fft(wavelet, n=T)

            conv = torch.fft.ifft(sig_f * wav_f[None, None, :], dim=-1)
            power = conv.abs()**2  # potenza
            specs.append(power.unsqueeze(2))  # [S, C, 1, T]

        specs = torch.cat(specs, dim=2)  # [S, C, F, T]
        return specs

    def __iter__(self):
        for file_path in self.files:
            folder = os.path.dirname(file_path)

            if not self.load_spectrograms and self.znorm:
                mean = np.load(os.path.join(folder, "mean.npy")).squeeze()
                std = np.load(os.path.join(folder, "standard_deviation.npy")).squeeze()

            with h5py.File(file_path, "r") as f:
                if self.load_spectrograms:
                    # carica spettrogrammi già normalizzati e log trasformati
                    data = f["spectrograms"][:]  # [S, C, F, T]
                else:
                    data = f["signals"][:]  # [S, C, T]

            # eventuale selezione canali per 2b
            if self.dataset == "2b":
                special_indices = [4, 5, 17]
                data = data[:, special_indices, :]  # operatore Ellipsis in modo da far combiare sia il caso raw che spectrogram senza specificare le dimensioni
                if not self.load_spectrograms and self.znorm:
                    mean = mean[special_indices]
                    std = std[special_indices]

            # se stiamo caricando i segnali grezzi, fai z-norm e Morlet
            if not self.load_spectrograms:
                if self.znorm:
                    data = (data - mean[None, :, None]) / (std[None, :, None] + 1e-10)
                # trasformazione Morlet
                data = torch.tensor(data, dtype=torch.float32, device='cuda')
                data = self.morlet_wavelet_fft(data, freqs=self.freqs,
                                        n_cycles=self.n_cycles,sfreq=self.sfreq)
                data = data.cpu()
                # data = tfr_array_morlet(data[0:1,:,:], sfreq=self.sfreq, freqs=self.freqs,
                #                 n_cycles=self.n_cycles, output='power', n_jobs=-1)
                
                data = np.log1p(data)
                print(data.shape)

            # converte in tensore PyTorch
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)

            # DataLoader per batch
            dataset_loader = DataLoader(
                TensorDataset(tensor),
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )

            for batch in dataset_loader:
                yield batch[0]  # batch[0] contiene lo spettrogramma pronto

paths = [
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUEP/Normal/REF"
]
train_loader = EEGFileLoader(paths, dataset="2b", znorm=True, batch_size=32, shuffle=True, mode="train", load_spectrograms = False)
for batch in train_loader:
    print("Batch train shape:", batch.shape)

#n_canali
#lunghezza segmento
#Ordine dei canali
#Shuffle sui file
#Normalizzazione spettrogramma
#salvare gli spettrogrammi in h5py