import os
import h5py
import numpy as np
from mne.time_frequency import tfr_array_morlet
from scipy.signal import butter, sosfiltfilt
from utils import append_to_log_file

def butter_bandpass_filter(data, lowcut=8, highcut=30, fs=250, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1)

def preprocess_folder(folder, sfreq=250, freqs=np.linspace(4,80,32), n_cycles=7,
                      apply_filter=False):
    print(f"Processing {folder}")
    append_to_log_file("./TUHSpectro.txt", f"Processing {folder}")

    mean = np.load(os.path.join(folder, "mean.npy")).squeeze()
    std = np.load(os.path.join(folder, "standard_deviation.npy")).squeeze()

    all_files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".h5")]
    special_indices = [4, 5, 17] #indici per il 2b

    # --- PASSO 1: calcolo spettrogrammi grezzi e li salvo ---
    raw_specs_paths = []
    for idx, fpath in enumerate(all_files[6:], start=6):
        append_to_log_file("./TUHSpectro.txt", f"fpath: {fpath}")
        with h5py.File(fpath, "r") as f:
            sig = f["signals"][:]
            sig = sig[:, special_indices, :]

        if apply_filter:
            sig = butter_bandpass_filter(sig, fs=sfreq)

        sig = (sig - mean[None, special_indices, None]) / (std[None, special_indices, None] + 1e-10)
        spec = tfr_array_morlet(sig, sfreq=sfreq, freqs=freqs,
                                n_cycles=n_cycles, output='power', n_jobs=1)
        spec = np.log1p(spec)
        append_to_log_file("./TUHSpectro.txt", f"spec: {spec.shape}")

        outname = os.path.join(folder, f"spec_raw_{idx:04d}.h5")
        with h5py.File(outname, "w") as h5:
            h5.create_dataset("spectrogram", data=spec, compression="gzip")
        raw_specs_paths.append(outname)
        append_to_log_file("./TUHSpectro.txt", f"Saved raw {outname}")

    # --- PASSO 2: calcolo media e std globali ---
    n_total = 0
    mean_spec = None
    M2 = None
    for path in raw_specs_paths:
        with h5py.File(path, "r") as f:
            spec = f["spectrogram"][:]
        S = spec.shape[0]
        if mean_spec is None:
            mean_spec = np.mean(spec, axis=0)
            M2 = np.sum((spec - mean_spec[None, :, :, :])**2, axis=0)
        else:
            mean_old = mean_spec.copy()
            mean_spec = (mean_old * n_total + np.sum(spec, axis=0)) / (n_total + S)
            M2 += np.sum((spec - mean_old[None, :, :, :])**2, axis=0)
        n_total += S

    std_spec = np.sqrt(M2 / n_total)
    np.save(os.path.join(folder, "mean_spec.npy"), mean_spec)
    np.save(os.path.join(folder, "std_spec.npy"), std_spec)
    append_to_log_file("./TUHSpectro.txt", "Saved mean_spec and std_spec")

    # --- PASSO 3: applico normalizzazione e salvo finali ---
    for path in raw_specs_paths:
        with h5py.File(path, "r") as f:
            spec = f["spectrogram"][:]
        spec_norm = (spec - mean_spec[None, :, :, :]) / (std_spec[None, :, :, :] + 1e-10)

        with h5py.File(path, "w") as h5:
            h5.create_dataset("spectrogram", data=spec_norm, compression="gzip")
        append_to_log_file("./TUHSpectro.txt", f"Saved normalized {outname}")


def calculate_mean(folder, raw_specs_paths):
    
    #--- calcolo media e std globali ---
    n_total = 0
    mean_spec = None
    M2 = None
    for path in raw_specs_paths:
        with h5py.File(path, "r") as f:
            spec = f["spectrogram"][:]
        S = spec.shape[0]
        if mean_spec is None:
            mean_spec = np.mean(spec, axis=0)
            M2 = np.sum((spec - mean_spec[None, :, :, :])**2, axis=0)
        else:
            mean_old = mean_spec.copy()
            mean_spec = (mean_old * n_total + np.sum(spec, axis=0)) / (n_total + S)
            M2 += np.sum((spec - mean_old[None, :, :, :])**2, axis=0)
        n_total += S

    std_spec = np.sqrt(M2 / n_total)
    np.save(os.path.join(folder, "mean_spec.npy"), mean_spec)
    np.save(os.path.join(folder, "std_spec.npy"), std_spec)
    append_to_log_file("./TUHSpectro.txt", "Saved mean_spec and std_spec")

def calculate_spectros(folder, sfreq=250, freqs=np.linspace(4,80,32), n_cycles=7,
                      apply_filter=False):
    print(f"Processing {folder}")
    append_to_log_file("./TUHSpectro.txt", f"Processing {folder}")

    mean = np.load(os.path.join(folder, "mean.npy")).squeeze()
    std = np.load(os.path.join(folder, "standard_deviation.npy")).squeeze()

    all_files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".h5")]
    special_indices = [4, 5, 17] #indici per il 2b

    # --- PASSO 1: calcolo spettrogrammi grezzi e li salvo ---
    raw_specs_paths = []
    for idx, fpath in enumerate(all_files[0:1]):
        append_to_log_file("./TUHSpectro.txt", f"fpath: {fpath}")
        with h5py.File(fpath, "r") as f:
            sig = f["signals"][:]
            sig = sig[:, special_indices, :]

        if apply_filter:
            sig = butter_bandpass_filter(sig, fs=sfreq)

        sig = (sig - mean[None, special_indices, None]) / (std[None, special_indices, None] + 1e-10)
        spec = tfr_array_morlet(sig, sfreq=sfreq, freqs=freqs,
                                n_cycles=n_cycles, output='power', n_jobs=3)
        spec = np.log1p(spec)
        append_to_log_file("./TUHSpectro.txt", f"spec: {spec.shape}")

        outname = os.path.join(folder, f"spec_raw_{idx:04d}.h5")
        with h5py.File(outname, "w") as h5:
            h5.create_dataset("spectrogram", data=spec, compression="gzip")
        raw_specs_paths.append(outname)
        append_to_log_file("./TUHSpectro.txt", f"Saved raw {outname}")
    

# --- USO ---
paths = [
#    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUEP/Normal/REF"
]

raw_specs_paths = [
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF/spec_raw_0000.h5",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF/spec_raw_0001.h5",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF/spec_raw_0002.h5",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF/spec_raw_0003.h5",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF/spec_raw_0004.h5",
    "../../../mnt/localstorage/cdeangelis/Dataset_bipolar_TUH/TUAB/Normal/REF/spec_raw_0005.h5"
]

#calculate_mean(paths[0], raw_specs_paths)
for p in paths:
    calculate_spectros(p, apply_filter = False)
