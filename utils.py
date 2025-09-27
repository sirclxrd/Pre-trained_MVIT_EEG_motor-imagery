import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import os
import yaml
import argparse
import torch
import csv
import seaborn as sns
import torch




LOW_FREQ = 8
HIGH_FREQ = 30

def plot_spectrogram(wavelet_features, epoch=0, channel=0, freqs=np.linspace(LOW_FREQ, HIGH_FREQ, 32), sfreq=250):
    print(f"LowFreq {LOW_FREQ}, HIG_FREQ {HIGH_FREQ}")
    power = wavelet_features[epoch, channel, :, :]
    times = np.arange(power.shape[1]) / sfreq
    plt.imshow(power, aspect='auto', origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]]) #sono il punto di inizio asse x, fine asse x, inizio y, fine y
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power')
    plt.title(f"Spectrogram - Epoch {epoch}, Channel {channel}")
    plt.show()

def visualize_patches_grid(x, patch_size = 16):
    """
    Visualizza le patch come una mappa (griglia) ricostruita dalla divisione dell'immagine.
    Mostra il primo esempio del batch.
    """
    B, C, H, W = x.shape
    ph, pw = patch_size, patch_size
    assert H % ph == 0 and W % pw == 0, "L'immagine deve essere divisibile per la patch size"
    n_h = H // ph
    n_w = W // pw
    # Estrai le patch grezze
    patches = rearrange(x, 'b c (h ph) (w pw) -> b (h w) ph pw c', ph=ph, pw=pw)
    patches = patches[0]  # primo batch
    fig, axes = plt.subplots(n_h, n_w, figsize=(n_w, n_h))
    for i in range(n_h):
        for j in range(n_w):
            idx = i * n_w + j
            patch = patches[idx]  # [ph, pw, C]

            if C == 1:
                axes[i, j].imshow(patch.squeeze(-1), cmap='gray')
            else:
                axes[i, j].imshow(patch[:,:,0], cmap="viridis") #visualizza solo il primo canale
            
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Patch map", y=1.02)
    plt.show()

def visualize_train_loss_acc(epoch_loss, epoch_acc, epoch_val_loss=[], epoch_val_acc=[], save_path="trainlossacc.png"):
    has_val = len(epoch_val_loss) != 0
    nrows = 2 if has_val else 1
    plt.figure(figsize=(12, 8 if has_val else 4))

    # Train Loss
    plt.subplot(nrows, 2, 1)
    plt.plot(epoch_loss, label='Train Loss', color='red')
    plt.xticks(range(0, len(epoch_loss), 10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss per Epoch')
    plt.grid(True)
    plt.legend()

    # Train Accuracy
    plt.subplot(nrows, 2, 2)
    plt.plot(epoch_acc, label='Train Accuracy', color='green')
    plt.xticks(range(0, len(epoch_acc), 10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy per Epoch')
    plt.grid(True)
    plt.legend()

    if has_val:
        # Validation Loss
        plt.subplot(nrows, 2, 3)
        plt.plot(epoch_val_loss, label='Validation Loss', color='blue')
        plt.xticks(range(0, len(epoch_val_loss), 5))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss per Epoch')
        plt.grid(True)
        plt.legend()

        # Validation Accuracy
        plt.subplot(nrows, 2, 4)
        plt.plot(epoch_val_acc, label='Validation Accuracy', color='orange')
        plt.xticks(range(0, len(epoch_val_acc), 5))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy per Epoch')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    #print(f"Grafico salvato in: {os.path.abspath(save_path)}")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def create_checkpoints_folders(config_path, single = True, pret = False, docker_prefix = ""):
    base_name = os.path.splitext(os.path.basename(config_path))[0]  # "single_config_16_2_2"
    folder_name = base_name.split("_")[-3:]  # ["16", "2", "2"]
    folder_name = "_".join(folder_name)      # "16_2_2"
    if single == True and pret == False:
        save_root = os.path.join(docker_prefix + "Single_checkpoints", folder_name)  # ad es. "Single_checkpoints/16_2_2"
    elif single == False and pret == False:
        save_root = os.path.join(docker_prefix + "Multi_checkpoints", folder_name)
    elif single == False and pret == True:
        save_root = os.path.join(docker_prefix + "Pretrained_Multi_checkpoints", folder_name)
    else:
        save_root = os.path.join(docker_prefix + "Pretrained_Single_checkpoints", folder_name)
    graphs_path = os.path.join(save_root, "graphs")
    log_path = os.path.join(save_root, "log_" + base_name + ".txt")
    os.makedirs(graphs_path, exist_ok=True)
    return save_root, graphs_path, log_path

def append_to_log_file(filepath, text):
    with open(filepath, 'a') as f:
        f.write(text + "\n")  

# togliere top1 che è per l'ensamble
def load_only_model(load_path, subject, model, val):
    if val == False:
        checkpoint = torch.load(load_path + "/" + subject + ".pth", map_location='cuda')
    else:
        checkpoint = torch.load(load_path + "/v_" + subject + ".pth", map_location='cuda')
        #checkpoint = torch.load(load_path + "/val_M" + subject + ".pth", map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

#val_ è quello pretrainato fino a graph3, ora provo v_
def save_model(val_loss, i, model, optimizer, scheduler, subject, save_path, sched_on, epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc ):
    if sched_on:
        torch.save({
                    'loss': val_loss,
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch_loss': epoch_loss,
                    'epoch_acc': epoch_acc,
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_acc': epoch_val_acc
        }, save_path + "/v_" +subject + ".pth")
    else:
        torch.save({
                    'loss': val_loss,
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
        }, save_path + "/v_" +subject + ".pth")

        

def average_models(model, model_paths, device='cuda'):

    # Carica tutti gli state dict
    state_dicts = [ torch.load(p, map_location=device)['model_state_dict'] for p in model_paths ]

    # Copia il primo come base
    avg_sd = state_dicts[0].copy()
    N = len(state_dicts)

    for key in avg_sd:
        if avg_sd[key].dtype.is_floating_point:
            # Stack e media
            stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
            avg_sd[key] = torch.mean(stacked, dim=0)

    # Crea il modello e carica i pesi mediati
    model.load_state_dict(avg_sd)
    return model

def flatten_dict(d, parent_key='', sep='.'):
    """Appiattisce un dizionario annidato."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def config_csv(config, csv_path = "logfile.csv", mean_accuracy = "0.25"):
    flat_config = flatten_dict(config)

    # aggiungi altre metriche, es. total_test_acc
    flat_config['test.mean_accuracy'] = mean_accuracy

    # Scrittura CSV (aggiunge intestazione solo se file non esiste)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=flat_config.keys())

        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_config)

def subject_csv(acc_list, csv_path="logfile_subjects.csv",testname="Test1"):
    subject_ids = [f"A0{i+1}" for i in range(len(acc_list))]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Scrive intestazione solo se il file non esiste
        if not file_exists:
            header = ["test_name"] + subject_ids
            writer.writerow(header)

        # Scrive la nuova riga di accuracy
        row = [testname] + acc_list
        writer.writerow(row)

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_probing_example(model, train_loader, mask_patch=400, channel=None, path="prob_example.png"):
    """
    Esegue probing su un singolo spettrogramma dal train_loader.
    Se `channel` è un int -> mostra solo quel canale
    Se `channel` è None -> mostra tutti i canali in colonne separate
    """
    model = model.cpu()
    batch, labels = next(iter(train_loader))
    x = batch[0:1]  # [1, C, H, W]
    C = x.shape[1]

    if channel is not None:
        channels_to_plot = [channel]
    else:
        channels_to_plot = list(range(C))

    fig, axs = plt.subplots(len(channels_to_plot), 3, figsize=(18, 5 * len(channels_to_plot)))

    if len(channels_to_plot) == 1:
        axs = np.expand_dims(axs, 0)

    for row_idx, ch in enumerate(channels_to_plot):
        with torch.no_grad():
            pred, masked = model.plot_patches(x, mask_patch, channel=ch)

        orig = x[0, ch].detach().cpu().numpy()
        masked_arr = masked[0, 0].detach().cpu().numpy()
        pred_arr = pred[0, 0].detach().cpu().numpy()

        mask_positions = (masked_arr == 99)
        correct_positions = (pred_arr == 99)

        overlay_pred = np.zeros((*orig.shape, 4), dtype=np.float32)
        overlay_pred[mask_positions & correct_positions] = [0, 1, 0, 0.4]
        overlay_pred[mask_positions & ~correct_positions] = [1, 0, 0, 0.4]

        overlay_mask = np.zeros((*orig.shape, 4), dtype=np.float32)
        overlay_mask[mask_positions] = [0, 0, 1, 0.4]

        axs[row_idx, 0].imshow(orig, aspect='auto', origin='lower', cmap='viridis')
        axs[row_idx, 0].set_title(f"Spectrogram (ch={ch})")
        axs[row_idx, 0].axis('off')

        axs[row_idx, 1].imshow(orig, aspect='auto', origin='lower', cmap='viridis')
        axs[row_idx, 1].imshow(overlay_mask, aspect='auto', origin='lower')
        axs[row_idx, 1].set_title("Masked patches")
        axs[row_idx, 1].axis('off')

        axs[row_idx, 2].imshow(orig, aspect='auto', origin='lower', cmap='viridis')
        axs[row_idx, 2].imshow(overlay_pred, aspect='auto', origin='lower')
        axs[row_idx, 2].set_title("Correct predictions")
        axs[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)

    model = model.to('cuda')







