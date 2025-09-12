from Preprocessing import prepare_dataloaders
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch
from models.MVIT import MultiChannelViT
from models.pret_MVIT import pret_MVIT
import torch.nn as nn
import time
from utils import (visualize_train_loss_acc, load_config, create_checkpoints_folders, 
                   append_to_log_file, load_only_model, config_csv, subject_csv, plot_confusion_matrix)
import random
import yaml
import argparse
from torch.utils.data import random_split
import os
from math import ceil
import math
from torch.utils.data import ConcatDataset
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

def apply_confidence_threshold(probs, preds, thr=0.6):
    out = preds.copy()
    for t in range(1, len(preds)):
        if probs[t].max() < thr:
            out[t] = out[t-1]
    return out


device = 'cuda'
def test_model(model, test_loader, criterion, log_file = "log.txt"):
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    inference_times = []
    batch = 0
    all_preds = []
    all_labels = []
    all_probs = []


    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).squeeze().long()

            #tempo per un batch di campioni
            start_time = time.time()
            outputs, token = model(inputs)
            end_time = time.time()

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1
            inference_times.append(end_time - start_time)


    avg_loss = running_loss / batch
    accuracy = correct / total
    avg_inference_time = np.mean(inference_times) / inputs.shape[0]  # per campione

    txt = f"[TEST] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Avg Inference Time: {avg_inference_time*1000:.2f} ms/sample"
    print(txt)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    thresholded_preds = apply_confidence_threshold(all_probs, all_preds, 0.3)
    thresholded_accuracy = (thresholded_preds == all_labels).mean()

    txt = f"[TEST] Accuracy (thresholded): {thresholded_accuracy:.4f}"
    append_to_log_file(log_file, txt)
    return avg_loss, accuracy, all_preds, all_labels

def cls_token_importance(fc_layer, embed_dim=768, n_channels=22):
    importances = []
    for i in range(n_channels):
        w_i = fc_layer.weight[:, i*embed_dim:(i+1)*embed_dim]  # [num_classes, D]
        imp_i = torch.norm(w_i, p=2).item()
        importances.append((i, imp_i))
    importances.sort(key=lambda x: x[1], reverse=True)
    
    print("Channel ranking by CLS token importance:")
    for idx, imp in importances:
        print(f"Channel {idx}: importance = {imp:.4f}")
    return importances

import torch
import matplotlib.pyplot as plt
import cv2

def compute_grid_from_tokens(num_patches, H):
    """
    Trova grid_h, grid_w dati num_patches e altezza H.
    Cerca il divisore più grande di num_patches che sia <= H.
    Fallback: sqrt-based.
    """
    # divisori
    divisors = [d for d in range(1, int(math.sqrt(num_patches)) + 1) if num_patches % d == 0]
    all_divs = set()
    for d in divisors:
        all_divs.add(d)
        all_divs.add(num_patches // d)
    all_divs = sorted(all_divs)

    # scegli il divisore più grande <= H
    candidates = [d for d in all_divs if d <= H]
    if len(candidates) > 0:
        grid_h = max(candidates)
        grid_w = num_patches // grid_h
        return grid_h, grid_w

    # fallback: cerca il divisore più vicino a sqrt
    grid_h = int(math.floor(math.sqrt(num_patches)))
    while grid_h > 0:
        if num_patches % grid_h == 0:
            grid_w = num_patches // grid_h
            return grid_h, grid_w
        grid_h -= 1

    # ultimate fallback: (1, num_patches)
    return 1, num_patches


def visualize_attention_on_image_fixed(image_tensor, attn_weights, save_path,
                                       target_size=(224,224), cmap_name="jet"):
    """
    Normalizza lo scalogramma, costruisce overlay e salva immagine (target_size).
    - image_tensor: tensor [C,H,W] or [H,W]
    - attn_weights: tensor 1D (num_patches) (ATTENZIONE: deve essere solo CLS->patches, senza CLS)
    """
    # to numpy
    image = image_tensor.detach().cpu().numpy()
    if image.ndim == 3:
        image = image[0]   # primo canale

    # log-scaled / normalizzazione per scalogrammi
    image_proc = np.log1p(np.abs(image))
    image_proc = (image_proc - image_proc.min()) / (image_proc.max() - image_proc.min() + 1e-8)

    num_patches = int(attn_weights.shape[0])

    # pad se necessario (ma con compute_grid_from_tokens normalmente non serve)
    # converti attn a cpu tensor
    attn = attn_weights.detach().cpu()
    # compute grid (user must supply actual H for compute_grid_from_tokens earlier, here we compute from target)
    # We only reshape later; caller must ensure tokens were computed on this image shape.
    # Here we do not guess grid: we expect caller to provide attn in correct token order and an image H to compute grid.
    # For safety, we'll compute grid by divisors using original image height:
    H_orig = image.shape[0]
    grid_h, grid_w = compute_grid_from_tokens(num_patches, H_orig)

    if grid_h * grid_w > num_patches:
        pad_size = grid_h * grid_w - num_patches
        attn = F.pad(attn, (0, pad_size), value=0.0)
        num_patches = attn.shape[0]

    attn_map = attn.cpu().numpy().reshape((grid_h, grid_w))

    # resize everything to target_size
    Ht, Wt = target_size
    image_resized = cv2.resize(image_proc, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    attn_resized = cv2.resize(attn_map, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    cmap = plt.get_cmap(cmap_name)
    attn_color = cmap(attn_resized)[:,:,:3]  # RGB

    overlay = 0.6 * np.expand_dims(image_resized, axis=-1) + 0.4 * attn_color
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.imsave(save_path, overlay)
    # alternative: plt.figure + plt.imshow(...); plt.savefig(..., dpi=300)
    print(f"[saved] {save_path} (target_size={target_size}, grid={grid_h}x{grid_w})")


def extract_and_visualize_attention_channels(model, dataloader, device="cuda",
                                                  save_dir=".", target_size=(224,224),
                                                  max_channels=None):
    """
    Per ciascun encoder (assunto corrispondente a un canale):
     - prende il primo sample del dataloader
     - passa al patch_embed del relativo encoder (solo quel canale)
     - esegue l'encoder locale per ottenere attn (CLS->patches)
     - ricostruisce la griglia usando compute_grid_from_tokens (basata sull'altezza reale del canale)
     - salva overlay in save_dir
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0] if isinstance(batch, (tuple,list)) else batch
            x_batch = x_batch.to(device)   # [B,C,H,W]
            sample = x_batch[0:1]          # [1,C,H,W]
            B, C, H, W = sample.shape
            n_channels = C

            if max_channels is None:
                max_ch = n_channels
            else:
                max_ch = min(max_channels, n_channels)

            for ch_idx, encoder in enumerate(model.encoders[:max_ch]):
                # prendi solo il canale ch_idx
                channel_input = sample[:, ch_idx:ch_idx+1, :, :]   # [1,1,H,W]

                # ottieni tokens [1, N, D]
                tokens = encoder.patch_embed(channel_input)       # attenzione: metodo patch_embed

                # ora fai forward dell'encoder locale (assumiamo supporti return_attn=True)
                res = encoder.encoder(tokens, return_attn=True)
                # res può essere (out, attn) o attn (dipende implementazione)
                if isinstance(res, tuple) and len(res) == 2:
                    _, attn_all = res
                else:
                    attn_all = res

                # attn_all può essere una lista di layers; prendi l'ultimo se è lista
                if isinstance(attn_all, (list, tuple)):
                    attn_tensor = attn_all[-1]
                else:
                    attn_tensor = attn_all

                # typical shape: [B, num_heads, N+1, N+1]
                # estrai CLS->patches (escludi CLS nella colonna 0)
                # robust extraction:
                attn_tensor = attn_tensor.detach().cpu()
                if attn_tensor.ndim == 4:
                    # [B, heads, N, N] or [B, heads, N+1, N+1]
                    # preferiamo la prima batch
                    batch0 = attn_tensor[0]   # [heads, N, N]
                    head0 = batch0[0]         # [N, N]
                    # se ha token CLS in posizione 0, escludi indice 0
                    if head0.shape[0] == tokens.shape[1] + 1:
                        cls_attn = head0[0, 1:]
                    else:
                        # qualche implementazione non include CLS: allora usa row 0 -> drop?
                        # fallback: usa prima riga senza escludere
                        cls_attn = head0[0, :tokens.shape[1]]
                elif attn_tensor.ndim == 3:
                    # [heads, N, N] or [N, N] etc.
                    head0 = attn_tensor[0]
                    if head0.shape[0] == tokens.shape[1] + 1:
                        cls_attn = head0[0, 1:]
                    else:
                        cls_attn = head0[0, :tokens.shape[1]]
                else:
                    raise RuntimeError(f"Unexpected attn tensor ndim={attn_tensor.ndim}")

                # ora cls_attn è un 1D tensor length == tokens.shape[1] (o close)
                # compute grid using actual H of the input channel
                num_patches = int(tokens.shape[1])
                grid_h, grid_w = compute_grid_from_tokens(num_patches, H)

                # sanity: if product mismatch, pad
                if grid_h * grid_w != num_patches:
                    pad_size = grid_h * grid_w - num_patches
                    if pad_size > 0:
                        cls_attn = F.pad(cls_attn, (0, pad_size), value=0.0)

                # immagine del canale (primo sample)
                channel_image = channel_input[0,0].cpu()  # [H,W]

                save_path = os.path.join(save_dir, f"attention_channel_{ch_idx}.png")
                visualize_attention_on_image_fixed(channel_image, cls_attn, save_path, target_size=target_size)

                

            break  # solo primo batch

def extract_and_visualize_attention_channels_grid(model, dataloader, device="cuda",
                                                  save_dir=".", target_size=(224,224),
                                                  max_channels=None,
                                                  grid_filename="attention_grid.png"):
    """
    Come extract_and_visualize_attention_channels, ma costruisce anche una griglia
    con tutte le immagini overlay dei canali in un'unica figura.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0] if isinstance(batch, (tuple,list)) else batch
            x_batch = x_batch.to(device)   # [B,C,H,W]
            sample = x_batch[0:1]          # [1,C,H,W]
            B, C, H, W = sample.shape
            n_channels = C

            if max_channels is None:
                max_ch = n_channels
            else:
                max_ch = min(max_channels, n_channels)

            overlays = []   # immagini overlay normalizzate
            titles   = []   # titoli canali

            for ch_idx, encoder in enumerate(model.encoders[:max_ch]):
                channel_input = sample[:, ch_idx:ch_idx+1, :, :]   # [1,1,H,W]

                # ottieni tokens
                tokens = encoder.patch_embed(channel_input)

                # forward con attn
                res = encoder.encoder(tokens, return_attn=True)
                if isinstance(res, tuple) and len(res) == 2:
                    _, attn_all = res
                else:
                    attn_all = res

                if isinstance(attn_all, (list, tuple)):
                    attn_tensor = attn_all[-1]
                else:
                    attn_tensor = attn_all

                attn_tensor = attn_tensor.detach().cpu()
                if attn_tensor.ndim == 4:
                    batch0 = attn_tensor[0]   # [heads, N, N]
                    head0  = batch0[0]        # [N, N]
                    if head0.shape[0] == tokens.shape[1] + 1:
                        cls_attn = head0[0, 1:]
                    else:
                        cls_attn = head0[0, :tokens.shape[1]]
                else:
                    raise RuntimeError(f"Unexpected attn tensor shape {attn_tensor.shape}")

                channel_image = channel_input[0,0].cpu()

                # usa funzione già pronta
                #save_path = os.path.join(save_dir, f"attention_channel_{ch_idx}.png")
                #visualize_attention_on_image_fixed(channel_image, cls_attn,
                #                                   save_path, target_size=target_size)

                # ricostruisci overlay come numpy per metterlo in griglia
                image = channel_image.detach().cpu().numpy()
                image_proc = np.log1p(np.abs(image))
                image_proc = (image_proc - image_proc.min()) / (image_proc.max() - image_proc.min() + 1e-8)

                # evita neri puri
                image_proc = 0.1 + 0.9 * image_proc

                num_patches = int(tokens.shape[1])
                grid_h, grid_w = compute_grid_from_tokens(num_patches, H)
                if grid_h * grid_w != num_patches:
                    pad_size = grid_h * grid_w - num_patches
                    if pad_size > 0:
                        cls_attn = F.pad(cls_attn, (0, pad_size), value=0.0)

                attn_map = cls_attn.cpu().numpy().reshape((grid_h, grid_w))
                Ht, Wt = target_size

                image_resized = cv2.resize(image_proc, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
                cmap_img = plt.get_cmap("magma")   # spettrogramma in colori
                image_resized = cmap_img(image_resized)[:, :, :3]

                attn_resized = cv2.resize(attn_map, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
                attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

                cmap = plt.get_cmap("jet")
                attn_color = cmap(attn_resized)[:,:,:3]
                overlay = 0.5 * image_resized + 0.5 * attn_color
                overlay = np.clip(overlay, 0.0, 1.0)

                overlays.append(overlay)
                titles.append(f"Ch {ch_idx}")

            # --- Costruzione griglia finale ---
            n_cols = 6
            n_rows = math.ceil(max_ch / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
            axes = axes.flatten()

            for idx, overlay in enumerate(overlays):
                axes[idx].imshow(overlay)
                axes[idx].set_title(titles[idx], fontsize=10)
                axes[idx].axis("off")

            for idx in range(len(overlays), len(axes)):
                axes[idx].axis("off")

            # barra colore
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
            fig.colorbar(sm, cax=cbar_ax, label="Attention level")

            grid_path = os.path.join(save_dir, grid_filename)
            plt.savefig(grid_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"[saved grid] {grid_path}")
            break  # solo primo batch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def extract_tsne(model, loader, subj, path):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to("cuda")  # se usi GPU
            out, feat = model(x)    # output del modello (prima del classificatore finale)
            features.append(feat.cpu())
            labels.append(y)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).squeeze().numpy()

    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"Classe {c}", alpha=0.7)

    plt.legend()
    plt.title("t-SNE delle feature della rete")
    plt.savefig(f"{graphs_path}/tsne_features_{subj}.png", dpi=300, bbox_inches='tight')  # <--- salva l'immagine


seed_n = 2025
print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/single_config_16_2_2.yaml')
args = parser.parse_args()
config = load_config(args.config)

s_accuracy = []
docker_prefix = "../../../mnt/localstorage/cdeangelis/"
save_path, graphs_path, log_path = create_checkpoints_folders(args.config, config["model"]["single"], docker_prefix = docker_prefix)
for n in range(9):
    subject_test = f"A0{n+1}"
    batch_size = 32
    #load_path = "Single_checkpoints/16_2_2/val_" + subject + ".pth"
    load_path = save_path + "/v_" +subject_test + ".pth"
    model = MultiChannelViT(**config["model"])
    #model = pret_MVIT(n_channels=22, img_height = 64, img_width = 1008, patch_size=PATCH_SIZE, embed_dim=768, num_classes=4, single=SINGLE)

    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss() #contiene già una softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    _, test_dataset = prepare_dataloaders(subject_id = subject_test, augment = False, filter=config["train"]["filter"], BCI = config["run"]["dataset"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    last_epoch = checkpoint['epoch'] + 1  # Per riprendere
    append_to_log_file(log_path, f"last epoch {last_epoch}")
    loss = checkpoint['loss']
    print("Epoch=", last_epoch)
    print("Train loss=", loss)

     
    avg_loss, avg_acc, all_preds, all_labels = test_model(model, test_loader, criterion)
    importances = cls_token_importance(model.concat_classifier[0])
    append_to_log_file(log_path, f"Importances {importances}")
    txt = f"Accuracy on {subject_test} is {avg_acc}"
    append_to_log_file(log_path, txt)
    s_accuracy.append(avg_acc)
    #cm = confusion_matrix(all_labels, all_preds)
    #plot_confusion_matrix(cm, graphs_path+"/cm_"+subject_test)
    #extract_tsne(model, test_loader, subject_test, graphs_path)
    #extract_and_visualize_attention_channels_grid(model, test_loader, grid_filename=graphs_path+"/attention_grid.png")



txt = f"Mean accuracy {np.mean(s_accuracy)}"
append_to_log_file(log_path, txt)
#subject_csv(s_accuracy, testname=config["info"]["test_name"])