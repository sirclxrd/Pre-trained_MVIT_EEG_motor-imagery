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

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).squeeze().long()
            if len(inputs.shape) == 3:
                print(inputs.shape)
                inputs = inputs.unsqueeze(1)
                print(inputs.shape)

            #tempo per un batch di campioni
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
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
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    #append_to_log_file(log_file, txt)
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

def visualize_attention_on_image(image_tensor, attn_weights, save_path="./attention_visualization.png", target_size=(224,224)):
    """
    Sovrappone la mappa di attenzione all'immagine.
    - image_tensor: [C,H,W] o [H,W]
    - attn_weights: [num_patches] tensor
    - target_size: dimensione finale immagine salvata
    """
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import cv2
    import math

    # Passa a CPU e numpy
    image = image_tensor.cpu().numpy()
    if image.ndim == 3:
        image = image[0]  # primo canale

    # Normalizza immagine tra 0 e 1
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Calcolo dimensione griglia patch
    num_patches = attn_weights.shape[0]
    grid_h = int(math.ceil(math.sqrt(num_patches)))
    grid_w = int(math.ceil(num_patches / grid_h))
    
    # Padding se necessario
    if grid_h * grid_w > num_patches:
        pad_size = grid_h * grid_w - num_patches
        attn_weights = F.pad(attn_weights, (0, pad_size), value=0)

    attn_map = attn_weights.cpu().numpy()
    attn_grid = attn_map.reshape((grid_h, grid_w))

    # Ridimensiona immagine e attention alla dimensione target
    H_target, W_target = target_size
    image_resized = cv2.resize(image, (W_target, H_target), interpolation=cv2.INTER_CUBIC)
    attn_resized = cv2.resize(attn_grid, (W_target, H_target), interpolation=cv2.INTER_CUBIC)

    # Normalizza attention
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    cmap = plt.get_cmap("jet")
    attn_color = cmap(attn_resized)[:,:,:3]
    overlay = 0.6*image_resized[:,:,None] + 0.4*attn_color
    overlay = np.clip(overlay, 0.0, 1.0)  # <-- importantissimo

    # Salva
    plt.imsave(save_path, overlay)
    print(f"Saved attention overlay to {save_path}")



def extract_and_visualize_attention_channels(model, dataloader, device="cuda", patch_grid_shape=(2,63), save_dir="."):
    """
    Estrae il primo batch dal dataloader, calcola le mappe di attenzione per tutti i canali e salva le immagini.
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
            x_batch = x_batch.to(device)
            
            # Lista per salvare le immagini
            B = x_batch.shape[0]
            
            # Itera sul batch (qui prendi solo il primo esempio)
            sample_image = x_batch[0]  # [C,H,W]
            
            # Ciclo su ciascun encoder (canale)
            for ch_idx, encoder in enumerate(model.encoders):
                # Patch embedding del canale
                channel_input = x_batch[0:1, ch_idx:ch_idx+1, :, :]
                token = encoder.patch_embed(channel_input)
                
                # Forward singolo encoder con ritorno attention
                _, attn = encoder.encoder(token, return_attn=True)
                
                # Prendi la prima testa e il CLS token
                cls_attn = attn[0][0, 0, 1:]  # esclude CLS
                
                # Prendi solo il canale corrispondente dall’immagine
                channel_image = sample_image[ch_idx:ch_idx+1, :, :]
                
                save_path = f"{save_dir}/attention_channel{ch_idx}.png"
                visualize_attention_on_image(channel_image, cls_attn, save_path)
            
            break  # prendi solo il primo batch


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
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, graphs_path+"/cm_"+subject_test)
    extract_and_visualize_attention_channels(model, test_loader, device="cuda", patch_grid_shape=(2,63), save_dir=graphs_path)
    # extract_and_visualize_attention_channel(
    # model,
    # test_loader,
    # device="cuda",
    # channel_idx=0,       # canale da visualizzare
    # patch_grid_shape=(2,63),
    # save_path=graphs_path+"attn_"+subject_test
    # )


txt = f"Mean accuracy {np.mean(s_accuracy)}"
append_to_log_file(log_path, txt)
#subject_csv(s_accuracy, testname=config["info"]["test_name"])