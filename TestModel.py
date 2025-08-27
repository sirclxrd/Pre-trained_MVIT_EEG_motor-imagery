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
    load_path = save_path + "/val_M" +subject_test + ".pth"
    model = MultiChannelViT(**config["model"])
    #model = pret_MVIT(n_channels=22, img_height = 64, img_width = 1008, patch_size=PATCH_SIZE, embed_dim=768, num_classes=4, single=SINGLE)

    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss() #contiene già una softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    test_dataset = prepare_dataloaders(subject_id = subject_test, augment = config["run"]["augment"], filter=config["train"]["filter"], BCI = config["run"]["dataset"], onlytest=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    last_epoch = checkpoint['epoch'] + 1  # Per riprendere
    loss = checkpoint['loss']
    print("Epoch=", last_epoch)
    print("Train loss=", loss)
    avg_loss, avg_acc, all_preds, all_labels = test_model(model, test_loader, criterion)
    txt = f"Accuracy on {subject_test} is {avg_acc}"
    append_to_log_file(log_path, txt)
    s_accuracy.append(avg_acc)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, graphs_path+"/cm_"+subject_test)

txt = f"Mean accuracy {np.mean(s_accuracy)}"
append_to_log_file(log_path, txt)
#subject_csv(s_accuracy, testname=config["info"]["test_name"])