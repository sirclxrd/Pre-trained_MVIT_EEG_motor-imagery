from Preprocessing import prepare_dataloaders
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch
from models.MVIT import MultiChannelViT
from models.pret_MVIT import pret_MVIT
import torch.nn as nn
import time
from utils import (
    visualize_train_loss_acc, load_config, create_checkpoints_folders, 
    append_to_log_file, load_only_model, config_csv, subject_csv, plot_confusion_matrix
)
import random
import yaml
import argparse
import os
from math import ceil
import math
from torch.utils.data import ConcatDataset
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Parametri globali
LAMBDA = 0.7
device = 'cuda'

# -------------------------------------------------------
# Funzione di test
# -------------------------------------------------------
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
                inputs = inputs.unsqueeze(1)

            start_time = time.time()
            outputs, out2 = model(inputs)
            end_time = time.time()

            loss = (1 - LAMBDA) * criterion(outputs, labels) + LAMBDA * criterion(out2, labels)
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

    return avg_loss, accuracy, all_preds, all_labels

def main(args,config, docker_prefix="../", root_2a = "../", root_2b = "../"):
    seed_n = 2025
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)


    # -------------------------------------------------------
    # Loop sui soggetti
    # -------------------------------------------------------
    s_accuracy = []
    s_kappa = []  # per salvare il k-score per soggetto

    save_path, graphs_path, log_path = create_checkpoints_folders(
        args.config,
        config["model"]["single"],
        docker_prefix=docker_prefix
    )

    for n in range(9):
        subject_test = f"A0{n+1}"
        batch_size = 32
        load_path = save_path + "/v_" + subject_test + ".pth"

        print(f"\n--- Testing on {subject_test} ---")
        model = MultiChannelViT(**config["model"], dataset=config["run"]["dataset"])
        model = model.to(device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Dataset e dataloader
        _, test_dataset = prepare_dataloaders(
            subject_id = subject_test,
            augment = False,
            filter = config["train"]["filter"],
            BCI = config["run"]["dataset"],
            onlytest = False,
            root= root_2a,
            root_2b = root_2b
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Carica checkpoint
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Epoch= {last_epoch}")
        print(f"Train loss= {loss}")

        # Test
        avg_loss, avg_acc, all_preds, all_labels = test_model(model, test_loader, criterion)

        # Kappa score per soggetto
        kappa = cohen_kappa_score(all_labels, all_preds)
        s_kappa.append(kappa)

        txt = f"Accuracy on {subject_test}: {avg_acc:.4f} | Kappa: {kappa:.4f}"
        print(txt)
        append_to_log_file(log_path, txt)

        s_accuracy.append(avg_acc)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, graphs_path + "/cm_" + subject_test)

    # -------------------------------------------------------
    # Statistiche finali
    # -------------------------------------------------------
    mean_acc = np.mean(s_accuracy)
    mean_kappa = np.mean(s_kappa)

    txt = f"\nMean accuracy: {mean_acc:.4f} | Mean Kappa: {mean_kappa:.4f}"
    print(txt)
    append_to_log_file(log_path, txt)

    # Se vuoi salvare i risultati per soggetto in CSV:
    # subject_csv(s_accuracy, testname=config["info"]["test_name"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/single_config_16_2_2.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    root_2a = "../Python/BciCompetitionIv2a/Train"
    root_2b = "../Python/BciCompetitionIv2b"
    main(args,config, docker_prefix="../", root_2a=root_2a, root_2b = root_2b)