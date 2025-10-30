from Preprocessing import prepare_dataloaders
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch
from models.MVIT import MultiChannelViT
import torch.nn as nn
import time
from utils import (visualize_train_loss_acc, load_config, create_checkpoints_folders, 
                   append_to_log_file, load_only_model, config_csv, subject_csv, save_model)
import random
import yaml
import argparse
from torch.utils.data import random_split
import os
from math import ceil
import math
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from Preprocessing_physionet import preprocess_physionet


VAL_EPOCH = 1
EARLY_STOP = 10
TOTAL_SUBJECTS = 9
SUBJECT = 1
LAMBDA = 0.7


device = 'cuda'

def frequency_masking(spectrogram, F=15):
    f = spectrogram.shape[-2]
    f0 = random.randint(0, f - F)
    spectrogram[:, :, f0:f0+F, :] = 0
    return spectrogram

def time_masking(spectrogram, T=30):
    t = spectrogram.shape[-1]
    t0 = random.randint(0, t - T)
    spectrogram[:, :, :, t0:t0+T] = 0
    return spectrogram

def channel_dropout(spectrogram, drop_prob=0.2):
    """Azzeramento casuale di alcuni canali"""
    mask = torch.rand(C) > drop_prob
    mask = mask.float().view(1, C, 1, 1)
    mask = mask.to(device='cuda')
    return spectrogram * mask

def random_augmentation(spectrogram):
    augmentations = [
        lambda x: frequency_masking(x, F=15),
        lambda x: time_masking(x, T=500),
        lambda x: frequency_masking(time_masking(x, T = 200)),        
        lambda x: x
    ]
    aug_func = random.choice(augmentations)
    return aug_func(spectrogram)

    


def training_epoch(model, train_loader, test_loader, val_loader ,criterion, optimizer,scheduler, epoch = 0, log_file = "log.txt"):
    running_loss = 0.0
    correct = 0
    total = 0
    batch = 0
    model.train()

    for inputs, labels in train_loader:
        #inputs = inputs.squeeze(2)
        if len(inputs.shape) == 3:
            print(inputs.shape)
            inputs = inputs.unsqueeze(1)
            print(inputs.shape)

        inputs = inputs.to(device).float()
        labels = labels.to(device).squeeze().long()
        inputs = random_augmentation(inputs)

        optimizer.zero_grad()
        #outputs, out2 = model(inputs)
        #loss = (1-LAMBDA) * criterion(outputs, labels) + LAMBDA*criterion(out2, labels)
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        
        if torch.isnan(outputs).any():
            print("NaN negli output del modello!")
            break
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        txt = f"Gradient norm: {total_norm.item()}"
        append_to_log_file(log_file, txt)
        optimizer.step()
        running_loss += loss.item()
        predicted = outputs.argmax(dim=1)# cerca il massimo sulle colonne
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch = batch + 1

    if scheduler is not None:
        scheduler.step()
    epoch_loss = running_loss / batch
    epoch_acc = correct / total
    txt = f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}"
    print(txt)
    append_to_log_file(log_file, txt)

    # Validation ogni 5 epoche
    if (epoch + 1) % VAL_EPOCH == 0 and val_loader is not None:
        batch = 0
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                if len(inputs.shape) == 3:
                    print(inputs.shape)
                    inputs = inputs.unsqueeze(1)
                    print(inputs.shape)
                inputs = inputs.to(device).float()
                #labels = labels.to(device).long()
                labels = labels.to(device).squeeze().long()

                # outputs, out2 = model(inputs)
                # loss = (1-LAMBDA) * criterion(outputs, labels) + LAMBDA*criterion(out2, labels)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = outputs.argmax(dim=1)# cerca il massimo sulle colonne
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                batch = batch + 1
                print("Val Batch: ", batch)

        val_epoch_loss = val_loss / batch
        val_epoch_acc = val_correct / val_total
        txt = f"Epoch {epoch+1} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}"
        print(txt)
        append_to_log_file(log_file,txt)
        return epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc

    return epoch_loss, epoch_acc

def test_model(model, test_loader, criterion, log_file = "log.txt"):
    model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0
    inference_times = []
    batch = 0

    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            if len(inputs.shape) == 3:
                print(inputs.shape)
                inputs = inputs.unsqueeze(1)
                print(inputs.shape)
            #inputs = inputs.squeeze(2)
            inputs = inputs.to(device).float()
            labels = labels.to(device).squeeze().long()

            start_time = time.time()
            #outputs, out2 = model(inputs)
            outputs, _ = model(inputs)
            end_time = time.time()

            #loss = (1-LAMBDA) * criterion(outputs, labels) + LAMBDA*criterion(out2, labels)

            loss = criterion(outputs, labels)
            running_loss += loss.item()


            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1
            inference_times.append(end_time - start_time)


    avg_loss = running_loss / batch
    accuracy = correct / total
    avg_inference_time = np.sum(inference_times) / total  # per campione

    txt = f"[TEST] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Avg Inference Time: {avg_inference_time*1000:.2f} ms/sample"
    print(txt)
    append_to_log_file(log_file, txt)
    return avg_loss, accuracy

def get_epoch_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return max(0.0, 1.0 - progress)  # scende linearmente fino a 0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args, config, docker_prefix = "../../../mnt/localstorage/cdeangelis/", root_2a = "", root_2b = ""):
    print(device)

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
    total_test_acc = []

    EPOCHS = config["train"]["epochs"]
    # creo il modello

    save_path, graphs_path, log_path = create_checkpoints_folders(args.config, config["model"]["single"], docker_prefix = docker_prefix)
    load_path = save_path
    save_path = save_path
    print("Save_path:",save_path)

    for n in range (TOTAL_SUBJECTS):
        early_stop = 0
        stopped = False

        model = MultiChannelViT(**config["model"], dataset=config["run"]["dataset"])
        model=model.to(device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay = 0.01        
            )

        if config["run"]["scheduler"]:
            scheduler = get_epoch_cosine_schedule_with_warmup(optimizer, warmup_epochs=0.05*EPOCHS, total_epochs=EPOCHS)
        else:
            scheduler = None


        txt=f"Current lr: {optimizer.param_groups[0]['lr']}"
        append_to_log_file(log_path, txt)

        last_epoch = 0
        best_loss = 100000
        val_best_loss = 100000
        val_best_acc = 0
        val_acc = 0
        epoch_loss = []
        epoch_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        # prendo train e test
        subject = "A0"+str(n+SUBJECT) ##########
        print("Subject ",subject)
        print("------------------")
        batch_size = config["train"]["batch_size"]

        if config["run"]["dataset"].lower() == "physionet":
            train_loader, val_loader, test_loader = preprocess_physionet(subject_id=subject, augment = config["run"]["augment"], filter = config["train"]["filter"], batch_size = batch_size)
        else:
            if config["run"]["augment"]:
                train_dataset, test_dataset, is_real = prepare_dataloaders(subject_id = subject, augment = config["run"]["augment"], filter=config["train"]["filter"], BCI = config["run"]["dataset"], root = root_2a, root_2b = root_2b) #choose if augment dataset
            else:
                train_dataset, test_dataset = prepare_dataloaders(subject_id = subject, augment = config["run"]["augment"], filter=config["train"]["filter"], BCI = config["run"]["dataset"], root = root_2a, root_2b = root_2b) #choose if augment dataset
        
            if config["run"]["val"]:
                if config["run"]["augment"]:
                    real_indices = np.where(is_real == 1)[0]
                    aug_indices  = np.where(is_real == 0)[0]

                    indices = np.random.permutation(len(real_indices))
                    train_len = int(0.8 * len(indices))
                    train_real = real_indices[:train_len]
                    val_real   = real_indices[train_len:]

                    train_indices = np.concatenate([train_real, aug_indices])
                    val_indices = val_real
                else:
                    indices = np.random.permutation(len(train_dataset))
                    train_len = int(0.8 * len(indices))
                    train_indices = indices[:train_len]
                    val_indices = indices[train_len:]

                train_subset = Subset(train_dataset, train_indices)
                val_subset = Subset(train_dataset, val_indices)

                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            else:
                val_loader = None
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            #################################
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if config["train"]["load"] == True:
            if config["run"]["val"] == True:
                checkpoint = torch.load(load_path + "/val_M" + subject + ".pth", map_location=device)
            else:
                checkpoint = torch.load(load_path + "/" + subject + ".pth", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            model.to(device)
            if 'epoch_loss' in checkpoint:
                epoch_loss= checkpoint['epoch_loss']
                epoch_acc= checkpoint['epoch_acc']
                epoch_val_loss= checkpoint['epoch_val_loss']
                epoch_val_acc= checkpoint['epoch_val_acc']
        append_to_log_file(log_path, f"Train on subject {subject}")
        for i in range(EPOCHS):
            if val_loader is not None and (i+1) % VAL_EPOCH == 0:
                loss, epoch_accuracy, val_loss, epoch_val_accuracy = training_epoch(model, train_loader, test_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path)
                if val_loss <= val_best_loss or epoch_val_accuracy >= val_best_acc:
                    early_stop = 0

                    if val_loss <= val_best_loss:
                        val_best_loss = val_loss
                    
                    if epoch_val_accuracy >= val_best_acc:
                        val_best_acc = epoch_val_accuracy
                        
                    save_model(val_loss, i, model, optimizer, scheduler, subject, save_path, config["run"]["scheduler"], epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc )
                else:
                    early_stop += 1

                if early_stop == EARLY_STOP and stopped == False:
                    append_to_log_file(log_path, f"Early stop at epoch {i}")
                    break

                epoch_val_loss.append(val_loss)
                epoch_val_acc.append(epoch_val_accuracy)
            else:
                loss, epoch_accuracy = training_epoch(model, train_loader, test_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path)
                
            epoch_loss.append(loss)
            epoch_acc.append(epoch_accuracy)     
            print("EPOCA"+ str(i)+ " finita ")
        

        _, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
        model = load_only_model(load_path, subject, model, config["run"]["val"])
        print("Test", subject)
        _, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
        total_test_acc.append(test_acc)
        visualize_train_loss_acc(epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc, save_path=graphs_path + "/" +subject)
        if config["run"]["save"] == False:
            os.remove(load_path + "/" + subject + ".pth")
    print("The mean accuracy is: ",np.mean(total_test_acc))
    txt = f"The mean accuracy is: {np.mean(total_test_acc)}"
    append_to_log_file(log_path, txt)
    txt = f"{args.config}, The mean accuracy is: {np.mean(total_test_acc)}"
    append_to_log_file("total.txt", txt) 
    config_csv(config, mean_accuracy=str(np.mean(total_test_acc)))
    subject_csv(total_test_acc, testname=config["info"]["test_name"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/single_config_16_2_2.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    root_2a = "../Python/BciCompetitionIv2a/Train"
    root_2b = "../Python/BciCompetitionIv2b"
    main(args,config, docker_prefix="../", root_2a=root_2a, root_2b = root_2b)