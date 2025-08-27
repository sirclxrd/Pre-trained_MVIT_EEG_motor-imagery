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
                   append_to_log_file, load_only_model, config_csv, subject_csv)
import random
import yaml
import argparse
from torch.utils.data import random_split
import os
from math import ceil
import math
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler




device = 'cuda'


def training_epoch(model, train_loader, test_loader, val_loader ,criterion, optimizer,scheduler, epoch = 0, log_file = "log.txt"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch = 0

    for inputs, labels in train_loader:
        #inputs = inputs.squeeze(2)
        if len(inputs.shape) == 3:
            print(inputs.shape)
            inputs = inputs.unsqueeze(1)
            print(inputs.shape)

        inputs = inputs.to(device).float()
        #labels = labels.to(device).long()
        labels = labels.to(device).squeeze().long()
        #print("Label min:", labels.min().item(), "max:", labels.max().item())

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        if torch.isnan(outputs).any():
            print("NaN negli output del modello!")
            break
        loss = criterion(outputs, labels)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if total_norm > 5.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        txt = f"Gradient norm: {total_norm.item()}"
        append_to_log_file(log_file, txt)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)# cerca il massimo sulle colonne

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch = batch + 1
        #print("Batch: ", batch)

    if scheduler is not None:
        scheduler.step()
    epoch_loss = running_loss / batch
    epoch_acc = correct / total
    txt = f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}"
    print(txt)
    append_to_log_file(log_file, txt)

    # Validation ogni 5 epoche
    if (epoch + 1) % 2 == 0 and val_loader is not None:
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

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)# cerca il massimo sulle colonne
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
    model.eval()

    running_loss = 0.0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    total = 0
    inference_times = []
    batch = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if len(inputs.shape) == 3:
                print(inputs.shape)
                inputs = inputs.unsqueeze(1)
                print(inputs.shape)
            #inputs = inputs.squeeze(2)
            inputs = inputs.to(device).float()
            labels = labels.to(device).squeeze().long()

            #tempo per un batch di campioni
            start_time = time.time()
            outputs, channels = model(inputs)
            end_time = time.time()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            _, predicted1 = torch.max(channels[0].data, 1)
            _, predicted2 = torch.max(channels[1].data, 1)
            _, predicted3 = torch.max(channels[2].data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            correct1 += (predicted1 == labels).sum().item()
            correct2 += (predicted2 == labels).sum().item()
            correct3 += (predicted3 == labels).sum().item()
            
            batch += 1
            inference_times.append(end_time - start_time)


    avg_loss = running_loss / batch
    accuracy = correct / total

    accuracy1 = correct1 / total
    accuracy2 = correct2 / total
    accuracy3 = correct3 / total

    avg_inference_time = np.mean(inference_times) / inputs.shape[0]  # per campione

    txt = f"[TEST] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Avg Inference Time: {avg_inference_time*1000:.2f} ms/sample"
    print(txt)
    append_to_log_file(log_file, txt)

    txt = f"Accuracy channel 1: {accuracy1:.4f}"
    append_to_log_file(log_file, txt)
    txt = f"Accuracy channel 2: {accuracy2:.4f}"
    append_to_log_file(log_file, txt)
    txt = f"Accuracy channel 3: {accuracy3:.4f}"
    append_to_log_file(log_file, txt)


    return avg_loss, accuracy

def lr_lambda(step):
    # Warmup lineare
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    # Cosine decay
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * decay_ratio))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

if __name__ == '__main__':
    print(device)
    #data_path = "/home/pfoggia/GenerativeAI/CELEBA/"
    #data_path = ".\CELEBA-20250604T155043Z-1-001\CELEBA"
    #save_path = "/home/C.DEANGELIS29/cond_test/VAE_models/"
    #load_path = "/home/C.DEANGELIS29/cond_test/VAE_models/ddpm3Cond98.pth"
    docker_prefix = "../../../mnt/localstorage/cdeangelis/"

    #seed_n = np.random.randint(2025)
    # seed_n = 2025
    # print('seed is ' + str(seed_n))
    # random.seed(seed_n)
    # np.random.seed(seed_n)
    # torch.manual_seed(seed_n)
    # torch.cuda.manual_seed(seed_n)
    # torch.cuda.manual_seed_all(seed_n)

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

    for n in range (9):
        early_stop = 0
        stopped = False

        if config["run"]["pret"] == False:
            model = MultiChannelViT(**config["model"])
            # from timm.models.vision_transformer import VisionTransformer

            # model = VisionTransformer(
            # img_size=(32, 1008),     # <-- la tua dimensione
            # patch_size=16,
            # in_chans=22,
            # num_classes=4,
            # embed_dim=768,
            # depth=2,
            # num_heads=2,
            # mlp_ratio=4,
            # drop_rate = 0.5,
            # attn_drop_rate = 0.5,
            # drop_path_rate = 0.5
            # )
            #init_weights(model)
        else:
            model = pret_MVIT(n_channels=config["model"]["n_channels"], img_height = config["model"]["img_height"], 
                          img_width = config["model"]["img_width"], patch_size=config["model"]["patch_size"], 
                          embed_dim=config["model"]["embed_dim"], num_classes=config["model"]["num_classes"], 
                          single=config["model"]["single"])
        model=model.to(device=device)
        criterion = nn.CrossEntropyLoss() #contiene già una softmax ###########
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["train"]["lr"], 
            betas = (0.85, 0.995),
            weight_decay=0.05  #weight decay 0.05, 0.005 è più lento ma l'andamento è giusto 
        )

        steps_per_epoch = ceil(288 / config["train"]["batch_size"])
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = int(0.1 * total_steps)

        if config["run"]["scheduler"]:
            # scheduler = LambdaLR(
            #     optimizer,
            #     lr_lambda=lr_lambda  # Warmup lineare
            # )
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
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
        subject = "A0"+str(n+1) ##########
        print("Subject ",subject)
        print("------------------")
        batch_size = config["train"]["batch_size"]

        if config["run"]["augment"]:
            train_dataset, test_dataset, is_real = prepare_dataloaders(subject_id = subject, augment = config["run"]["augment"], filter=config["train"]["filter"], BCI = config["run"]["dataset"]) #choose if augment dataset
        else:
            train_dataset, test_dataset = prepare_dataloaders(subject_id = subject, augment = config["run"]["augment"], filter=config["train"]["filter"], BCI = config["run"]["dataset"]) #choose if augment dataset

        #splitto in train e validation solo se specificato nel file
        if config["run"]["val"]:
            if config["run"]["augment"]:
                real_indices = np.where(is_real == 1)[0]
                aug_indices  = np.where(is_real == 0)[0]

                # Splitto solo i dati reali per creare il validation
                indices = np.random.permutation(len(real_indices))
                train_len = int(0.8 * len(indices))
                train_real = real_indices[:train_len]
                val_real   = real_indices[train_len:]

                # Aggiungo dati augmentati solo al training
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

        # per caricare il modello
        if config["train"]["load"] == True:
            if config["run"]["val"] == True:
                checkpoint = torch.load(load_path + "/val_" + subject + ".pth", map_location=device)
            else:
                checkpoint = torch.load(load_path + "/" + subject + ".pth", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            last_epoch = checkpoint['epoch'] + 1  # Per riprendere
        

        save_interval = 50 * 60
        last_save_time = time.time()

        for i in range(EPOCHS):
            if val_loader is not None and (i+1) % 2 == 0:
                loss, epoch_accuracy, val_loss, epoch_val_accuracy = training_epoch(model, train_loader, test_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path)
                if val_loss < val_best_loss or epoch_val_accuracy > val_best_acc + 1e-3:
                    early_stop = 0

                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                    
                    if epoch_val_accuracy > val_best_acc + 1e-3:
                        val_best_acc = epoch_val_accuracy
                        
                    torch.save({
                        'loss': val_loss,
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }, save_path + "/val_" +subject + ".pth")
                else:
                    early_stop += 1

                if early_stop == 15 and stopped == False:
                    append_to_log_file(log_path, f"Early stop at epoch {i}")
                    torch.save({
                        'loss': val_loss,
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }, save_path + "/val_" +subject + ".pth")
                    #_, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
                    break
                    stopped = True

                epoch_val_loss.append(val_loss)
                epoch_val_acc.append(epoch_val_accuracy)
            else:
                loss, epoch_accuracy = training_epoch(model, train_loader, test_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path)
                

            epoch_loss.append(loss)
            epoch_acc.append(epoch_accuracy)

            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'loss': loss,
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, save_path + "/" +subject + ".pth")

            
            print("EPOCA"+ str(i)+ " finita ")
            # current_time = time.time()
            # if current_time - last_save_time >= 50*60 or i == (EPOCHS-1):
            #     torch.save({
            #         'loss': loss,
            #         'epoch': i,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler_state_dict': scheduler.state_dict()
            #     }, save_path + "vae_new" + str(i + last_epoch) + ".pth")
            #     print(f'Salvataggio epoca ', i, ' completato.')
            #     last_save_time = current_time
        

        _, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
        model = load_only_model(load_path, subject, model, config["run"]["val"]) #carica il modello con la best loss
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
    append_to_log_file("total.txt", txt) #per un insieme di tutti i risultati
    config_csv(config, mean_accuracy=str(np.mean(total_test_acc))) #scrive i risultati su un file csv
    subject_csv(total_test_acc, testname=config["info"]["test_name"])