from Preprocessing import prepare_dataloaders
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import torch
from models.MVIT import MultiChannelViT
from models.pret_MVIT import pret_MVIT
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
from timm.models.vision_transformer import VisionTransformer




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
    B, C, H, W = spectrogram.shape
    mask = torch.rand(C) > drop_prob
    mask = mask.float().view(1, C, 1, 1)
    mask = mask.to(device='cuda')
    return spectrogram * mask

def random_augmentation(spectrogram):
    """
    Applica casualmente una di queste 3 augmentations oppure nessuna:
    - additive_noise
    - frequency_masking
    - time_masking
    
    Args:
        spectrogram: tensor [B, C, H, W]
        
    Returns:
        spectrogram trasformato
    """
    augmentations = [
        lambda x: frequency_masking(x, F=15),
        lambda x: time_masking(x, T=500),
        lambda x: frequency_masking(time_masking(x, T = 200)),        
        lambda x: x
    ]
    aug_func = random.choice(augmentations)
    return aug_func(spectrogram)

def training_epoch(model, train_loader, val_loader ,criterion, optimizer,scheduler, epoch = 0, log_file = "log.txt"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch = 0

    for inputs, labels in train_loader:
        #inputs = inputs.squeeze(2)
        inputs = inputs.to(device).float()
        #labels = labels.to(device).long()
        labels = labels.to(device).squeeze().long()
        inputs = random_augmentation(inputs)

        optimizer.zero_grad()
        outputs, token = model(inputs)

        if torch.isnan(outputs).any():
            print("NaN negli output del modello!")
            break
        loss = criterion(outputs, labels)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #######
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if total_norm > 5.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        print(f"Gradient norm: {total_norm.item()}")
        optimizer.step()
        # if scheduler is not None:
        #     scheduler.step()

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

    # Validation ogni 2 epoche
    if (epoch + 1) % 5 == 0 and val_loader is not None:
        batch = 0
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                #labels = labels.to(device).long()
                labels = labels.to(device).squeeze().long()

                outputs, token = model(inputs)
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
    total = 0
    inference_times = []
    batch = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            #inputs = inputs.squeeze(2)
            inputs = inputs.to(device).float()
            labels = labels.to(device).squeeze().long()

            #tempo per un batch di campioni
            start_time = time.time()
            outputs, token = model(inputs)
            end_time = time.time()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1
            inference_times.append(end_time - start_time)


    avg_loss = running_loss / batch
    accuracy = correct / total
    avg_inference_time = np.mean(inference_times) / inputs.shape[0]  # per campione

    txt = f"[TEST] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Avg Inference Time: {avg_inference_time*1000:.2f} ms/sample"
    print(txt)
    append_to_log_file(log_file, txt)
    return avg_loss, accuracy

def get_epoch_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        # Warmup lineare
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # Decadimento lineare dopo il warmup
        else:
            progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return max(0.0, 1.0 - progress)  # scende linearmente fino a 0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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


def main(args, config, docker_prefix="../../../mnt/localstorage/cdeangelis/", root_2a="./BciCompetitionIv2a/Train", root_2b = "./BciCompetitionIv2b"):
    print(device)
    #data_path = "/home/pfoggia/GenerativeAI/CELEBA/"
    #data_path = ".\CELEBA-20250604T155043Z-1-001\CELEBA"
    #save_path = "/home/C.DEANGELIS29/cond_test/VAE_models/"
    #load_path = "/home/C.DEANGELIS29/cond_test/VAE_models/ddpm3Cond98.pth"

    #seed_n = np.random.randint(2025)
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

    

    if config["run"]["pret"] == False:
        model = MultiChannelViT(**config["model"])

        # model = VisionTransformer(
        # img_size=(32, 1008),     # <-- la tua dimensione
        # patch_size=16,
        # in_chans=22,
        # num_classes=4,
        # embed_dim=768,
        # depth=2,
        # num_heads=2,
        # mlp_ratio=4,
        # drop_rate = 0.3,
        # attn_drop_rate = 0.3,
        # drop_path_rate = 0.3
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
            weight_decay = 0.01        
            )

    # steps_per_epoch = ceil(288 / config["train"]["batch_size"])
    # total_steps = steps_per_epoch * EPOCHS
    # warmup_steps = int(0.1 * total_steps)

    if config["run"]["scheduler"]:
            scheduler = get_epoch_cosine_schedule_with_warmup(optimizer, warmup_epochs=0.1*EPOCHS, total_epochs=EPOCHS)
            #scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-4)
    else:
        scheduler = None

    txt=f"Current lr: {optimizer.param_groups[0]['lr']}"
    append_to_log_file(log_path, txt)
    early_stop = 0
    stopped = False
    last_epoch = 0
    best_loss = 100000
    val_best_loss = 100000
    epoch_loss = []
    epoch_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    batch_size = config["train"]["batch_size"]
    val_best_acc = 0
    val_acc = 0

    train_subjects = [f"A0{i+1}" for i in range(9)]
    train_datasets = [
        prepare_dataloaders(subject_id=subj, augment=config["run"]["augment"], filter=config["train"]["filter"], BCI = config["run"]["dataset"], root=root_2a, root_2b = root_2b)[0]
        for subj in train_subjects
    ]

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)

    if config["run"]["val"]:
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


    # per caricare il modello
    if config["train"]["load"] == True:
        if config["run"]["val"] == True:
            checkpoint = torch.load(load_path + "/val_model" + ".pth", map_location=device)
        else:
            checkpoint = torch.load(load_path + "/model" + ".pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch'] + 1  # Per riprendere
        

    val_loss_history = []
    
    for i in range(EPOCHS):
        if val_loader is not None and (i+1) % 5 == 0:
            loss, epoch_accuracy, val_loss, epoch_val_accuracy = training_epoch(model, train_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path)

            val_loss_history.append(val_loss) #media mobile per early stop
            if len(val_loss_history) > 5:
                val_loss_history.pop(0)
            moving_avg = sum(val_loss_history) / len(val_loss_history)

            if val_loss < val_best_loss or epoch_val_accuracy > val_best_acc + 1e-3:
                early_stop = 0
                
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    
                if epoch_val_accuracy > val_best_acc + 1e-3:
                    val_best_acc = epoch_val_accuracy

                save_model(val_loss, i, model, optimizer, scheduler,"1" ,save_path, config["run"]["scheduler"],epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc )
            else:
                early_stop += 1

            if early_stop >= 10 and stopped == False:
                append_to_log_file(log_path, f"Early stop at epoch {i}")
                #_, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
                stopped = True
                break




            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(epoch_val_accuracy)
        else:
            loss, epoch_accuracy = training_epoch(model, train_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path)
            

        epoch_loss.append(loss)
        epoch_acc.append(epoch_accuracy)

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'loss': loss,
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                #'scheduler_state_dict': scheduler.state_dict()
            }, save_path + "/model" +".pth")

        
        print("EPOCA"+ str(i)+ " finita ")
    visualize_train_loss_acc(epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc, save_path=graphs_path + "/model_loss")

    for n in range (9):
        # prendo train e test
        subject = "A0"+str(n+1) ##########
        print("Subject ",subject)
        print("------------------")
        test_dataset = prepare_dataloaders(subject_id = subject, augment = config["run"]["augment"], filter=config["train"]["filter"], onlytest=True, BCI = config["run"]["dataset"]) #choose if augment dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        #append_to_log_file(log_path, f"Last Epoch model subject {subject}")
        #_, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)

        print("Test", subject)
        model = load_only_model(load_path, "model", model, config["run"]["val"]) #carica il modello con la best loss
        txt = f"Test on subj {subject}"
        append_to_log_file(log_path, txt)
        _, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
        total_test_acc.append(test_acc)

        # if config["run"]["save"] == False:
        #     os.remove(load_path + "/" + "model" + ".pth")
    print("The mean accuracy is: ",np.mean(total_test_acc))
    txt = f"The mean accuracy is: {np.mean(total_test_acc)}"
    append_to_log_file(log_path, txt)
    txt = f"{args.config}, The mean accuracy is: {np.mean(total_test_acc)}"
    append_to_log_file("total.txt", txt) #per un insieme di tutti i risultati
    config_csv(config, mean_accuracy=str(np.mean(total_test_acc))) #scrive i risultati su un file csv
    subject_csv(total_test_acc, testname=config["info"]["test_name"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/single_config_16_2_2.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    root_2a = "../Python/BciCompetitionIv2a/Train"
    root_2b = "../Python/BciCompetitionIv2b"
    main(args,config, docker_prefix="../", root_2a=root_2a, root_2b = root_2b)