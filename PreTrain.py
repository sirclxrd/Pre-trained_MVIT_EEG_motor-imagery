from Preprocessing import prepare_dataloaders
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch
from models.MVIT import MultiChannelViTSelfSupervised, MultiChannelViT
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
import copy
import torch.nn.functional as F







import torch
import torch.nn.functional as F
import random

device = 'cuda'
def generate_views(x, 
                   num_global=2, 
                   num_local=8, 
                   global_ratio_range=(0.5, 1.0), 
                   resize_global=(32, 504), 
                   resize_local=(16, 252)
                  ):
    B, C, H, W = x.shape
    global_views = []
    local_views = []

    def random_crop(img, crop_h, crop_w):
        max_y = H - crop_h
        max_x = W - crop_w
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)
        return img[:, :, top:top + crop_h, left:left + crop_w]

    # Genera global views
    global_crop_sizes = []
    for _ in range(num_global):
        ratio = random.uniform(*global_ratio_range)
        crop_w = int(W * ratio)
        crop_h = H  # manteniamo altezza costante per ora
        global_crop_sizes.append( (crop_h, crop_w) )
        cropped = random_crop(x, crop_h, crop_w)
        resized = F.interpolate(cropped, size=resize_global, mode='bilinear', align_corners=False)
        global_views.append(resized)

    # Genera local views: 8 viste distribuite sulle 2 global crop_sizes
    for i in range(num_local):
        # Scegli quale global crop usare per la local view (ad es 2 local views per global crop)
        idx_global = i % num_global
        crop_h_global, crop_w_global = global_crop_sizes[idx_global]

        # Calcola crop locale 1/2 in altezza e larghezza per avere area 1/4
        crop_h_local = crop_h_global // 2
        crop_w_local = crop_w_global // 2

        cropped = random_crop(x, crop_h_local, crop_w_local)
        resized = F.interpolate(cropped, size=resize_local, mode='bilinear', align_corners=False)
        local_views.append(resized)

    return global_views, local_views

def sharpen(p, T):
    """
    p: (B, D) feature vectors
    T: temperature (<1 for sharpening)
    """
    p_sharp = p ** (1.0 / T)
    return p_sharp / (p_sharp.sum(dim=-1, keepdim=True) + 1e-6)

def view_prediction_loss(student_output, teacher_output, center, epoch, student_temp=0.1, teacher_temp_schedule=None, ncrops=10):
    """
    student_output: tensor [ncrops, B, D] (2 global + 8 local)
    teacher_output: tensor [2, B, D] (2 global)
    center: tensor [1, D]
    epoch: int
    teacher_temp_schedule: list or array with temperature per epoch
    """

    # divido student output in list di tensor [B, D]
    student_out = (student_output / student_temp).chunk(ncrops)

    # centro e applico temperatura al teacher
    temp = teacher_temp_schedule[epoch]
    #center = center.unsqueeze(1).expand(-1, teacher_output.shape[1], -1)
    teacher_out = F.softmax((teacher_output - center) / temp, dim=-1).detach().chunk(2)


    n_loss_terms = 0
    total_loss = 0

    for iq, q in enumerate(teacher_out):         # 2 teacher views
        for v in range(len(student_out)):        # 10 student views
            if v == iq:  # escludo le viste student corrispondenti a quelle teacher
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1

    center_momentum = 0.9
    total_loss /= n_loss_terms
    batch_center = teacher_output.mean(dim=(0,1), keepdim=True)  # [1, 1, D]
    batch_center = batch_center.squeeze(1)  # [1, D]
    new_center = center * center_momentum + batch_center * (1 - center_momentum)


    return total_loss, new_center

def get_teacher_momentum(base_m=0.996, final_m=1.0, step=0, max_steps=10000):
    """ Cosine schedule for momentum """
    return final_m - (final_m - base_m) * (
        (1 + torch.cos(torch.tensor(step * 3.14159265359 / max_steps))) / 2
    )

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data = (
            momentum * teacher_param.data + (1. - momentum) * student_param.data
        )

def additive_noise(spectrogram, noise_level=0.01):
    """
    Aggiunge rumore gaussiano allo spettrogramma.
    
    Args:
        spectrogram: tensor, forma [B, C, H, W] (o [B, H, W] a seconda del tuo input)
        noise_level: float, deviazione standard del rumore
    
    Returns:
        spectrogram con rumore aggiunto, stessa shape dell'input
    """
    noise = torch.randn_like(spectrogram) * noise_level
    return spectrogram + noise

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
        lambda x: additive_noise(x, noise_level=0.1),
        lambda x: frequency_masking(x, F=15),
        lambda x: time_masking(x, T=30),
        lambda x: x  # nessuna trasformazione
    ]
    aug_func = random.choice(augmentations)
    return aug_func(spectrogram)

class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, warmup_epochs, total_epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
        else:
            # Cosine decay
            t = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
        return lr


def training_epoch(model, model_test, teacher, train_loader, test_loader, val_loader ,criterion, optimizer,scheduler, epoch = 0, log_file = "log.txt", ema_decay=0.99, global_step = 0, center=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch = 0
    total_steps = EPOCHS * len(train_loader)
    teacher_schedule = torch.linspace(0.04, 0.07, steps=EPOCHS)


    for inputs, _ in train_loader:
        #inputs = inputs.squeeze(2)
        inputs = inputs.to(device).float()
        #labels = labels.to(device).long()
        #print("Label min:", labels.min().item(), "max:", labels.max().item())
        
        #inputs = random_augmentation(inputs)
        global_views, local_views = generate_views(inputs)

        with torch.no_grad():
            teacher_outs = [teacher(view) for view in global_views]
        all_student_views = global_views + local_views
        student_outs = [model(view) for view in all_student_views]

        student_outs_tensor = torch.stack(student_outs)
        teacher_outs_tensor = torch.stack(teacher_outs)
        loss, center = view_prediction_loss(student_outs_tensor, teacher_outs_tensor, center, epoch, student_temp=0.1, teacher_temp_schedule=teacher_schedule, ncrops=10)
        #append_to_log_file(log_file, f"{loss.item()} and {center.shape}")

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if total_norm > 5.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        txt = f"Gradient norm: {total_norm.item()}"
        #append_to_log_file(log_file, txt)
        optimizer.step()
        momentum = get_teacher_momentum(
        base_m=0.996,
        final_m=1.0,
        step=global_step,
        max_steps=total_steps
        )
        update_teacher(model, teacher, momentum)

        running_loss += loss.item()
        batch = batch + 1
        global_step += 1
        #print("Batch: ", batch)
        
        #ogni step
        if scheduler is not None:
            scheduler.step()
    epoch_loss = running_loss / batch
    #std = student_outs.std(dim=0).mean().item()
    txt = f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}"
    print(txt)
    append_to_log_file(log_file, txt)

    # Validation ogni 5 epoche
    if (epoch + 1) % 5 == 0 and val_loader is not None:
        model.eval()
        teacher.eval()
        batch = 0
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_batches = 0
        mean_cos_sim = 0.0
        mean_var = 0.0
        val_epoch_acc = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                #labels = labels.to(device).long()
                labels = labels.to(device).squeeze().long()
                global_views, local_views = generate_views(inputs)
                student_out = [model(view) for view in (global_views + local_views)]
                # forward teacher (solo viste globali, ad es. prime 2)
                teacher_out = [teacher(view) for view in global_views[:2]]
                student_out_tensor = torch.stack(student_out)
                teacher_out_tensor = torch.stack(teacher_out)
                loss, center = view_prediction_loss(student_out_tensor, teacher_out_tensor, center, epoch, student_temp=0.1, teacher_temp_schedule=teacher_schedule, ncrops=10)

                val_loss += loss.item()
                val_batches += 1

                # esempio: similarità cosine media tra teacher e student
                cos_sim = torch.nn.functional.cosine_similarity(
                    student_out[0], teacher_out[0], dim=-1
                ).mean()
                mean_cos_sim += cos_sim.item()

                # esempio: varianza della rappresentazione student
                repr_var = student_out[0].var(dim=0).mean()
                mean_var += repr_var.item()

        val_epoch_loss = val_loss / val_batches
        cos_sim = mean_cos_sim / val_batches
        variance = mean_var / val_batches
        txt = f"Epoch {epoch+1} | Val Loss: {val_epoch_loss:.4f} | Mean Cos sim: {cos_sim:.4f} | Mean Variance: {variance:.4f}"
        print(txt)
        append_to_log_file(log_file,txt)
        return epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc

    return epoch_loss, epoch_acc, global_step, center

def test_model(model, test_loader, criterion, log_file = "log.txt"):
    model.to(device)
    model_test.to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    inference_times = []
    batch = 0

    with torch.no_grad():
        model_test.eval()
        for inputs, labels in test_loader:
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            #inputs = inputs.squeeze(2)
            inputs = inputs.to(device).float()
            labels = labels.to(device).squeeze().long()

            #tempo per un batch di campioni
            start_time = time.time()
            outputs = model_test(inputs)
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

def save_model(val_loss, i, model, optimizer, scheduler, subject, save_path, sched_on ):
    if sched_on:
        torch.save({
                    'loss': val_loss,
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler
        }, save_path + "/val_" +subject + ".pth")
    else:
        torch.save({
                    'loss': val_loss,
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
        }, save_path + "/val_" +subject + ".pth")

if __name__ == '__main__':
    print(device)
    #data_path = "/home/pfoggia/GenerativeAI/CELEBA/"
    #data_path = ".\CELEBA-20250604T155043Z-1-001\CELEBA"
    #save_path = "/home/C.DEANGELIS29/cond_test/VAE_models/"
    #load_path = "/home/C.DEANGELIS29/cond_test/VAE_models/ddpm3Cond98.pth"
    docker_prefix = "../../../mnt/localstorage/cdeangelis/"

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

    for n in range (9):
        center = torch.zeros(1, 1, 192).to(device) #FEATURE DIM
        global_step = 0
        early_stop = 0
        stopped = False
        if config["run"]["pret"] == False:
            model_test = MultiChannelViT(**config["model"])
            model_test.to(device)

            model = MultiChannelViTSelfSupervised(**config["model"])
            teacher = MultiChannelViTSelfSupervised(**config["model"])
            teacher.to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
        else:
            model = pret_MVIT(n_channels=config["model"]["n_channels"], img_height = config["model"]["img_height"], 
                          img_width = config["model"]["img_width"], patch_size=config["model"]["patch_size"], 
                          embed_dim=config["model"]["embed_dim"], num_classes=config["model"]["num_classes"], 
                          single=config["model"]["single"])
        model=model.to(device=device)
        criterion = nn.CrossEntropyLoss() #contiene già una softmax ###########

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
        append_to_log_file(log_path, f"Train subject {subject}")


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
                train_len = int(0.9 * len(indices))
                train_real = real_indices[:train_len]
                val_real   = real_indices[train_len:]

                # Aggiungo dati augmentati solo al training
                train_indices = np.concatenate([train_real, aug_indices])
                val_indices = val_real
            else:
                indices = np.random.permutation(len(train_dataset))
                train_len = int(0.9 * len(indices))
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
        append_to_log_file(log_path, f"Fatto")

        base_lr = 5e-4 * (config["train"]["batch_size"] / 32)  # scaling lineare
        total_epochs = EPOCHS
        warmup_epochs = 0.1 * EPOCHS
        steps_per_epoch = len(train_loader)

        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        if config["run"]["scheduler"]:
            # scheduler = LambdaLR(
            #     optimizer,
            #     lr_lambda=lr_lambda  # Warmup lineare
            # )
            scheduler = WarmupCosineScheduler(
            optimizer,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch
            )
        else:
            scheduler = None
        
        txt=f"Current lr: {optimizer.param_groups[0]['lr']}"
        append_to_log_file(log_path, txt)

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
            if val_loader is not None and (i+1) % 5 == 0:
                loss, epoch_accuracy, val_loss, epoch_val_accuracy = training_epoch(model, model_test, teacher, train_loader, test_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path, global_step = global_step, center = center)
                if val_loss <= val_best_loss or epoch_val_accuracy >= val_best_acc:
                    early_stop = 0

                    if val_loss <= val_best_loss:
                        val_best_loss = val_loss
                    
                    if epoch_val_accuracy >= val_best_acc:
                        val_best_acc = epoch_val_accuracy
                        
                    save_model(val_loss, i, teacher, optimizer, scheduler, subject, save_path, config["run"]["scheduler"] )

                    if val_loss < -0.98:
                        break
                else:
                    early_stop += 1

                if early_stop == 20 and stopped == False:
                    append_to_log_file(log_path, f"Early stop at epoch {i}")
                    #_, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
                    break
                    stopped = True

                # salvo quando sta per fermarsi
                # if early_stop == 1:
                #     save_model(val_loss, i, model, optimizer, scheduler, subject, save_path, config["run"]["scheduler"] )

                epoch_val_loss.append(val_loss)
                epoch_val_acc.append(epoch_val_accuracy)
            else:
                loss, epoch_accuracy, global_step, center = training_epoch(model, model_test, teacher, train_loader, test_loader, val_loader ,criterion, optimizer, scheduler, epoch=i, log_file = log_path, global_step = global_step, center = center)
                

            epoch_loss.append(loss)
            epoch_acc.append(epoch_accuracy)

            # if loss < best_loss:
            #     best_loss = loss
            #     torch.save({
            #         'loss': loss,
            #         'epoch': i,
            #         'model_state_dict': teacher.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict()
            #     }, save_path + "/val_" +subject + ".pth")

            

            
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
        

       # _, test_acc = test_model(model, test_loader=test_loader, criterion=criterion, log_file = log_path)
        #visualize_train_loss_acc(epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc, save_path=graphs_path + "/" +subject)
        if config["run"]["save"] == False:
            os.remove(load_path + "/" + subject + ".pth")

