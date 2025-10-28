from Preprocessing import prepare_dataloaders
import numpy as np
from torch.utils.data import DataLoader
import torch
from models.MVIT import MultiChannelViT
import torch.nn as nn
import time
from utils import (load_config, create_checkpoints_folders, 
                   append_to_log_file, plot_confusion_matrix)
import random
import argparse
import math
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


device = 'cuda'

def test_model(model, test_loader, criterion, log_file="log.txt"):
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

            start_time = time.time()
            outputs = model(inputs)
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
    avg_inference_time = np.mean(inference_times) / inputs.shape[0]

    txt = f"[TEST] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Avg Inference Time: {avg_inference_time*1000:.2f} ms/sample"
    print(txt)
    append_to_log_file(log_file, txt)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    kappa = cohen_kappa_score(all_labels, all_preds)
    txt = f"[TEST] Cohen's Kappa: {kappa:.4f}"
    print(txt)
    append_to_log_file(log_file, txt)

    return avg_loss, accuracy, all_preds, all_labels, kappa


def compute_grid_from_tokens(num_patches, H):
    divisors = [d for d in range(1, int(math.sqrt(num_patches)) + 1) if num_patches % d == 0]
    all_divs = set()
    for d in divisors:
        all_divs.add(d)
        all_divs.add(num_patches // d)
    all_divs = sorted(all_divs)
    candidates = [d for d in all_divs if d <= H]
    if len(candidates) > 0:
        grid_h = max(candidates)
        grid_w = num_patches // grid_h
        return grid_h, grid_w
    grid_h = int(math.floor(math.sqrt(num_patches)))
    while grid_h > 0:
        if num_patches % grid_h == 0:
            grid_w = num_patches // grid_h
            return grid_h, grid_w
        grid_h -= 1
    return 1, num_patches


def visualize_attention_on_image_fixed(image_tensor, attn_weights, save_path,
                                       target_size=(224,224), cmap_name="jet"):
    image = image_tensor.detach().cpu().numpy()
    if image.ndim == 3:
        image = image[0]
    image_proc = np.log1p(np.abs(image))
    image_proc = (image_proc - image_proc.min()) / (image_proc.max() - image_proc.min() + 1e-8)
    num_patches = int(attn_weights.shape[0])
    attn = attn_weights.detach().cpu()
    H_orig = image.shape[0]
    grid_h, grid_w = compute_grid_from_tokens(num_patches, H_orig)
    if grid_h * grid_w > num_patches:
        pad_size = grid_h * grid_w - num_patches
        attn = F.pad(attn, (0, pad_size), value=0.0)
        num_patches = attn.shape[0]
    attn_map = attn.cpu().numpy().reshape((grid_h, grid_w))
    Ht, Wt = target_size
    image_resized = cv2.resize(image_proc, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    attn_resized = cv2.resize(attn_map, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    attn_color = cmap(attn_resized)[:,:,:3]
    overlay = 0.6 * np.expand_dims(image_resized, axis=-1) + 0.4 * attn_color
    overlay = np.clip(overlay, 0.0, 1.0)
    plt.imsave(save_path, overlay)
    print(f"[saved] {save_path} (target_size={target_size}, grid={grid_h}x{grid_w})")


def extract_tsne(model, loader, subj, path):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to("cuda")
            out, feat = model(x, True)
            features.append(feat.cpu())
            labels.append(y)
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).squeeze().numpy()
    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"Class {c}", alpha=0.7)
    plt.legend()
    plt.title("t-SNE")
    plt.savefig(f"{path}/tsne_features_{subj}.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_loss_and_accuracy(checkpoint, subject, save_path, val_interval=5):
    train_loss = checkpoint['epoch_loss']
    train_acc = checkpoint['epoch_acc']
    val_loss = checkpoint['epoch_val_loss']
    val_acc = checkpoint['epoch_val_acc']

    epochs = list(range(1, len(train_loss) + 1))

    val_epochs = [0] + list(range(val_interval, val_interval * len(val_loss) + 1, val_interval))
    val_loss = [val_loss[0]] + val_loss
    val_acc = [val_acc[0]] + val_acc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- LOSS ---
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue')
    axes[0].plot(val_epochs, val_loss, label='Val Loss', color='orange')
    axes[0].set_title(f"Loss - {subject}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- ACCURACY ---
    axes[1].plot(epochs, train_acc, label='Train Accuracy', color='blue')
    axes[1].plot(val_epochs, val_acc, label='Val Accuracy', color='orange')
    axes[1].set_title(f"Accuracy - {subject}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"loss_accuracy_{subject}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(args, config, docker_prefix="../", root_2a="../", root_2b="../"):
    seed_n = 2025
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    s_accuracy = []
    s_kappa = []
    save_path, graphs_path, log_path = create_checkpoints_folders(args.config, config["model"]["single"], docker_prefix=docker_prefix)

    mean_cm = None

    for n in range(9):
        subject_test = f"A0{n+1}"
        batch_size = 32
        load_path = save_path + "/v_" + subject_test + ".pth"

        model = MultiChannelViT(**config["model"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-4)

        _, test_dataset = prepare_dataloaders(
            subject_id=subject_test, 
            augment=False, 
            filter=config["train"]["filter"], 
            BCI=config["run"]["dataset"], 
            root=root_2a, 
            root_2b=root_2b
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Epoch=", checkpoint['epoch']+1)
        print("Train loss=", checkpoint['loss'])

        # Plot Loss + Accuracy
        # plot_loss_and_accuracy(checkpoint, subject_test, graphs_path)

        avg_loss, avg_acc, all_preds, all_labels, kappa = test_model(model, test_loader, criterion)

        txt = f"Accuracy on {subject_test} is {avg_acc:.4f} | Kappa: {kappa:.4f}"
        append_to_log_file(log_path, txt)

        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, graphs_path+"/cm_"+subject_test)
        extract_tsne(model, test_loader, subject_test, graphs_path)

        s_accuracy.append(avg_acc)
        s_kappa.append(kappa)

        if mean_cm is None:
            mean_cm = cm.astype(np.float64)
        else:
            mean_cm += cm

    mean_cm /= 9
    plot_confusion_matrix(mean_cm, graphs_path + "/mean_confusion_matrix")

    subjects = [f"S{i+1}" for i in range(9)]
    plt.figure(figsize=(10, 5))
    plt.bar(subjects, s_accuracy)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Subject")
    plt.title("Accuracy per Subject")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(graphs_path + "/accuracy_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    txt = f"Mean accuracy {np.mean(s_accuracy):.4f} | Mean Cohen's Kappa {np.mean(s_kappa):.4f}"
    print(txt)
    append_to_log_file(log_path, txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/single_config_16_2_2.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    root_2a = "../Python/BciCompetitionIv2a/Train"
    root_2b = "../Python/BciCompetitionIv2b"
    main(args, config, docker_prefix="../", root_2a=root_2a, root_2b=root_2b)
