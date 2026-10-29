import os
import sys
import gc
import json
import time
import math
import argparse
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

# Configuração de Backend Gráfico para ambiente headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch e Torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms, models

# Algoritmos Clássicos e Métricas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, top_k_accuracy_score, f1_score
)

# UMAP
try:
    import umap
except ImportError:
    try:
        import umap.umap_ as umap
    except ImportError:
        umap = None


@dataclass
class Config:
    seed: int = 42
    run_id: str = "run_auditada"
    base_dir: str = "./resultados_botanico"
    data_dir: str = "./data"
    batch_size_train: int = 32
    batch_size_eval: int = 64
    num_workers: int = 2
    num_classes: int = 102
    img_size: int = 224
    resize_size: int = 256
    
    # Fase 1: Treinamento da Cabeça MLP
    epochs_phase1: int = 8
    lr_phase1: float = 1e-3
    freeze_backbone_bn: bool = True  # Mantém BatchNorm do backbone em eval()
    
    # Fase 2: Fine-Tuning Progressivo
    epochs_phase2: int = 4
    lr_phase2_backbone: float = 1e-4
    lr_phase2_head: float = 5e-4
    weight_decay: float = 1e-4
    
    # Avaliação e TTA
    run_tta_test: bool = True        # Executa TTA no teste completo se True
    n_bootstrap: int = 1000
    n_calibration_bins: int = 10
    
    # Hardware & Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()


def seed_everything(seed: int = 42):
   
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Relação oficial das 102 espécies botânicas (índices 0 a 101)
FLOWER_NAMES = [
    "Pink Primrose", "Hard-Leaved Pocket Orchid", "Canterbury Bells", "Sweet Pea",
    "English Marigold", "Tiger Lily", "Moon Orchid", "Bird of Paradise",
    "Monkshood", "Globe Thistle", "Snapdragon", "Colt's Foot",
    "King Protea", "Spear Thistle", "Yellow Iris", "Globe-Flower",
    "Purple Coneflower", "Peruvian Lily", "Balloon Flower", "Giant White Arum Lily",
    "Fire Lily", "Pincushion Flower", "Fritillary", "Red Ginger",
    "Grape Hyacinth", "Corn Poppy", "Prince of Wales Feathers", "Stemless Gentian",
    "Artichoke", "Sweet William", "Carnation", "Garden Phlox",
    "Love in the Mist", "Mexican Aster", "Alpine Sea Holly", "Ruby-Lipped Cattleya",
    "Cape Flower", "Great Masterwort", "Siam Tulip", "Lenten Rose",
    "Barbeton Daisy", "Daffodil", "Sword Lily", "Poinsettia",
    "Bolero Deep Blue", "Wallflower", "Marigold", "Buttercup",
    "Oxeye Daisy", "Common Dandelion", "Petunia", "Wild Pansy",
    "Primula", "Sunflower", "Pelargonium", "Bishop of Llandaff",
    "Gaura", "Geranium", "Orange Dahlia", "Pink-Yellow Dahlia",
    "Cautleya Spicata", "Japanese Anemone", "Black-Eyed Susan", "Silverbush",
    "Californian Poppy", "Osteospermum", "Spring Crocus", "Bearded Iris",
    "Windflower", "Tree Poppy", "Gazania", "Azalea",
    "Water Lily", "Rose", "Thorn Apple", "Morning Glory",
    "Passion Flower", "Lotus", "Toad Lily", "Anthurium",
    "Frangipani", "Clematis", "Hibiscus", "Columbine",
    "Desert-Rose", "Tree Mallow", "Magnolia", "Cyclamen",
    "Watercress", "Canna Lily", "Hippeastrum", "Bee Balm",
    "Ball Moss", "Foxglove", "Bougainvillea", "Camellia",
    "Mallow", "Mexican Petunia", "Bromelia", "Blanket Flower",
    "Trumpet Creeper", "Blackberry Lily"
]

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def denormalize(tensor: torch.Tensor, mean: List[float] = MEAN, std: List[float] = STD) -> torch.Tensor:
  
    t = tensor.clone().detach().cpu()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return torch.clamp(t, 0.0, 1.0)


def get_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
 
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(cfg.resize_size, antialias=True),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, val_test_transform

class FlowerClassifier(nn.Module):
   
    def __init__(self, backbone: nn.Module, num_classes: int = 102):
        super().__init__()
        self.backbone = backbone
        in_features = self.backbone.classifier[1].in_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.features(x)
        pooled = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.head(pooled)


def compute_classification_metrics(y_true: np.ndarray, y_probs: np.ndarray, k: int = 5, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
   
    y_preds = np.argmax(y_probs, axis=1)
    n_classes = y_probs.shape[1]
    if labels is None:
        labels = np.arange(n_classes)
        
    top1 = float(accuracy_score(y_true, y_preds)) * 100.0
    topk = float(top_k_accuracy_score(y_true, y_probs, k=k, labels=labels)) * 100.0
    f1_macro = float(f1_score(y_true, y_preds, average="macro", zero_division=0)) * 100.0
    return {
        "top1": top1,
        f"top{k}": topk,
        "f1_macro": f1_macro
    }


def compute_contingency_table(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Tuple[int, int]:
  
    corr_a = (preds_a == y_true)
    corr_b = (preds_b == y_true)
    b = int(np.sum(~corr_a & corr_b))
    c = int(np.sum(corr_a & ~corr_b))
    return b, c


def compute_mcnemar(b: int, c: int, correction: bool = True) -> Dict[str, Any]:
  
    total_disc = b + c
    if total_disc == 0:
        return {"b": b, "c": c, "chi2": 0.0, "df": 1, "p_value": 1.0, "significant_005": False}
    
    diff = abs(b - c)
    if correction:
        numerator = max(0.0, diff - 1.0) ** 2
    else:
        numerator = float(diff ** 2)
        
    chi2 = numerator / float(total_disc)
    p_val = float(stats.chi2.sf(chi2, df=1))
    
    return {
        "b": b,
        "c": c,
        "chi2": float(chi2),
        "df": 1,
        "p_value": p_val,
        "significant_005": bool(p_val < 0.05)
    }


def compute_paired_bootstrap(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray,
                             n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
   
    rng = np.random.RandomState(seed)
    n = len(y_true)
    corr_a = (preds_a == y_true).astype(float)
    corr_b = (preds_b == y_true).astype(float)
    
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        acc_a = np.mean(corr_a[idx])
        acc_b = np.mean(corr_b[idx])
        deltas.append((acc_b - acc_a) * 100.0)
        
    deltas = np.array(deltas)
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "std_delta": float(np.std(deltas))
    }


def compute_bootstrap_ci(y_true: np.ndarray, preds: np.ndarray,
                         n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    
    rng = np.random.RandomState(seed)
    n = len(y_true)
    corr = (preds == y_true).astype(float)
    
    accs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs.append(np.mean(corr[idx]) * 100.0)
        
    accs = np.array(accs)
    return {
        "mean_acc": float(np.mean(accs)),
        "ci_low": float(np.percentile(accs, 2.5)),
        "ci_high": float(np.percentile(accs, 97.5)),
        "std_acc": float(np.std(accs))
    }


def compute_calibration_metrics(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
  
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    n_samples = len(y_true)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Atribuição aos bins garantindo que 1.0 seja alocado no último bin
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(accuracies[mask]))
            conf = float(np.mean(confidences[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / n_samples) * abs(acc - conf)
        else:
            bin_accs.append(None)  # None indica bin vazio (evita viés visual de acc=0)
            bin_confs.append(float(bin_centers[b]))
            
    # Multiclass Brier Score: 1/N sum_i sum_c (p_ic - y_ic)^2
    y_one_hot = np.zeros_like(y_probs)
    y_one_hot[np.arange(n_samples), y_true] = 1.0
    brier = float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))
    
    # Negative Log-Likelihood (NLL)
    eps = 1e-12
    p_true = y_probs[np.arange(n_samples), y_true]
    nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    
    return {
        "ece": float(ece),
        "ece_percentage": float(ece * 100.0),
        "brier_score": brier,
        "nll": nll,
        "bin_centers": bin_centers.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts
    }

@torch.no_grad()
def extract_embeddings(model_bb: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
   
    model_bb.eval()
    feats, labs = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", enabled=(device.type == "cuda")):
            out = model_bb.features(imgs)
            emb = nn.functional.adaptive_avg_pool2d(out.float(), 1).flatten(1)
        feats.append(emb.cpu())
        labs.append(lbls)
    return torch.cat(feats), torch.cat(labs)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              optimizer: Optional[optim.Optimizer] = None,
              scaler: Optional[GradScaler] = None,
              device: torch.device = torch.device("cpu"),
              is_phase1: bool = False) -> Tuple[float, float]:
   
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if is_train and is_phase1:
        # Congelamento estrito de BatchNorm no backbone durante a Fase 1
        model.backbone.eval()
        
    loss_sum, correct, total = 0.0, 0, 0
    use_amp = (device.type == "cuda")
    
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            
        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, lbls)
            
        if is_train:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == lbls).sum().item()
        total    += imgs.size(0)
        
    return loss_sum / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    
    model.eval()
    probs_list, labels_list = [], []
    use_amp = (device.type == "cuda")
    
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)
        probs_list.append(probs.cpu())
        labels_list.append(lbls)
        
    all_probs = torch.cat(probs_list).numpy()
    all_labels = torch.cat(labels_list).numpy()
    return all_probs, all_labels


@torch.no_grad()
def evaluate_tta_full(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
   
    model.eval()
    probs_list, labels_list = [], []
    use_amp = (device.type == "cuda")
    
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            p_std = torch.softmax(model(imgs).float(), dim=1)
            imgs_flipped = torch.flip(imgs, dims=[3])
            p_flip = torch.softmax(model(imgs_flipped).float(), dim=1)
            p_avg = (p_std + p_flip) / 2.0
        probs_list.append(p_avg.cpu())
        labels_list.append(lbls)
        
    all_probs = torch.cat(probs_list).numpy()
    all_labels = torch.cat(labels_list).numpy()
    return all_probs, all_labels

class GradCAM:
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._fwd_hook)
        self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor, device: torch.device, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        self.model.zero_grad()
        x = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        
        with torch.enable_grad():
            feats = self.model.backbone.features(x)
            pooled = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            logits = self.model.head(pooled)
            if class_idx is None:
                class_idx = logits.argmax(1).item()
            score = logits[0, class_idx]
            score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = nn.functional.relu((weights * self.activations).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=img_tensor.shape[1:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        pred_prob = torch.softmax(logits, dim=1)[0, class_idx].item()
        return cam, class_idx, pred_prob

def generate_all_plots(results: Dict[str, Any], plots_dir: str, test_dataset: Dataset,
                       model: Optional[FlowerClassifier] = None, device: Optional[torch.device] = None):
    """Gera e salva as 11 figuras com rigor estético e rastreabilidade numérica."""
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    

    # F01: Grid de Amostras Reais (3x6)

    print("[1/11] Gerando F01: Grid de Amostras Reais (Oxford Flowers-102)...")
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=300)
    indices = np.linspace(0, len(test_dataset) - 1, 18, dtype=int)
    for ax, idx in zip(axes.ravel(), indices):
        img_t, lbl = test_dataset[idx]
        img_np = denormalize(img_t).permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        c_name = FLOWER_NAMES[lbl] if lbl < len(FLOWER_NAMES) else f"Cl.{lbl}"
        ax.set_title(f"Cl.{lbl}: {c_name}", fontsize=8.5, fontweight="bold", pad=4)
        ax.axis("off")
    fig.suptitle("Figura 1 — Variabilidade Morfológica e Diversidade Fenotípica Real no Dataset Oxford Flowers-102",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "F01_amostras_dataset.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # F02: Pipeline de Aumento de Dados

    print("[2/11] Gerando F02: Pipeline de Aumento de Dados...")
    sample_raw_img, sample_lbl = test_dataset[10]
    sample_np = denormalize(sample_raw_img).permute(1, 2, 0).numpy()
    t_crop = transforms.RandomResizedCrop(224, scale=(0.7, 0.75), ratio=(1.0, 1.0))
    t_rot  = transforms.RandomRotation(degrees=(25, 25))
    t_jit  = transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35)
    
    img_crop = t_crop(sample_raw_img)
    img_flip = torch.flip(sample_raw_img, dims=[2])
    img_rot  = t_rot(sample_raw_img)
    img_jit  = t_jit(sample_raw_img)
    img_norm = (sample_raw_img - sample_raw_img.min()) / (sample_raw_img.max() - sample_raw_img.min() + 1e-8)
    
    aug_panels = [
        ("Original\n(224×224)", sample_np),
        ("RandomResizedCrop\n(escala=0.72)", denormalize(img_crop).permute(1, 2, 0).numpy()),
        ("HorizontalFlip\n(p = 0.50)", denormalize(img_flip).permute(1, 2, 0).numpy()),
        ("RandomRotation\n(θ = +25°)", denormalize(img_rot).permute(1, 2, 0).numpy()),
        ("ColorJitter\n(Perturbação Solar)", denormalize(img_jit).permute(1, 2, 0).numpy()),
        ("Normalizado\n(ImageNet Mean/Std)", img_norm.permute(1, 2, 0).numpy())
    ]
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.8), dpi=300)
    for ax, (title, img_v) in zip(axes, aug_panels):
        ax.imshow(np.clip(img_v, 0.0, 1.0))
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
    fig.suptitle("Figura 2 — Pipeline de Pré-processamento e Aumento Estocástico Aplicado à Mesma Foto Real",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "F02_data_augmentation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


    # F03: Hierarquia de Features e Filtros

    if model is not None and device is not None:
        print("[3/11] Gerando F03: Filtros Convolucionais 3x3 e Ativações Reais da MobileNetV2...")
        fig = plt.figure(figsize=(16, 8.5), dpi=300)
        gs = fig.add_gridspec(2, 6, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.15)
        first_conv_weights = model.backbone.features[0][0].weight.detach().cpu()
        sub_gs = gs[0, :].subgridspec(2, 6, hspace=0.15, wspace=0.15)
        for i in range(12):
            sub_ax = fig.add_subplot(sub_gs[i // 6, i % 6])
            kw = first_conv_weights[i]
            kw_norm = (kw - kw.min()) / (kw.max() - kw.min() + 1e-8)
            sub_ax.imshow(kw_norm.permute(1, 2, 0).numpy(), interpolation="nearest")
            sub_ax.set_title(f"Filtro #{i+1}", fontsize=8)
            sub_ax.axis("off")

        with torch.no_grad():
            x_probe = sample_raw_img.unsqueeze(0).to(device)
            probe_stages = []
            curr = x_probe
            for idx_b, blk in enumerate(model.backbone.features):
                curr = blk(curr)
                if idx_b in [0, 3, 6, 13, 18]:
                    probe_stages.append((idx_b, curr.detach().cpu().mean(dim=1)[0].numpy()))

        st_info = [
            ("Entrada Real\n(224×224×3)", sample_np, None),
            ("features.0\n(Bordas Finas)", probe_stages[0][1], "viridis"),
            ("features.3\n(Gradientes)", probe_stages[1][1], "viridis"),
            ("features.6\n(Texturas)", probe_stages[2][1], "magma"),
            ("features.13\n(Órgãos Florais)", probe_stages[3][1], "plasma"),
            ("features.18\n(Semântica)", probe_stages[4][1], "inferno")
        ]
        for j, (st_name, st_mat, cmp) in enumerate(st_info):
            ax_st = fig.add_subplot(gs[1, j])
            if cmp is None:
                ax_st.imshow(st_mat)
            else:
                st_n = (st_mat - st_mat.min()) / (st_mat.max() - st_mat.min() + 1e-8)
                ax_st.imshow(st_n, cmap=cmp)
            ax_st.set_title(st_name, fontsize=9, fontweight="bold", pad=4)
            ax_st.axis("off")

        fig.suptitle("Figura 3 — Extração Hierárquica Real: Filtros Convolucionais 3×3 e Mapas de Ativação Profundos",
                     fontsize=13, fontweight="bold", y=0.98)
        fig.savefig(os.path.join(plots_dir, "F03_hierarquia_features.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F04: Projeção UMAP 2D dos Embeddings (Sem Famílias Artificiais)

    if "umap_2d" in results:
        print("[4/11] Gerando F04: Projeção UMAP 2D dos Embeddings Reais (Sem Famílias Artificiais)...")
        umap_data = results["umap_2d"]
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        scatter = ax.scatter(
            umap_data["x"], umap_data["y"],
            c=umap_data["labels"], cmap="viridis",
            s=22, alpha=0.8, edgecolors="none"
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Índice da Classe Botânica (0 a 101)", fontsize=9, fontweight="bold")
        ax.set_title("Figura 4 — Projeção UMAP 2D dos Embeddings Reais MobileNetV2 (1280-d)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Dimensão UMAP 1", fontsize=10, fontweight="bold")
        ax.set_ylabel("Dimensão UMAP 2", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F04_umap_embeddings.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F05: Desempenho Comparativo de Classificadores

    print("[5/11] Gerando F05: Desempenho Comparativo de Classificadores...")
    classic_res = results.get("models_summary", [])
    if classic_res:
        comp_df = pd.DataFrame(classic_res)
        fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)
        sns.barplot(data=comp_df, x="model_name", y="top1", hue="model_name", palette="Blues_r", ax=ax, legend=False)
        p1_acc = results.get("metrics_p1", {}).get("top1", 0.0)
        if p1_acc > 0:
            ax.axhline(p1_acc, color="#C0392B", linestyle="--", linewidth=1.2, label=f"Baseline Fase 1 ({p1_acc:.2f}%)")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Acurácia Top-1 no Teste (%)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Classificador / Paradigma de Inferência", fontsize=10, fontweight="bold")
        ax.set_title("Figura 5 — Desempenho Empírico Comparativo sobre Embeddings Reais MobileNetV2",
                     fontsize=12, fontweight="bold", pad=12)
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h + 1.2),
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1B365D")
        ax.legend(loc="lower right", fontsize=9)
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F05_comparativo_classificadores.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F06: Curvas de Aprendizado da Fase 1

    hist_p1 = results.get("history_p1", {})
    if hist_p1 and "train_loss" in hist_p1:
        print("[6/11] Gerando F06: Curvas Reais de Aprendizado da Fase 1...")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
        ep_p1 = range(1, len(hist_p1["train_loss"]) + 1)
        axes[0].plot(ep_p1, hist_p1["train_loss"], marker="o", color="#1F77B4", label="Treino (Cross-Entropy)")
        axes[0].plot(ep_p1, hist_p1["val_loss"], marker="s", color="#FF7F0E", label="Validação")
        axes[0].set_title("Dinâmica da Perda (Fase 1: Cabeça MLP)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Época", fontsize=9, fontweight="bold")
        axes[0].set_ylabel("Perda", fontsize=9, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(ep_p1, [a * 100 for a in hist_p1["train_acc"]], marker="o", color="#2CA02C", label="Treino Top-1")
        axes[1].plot(ep_p1, [a * 100 for a in hist_p1["val_acc"]], marker="s", color="#D62728", label="Validação Top-1")
        axes[1].set_title("Convergência da Acurácia (Fase 1: Cabeça MLP)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Época", fontsize=9, fontweight="bold")
        axes[1].set_ylabel("Acurácia Top-1 (%)", fontsize=9, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Figura 6 — Curvas Empíricas de Otimização na Fase 1 com Backbone MobileNetV2 Congelado",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F06_curvas_aprendizado_fase1.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F07: Curvas Consolidadas Fase 1 + Fine-Tuning

    hist_ft = results.get("history_ft", {})
    if hist_p1 and hist_ft and "train_loss" in hist_ft:
        print("[7/11] Gerando F07: Curvas Consolidadas Fase 1 + Fine-Tuning...")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
        n_p1 = len(hist_p1["train_loss"])
        n_ft = len(hist_ft["train_loss"])
        split_point = n_p1 + 0.5

        axes[0].plot(range(1, n_p1 + 1), hist_p1["train_loss"], marker="o", color="#1F77B4", label="Treino (Fase 1)")
        axes[0].plot(range(1, n_p1 + 1), hist_p1["val_loss"], marker="s", color="#FF7F0E", label="Validação (Fase 1)")
        axes[0].plot(range(n_p1 + 1, n_p1 + n_ft + 1), hist_ft["train_loss"], marker="o", linestyle="--", color="#1F77B4", label="Treino (Fine-Tuning)")
        axes[0].plot(range(n_p1 + 1, n_p1 + n_ft + 1), hist_ft["val_loss"], marker="s", linestyle="--", color="#FF7F0E", label="Validação (Fine-Tuning)")
        axes[0].axvline(split_point, color="#7F7F7F", linestyle=":", linewidth=1.5, label="Descongelamento features.16-18")
        axes[0].set_title("Perda Global Consolidada", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Época Global", fontsize=9, fontweight="bold")
        axes[0].set_ylabel("Perda", fontsize=9, fontweight="bold")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(range(1, n_p1 + 1), [a * 100 for a in hist_p1["train_acc"]], marker="o", color="#2CA02C", label="Treino (Fase 1)")
        axes[1].plot(range(1, n_p1 + 1), [a * 100 for a in hist_p1["val_acc"]], marker="s", color="#D62728", label="Validação (Fase 1)")
        axes[1].plot(range(n_p1 + 1, n_p1 + n_ft + 1), [a * 100 for a in hist_ft["train_acc"]], marker="o", linestyle="--", color="#2CA02C", label="Treino (Fine-Tuning)")
        axes[1].plot(range(n_p1 + 1, n_p1 + n_ft + 1), [a * 100 for a in hist_ft["val_acc"]], marker="s", linestyle="--", color="#D62728", label="Validação (Fine-Tuning)")
        axes[1].axvline(split_point, color="#7F7F7F", linestyle=":", linewidth=1.5, label="Descongelamento features.16-18")
        axes[1].set_title("Evolução da Acurácia Top-1 (%)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Época Global", fontsize=9, fontweight="bold")
        axes[1].set_ylabel("Acurácia (%)", fontsize=9, fontweight="bold")
        axes[1].legend(fontsize=8, loc="lower right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Figura 7 — Dinâmica de Treinamento em Dois Estágios: Congelamento e Fine-Tuning Progressivo",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F07_curvas_consolidadas_finetuning.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F08: Matriz de Confusão Global (102x102)

    if "confusion_matrix" in results:
        print("[8/11] Gerando F08: Matriz de Confusão Global (102x102)...")
        cm = np.array(results["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
        sns.heatmap(cm, cmap="Blues", cbar=True, square=True, xticklabels=10, yticklabels=10,
                    cbar_kws={"shrink": 0.8, "label": "Número Real de Amostras de Teste"}, ax=ax)
        ax.set_title("Figura 8 — Matriz de Confusão Global Empírica (102 Classes Botânicas) no Teste",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Classe Predita (Índice 0 a 101)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Classe Real / Ground Truth (Índice 0 a 101)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F08_matriz_confusao.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F09: Interpretabilidade Visual Grad-CAM Real

    if model is not None and device is not None:
        print("[9/11] Gerando F09: Interpretabilidade Visual com Grad-CAM Real...")
        gradcam = GradCAM(model, model.backbone.features[-1])
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
        rng_cam = np.random.RandomState(42)
        cam_indices = rng_cam.choice(len(test_dataset), size=4, replace=False)
        for col, idx in enumerate(cam_indices):
            img_t, true_lbl = test_dataset[idx]
            cam_map, pred_lbl, pred_prob = gradcam.generate(img_t, device)
            img_raw = denormalize(img_t).permute(1, 2, 0).numpy()
            is_hit = (pred_lbl == true_lbl)
            status_str = "ACERTO" if is_hit else "ERRO"
            hdr_color = "darkgreen" if is_hit else "crimson"
            name_gt = FLOWER_NAMES[true_lbl] if true_lbl < len(FLOWER_NAMES) else f"Cl.{true_lbl}"
            name_pr = FLOWER_NAMES[pred_lbl] if pred_lbl < len(FLOWER_NAMES) else f"Cl.{pred_lbl}"

            axes[0, col].imshow(img_raw)
            axes[0, col].set_title(f"GT: {name_gt}\nPred: {name_pr} ({pred_prob*100:.1f}%) [{status_str}]",
                                   fontsize=8.5, fontweight="bold", color=hdr_color, pad=4)
            axes[0, col].axis("off")

            axes[1, col].imshow(img_raw)
            axes[1, col].imshow(cam_map, cmap="jet", alpha=0.55)
            axes[1, col].set_title("Grad-CAM: Relevância Espacial", fontsize=8.5, pad=4)
            axes[1, col].axis("off")

        fig.suptitle("Figura 9 — Interpretabilidade Qualitativa Grad-CAM: Regiões de Relevância Espacial Convolucional",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F09_gradcam_interpretabilidade.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F10: Calibração Probabilística e Reliability Diagram

    if "calibration" in results:
        print("[10/11] Gerando F10: Calibração Probabilística e Diagrama de Confiabilidade...")
        calib = results["calibration"]
        probs_max = np.array(results.get("confidences_test", []))
        correct_mask = np.array(results.get("correct_mask_test", []))
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
        if len(probs_max) > 0 and len(correct_mask) > 0:
            axes[0].hist(probs_max[correct_mask], bins=25, alpha=0.75, color="#2CA02C",
                         label=f"Corretas ({correct_mask.sum()})", density=True)
            axes[0].hist(probs_max[~correct_mask], bins=25, alpha=0.75, color="#D62728",
                         label=f"Incorretas ({(~correct_mask).sum()})", density=True)
            axes[0].set_title("Distribuição da Confiança Máxima Pmax", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Confiança Máxima Softmax P(y|X)", fontsize=9, fontweight="bold")
            axes[0].set_ylabel("Densidade", fontsize=9, fontweight="bold")
            axes[0].legend(fontsize=9)
            axes[0].grid(True, linestyle="--", alpha=0.5)

        # Diagrama de Confiabilidade
        centers = calib["bin_centers"]
        accs = [a if a is not None else np.nan for a in calib["bin_accs"]]
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="#7F7F7F", linewidth=1.5, label="Calibração Perfeita (y = x)")
        valid_mask = [a is not None for a in calib["bin_accs"]]
        valid_centers = [c for c, v in zip(centers, valid_mask) if v]
        valid_accs = [a for a, v in zip(calib["bin_accs"], valid_mask) if v]
        
        axes[1].bar(valid_centers, valid_accs, width=0.08, alpha=0.65, color="#1F77B4",
                    edgecolor="#1B365D", label=f"Acurácia Observada (ECE = {calib['ece_percentage']:.2f}%)")
        axes[1].plot(valid_centers, valid_accs, marker="o", color="#1B365D", linewidth=1.5)
        axes[1].set_title(f"Diagrama de Confiabilidade (Reliability Diagram)\nECE = {calib['ece_percentage']:.2f}% | Brier = {calib['brier_score']:.4f}",
                          fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Confiança Média do Bin", fontsize=9, fontweight="bold")
        axes[1].set_ylabel("Acurácia Observada", fontsize=9, fontweight="bold")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=9, loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Figura 10 — Avaliação de Calibração Preditiva e Incerteza no Conjunto de Teste",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F10_calibracao_confiabilidade.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


    # F11: Análise de Erros por Classe e Pares Críticos

    if "confusion_matrix" in results:
        print("[11/11] Gerando F11: Desagregação por Classe e Pares Críticos de Confusão...")
        cm = np.array(results["confusion_matrix"])
        class_acc = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None) * 100.0
        sorted_indices = np.argsort(class_acc)
        mean_acc_val = float(np.mean(class_acc))

        off_diag = cm.copy().astype(float)
        np.fill_diagonal(off_diag, 0)
        flat_errors = []
        for i in range(len(class_acc)):
            for j in range(len(class_acc)):
                if off_diag[i, j] > 0:
                    flat_errors.append((i, j, int(off_diag[i, j])))
        flat_errors.sort(key=lambda x: -x[2])
        top_pairs_real = flat_errors[:6]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=300, gridspec_kw={"height_ratios": [1.4, 1.0]})
        colors_bars = ["#D62728" if class_acc[idx] < 60.0 else "#1F77B4" for idx in sorted_indices]
        axes[0].bar(range(len(class_acc)), class_acc[sorted_indices], color=colors_bars, width=0.8)
        axes[0].axhline(mean_acc_val, color="#1B365D", linestyle="--", linewidth=1.5,
                        label=f"Média Top-1: {mean_acc_val:.1f}%")
        axes[0].axhline(60.0, color="#D62728", linestyle=":", linewidth=1.2, label="Limiar Crítico (<60%)")
        axes[0].set_title("A. Acurácia Individual por Classe (Ordenada da Menor para Maior)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Acurácia Top-1 (%)", fontsize=9, fontweight="bold")
        axes[0].set_xlabel("Classes Botânicas (102 Classes Reordenadas)", fontsize=9, fontweight="bold")
        axes[0].set_ylim(0, 105)
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        pair_labels = [f"{FLOWER_NAMES[ci]} (Cl.{ci}) → {FLOWER_NAMES[cj]} (Cl.{cj})" for ci, cj, _ in top_pairs_real]
        pair_counts = [count for _, _, count in top_pairs_real]
        axes[1].barh(range(len(pair_labels)), pair_counts, color="#FF7F0E", edgecolor="#D35400", height=0.6)
        axes[1].set_yticks(range(len(pair_labels)))
        axes[1].set_yticklabels(pair_labels, fontsize=8, fontweight="bold")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Frequência de Amostras de Teste Confundidas", fontsize=9, fontweight="bold")
        axes[1].set_title("B. Pares Taxonômicos com Maior Frequência de Confusão", fontsize=11, fontweight="bold")
        for idx_b, c_val in enumerate(pair_counts):
            axes[1].text(c_val + 0.3, idx_b, f"{c_val} amostras", va="center", fontsize=8, fontweight="bold", color="#1B365D")
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)

        fig.suptitle("Figura 11 — Diagnóstico de Granularidade Fina: Desagregação por Classe e Pares de Confusão",
                     fontsize=12, fontweight="bold", y=0.99)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "F11_analise_classes_erros.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("[OK] Figuras geradas com sucesso em:", plots_dir)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Auditado de Classificação Botânica (Oxford Flowers-102)")
    parser.add_argument("--base-dir", type=str, default="./resultados_botanico", help="Diretório base de saída")
    parser.add_argument("--data-dir", type=str, default="./data", help="Diretório do dataset Flowers-102")
    parser.add_argument("--epochs-p1", type=int, default=8, help="Épocas na Fase 1")
    parser.add_argument("--epochs-p2", type=int, default=4, help="Épocas no Fine-Tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size de treino")
    parser.add_argument("--seed", type=int, default=42, help="Semente de reprodutibilidade")
    parser.add_argument("--no-tta", action="store_true", help="Desativa TTA na avaliação de teste")
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        base_dir=args.base_dir,
        data_dir=args.data_dir,
        epochs_phase1=args.epochs_p1,
        epochs_phase2=args.epochs_p2,
        batch_size_train=args.batch_size,
        run_tta_test=not args.no_tta
    )

    seed_everything(cfg.seed)
    os.makedirs(cfg.base_dir, exist_ok=True)
    plots_dir = os.path.join(cfg.base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("PIPELINE AUDITADO DE CLASSIFICAÇÃO BOTÂNICA (OXFORD FLOWERS-102)")
    print(f"Dispositivo: {cfg.device}")
    print(f"Diretório de Artefatos: {cfg.base_dir}")
    print(f"Diretório de Figuras: {plots_dir}")
    print("=" * 80)

    # 1. Carregamento de Dados
    print("\n[INFO] Carregando partições do Oxford Flowers-102...")
    train_trans, val_test_trans = get_transforms(cfg)
    train_dataset = datasets.Flowers102(root=cfg.data_dir, split="train", download=True, transform=train_trans)
    val_dataset   = datasets.Flowers102(root=cfg.data_dir, split="val",   download=True, transform=val_test_trans)
    test_dataset  = datasets.Flowers102(root=cfg.data_dir, split="test",  download=True, transform=val_test_trans)

    train_eval_dataset = datasets.Flowers102(root=cfg.data_dir, split="train", download=True, transform=val_test_trans)

    train_loader      = DataLoader(train_dataset, batch_size=cfg.batch_size_train, shuffle=True,  num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    val_loader        = DataLoader(val_dataset,   batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader       = DataLoader(test_dataset,  batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    print(f"[OK] Amostras carregadas: Treino={len(train_dataset)} | Validação={len(val_dataset)} | Teste={len(test_dataset)}")

    # 2. Inicialização do Backbone
    print("\n[INFO] Inicializando backbone MobileNetV2 (ImageNet-1K V1)...")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    backbone = models.mobilenet_v2(weights=weights).to(cfg.device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # 3. Extração Determinística de Embeddings para Modelos Clássicos
    print("\n[INFO] Extraindo embeddings latentes de 1280-d de forma determinística...")
    X_train_emb, y_train_emb = extract_embeddings(backbone, train_eval_loader, torch.device(cfg.device))
    X_test_emb,  y_test_emb  = extract_embeddings(backbone, test_loader,       torch.device(cfg.device))
    print(f"[OK] Embeddings extraídos: X_train={X_train_emb.shape}, X_test={X_test_emb.shape}")

    # Projeção UMAP 2D
    umap_res = {}
    if umap is not None:
        print("[INFO] Calculando projeção UMAP 2D sobre amostras do conjunto de teste...")
        rng_u = np.random.RandomState(cfg.seed)
        n_umap_samples = min(1500, len(X_test_emb))
        idx_u = rng_u.choice(len(X_test_emb), size=n_umap_samples, replace=False)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=2, random_state=cfg.seed)
        X_test_2d = reducer.fit_transform(X_test_emb[idx_u].numpy())
        umap_res = {
            "x": X_test_2d[:, 0].tolist(),
            "y": X_test_2d[:, 1].tolist(),
            "labels": y_test_emb[idx_u].numpy().tolist()
        }

    # 4. Benchmarking de Classificadores Clássicos
    print("\n[INFO] Treinando classificadores clássicos sobre embeddings de treino...")
    classic_models = [
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ("Random Forest (100)", RandomForestClassifier(n_estimators=100, max_depth=20, random_state=cfg.seed, n_jobs=-1)),
        ("Regressão Logística L2", LogisticRegression(max_iter=300, C=1.0, random_state=cfg.seed))
    ]
    classic_predictions = {}
    classic_summary = []

    for name, clf in classic_models:
        t0 = time.time()
        clf.fit(X_train_emb.numpy(), y_train_emb.numpy())
        preds = clf.predict(X_test_emb.numpy())
        classic_predictions[name] = preds
        acc = float(accuracy_score(y_test_emb.numpy(), preds)) * 100.0
        f1 = float(f1_score(y_test_emb.numpy(), preds, average="macro", zero_division=0)) * 100.0
        fit_time = time.time() - t0
        classic_summary.append({
            "model_name": name,
            "top1": acc,
            "f1_macro": f1,
            "time_s": fit_time
        })
        print(f"   {name:<24} | Top-1: {acc:6.2f}% | Macro F1: {f1:6.2f}% | Tempo: {fit_time:4.1f}s")

    # 5. Modelo Neural End-to-End: Fase 1 (Cabeça MLP, Backbone Congelado)
    print("\n[INFO] Iniciando Treinamento da Fase 1 (Cabeça MLP, Backbone Congelado)...")
    model = FlowerClassifier(backbone, cfg.num_classes).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=cfg.use_amp)
    optimizer_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg.lr_phase1, weight_decay=cfg.weight_decay)

    history_p1 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss_p1 = float("inf")
    best_state_p1 = None

    for epoch in range(1, cfg.epochs_phase1 + 1):
        t0_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer_p1, scaler,
                                    torch.device(cfg.device), is_phase1=cfg.freeze_backbone_bn)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device=torch.device(cfg.device))
        history_p1["train_loss"].append(tr_loss); history_p1["train_acc"].append(tr_acc)
        history_p1["val_loss"].append(vl_loss);   history_p1["val_acc"].append(vl_acc)
        if vl_loss < best_val_loss_p1:
            best_val_loss_p1 = vl_loss
            best_state_p1 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"   [Fase 1] Época {epoch:02d}/{cfg.epochs_phase1:02d} | Train: Loss={tr_loss:.4f}, Acc={tr_acc*100:5.2f}% | Val: Loss={vl_loss:.4f}, Acc={vl_acc*100:5.2f}% ({time.time()-t0_ep:.1f}s)")

    if best_state_p1 is not None:
        model.load_state_dict(best_state_p1)

    print("\n[INFO] Avaliando Fase 1 no conjunto de teste...")
    test_probs_p1, test_labels_real = evaluate_model(model, test_loader, torch.device(cfg.device))
    test_preds_p1 = np.argmax(test_probs_p1, axis=1)
    metrics_p1 = compute_classification_metrics(test_labels_real, test_probs_p1)
    print(f"[OK] Desempenho Fase 1 (Teste): Top-1 = {metrics_p1['top1']:.2f}% | Top-5 = {metrics_p1['top5']:.2f}% | Macro F1 = {metrics_p1['f1_macro']:.2f}%")

    classic_summary.append({
        "model_name": "Cabeça MLP (Fase 1)",
        "top1": metrics_p1["top1"],
        "f1_macro": metrics_p1["f1_macro"],
        "time_s": 0.0
    })

    # 6. Modelo Neural End-to-End: Fase 2 (Fine-Tuning Progressivo)
    print("\n[INFO] Iniciando Fase 2: Fine-Tuning Progressivo dos Blocos features.16-18...")
    for name, param in model.backbone.named_parameters():
        if any(k in name for k in ["features.16", "features.17", "features.18"]):
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parâmetros treináveis após descongelamento cirúrgico: {trainable_params:,}")

    optimizer_ft = optim.AdamW([
        {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": cfg.lr_phase2_backbone},
        {"params": model.head.parameters(), "lr": cfg.lr_phase2_head}
    ], weight_decay=cfg.weight_decay)

    history_ft = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss_ft = float("inf")
    best_state_ft = None

    for epoch in range(1, cfg.epochs_phase2 + 1):
        t0_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer_ft, scaler,
                                    torch.device(cfg.device), is_phase1=False)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device=torch.device(cfg.device))
        history_ft["train_loss"].append(tr_loss); history_ft["train_acc"].append(tr_acc)
        history_ft["val_loss"].append(vl_loss);   history_ft["val_acc"].append(vl_acc)
        if vl_loss < best_val_loss_ft:
            best_val_loss_ft = vl_loss
            best_state_ft = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"   [Fine-Tuning] Época {epoch:02d}/{cfg.epochs_phase2:02d} | Train: Loss={tr_loss:.4f}, Acc={tr_acc*100:5.2f}% | Val: Loss={vl_loss:.4f}, Acc={vl_acc*100:5.2f}% ({time.time()-t0_ep:.1f}s)")

    if best_state_ft is not None:
        model.load_state_dict(best_state_ft)

    print("\n[INFO] Avaliando Modelo Final (Fine-Tuning) no conjunto de teste completo (6.149 imagens)...")
    test_probs_ft, test_labels_real = evaluate_model(model, test_loader, torch.device(cfg.device))
    test_preds_ft = np.argmax(test_probs_ft, axis=1)
    metrics_ft = compute_classification_metrics(test_labels_real, test_probs_ft)
    cm = confusion_matrix(test_labels_real, test_preds_ft, labels=list(range(cfg.num_classes)))

    print(f"[OK] Desempenho Modelo Final (Teste):")
    print(f"     Top-1 Accuracy: {metrics_ft['top1']:.2f}%")
    print(f"     Top-5 Accuracy: {metrics_ft['top5']:.2f}%")
    print(f"     Macro F1-Score: {metrics_ft['f1_macro']:.2f}%")

    classic_summary.append({
        "model_name": "MobileNetV2 (Fine-Tuning)",
        "top1": metrics_ft["top1"],
        "f1_macro": metrics_ft["f1_macro"],
        "time_s": 0.0
    })

    # 7. Test-Time Augmentation (TTA) no Teste Completo
    metrics_tta = None
    test_probs_tta = None
    if cfg.run_tta_test:
        print("\n[INFO] Executando TTA no conjunto de teste completo (sem subamostragem ou piso forçado)...")
        test_probs_tta, _ = evaluate_tta_full(model, test_loader, torch.device(cfg.device))
        test_preds_tta = np.argmax(test_probs_tta, axis=1)
        metrics_tta = compute_classification_metrics(test_labels_real, test_probs_tta)
        delta_tta_real = metrics_tta["top1"] - metrics_ft["top1"]
        print(f"[OK] TTA no teste completo: Top-1 = {metrics_tta['top1']:.2f}% (ΔTTA = {delta_tta_real:+.2f}%)")

    # 8. Validação Estatística Não-Paramétrica
    print("\n[INFO] Executando Testes Estatísticos Não-Paramétricos...")
    
    # McNemar entre Fase 1 e Fine-Tuning
    b_ft, c_ft = compute_contingency_table(test_labels_real, test_preds_p1, test_preds_ft)
    mcnemar_ft = compute_mcnemar(b_ft, c_ft, correction=True)
    
    # Bootstrap pareado do delta de Fine-Tuning
    boot_delta_ft = compute_paired_bootstrap(test_labels_real, test_preds_p1, test_preds_ft,
                                             n_boot=cfg.n_bootstrap, seed=cfg.seed)
    # Bootstrap IC 95% do modelo final
    boot_ci_ft = compute_bootstrap_ci(test_labels_real, test_preds_ft,
                                      n_boot=cfg.n_bootstrap, seed=cfg.seed)

    # McNemar para cada modelo clássico pareado contra a Fase 1
    mcnemar_classics = {}
    for name, c_preds in classic_predictions.items():
        b_c, c_c = compute_contingency_table(test_labels_real, c_preds, test_preds_p1)
        mcnemar_classics[name] = compute_mcnemar(b_c, c_c, correction=True)

    print(f"[*] Delta Fine-Tuning: +{metrics_ft['top1'] - metrics_p1['top1']:.2f}%")
    print(f"[*] Bootstrap Pareado Δ(FT - P1): {boot_delta_ft['mean_delta']:+.2f}% [IC 95%: {boot_delta_ft['ci_low']:+.2f}%, {boot_delta_ft['ci_high']:+.2f}%]")
    print(f"[*] Bootstrap IC 95% Final: [{boot_ci_ft['ci_low']:.2f}%, {boot_ci_ft['ci_high']:.2f}%]")
    print(f"[*] McNemar Edwards (P1 vs FT): b={b_ft}, c={c_ft}, chi2={mcnemar_ft['chi2']:.2f}, p={mcnemar_ft['p_value']:.4e}")

    # 9. Calibração Probabilística
    print("\n[INFO] Calculando métricas de calibração probabilística...")
    calib_metrics = compute_calibration_metrics(test_labels_real, test_probs_ft, n_bins=cfg.n_calibration_bins)
    print(f"[*] Expected Calibration Error (ECE): {calib_metrics['ece_percentage']:.2f}%")
    print(f"[*] Brier Score: {calib_metrics['brier_score']:.4f}")
    print(f"[*] Negative Log-Likelihood (NLL): {calib_metrics['nll']:.4f}")

    # 10. Tabela de Síntese Comparativa
    table_rows = []
    # Clássicos
    for name in ["KNN (k=5)", "Random Forest (100)", "Regressão Logística L2"]:
        c_acc = next(item["top1"] for item in classic_summary if item["model_name"] == name)
        c_f1  = next(item["f1_macro"] for item in classic_summary if item["model_name"] == name)
        mcn_c = mcnemar_classics[name]
        p_str = f"p = {mcn_c['p_value']:.2e}" if mcn_c['p_value'] >= 1e-10 else "p < 10⁻¹⁰"
        table_rows.append([
            name, f"{c_acc:.2f}%", "—", f"{c_f1:.2f}%", "—",
            f"{c_acc - metrics_p1['top1']:+.2f}%", p_str
        ])

    # Fase 1
    table_rows.append([
        "Cabeça MLP (Fase 1)", f"{metrics_p1['top1']:.2f}%", f"{metrics_p1['top5']:.2f}%",
        f"{metrics_p1['f1_macro']:.2f}%", "—", "Referência", "Referência"
    ])

    # Fine-Tuning
    p_ft_str = f"p = {mcnemar_ft['p_value']:.2e}" if mcnemar_ft['p_value'] >= 1e-10 else "p < 10⁻¹⁰"
    table_rows.append([
        "MobileNetV2 (Fine-Tuning)", f"{metrics_ft['top1']:.2f}%", f"{metrics_ft['top5']:.2f}%",
        f"{metrics_ft['f1_macro']:.2f}%", f"[{boot_ci_ft['ci_low']:.2f}%, {boot_ci_ft['ci_high']:.2f}%]",
        f"+{metrics_ft['top1'] - metrics_p1['top1']:.2f}%", p_ft_str
    ])

    # TTA se disponível
    if metrics_tta is not None:
        table_rows.append([
            "Com TTA (Flip Completo)", f"{metrics_tta['top1']:.2f}%", f"{metrics_tta['top5']:.2f}%",
            f"{metrics_tta['f1_macro']:.2f}%", "—", f"+{metrics_tta['top1'] - metrics_p1['top1']:.2f}%", "—"
        ])

    # 11. Consolidação e Persistência de Resultados
    results_dict = {
        "metadata": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": cfg.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": cfg.seed
        },
        "config": asdict(cfg),
        "models_summary": classic_summary,
        "metrics_p1": metrics_p1,
        "metrics_ft": metrics_ft,
        "metrics_tta": metrics_tta,
        "mcnemar_ft": mcnemar_ft,
        "mcnemar_classics": mcnemar_classics,
        "boot_delta_ft": boot_delta_ft,
        "boot_ci_ft": boot_ci_ft,
        "calibration": calib_metrics,
        "history_p1": history_p1,
        "history_ft": history_ft,
        "confusion_matrix": cm.tolist(),
        "confidences_test": np.max(test_probs_ft, axis=1).tolist(),
        "correct_mask_test": (test_preds_ft == test_labels_real).tolist(),
        "umap_2d": umap_res,
        "table_summary": table_rows
    }

    # Salva arquivos de auditoria em disco
    metrics_path = os.path.join(cfg.base_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Métricas e metadados estruturados salvos em: {metrics_path}")

    predictions_path = os.path.join(cfg.base_dir, "predictions.npz")
    np.savez_compressed(
        predictions_path,
        y_true=test_labels_real,
        probs_p1=test_probs_p1,
        preds_p1=test_preds_p1,
        probs_ft=test_probs_ft,
        preds_ft=test_preds_ft,
        probs_tta=test_probs_tta if test_probs_tta is not None else np.array([]),
        preds_classic_knn=classic_predictions.get("KNN (k=5)", np.array([])),
        preds_classic_rf=classic_predictions.get("Random Forest (100)", np.array([])),
        preds_classic_lr=classic_predictions.get("Regressão Logística L2", np.array([]))
    )
    print(f"[OK] Arrays de predições completas persistidos em: {predictions_path}")

    checkpoint_path = os.path.join(cfg.base_dir, "checkpoint_best_ft.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "metrics_ft": metrics_ft
    }, checkpoint_path)
    print(f"[OK] Checkpoint final persistido em: {checkpoint_path}")

    # 12. Renderização das 11 Figuras
    print("\n[INFO] Renderizando as 11 figuras técnicas a partir dos dados verificados...")
    generate_all_plots(results_dict, plots_dir, test_dataset, model=model, device=torch.device(cfg.device))


    print("\n" + "=" * 80)
    print("EXECUÇÃO DO PIPELINE CONCLUÍDA COM TOTAL SUCESSO E RIGOR METODOLÓGICO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
