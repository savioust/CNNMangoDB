#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — versão completa com:
- treino do zero (weights=None)
- early stopping
- ReduceLROnPlateau
- salvamento automático do melhor modelo
- scheduler e seed reprodutível
- logging de histórico em JSON e plots (loss / val_f1)
- compatibilidade com dataset_loader atualizado (RandomResizedCrop / CenterCrop / Normalize)
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from dataset_loader import load_dataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

(Path("models")).mkdir(parents=True, exist_ok=True)
(Path("figures")).mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() - 1 if os.cpu_count() is not None else 0) or 0
PIN_MEMORY = True if torch.cuda.is_available() else False
SEED = 42

PATIENCE_ES = 8               
PATIENCE_SCHED = 4            
FACTOR_SCHED = 0.5            
MIN_LR = 1e-7
CLIP_GRAD_NORM = 5.0         

(Path("models")).mkdir(parents=True, exist_ok=True)
(Path("figures")).mkdir(parents=True, exist_ok=True)

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False

print(f"Device: {DEVICE}")
print(f"Num workers: {NUM_WORKERS} | Pin memory: {PIN_MEMORY}")
print("Transforms used (from dataset_loader): RandomResizedCrop(224) for training; CenterCrop(224) for val/test; Normalize(ImageNet)")

def get_model(model_name: str, num_classes: int = NUM_CLASSES):
    """Retorna o modelo instanciado. Nesta versão treinamos do zero (weights=None)."""
    if model_name == "AlexNet":
        model = models.alexnet(weights=None, num_classes=num_classes)
        return model

    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None, num_classes=num_classes)
        return model

    else:
        raise ValueError(f"Modelo {model_name} não implementado")

class EarlyStopping:
    """Simples early stopping que monitora a métrica (val_f1) e interrompe se não melhorar."""
    def __init__(self, patience=PATIENCE_ES, mode='max'):
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.mode = mode

    def step(self, value):
        if self.best is None:
            self.best = value
            self.num_bad_epochs = 0
            return False

        improved = (value > self.best) if self.mode == 'max' else (value < self.best)
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                return True
            return False


def save_history(history: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)


def plot_history(history: dict, model_name: str):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], label='val_f1')
    plt.plot(epochs, history['val_accuracy'], label='val_acc')
    plt.title(f'{model_name} Metrics')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    out = Path('figures') / f'{model_name}_history.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved history plot to: {out}")

def evaluate(model, data_loader, device=DEVICE, return_all=False, show_confusion=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if show_confusion:
        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        cm = confusion_matrix(all_labels, all_preds)
        print("Matriz de Confusão:")
        print(cm)

    if return_all:
        return acc, precision, recall, f1
    return acc


def train(model, model_name, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR_SCHED,
                                                     patience=PATIENCE_SCHED, min_lr=MIN_LR)

    early_stopper = EarlyStopping(patience=PATIENCE_ES, mode='max')

    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'best_val_f1': 0.0,
        'transforms': {
            'train': 'RandomResizedCrop(224) + Normalize(ImageNet)',
            'val/test': 'Resize(256) + CenterCrop(224) + Normalize(ImageNet)'
        }
    }

    best_val_f1 = 0.0
    best_model_path = Path('models') / f'best_{model_name}.pt'

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"{model_name} | Epoch {epoch}/{epochs}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if CLIP_GRAD_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            current_loss = running_loss / ((progress_bar.n + 1) * train_loader.batch_size)
            progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)

        acc, precision, recall, f1 = evaluate(model, val_loader, return_all=True, show_confusion=False)

        scheduler.step(epoch_loss)

        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(epoch_loss)
        history['val_accuracy'].append(acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)

        print(f"\n{model_name} - Epoch {epoch}/{epochs}")
        print(
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        print(f"Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            history['best_val_f1'] = best_val_f1
            print(f"Novo melhor modelo salvo (val_f1 = {best_val_f1:.4f}) -> {best_model_path}")

        stop = early_stopper.step(f1)
        if stop:
            print(f"Parando cedo após {epoch} épocas (sem melhoria por {PATIENCE_ES} épocas).")
            break

        history_file = Path(f"history_{model_name}.json")
        save_history(history, history_file)

    plot_history(history, model_name)
    save_history(history, Path(f"history_{model_name}.json"))

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model

if __name__ == '__main__':
    models_to_train = ["AlexNet", "EfficientNetB0"]

    train_loader, val_loader, test_loader, classes = load_dataset(DATA_DIR, batch_size=BATCH_SIZE)
    print("Classes detectadas:", classes)

    for model_name in models_to_train:
        print(f"Treinando modelo: {model_name}")

        model = get_model(model_name)

        model = train(
            model,
            model_name,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=DEVICE
        )

        print(f"Avaliação final do modelo {model_name} no conjunto de teste:")
        acc, precision, recall, f1 = evaluate(model, test_loader, return_all=True)
        print(f"Teste -> Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        save_path = Path("models") / f"{model_name}_final.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Pesos finais salvos em: {save_path}")

    print("Treinamento concluído.")