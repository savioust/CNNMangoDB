from dataset_loader import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

def denormalize(img_tensor):
    """
    img_tensor: Tensor (C, H, W)
    retorna: numpy array (H, W, C) desnormalizado
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = img_tensor.numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)

    return np.transpose(img, (1, 2, 0))

data_dir = Path("data")

print("📂 Caminho absoluto:", data_dir.resolve())
print("📁 Existe?", data_dir.exists())

train_loader, val_loader, test_loader, classes = load_dataset(data_dir, batch_size=32)

print("📂 Classes detectadas:", classes)
print(f"🧩 Total de classes: {len(classes)}")

if train_loader:
    print(f"🟩 Treino: {len(train_loader.dataset)} imagens")
if val_loader:
    print(f"🟦 Validação: {len(val_loader.dataset)} imagens")
if test_loader:
    print(f"🟥 Teste: {len(test_loader.dataset)} imagens")

if train_loader:
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    plt.figure(figsize=(12, 6))

    for i in range(8):
        img = denormalize(images[i])
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')

    plt.suptitle("Amostras do conjunto de treino (transformadas + desnormalizadas)")
    plt.show()
