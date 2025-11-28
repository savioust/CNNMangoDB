from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def load_dataset(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        print("📁 Detecção: pastas de treino, validação e teste já existem.")
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        classes = train_dataset.classes
    else:
        print("⚙️ Pastas separadas não detectadas — dividindo automaticamente o conjunto.")
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        classes = dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes
