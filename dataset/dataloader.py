import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform

def get_dataloaders(config):
    """Initializes and returns train and validation dataloaders for the chosen dataset."""
    train_transform, test_transform = get_transforms(image_size=224)
    os.makedirs(config.data_dir, exist_ok=True)

    if config.dataset == "fgvc_aircraft":
        train_dataset = datasets.FGVCAircraft(
            root=config.data_dir, split='trainval', download=True, transform=train_transform
        )
        test_dataset = datasets.FGVCAircraft(
            root=config.data_dir, split='test', download=True, transform=test_transform
        )
        class_names = train_dataset.classes

    elif config.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=config.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=config.data_dir, train=False, download=True, transform=test_transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )

    return train_loader, test_loader, class_names