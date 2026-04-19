import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import medmnist
from medmnist import INFO

class MedMNISTDataset(Dataset):
    """Custom Dataset class wrapping MedMNIST for standard PyTorch usage."""
    def __init__(self, data_flag='bloodmnist', split='train', download=True, transform=None):
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(split=split, transform=transform, download=download)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_medmnist_dataloaders(data_flag='bloodmnist', batch_size=128):
    info = INFO[data_flag]

    # BloodMNIST is 28x28 RGB images. We resize to 32x32 to play nicely with standard ResNet padding.
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MedMNISTDataset(data_flag=data_flag, split='train', transform=data_transform, download=True)
    val_dataset = MedMNISTDataset(data_flag=data_flag, split='val', transform=data_transform, download=True)
    test_dataset = MedMNISTDataset(data_flag=data_flag, split='test', transform=data_transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, info['n_channels'], len(info['label'])

def get_chestxray_dataloaders(data_dir, batch_size=128):
    """
    Loads Chest X-Ray images using ImageFolder.
    Assumes standard Kaggle directory structure:
        data_dir/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=eval_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=eval_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, 3, 2