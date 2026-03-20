import torch
import torchvision.transforms as transforms
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