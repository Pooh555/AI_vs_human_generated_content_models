import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Image transformation for all inputs
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_dataloaders(data_dir='data', batch_size=32):
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
