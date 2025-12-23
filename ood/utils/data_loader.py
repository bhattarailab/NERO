import torch
import torchvision.transforms as transforms
from torchvision import transforms , datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image to 224x224 as required by VGG16
        transforms.ToTensor(),          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

def get_loader(args):
    
    train_dataset = datasets.ImageFolder(root=args.id_path_train, transform=transform)
    id_dataset = datasets.ImageFolder(root=args.id_path_valid, transform=transform)
    ood_dataset = datasets.ImageFolder(root=args.ood_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return (train_loader, id_loader, ood_loader)