import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


def initialize_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument('--dataset', type=str, default='gastrovision', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--val_iter', type=int, default=5, help='Validation interval')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'deit'], help='Model architecture')
    parser.add_argument('--torch_path', type=str,  help='Path for Torch models')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--train_dir', type=str, help='Training data directory')
    parser.add_argument('--test_dir', type=str, help='Testing data directory')
    parser.add_argument('--seed', type=int, default=42, help='Seed value')

    return parser.parse_args()


def get_data_loaders(train_dir, test_dir, batch_size):
    k_transform_train = transforms.Compose(
                    [
                        transforms.Resize(size=(224, 224)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomInvert(0.3),
                        transforms.RandomEqualize(0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #kvasir norm
                    ]
                ) 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=k_transform_train)
    val_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    return train_loader, val_loader

def get_model(model_name, num_classes, torch_path):
    os.environ['TORCH_HOME'] = torch_path
    if model_name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'deit':
        os.environ["HUGGINGFACE_HUB_CACHE"] = torch_path
        from timm import create_model
        model=create_model('deit_small_patch16_224.fb_in1k', pretrained=True,num_classes=num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, val_iter, checkpoint_path):
    model.to(device)
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        
        if (epoch+1) % val_iter == 0:
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                print(f"Validation Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model, os.path.join(checkpoint_path, f"{args.dataset}.pt"))
        # scheduler.step(epoch-1) 

if __name__ == "__main__":
    args = parse_args()
    initialize_seed(args.seed)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(args.train_dir, args.test_dir, args.batch_size)
    model = get_model(args.model_name, args.num_classes, args.torch_path)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,eta_min=1e-7)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.val_iter, checkpoint_path)