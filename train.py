import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import ButterflyDataset, get_transforms, build_label_mapping

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using NVIDIA CUDA GPU acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS GPU acceleration.")
    else:
        device = torch.device('cpu')
        print("Using CPU execution.")
    return device

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, 'data', 'Training_set.csv')
    default_img_dir = os.path.join(base_dir, 'data', 'train')
    default_save_path = os.path.join(base_dir, 'best_model.pth')

    parser = argparse.ArgumentParser(description="Train Image Classification Model (ResNet-18)")
    parser.add_argument('--csv_path', type=str, default=default_csv, help='Path to training CSV file')
    parser.add_argument('--img_dir', type=str, default=default_img_dir, help='Path to training images directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloaders')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default=default_save_path, help='Path to save best model checkpoint')
    args = parser.parse_args()

    device = get_device()

    # Load dataset CSV
    df = pd.read_csv(args.csv_path)
    label2idx, idx2label = build_label_mapping(df)
    num_classes = len(label2idx)
    print(f"Dataset loaded: {len(df)} samples, {num_classes} unique classes.")

    # Stratified Train/Val Split (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_transform, val_transform = get_transforms()

    train_dataset = ButterflyDataset(train_df, args.img_dir, transform=train_transform, label2idx=label2idx)
    val_dataset = ButterflyDataset(val_df, args.img_dir, transform=val_transform, label2idx=label2idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Pretrained ResNet-18 Model
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Replace final classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0

    print("\nStarting Training Pipeline...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f" -> Saved new best model checkpoint (Val Acc: {best_val_acc*100:.2f}%) to {args.save_path}")

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    main()
