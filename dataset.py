import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ButterflyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images from CSV annotations.
    """
    def __init__(self, df, img_dir, transform=None, label2idx=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label2idx = label2idx
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            label_str = row['label']
            label_idx = self.label2idx[label_str]
            return image, torch.tensor(label_idx, dtype=torch.long)

def get_transforms(img_size=224):
    """
    Data augmentation for training and normalization for validation/test.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def build_label_mapping(df, save_path='label_mapping.json'):
    """
    Maps string labels to unique integer indices and saves mapping to JSON.
    """
    unique_labels = sorted(df['label'].unique().tolist())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for idx, label in enumerate(unique_labels)}
    
    mapping_data = {
        'label2idx': label2idx,
        'idx2label': idx2label
    }
    
    with open(save_path, 'w') as f:
        json.dump(mapping_data, f, indent=4)
        
    return label2idx, idx2label
