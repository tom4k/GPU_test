import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import ButterflyDataset, get_transforms

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def predict():
    parser = argparse.ArgumentParser(description="Generate Predictions for Image Dataset")
    parser.add_argument('--csv_path', type=str, default='data/Testing_set.csv', help='Path to test CSV file')
    parser.add_argument('--img_dir', type=str, default='data/test', help='Path to test images directory')
    parser.add_argument('--mapping_path', type=str, default='label_mapping.json', help='Path to label mapping JSON file')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Path to save output predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load label mapping
    if not os.path.exists(args.mapping_path):
        raise FileNotFoundError(f"Label mapping file not found at {args.mapping_path}. Please train the model first.")

    with open(args.mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # idx2label has integer keys stored as strings in JSON
    idx2label = {int(k): v for k, v in mapping_data['idx2label'].items()}
    num_classes = len(idx2label)

    # Load test DataFrame
    test_df = pd.read_csv(args.csv_path)
    print(f"Test samples to predict: {len(test_df)}")

    _, val_transform = get_transforms()
    test_dataset = ButterflyDataset(test_df, args.img_dir, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model architecture and load weights
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}. Please train the model first.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    filenames_list = []
    predictions_list = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds_cpu = preds.cpu().numpy()
            for fn, pred_idx in zip(filenames, preds_cpu):
                filenames_list.append(fn)
                predictions_list.append(idx2label[pred_idx])

    submission_df = pd.DataFrame({
        'filename': filenames_list,
        'label': predictions_list
    })

    submission_df.to_csv(args.output_path, index=False)
    print(f"\nPredictions saved successfully to {args.output_path}!")
    print("Sample predictions preview:")
    print(submission_df.head(10))

if __name__ == '__main__':
    predict()
