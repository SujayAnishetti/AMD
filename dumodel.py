import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
import cv2
import os
import numpy as np

# Custom Dataset
class LineDetectionDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations
        with open(json_path, 'r') as file:
            self.annotations = json.load(file)

        self.image_files = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Get annotation and create a mask
        annotation = self.annotations[self.image_files[idx]]
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for line in annotation['lines']:
            cv2.line(mask, tuple(line[0]), tuple(line[1]), color=1, thickness=2)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# DU-Net++ Model
class DUNetPP(nn.Module):
    def __init__(self):
        super(DUNetPP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training Function
def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example Usage
if __name__ == '__main__':
    image_dir = 'images//11.PNG'
    json_path = 'annotations//debug_rpe_mask_annotations.json'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    dataset = LineDetectionDataset(image_dir, json_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DUNetPP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, optimizer, criterion, num_epochs=10)
