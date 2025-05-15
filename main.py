import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from dataset import LineDataset
from model import DU_NetPlusPlus
from utils import visualize_predictions, save_model, load_model

def train_model(model, dataloader, optimizer, criterion, epochs=20, device="cuda"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    save_model(model, "models/model.pth")  

def evaluate_model(model, dataloader, device="cuda"):
    model.to(device)  
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            print(f"Prediction shape: {predicted.shape}")
            visualize_predictions(images[0].cpu().permute(1, 2, 0).numpy(), masks[0], predicted[0])

# Configuration
data_dir = "data/"
image_paths = [os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images"))]
json_paths = [os.path.join(data_dir, "annotations", f) for f in os.listdir(os.path.join(data_dir, "annotations"))]

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

dataset = LineDataset(image_paths, json_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = DU_NetPlusPlus()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Segmentation

# Train and Evaluate
train_model(model, dataloader, optimizer, criterion, epochs=1000)

# Load and Test Model on New Images
model = DU_NetPlusPlus()
model = load_model(model, "models/model.pth")

evaluate_model(model, dataloader)
