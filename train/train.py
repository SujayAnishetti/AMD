import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet_plus_plus import DenseUNetPlusPlus
from utils.dataset import OCTDataset

# Hyperparameters
batch_size = 8
epochs = 20
learning_rate = 1e-4

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize the image
])

# Dataset and DataLoader setup
train_dataset = OCTDataset(image_dir="data/train_images", mask_dir="data/train_masks", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = DenseUNetPlusPlus(encoder_name="densenet121", encoder_weights="imagenet", in_channels=1, classes=2).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # You can also use DiceLoss from smp.losses
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, supervision_outputs = model(inputs)

        # Compute the loss for the final output
        loss = criterion(outputs, masks)

        # Add loss for deep supervision outputs
        for sup_output in supervision_outputs:
            loss += criterion(sup_output, masks) * 0.2  # Weight deep supervision losses

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
