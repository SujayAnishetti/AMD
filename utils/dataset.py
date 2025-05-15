import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert('L')  # Load image as grayscale
        mask = Image.open(mask_path).convert('L')  # Load mask as grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
