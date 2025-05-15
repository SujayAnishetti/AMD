import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import cv2
import numpy as np

class LineDataset(Dataset):
    def __init__(self, image_paths, json_paths, transform=None):
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        with open(self.json_paths[idx], 'r') as f:
            annotations = json.load(f)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for annotation in annotations:
            if annotation["label"] == "RPE" and annotation["type"] == "polygon":
                points = np.array(annotation["points"], np.int32)
                if len(points) == 2:
                    cv2.line(mask, tuple(points[0]), tuple(points[1]), 255, 3)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask.unsqueeze(0) / 255.0  # Normalize mask to [0, 1]
