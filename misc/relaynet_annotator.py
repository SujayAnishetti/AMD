import os
import numpy as np
import cv2
from label_studio_sdk import Client
from relaynet_model import load_model, predict_image  # Assume you'll wrap this part
from utils import convert_to_labelstudio_mask_format  # Helper function you’ll write

# CONFIG
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA1Mzk5MTAxMywiaWF0IjoxNzQ2NzkxMDEzLCJqdGkiOiI0ZjEzMTBmZDE4Mjk0OWQyOWQ0NzJjZTNiN2U5NGU4MCIsInVzZXJfaWQiOjF9.-I7of1DmAesIOZ7RBnlhPCb-8uXMxkymfpzNgc1oECo'
PROJECT_ID = 1  # Change to match your project
IMAGE_DIR = 'images_to_annotate'

# Connect to Label Studio
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
project = ls.get_project(PROJECT_ID)

# Load ReLayNet model
model = load_model()

# Get all tasks
tasks = project.get_tasks()

for task in tasks:
    image_url = task['data']['image']
    image_path = download_image(image_url, IMAGE_DIR)  # You’ll write this
    oct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Predict segmentation
    pred_mask = predict_image(model, oct_image)  # Output: np.array with layer indices

    # Convert to RLE or PNG for Label Studio
    annotation_result = convert_to_labelstudio_mask_format(pred_mask)

    # Send pre-annotation back to LS
    project.create_annotation(
        task_id=task['id'],
        annotation={
            "result": [annotation_result]
        }
    )
