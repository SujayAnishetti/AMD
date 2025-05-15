import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def save_model(model, filepath):
    """
    Save the trained model to the given file path.
    """
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(model, filepath):
    """
    Load the trained model from the given file path.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"✅ Model loaded from {filepath}")
    return model

def visualize_predictions(image, mask, prediction):
    """
    Overlay the predicted mask on the original image.
    """
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # Single channel image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    mask = mask.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()
    
    # Convert to binary masks
    mask = (mask > 0.5).astype(np.uint8) * 255
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    
    # Overlay masks in green (actual) and red (predicted)
    overlay = image.copy()
    overlay[mask == 255] = [0, 255, 0]   # Green for ground truth
    overlay[prediction == 255] = [0, 0, 255]  # Red for prediction

    # Display
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.title("Overlay (Green: GT, Red: Prediction)")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Prediction Mask")

    plt.show()
