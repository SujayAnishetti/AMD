import cv2
import numpy as np
import os
import json

def convert_mask_to_polygon(mask_image_path):
    # Load the mask image (grayscale)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"❌ Failed to read mask: {mask_image_path}")
        return []

    # Find contours (edges of the mask regions)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    
    for contour in contours:
        # Approximate the contour to a polygon (simplify the shape)
        epsilon = 0.04 * cv2.arcLength(contour, True)  # You can adjust this value to control the precision
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Convert the polygon points into a list of tuples with regular Python ints
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx_polygon]
        polygons.append(polygon)
    
    return polygons

def save_polygons_as_json(polygons, output_json_path):
    # Create a dictionary for the Label Studio annotation format
    annotations = []
    
    for polygon in polygons:
        annotations.append({
            "label": "RPE",  # Replace with "PR2" if it's for the PR2 line
            "points": polygon,
            "type": "polygon"
        })

    # Save as a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)

    print(f"✅ Saved polygons to {output_json_path}")

def process_masks_in_directory(mask_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for mask_filename in os.listdir(mask_dir):
        mask_image_path = os.path.join(mask_dir, mask_filename)

        if not mask_filename.endswith("_mask.png"):
            continue  # Skip files that are not mask images

        print(f"Processing mask: {mask_filename}")

        # Convert mask to polygons
        polygons = convert_mask_to_polygon(mask_image_path)

        if polygons:
            # Define the output JSON path
            output_json_path = os.path.join(output_dir, f"{os.path.splitext(mask_filename)[0]}_annotations.json")

            # Save the polygons in Label Studio's annotation format
            save_polygons_as_json(polygons, output_json_path)

# === Run the processing ===
if __name__ == "__main__":
    mask_directory = "./masks"  # Replace with your directory containing mask images
    output_directory = "./annotations"  # Directory to save the JSON annotations

    process_masks_in_directory(mask_directory, output_directory)