import os
import json
from PIL import Image
import shutil

# === Paths ===
image_dir = "images"
json_dir = "annotations"
output_dir = "dataset"
image_out = os.path.join(output_dir, "images", "train")
label_out = os.path.join(output_dir, "labels", "train")
os.makedirs(image_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)

# === Label mapping ===
label_map = {"RPE": 0, "PR2": 1}

# === Convert each image/json pair ===
for filename in os.listdir(json_dir):
    if not filename.endswith(".json"):
        continue

    stem = os.path.splitext(filename)[0]
    json_path = os.path.join(json_dir, filename)
    img_path = os.path.join(image_dir, stem + ".PNG")

    if not os.path.exists(img_path):
        print(f"Image not found for: {stem}")
        continue

    # Copy image to dataset folder
    shutil.copy2(img_path, os.path.join(image_out, f"{stem}.PNG"))

    # Open image to get dimensions
    img = Image.open(img_path)
    W, H = img.size

    # Read annotations
    with open(json_path, "r") as f:
        annotations = json.load(f)

    label_file = os.path.join(label_out, f"{stem}.txt")
    with open(label_file, "w") as f_out:
        for obj in annotations:
            label = obj.get("label")
            points = obj.get("points", [])
            if label not in label_map or len(points) < 3:
                continue  # YOLO requires polygons with at least 3 points

            class_id = label_map[label]
            flat = []
            for x, y in points:
                flat.extend([x / W, y / H])
            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in flat) + "\n"
            f_out.write(line)

print("✅ Conversion complete!")
