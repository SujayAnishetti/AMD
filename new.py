import cv2
import numpy as np
import os

def extract_lines(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    # --- Detect RPE line (yellow-green) ---
    lower_rpe = np.array([190, 190, 0])
    upper_rpe = np.array([255, 255, 50])
    rpe_mask = cv2.inRange(img, lower_rpe, upper_rpe)

    # --- Detect PR2 line (white-gray with green bias) ---
    lower_pr2 = np.array([235, 250, 235])
    upper_pr2 = np.array([245, 255, 245])  # 260 is invalid, max is 255
    pr2_mask = cv2.inRange(img, lower_pr2, upper_pr2)

    # --- Combine masks safely ---
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask[rpe_mask > 0] = 1
    mask[pr2_mask > 0] = 2
    mask[(rpe_mask > 0) & (pr2_mask > 0)] = 2  # Optional: PR2 priority

    # Save debug masks
    cv2.imwrite("debug_rpe_mask.png", rpe_mask)
    cv2.imwrite("debug_pr2_mask.png", pr2_mask)

    # Save combined mask
    mask_filename = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
    cv2.imwrite(os.path.join(save_path, mask_filename), mask)

    print(f"✅ Saved mask to {os.path.join(save_path, mask_filename)}")

    # Optional visualization mask (for checking)
    vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    vis_mask[mask == 1] = (255, 0, 0)   # RPE as Blue
    vis_mask[mask == 2] = (0, 255, 0)   # PR2 as Green

    cv2.imwrite("vis_mask.png", vis_mask)


if __name__ == "__main__":
    image_path = "11.PNG"
    save_dir = "./masks"
    os.makedirs(save_dir, exist_ok=True)
    extract_lines(image_path, save_dir)
