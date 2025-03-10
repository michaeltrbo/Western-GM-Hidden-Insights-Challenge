import os
import cv2

# Define paths
labels_txt = "thermal_dataset/8_bit_dataset/8_bit_dataset/train_labels_8_bit.txt"
images_folder = "thermal_dataset/8_bit_dataset/8_bit_dataset/dataset/train/images"
output_labels_folder = "thermal_dataset/8_bit_dataset/8_bit_dataset/dataset/train/labels"

os.makedirs(output_labels_folder, exist_ok=True)

with open(labels_txt, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()

    # ✅ Ensure the line has exactly 6 parts
    if len(parts) != 6:
        print(f"Skipping malformed line: {line.strip()} (found {len(parts)} parts)")
        continue

    image_name, class_id, x_min, y_min, x_max, y_max = parts
    class_id = int(class_id)

    # ✅ Convert class ID by decrementing it by 1
    class_id -= 1

    # ✅ Ensure class label is valid after decrementing
    if class_id not in {0, 1, 2}:
        print(f"Skipping invalid class label ({class_id}) in: {line.strip()}")
        continue

    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    image_path = os.path.join(images_folder, image_name)
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found!")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Cannot open image {image_path}")
        continue

    height, width = img.shape[:2]

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height

    label_file_path = os.path.join(output_labels_folder, image_name.replace(".jpeg", ".txt"))
    with open(label_file_path, "a") as label_file:
        label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("✅ Conversion complete (Class labels decremented by 1).")