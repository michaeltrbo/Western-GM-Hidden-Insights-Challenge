import cv2
import torch
import numpy as np
import os
import glob
from ultralytics import YOLO

# Paths
pt_model_path = r"Western-GM-Hidden-Insights-Challenge\runs\train\yolov8n_scratch8\weights\best.onnx"  # Path to the YOLO .pt model
image_folder = r"Western-GM-Hidden-Insights-Challenge\dataset\val\images"  # Change to your image directory
output_folder = r"Western-GM-Hidden-Insights-Challenge\Processed_Images"  # Folder to save processed images
results_file = "results.txt"

# Define class labels (adjust according to your model's labels)
class_names = {0: "1", 1: "2", 2: "3"}  # Example labels
# 1: Person
# 2: Bike
# 3: Car

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO(pt_model_path, task="detect")
print("Model loaded successfully.")

# Open results file
with open(results_file, "w") as f:
    # Check if image folder exists and contains images
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpeg")))
    if not image_files:
        print("No images found in the folder.")
    
    for img_path in image_files:
        print(f"Processing {img_path}...")
        image_file = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Skipping {image_file}, unable to read the image.")
            continue

        H, W, _ = image.shape  # Original image size
        print(f"Image shape: {H}x{W}")

        # Run inference using YOLO model
        results = model(img_path)[0]  # Run inference and get the first result
        print(f"Detections found: {len(results.boxes)}")

        # Process detections
        for box in results.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Class index
            conf = box.conf[0].item()  # Confidence score
            
            if conf > 0.5:  # Confidence threshold
                # Map class ID to expected format (if needed)
                mapped_class_id = class_names.get(class_id, "Unknown")

                # Write to results file
                f.write(f"{image_file} {mapped_class_id} {conf:.2f} {x_min} {y_min} {x_max} {y_max}\n")
                
                # Draw bounding box and label
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0) if class_id == 1 else (0, 0, 255)  # Different colors for different classes
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{mapped_class_id} {conf:.2f}"
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image while processing
        cv2.imshow("Processing Image", image)
        
        # Wait for 30ms, allow user to break with 'q' key
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        # Save processed image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

# Cleanup
cv2.destroyAllWindows()

print("Processing complete. Results saved in results.txt and output_images folder.")
