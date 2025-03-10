import cv2
import torch
import os
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.onnx")  # Replace with your model file

# Define class labels
class_names = {0: "Person", 1: "Bicycle", 2: "Car"}

# Set image folder path
image_folder = "dataset/val/images"  # Replace with your image folder
image_files = sorted(os.listdir(image_folder))

# Define colors for bounding boxes
colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}  # Green for person, Blue for bike, Red for car

# Loop through images like a video
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = cv2.imread(image_path)

    if img is None:
        continue

    # Run inference
    results = model(image_path)[0]  # Get first result

    # Draw bounding boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0])  # Class index
        conf = box.conf[0].item()  # Confidence score

        if class_id in class_names:  # Only draw if class is in (0,1,2)
            label = f"{class_names[class_id]} {conf:.2f}"
            color = colors[class_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show image
    cv2.imshow("YOLO Object Detection", img)

    # Wait 30ms and break if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

    # Simulate video effect (adjust as needed)
    time.sleep(0.03)

# Cleanup
cv2.destroyAllWindows()