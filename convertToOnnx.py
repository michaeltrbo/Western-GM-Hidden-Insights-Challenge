from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/yolov8n_scratch8/weights/best.pt")

# Export to ONNX format (this will produce best.onnx in the current directory)
model.export(format="onnx")
