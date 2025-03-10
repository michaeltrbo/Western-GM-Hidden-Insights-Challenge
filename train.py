from ultralytics import YOLO
import torch

def main():
    # Initialize YOLOv8n from scratch (no pretrained weights)
    model = YOLO("yolov8n.yaml")  # Model architecture

    # Train YOLOv8n on your new dataset structure
    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,
        optimizer="Adam",
        lr0=0.01,
        save=True,
        save_period=5,
        project="runs/train",
        name="yolov8n_scratch"
    )

    print("✅ Training complete! Best model saved in 'runs/train/yolov8n_scratch/weights/best.pt'")

if __name__ == "__main__":
    # Required for Windows multiprocessing support
    torch.multiprocessing.freeze_support()
    main()