## Western Automotive Challenge - Thermal Object Detection  

This project is part of the **Western Engineering Automotive Innovation Challenge**, which focuses on improving road safety for autonomous vehicles using thermal imaging. The goal is to develop a lightweight embedded system that detects road users—cars, pedestrians, and cyclists—by processing thermal camera frames.  

### **Challenge Overview**  
Ensuring road user safety is critical for autonomous systems, especially in low visibility conditions like nighttime, fog, or adverse weather. Traditional cameras struggle in these scenarios, whereas thermal cameras capture heat signatures, making them valuable for obstacle detection.  

### **Solution**  
This project implements **YOLOv8 Nano** for real-time object detection on thermal images. The system runs on a **Raspberry Pi 5**, fetching thermal image frames from a provided dataset, processing them to detect objects, and returning bounding boxes with confidence scores.  

### **Deployment**  
- The embedded system requests thermal images from a dataset.  
- YOLOv8 Nano processes the frames and identifies road users.  
- The system sends detection results back with bounding boxes and confidence levels.  

This solution enhances autonomous vehicle perception by enabling reliable detection of obstacles in challenging environments.


## Getting Started

### Dependencies

- Python 3.9.13 or higher
- PyTorch (newest)
- OpenCV
- Other dependencies as listed in requirements.txt

### Installing

Clone the repository:

~~~
git clone https://github.com/michaeltrbo/Western-GM-Hidden-Insights-Challenge.git
~~~

Get the requirements:

~~~
pip install -r requirements.txt
~~~
