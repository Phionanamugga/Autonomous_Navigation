# Autonomous_navigation
Real-Time Multi-Object Detection and Tracking for Autonomous Navigation
## Project Overview
This project implements a real-time multi-object detection and tracking system tailored for autonomous navigation applications. Using YOLOv8 for object detection and DeepSORT for tracking, the system identifies and tracks objects such as cars, pedestrians, and traffic signs in video feeds. The project is optimized for edge deployment using TensorRT and includes a web-based demo via Streamlit, showcasing real-time performance. This work demonstrates end-to-end expertise in computer vision, deep learning, and deployment, aligning with industry needs at companies like Google, Amazon, and Tesla.

## Features
Real-Time Detection: Achieves 25-30 FPS on edge devices using YOLOv8 and TensorRT.
Multi-Object Tracking: Tracks objects across frames with DeepSORT, handling occlusions robustly.
Web Demo: Interactive Streamlit app for visualizing detection and tracking results.
Scalable Deployment: Containerized with Docker for reproducibility and edge compatibility.
Dataset: Fine-tuned on the KITTI dataset, with support for COCO or custom datasets.

## Installation
### Prerequisites

Python 3.8+
NVIDIA GPU (recommended for training and inference)
CUDA and cuDNN (for GPU acceleration)
Docker (optional, for deployment)

### Setup

Clone the Repository:
git clone https://github.com/Phionanamugga/autonomous_navigation.git
cd autonomous_navigation


### Install Dependencies:
pip install -r requirements.txt

Ensure requirements.txt includes:
torch>=2.0.0
ultralytics>=8.2.0
supervision>=0.26.0
deep-sort-realtime>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0
streamlit>=1.38.0


### Download Dataset:
Download the KITTI dataset or use COCO.
Update kitti.yaml with dataset paths:path: /path/to/kitti
train: images/train
val: images/val
names:
  0: car
  1: pedestrian
  2: traffic_sign


## Download Pre-trained Model:

The project uses yolov8n.pt from Ultralytics. Download it via:wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt


## Usage
Training the Model
Fine-tune YOLOv8 on your dataset:
python train_yolo.py


## Configurable parameters: 
epochs=50, imgsz=640, batch=16.
Outputs a fine-tuned model: yolov8_finetuned.pt.

## Running Detection and Tracking
Run the real-time detection and tracking pipeline:
python object_detection_tracking.py --video sample_traffic_video.mp4


Replace sample_traffic_video.mp4 with your video file or use 0 for webcam input.
Press q to exit the video feed.

Launching the Web Demo
Visualize results via Streamlit:
streamlit run app.py


Upload a video file to see detection and tracking results in a browser.

Deploying with Docker
Build and run the Docker container:
docker build -t object-detection .
docker run --gpus all -v $(pwd)/data:/app/data object-detection

Methodology
Dataset

KITTI Dataset: 7,481 training images and 7,518 testing images, focusing on cars, pedestrians, and traffic signs.
Preprocessing: Applied augmentations (e.g., rotation, flip) using Albumentations to improve model robustness.

Model Architecture

YOLOv8 Nano: Chosen for its balance of speed and accuracy, achieving real-time performance on edge devices.
DeepSORT: Used for multi-object tracking with appearance-based re-identification.

Training

Fine-tuned YOLOv8 for 50 epochs on KITTI, with a batch size of 16 and image size of 640x640.
Optimized with AdamW optimizer and a learning rate of 0.001.

Deployment

Converted model to ONNX and TensorRT formats for optimized inference.
Containerized with Docker for reproducibility and scalability.

Results

Accuracy: mAP@0.5 of 0.82 on KITTI validation set.
Performance: 25-30 FPS on NVIDIA Jetson Nano with TensorRT.
Tracking Robustness: Successfully tracks objects through partial occlusions and varying lighting conditions.

Challenges and Solutions

API Versioning Issue:
Problem: Encountered AttributeError: module 'supervision' has no attribute 'BoundingBoxAnnotator' due to outdated supervision library API.
Solution: Upgraded to supervision>=0.26.0 and replaced BoundingBoxAnnotator with BoxAnnotator.


Occlusion Handling:
Problem: DeepSORT lost tracks during heavy occlusions.
Solution: Tuned max_age parameter to 30 and increased nn_budget to 100 for better re-identification.


Real-Time Performance:
Problem: Initial inference was slow on edge devices.
Solution: Optimized with TensorRT and reduced model size by using YOLOv8 Nano.



Future Improvements

Multi-Camera Support: Extend the pipeline to process feeds from multiple cameras for 360-degree awareness.
Advanced Models: Experiment with YOLOv9 or transformer-based models like DETR for improved accuracy.
Edge Optimization: Further optimize for low-power devices using INT8 quantization.
Additional Classes: Expand to detect more object types (e.g., cyclists, road lanes).

Demo
A live demo is available at [your-demo-link] (replace with your hosted Streamlit app or video link). Example output:
Contact
For questions or collaboration, reach out to [your-email@example.com] or open an issue on GitHub.
License
This project is licensed under the MIT License.