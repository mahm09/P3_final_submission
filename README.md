# Computer Vision Model Comparison: YOLOv8 vs Faster R-CNN vs MobileNet

## Project Overview
This repository contains the final implementation and trained models for a comparative study of object detection and pose estimation architectures. The project evaluates the performance difference between single-stage detectors (YOLOv8), two-stage detectors (Faster R-CNN), and lightweight architectures (MobileNet/MediaPipe).

## Included Models
The `P3_Final_Submission` folder includes the best-performing weights from our training sessions:

* **`best_yolo.pt`**: The optimal weights obtained from training **YOLOv8** (via Ultralytics). This model offers a balance of high speed and accuracy.
* **`best_frcnn.pth`**: The best checkpoint for the **Faster R-CNN** (ResNet50 backbone) model, trained using PyTorch. This model typically provides higher accuracy for small objects.
* **`best_mobilenet.pth`**: The trained weights for the **MobileNet** based architecture, optimized for efficient pose estimation and lightweight detection.

## Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies**: Ensure you have Python 3.8+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Key Scripts & Usage

### 1. Training & Inference
* **`yolotrain.py`**: Script used to train the YOLOv8 model on the dataset.
* **`fasterrcnntest.py`**: Evaluation script for the Faster R-CNN model. Loads `best_frcnn.pth` to run inference on test images.
* **`pose-mobilnet.py` / `pose-mobilnet_v2.py`**: Scripts for running the MobileNet-based pose estimation pipeline.

### 2. Visualization & Analysis
* **`generate_advanced_plots.py`**: Generates comprehensive performance graphs (Confusion Matrices, Precision-Recall curves, Loss curves) comparing the models.
* **`plot_results.py`**: Helper script to visualize specific detection results.

### Example Usage
To run the advanced plotting script:
```bash
python generate_advanced_plots.py
