# Computer Vision Model Comparison: YOLOv8 vs Faster R-CNN vs MobileNet

## Project Overview
This repository contains the final implementation and trained models for a comparative study of object detection and pose estimation architectures. The project evaluates the performance difference between single-stage detectors (YOLOv8) and two-stage detectors (Faster R-CNN with a MobileNetV2 backbone (itself with a MediaPipe wrist cropping system)). During the project, another repository was used. This was cleaned up for submission, with some testing files, the dataset, and various other programs not making the final cut. As a result, it is possible that the programs do not currently have full functionality, as they did in our working project folder. These programs are merely intended to showcase the implementation.

The repository also contains a folder with an example implementation of the deployment method described in the report, with it's own requirements.txt file, a sample image to try, and a sample result recieved from the API server, among other things. The Docker server host must be running for this to work.

## Included Models
The `P3_Final_Submission` folder includes the best-performing weights from our training sessions:

* **`best_yolo.pt`**: The optimal weights obtained from training **YOLOv8** (via Ultralytics). This model offers a balance of high speed and accuracy.
* **`best_frcnn.pth`**: The best checkpoint for the **Faster R-CNN** (MobileNetV2 backbone) model, trained using PyTorch. 
* **`best_mobilenet.pth`**: The trained weights for the **MobileNet** based architecture, (theoretically) optimized for efficient pose estimation and detection.

## Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies**: Ensure you have Python 3.8+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Key Scripts

### 1. Training & Inference
* **`yolotrain.py`**: Script used to train the YOLOv8 model on the dataset.
* **`fasterrcnntest.py`**: Evaluation script for the Faster R-CNN model. Loads `best_frcnn.pth` to run inference on test images.
* **`pose-mobilnet.py` / `pose-mobilnet_v2.py`**: Scripts for running the MobileNet-based pose estimation pipeline.

### 2. Visualization & Analysis
* **`plot_results.py`**: Example helper script to visualize specific detection results.

### Example Usage
To run the advanced plotting script:
```bash
python generate_advanced_plots.py
