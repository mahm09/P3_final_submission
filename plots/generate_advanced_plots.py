# ==============================================================================
# DISCLAIMER:
# The boilerplate for this script was written with the help of AI.
# It was reviewed, tested, and modified by the project group to suit 
# project-specific data structures and requirements.
# ==============================================================================

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= CONFIGURATION =================
# Checks for CUDA-enabled GPU availability for accelerated inference; defaults to CPU otherwise.
# Defines file paths for the trained model checkpoint and validation dataset.
# Note: 'MODEL_PATH' must point to the valid location of the saved state dictionary.
MODEL_PATH = "frcnn_output/best_model.pth"
IMG_DIR = "billeddata/val/images"

# Automated detection of annotation format.
# The logic prioritizes the existence of a 'labels' directory (indicating YOLO format)
# and falls back to an 'annotations' directory (indicating Pascal VOC/XML format).
if os.path.exists("billeddata/val/labels"):
    LABEL_DIR = "billeddata/val/labels" # YOLO format
    LABEL_TYPE = "txt"
else:
    LABEL_DIR = "billeddata/val/annotations" # XML format
    LABEL_TYPE = "xml"

CSV_PATH = "frcnn_output/training_losses.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IoU Threshold for Evaluation: Defines the minimum Intersection over Union required
# to classify a prediction as a True Positive (TP) during metric calculation.
IOU_THRESHOLD = 0.5 
# ===============================================

class GunDataset(torch.utils.data.Dataset):
    """
    Custom Dataset Loader extending torch.utils.data.Dataset.
    
    [BOILERPLATE NOTE]: This class implements the standard PyTorch Dataset protocol.
    It is responsible for traversing the directory structure, loading images, and 
    parsing label files (XML or TXT) into the canonical coordinate format required 
    by torchvision models: [xmin, ymin, xmax, ymax].
    """
    def __init__(self, img_dir, label_dir, label_type):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_type = label_type
        # Generates an index of all JPEG images to support random access via __getitem__.
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"--- DATASET: Found {len(self.imgs)} images. Using {label_type.upper()} labels from {label_dir} ---")

    def __getitem__(self, idx):
        # Loads image via PIL and converts to RGB to ensure 3-channel consistency.
        # Dimensions are captured to facilitate un-normalization of YOLO coordinates.
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img_tensor = transforms.functional.to_tensor(img)

        boxes = []
        
        # [BOILERPLATE NOTE]: XML parsing utilizes the standard Python ElementTree library.
        # This block extracts bounding box coordinates from <bndbox> tags found in Pascal VOC annotations.
        if self.label_type == "xml":
            xml_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.xml'))
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    # Extracts raw pixel coordinates directly.
                    boxes.append([
                        float(bbox.find("xmin").text), float(bbox.find("ymin").text),
                        float(bbox.find("xmax").text), float(bbox.find("ymax").text)
                    ])
                    
        # [BOILERPLATE NOTE]: Implementation of the standard geometric conversion formula
        # to transform normalized YOLO coordinates (center_x, center_y, width, height) 
        # into absolute boundary coordinates (x_min, y_min, x_max, y_max).
        elif self.label_type == "txt":
            txt_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    for line in f.readlines():
                        parts = list(map(float, line.strip().split()))
                        # YOLO format: class_id, center_x, center_y, width, height (normalized 0-1)
                        cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                        
                        # Conversion logic to absolute pixels
                        xmin = (cx - w/2) * width
                        ymin = (cy - h/2) * height
                        xmax = (cx + w/2) * width
                        ymax = (cy + h/2) * height
                        boxes.append([xmin, ymin, xmax, ymax])

        # Converts list to PyTorch Tensor. Ensures a shape of (0, 4) is returned 
        # if no objects are present, preventing runtime errors in the model.
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes_tensor) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            
        target = {"boxes": boxes_tensor}
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

def get_model():
    """
    Model Architecture Construction.
    
    [BOILERPLATE NOTE]: This function reconstructs the Faster R-CNN architecture 
    using torchvision primitives. Replacing the backbone and anchor generator is a 
    standard design pattern for creating lightweight custom object detectors.
    """
    # 1. Instantiates MobileNetV2 to serve as the feature extraction backbone.
    model = models.mobilenet_v2(weights=None)
    
    # 2. Replaces the classification head with an Identity layer, retaining only 
    # the feature maps for downstream tasks.
    model.classifier = nn.Identity() 
    backbone = model.features
    backbone.out_channels = 1280 # Specifies output channel dimension for MobileNetV2.
    
    # 3. Configures the Anchor Generator.
    # Defines the scales and aspect ratios of reference boxes used by the RPN 
    # to propose candidate object regions.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    
    # 4. Configures the Region of Interest (RoI) Aligner.
    # Performs spatial pooling to extract fixed-size feature maps from the backbone 
    # corresponding to RPN proposals.
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    
    # 5. Assembles the components into a complete Faster R-CNN detector.
    return FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, min_size=224, max_size=896)

def evaluate_model(model, dataloader, device):
    """
    Inference Execution Loop.
    
    [BOILERPLATE NOTE]: Executes a standard PyTorch evaluation pass. Iterates 
    through the dataset batches, transfers data to the target device (GPU/CPU), 
    and aggregates raw model predictions for offline analysis.
    """
    model.eval() # Sets model to evaluation mode (disables Dropout/BatchNorm updates).
    all_preds = []
    all_gts = []
    print("--- Running Inference ---")
    
    with torch.no_grad(): # Disables gradient computation to reduce memory footprint.
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            # Unpacks batch results.
            for i, output in enumerate(outputs):
                # Transfers tensors to CPU immediately to free GPU VRAM.
                all_preds.append({'boxes': output['boxes'].cpu(), 'scores': output['scores'].cpu()})
                all_gts.append(targets[i]['boxes'])
    return all_preds, all_gts

def calculate_metrics(all_preds, all_gts):
    """
    Quantitative Metrics Calculation.
    
    Computes Precision, Recall, and F1 scores by comparing predictions against 
    ground truth.
    
    [BOILERPLATE NOTE]: This function implements a standard algorithm for object 
    detection evaluation, matching bounding boxes based on IoU overlap and 
    accumulating True/False Positives.
    """
    tp_list, fp_list = [], []
    num_gt = 0
    
    # Iterates through each image in the validation set.
    for pred, gt in zip(all_preds, all_gts):
        num_gt += len(gt) # Accumulates total count of ground truth objects.
        if len(pred['boxes']) == 0: continue
        
        # Sorts predictions by confidence score (descending).
        # This sorting is critical for the correct generation of the Precision-Recall curve.
        sorted_indices = torch.argsort(pred['scores'], descending=True)
        p_boxes = pred['boxes'][sorted_indices]
        p_scores = pred['scores'][sorted_indices]
        
        # If no ground truth exists, all predictions are False Positives.
        if len(gt) == 0:
            for score in p_scores: fp_list.append((score.item(), 1))
            continue
            
        # Calculates the Intersection over Union (IoU) matrix.
        # Shape: (N_pred, M_gt)
        ious = box_iou(p_boxes, gt)
        gt_matched = set()
        
        # Iterates through sorted predictions to assign matches.
        for i, score in enumerate(p_scores):
            if ious.shape[1] > 0:
                # Identifies the ground truth box with maximum overlap for the current prediction.
                max_iou, max_idx = torch.max(ious[i], dim=0)
                
                # Match Logic:
                # 1. Overlap must exceed the defined IOU_THRESHOLD (0.5).
                # 2. The ground truth object must not have been previously matched (prevents duplicate counting).
                if max_iou >= IOU_THRESHOLD and max_idx.item() not in gt_matched:
                    tp_list.append((score.item(), 1)) # True Positive
                    gt_matched.add(max_idx.item())
                else: 
                    fp_list.append((score.item(), 1)) # False Positive
            else: 
                fp_list.append((score.item(), 1))

    print(f"DEBUG: Found {num_gt} Ground Truths and {len(tp_list)} True Positives.")
    if num_gt == 0: return np.array([]), np.array([]), np.array([]), np.array([]), 0, np.array([]), np.array([])

    # Converts lists to NumPy arrays for vectorized computation.
    tp_data = np.array(tp_list) if tp_list else np.zeros((0, 2))
    fp_data = np.array(fp_list) if fp_list else np.zeros((0, 2))
    
    # Structures data as [Score, is_TP, is_FP].
    tp_block = np.column_stack([tp_data[:, 0], np.ones(len(tp_data)), np.zeros(len(tp_data))]) if len(tp_data)>0 else np.zeros((0,3))
    fp_block = np.column_stack([fp_data[:, 0], np.zeros(len(fp_data)), np.ones(len(fp_data))]) if len(fp_data)>0 else np.zeros((0,3))

    # Concatenates and sorts all detections by confidence score.
    all_dets = np.vstack([tp_block, fp_block])
    if len(all_dets) > 0: all_dets = all_dets[np.argsort(all_dets[:, 0])[::-1]]
        
    # Calculates cumulative sums to derive Precision and Recall arrays.
    tps = np.cumsum(all_dets[:, 1])
    fps = np.cumsum(all_dets[:, 2])
    
    precisions = tps / (tps + fps + 1e-6) # Precision = TP / (TP + FP)
    recalls = tps / (num_gt + 1e-6)       # Recall = TP / Total Ground Truths
    scores = all_dets[:, 0]
    
    # Computes F1 score at every threshold point.
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    
    return scores, precisions, recalls, f1s, num_gt, tps, fps

def plot_results(scores, precisions, recalls, f1s, num_gt, tps, fps, csv_path):
    """
    Visualization Generation.
    
    Produces four key analytical plots: Loss Curves, F1-Confidence Curve, 
    Precision-Recall Curve, and Confusion Matrices.
    
    [BOILERPLATE NOTE]: Utilizes standard Matplotlib and Seaborn boilerplate for 
    generating scientific plots and heatmaps.
    """
    sns.set_style("whitegrid")
    
    # --- A. LOSS CURVES (SMOOTHED) ---
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            plt.figure(figsize=(10, 6))
            
            # Applies rolling window smoothing to visualize trends amidst training noise.
            def smooth(data, window=5):
                return data.rolling(window=window, min_periods=1).mean()

            plt.plot(df['epoch'], smooth(df['box_loss']), label='Train Box Loss', color='tab:blue', linewidth=2.5)
            plt.plot(df['epoch'], smooth(df['cls_loss']), label='Train Cls Loss', color='tab:orange', linewidth=2.5)
            # Overlays raw data with transparency for context.
            plt.plot(df['epoch'], df['box_loss'], color='tab:blue', alpha=0.2)
            plt.plot(df['epoch'], df['cls_loss'], color='tab:orange', alpha=0.2)

            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper left')
            
            # Plots Validation IoU on a secondary Y-axis to correlate loss with accuracy.
            ax2 = plt.gca().twinx()
            ax2.plot(df['epoch'], smooth(df['val_mean_iou']), color='black', linestyle='--', linewidth=2, label='Val Mean IoU')
            ax2.set_ylabel('IoU (Higher is Better)')
            ax2.legend(loc='upper right')
            
            plt.title('R-CNN Training Losses (Smoothed)')
            plt.tight_layout()
            plt.savefig("graph_losses.png")
            print("Saved graph_losses.png")
        except: print("Could not read CSV.")

    if len(scores) == 0: return

    # --- B. F1 CURVE ---
    # Visualizes the F1 score across thresholds to identify the optimal confidence cutoff.
    plt.figure()
    plt.plot(scores, f1s, linewidth=2.5, color='tab:purple')
    plt.title('F1-Confidence Curve')
    plt.xlabel('Confidence'); plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graph_f1_curve.png")
    
    # --- C. PRECISION-RECALL CURVE ---
    # Standard metric for evaluating object detection performance.
    plt.figure()
    plt.plot(recalls, precisions, linewidth=2.5, color='tab:blue')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graph_pr_curve.png")

    # --- D. CONFUSION MATRICES ---
    # Generates matrices for specific confidence thresholds (0.5, 0.65, 0.8) 
    # to analyze the trade-off between False Positives and False Negatives.
    thresholds = [0.5, 0.65, 0.8]
    
    for conf in thresholds:
        threshold_mask = scores > conf
        final_tp, final_fp = 0, 0
        
        # Retrieves accumulated TP/FP counts at the specific confidence index.
        if np.any(threshold_mask):
            idx = np.where(threshold_mask)[0][-1]
            final_tp = int(tps[idx])
            final_fp = int(fps[idx])
            
        final_fn = int(num_gt - final_tp)
        final_tn = 0 # True Negatives are undefined/infinite in object detection.
        
        matrix = [[final_tp, final_fn], [final_fp, final_tn]]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Gun (Pred)", "Bg (Pred)"], 
                    yticklabels=["Gun (True)", "Bg (True)"])
        
        plt.title(f"Confusion Matrix (Conf > {conf})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        filename = f"graph_confusion_matrix_{conf}.png"
        plt.savefig(filename)
        plt.close() # Explicitly closes figure to release memory.
        print(f"Saved {filename}")

    print("Saved all graphs.")

def main():
    # 1. Dataset Initialization
    dataset = GunDataset(IMG_DIR, LABEL_DIR, LABEL_TYPE)
    # Uses a custom lambda collate_fn to handle variable-length bounding box lists.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=lambda x: tuple(zip(*x)))
    
    # 2. Model Reconstruction & Loading
    print("Loading model...")
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    # 3. Inference Execution
    all_preds, all_gts = evaluate_model(model, dataloader, DEVICE)
    
    # 4. Metric Computation
    print("Calculating metrics...")
    scores, precisions, recalls, f1s, num_gt, tps, fps = calculate_metrics(all_preds, all_gts)
    
    # 5. Result Visualization
    plot_results(scores, precisions, recalls, f1s, num_gt, tps, fps, CSV_PATH)

if __name__ == "__main__":
    main()
