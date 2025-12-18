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
# ### Checks for GPU availability and sets file paths. 
# ### Note: Ensure 'MODEL_PATH' matches where your file actually is (e.g., 'faster_rcnn/best_frcnn.pth')
MODEL_PATH = "frcnn_output/best_model.pth"
IMG_DIR = "billeddata/val/images"

# ### Auto-detects if you are using YOLO labels (.txt) or XML annotations
if os.path.exists("billeddata/val/labels"):
    LABEL_DIR = "billeddata/val/labels" # YOLO format
    LABEL_TYPE = "txt"
else:
    LABEL_DIR = "billeddata/val/annotations" # XML format
    LABEL_TYPE = "xml"

CSV_PATH = "frcnn_output/training_losses.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_THRESHOLD = 0.5
# ===============================================

class GunDataset(torch.utils.data.Dataset):
    """
    ### Custom Dataset Loader
    ### This class handles finding images and converting label files (XML or TXT) 
    ### into the format PyTorch expects (boxes: [xmin, ymin, xmax, ymax]).
    """
    def __init__(self, img_dir, label_dir, label_type):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_type = label_type
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"--- DATASET: Found {len(self.imgs)} images. Using {label_type.upper()} labels from {label_dir} ---")

    def __getitem__(self, idx):
        # ### Load the image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img_tensor = transforms.functional.to_tensor(img)

        boxes = []
        # ### PARSING LOGIC: Extracts coordinates based on file type
        if self.label_type == "xml":
            xml_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.xml'))
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    boxes.append([
                        float(bbox.find("xmin").text), float(bbox.find("ymin").text),
                        float(bbox.find("xmax").text), float(bbox.find("ymax").text)
                    ])
        elif self.label_type == "txt":
            txt_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    for line in f.readlines():
                        parts = list(map(float, line.strip().split()))
                        # ### Convert YOLO (center_x, center_y, w, h) to (xmin, ymin, xmax, ymax)
                        cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                        xmin = (cx - w/2) * width
                        ymin = (cy - h/2) * height
                        xmax = (cx + w/2) * width
                        ymax = (cy + h/2) * height
                        boxes.append([xmin, ymin, xmax, ymax])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes_tensor) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        target = {"boxes": boxes_tensor}
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

def get_model():
    """
    ### Model Architecture Builder
    ### Reconstructs the exact Faster R-CNN + MobileNetV2 architecture used in training.
    """
    # ### Load MobileNetV2 backbone (feature extractor)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Identity() # Remove classification layer
    backbone = model.features
    backbone.out_channels = 1280
    
    # ### Configure Anchor Generator (sizes of boxes the model looks for)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    
    # ### Configure Region of Interest (RoI) Aligner
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    
    # ### Combine everything into the full Faster R-CNN model
    return FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, min_size=224, max_size=896)

def evaluate_model(model, dataloader, device):
    """
    ### Inference Loop
    ### Runs the model on the test/validation set to get predictions.
    """
    model.eval() # Set to evaluation mode
    all_preds = []
    all_gts = []
    print("--- Running Inference ---")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                # ### Store predictions (CPU) to save GPU memory
                all_preds.append({'boxes': output['boxes'].cpu(), 'scores': output['scores'].cpu()})
                all_gts.append(targets[i]['boxes'])
    return all_preds, all_gts

def calculate_metrics(all_preds, all_gts):
    """
    ### Metrics Calculator
    ### Compares Predictions vs Ground Truths to calculate Precision, Recall, and F1.
    """
    tp_list, fp_list = [], []
    num_gt = 0
    
    for pred, gt in zip(all_preds, all_gts):
        num_gt += len(gt)
        if len(pred['boxes']) == 0: continue
        
        # ### Sort predictions by confidence score (high to low)
        sorted_indices = torch.argsort(pred['scores'], descending=True)
        p_boxes = pred['boxes'][sorted_indices]
        p_scores = pred['scores'][sorted_indices]
        
        if len(gt) == 0:
            for score in p_scores: fp_list.append((score.item(), 1))
            continue
            
        # ### Calculate Intersection over Union (IoU)
        ious = box_iou(p_boxes, gt)
        gt_matched = set()
        
        for i, score in enumerate(p_scores):
            if ious.shape
