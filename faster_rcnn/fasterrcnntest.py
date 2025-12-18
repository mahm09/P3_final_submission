# ==============================================================================
# DISCLAIMER:
# The boilerplate for this script was written with the help of AI.
# It was reviewed, tested, and modified by the project group to suit 
# project-specific data structures and requirements.
# ==============================================================================

import os
import csv
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ### Dataset Paths
DATA_DIR = os.path.join(PROJECT_DIR, "billeddata")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")

# ### Path to the Pre-trained Backbone (MobileNet)
# ### This assumes you have already trained a classifier and saved it here.
MOBILENET_PATH = os.path.join(PROJECT_DIR, "output", "best_model.pth")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "frcnn_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ### Hyperparameters
NUM_CLASSES = 2  # 2 classes: 0=background, 1=gun
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.0003

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CUSTOM DATASET: YOLO to FASTER R-CNN
# ============================================================
class YOLO2FRCNNDataset(Dataset):
    """
    ### This class loads images and converts YOLO labels (.txt) 
    ### into the Pascal VOC format (xmin, ymin, xmax, ymax) required by Faster R-CNN.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.lbl_dir = os.path.join(root_dir, "labels")

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # ### Assumes label file has same name as image but with .txt extension
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    # ### Read normalized YOLO coordinates (center_x, center_y, width, height)
                    cls, cx, cy, bw, bh = map(float, line.strip().split())
                    
                    # ### Convert to absolute pixel coordinates (xmin, ymin, xmax, ymax)
                    xmin = (cx - bw / 2) * w
                    ymin = (cy - bh / 2) * h
                    xmax = (cx + bw / 2) * w
                    ymax = (cy + bh / 2) * h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # Always Class 1 for "Gun" (0 is reserved for background)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return torchvision.transforms.functional.to_tensor(img), target


def collate_fn(batch):
    """
    ### Custom batch handler needed because images/boxes have different sizes.
    """
    return tuple(zip(*batch))

# ============================================================
# BACKBONE LOADER
# ============================================================
def try_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=False)
        return True, "loaded non-strict"
    except Exception as e:
        return False, str(e)


def load_mobilenet_backbone(mobilenet_path, device):
    """
    ### Loads the pre-trained MobileNetV2 model to use as the 'eyes' (Feature Extractor)
    ### for the Faster R-CNN detector.
    """
    if not os.path.exists(mobilenet_path):
        raise FileNotFoundError(f"MobileNet checkpoint not found: {mobilenet_path}")

    # ### Load weights
    raw = torch.load(mobilenet_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise RuntimeError("Unrecognized checkpoint format.")

    # ### Rebuild the backbone architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Identity()  # Remove the old classification head

    ok, msg = try_load_state_dict(model, state_dict)
    if not ok:
        raise RuntimeError(f"Failed to load MobileNet checkpoint: {msg}")

    model = model.to(device)
    backbone = model.features
    
    # ### Determine output channels (usually 1280 for MobileNetV2)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = backbone(dummy)
    backbone.out_channels = out.shape[1]

    return backbone, backbone.out_channels

# ============================================================
# UTILITY: IoU Calculation
# ============================================================
def compute_iou(box1, box2):
    """
    ### Calculates Intersection over Union (IoU) to validate accuracy.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

# ============================================================
# MAIN TRAINING ROUTINE
# ============================================================
def main():
    print("Loading MobileNet backbone...")
    backbone, out_channels = load_mobilenet_backbone(MOBILENET_PATH, DEVICE)
    print(f"Backbone loaded, out_channels={out_channels}")

    print("Creating datasets...")
    train_dataset = YOLO2FRCNNDataset(TRAIN_DIR)
    val_dataset = YOLO2FRCNNDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, collate_fn=collate_fn
    )

    print("Building FasterRCNN model...")
    # ### Configure the Anchor Generator (box sizes the model will try to fit)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # ### Configure the RoI Pooler (aligns features to boxes)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # ### Assemble the full detector
    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=224,
        max_size=896
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ### Create CSV log file
    loss_log_path = os.path.join(OUTPUT_DIR, "training_losses.csv")
    with open(loss_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "cls_loss", "box_loss", "rpn_loss", "val_mean_iou"])

    best_iou = 0.0
    print("Starting training...")

    # ### ---------------- TRAINING LOOP ---------------- ###
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for imgs, targets in train_loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # ### Calculate Loss
            loss_dict = model(imgs, targets)
            losses = sum(loss_dict.values())

            # ### Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        scheduler.step()

        # ### ---------------- VALIDATION LOOP ---------------- ###
        model.eval()
        iou_scores = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(DEVICE) for img in imgs]
                outputs = model(imgs)

                # ### Compare Predictions to Ground Truth
                for out, tgt in zip(outputs, targets):
                    pred_boxes = out["boxes"].cpu().numpy()
                    gt_boxes = tgt["boxes"].numpy()
                    for pb in pred_boxes:
                        # ### Calculate Max IoU for this prediction
                        iou = max(compute_iou(pb, gt_box) for gt_box in gt_boxes) if len(gt_boxes) > 0 else 0
                        iou_scores.append(iou)

        mean_iou = np.mean(iou_scores)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Total loss: {epoch_loss:.4f}, Validation mean IoU: {mean_iou:.4f}")

        # ### Save Best Model
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with IoU {best_iou:.4f} to {best_path}")

        # ### Log Metrics to CSV
        with open(loss_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                epoch_loss, # Sum of losses for the entire epoch
                loss_dict.get("loss_classifier", torch.tensor(0)).item(), # Snapshot of last batch
                loss_dict.get("loss_box_reg", torch.tensor(0)).item(),
                loss_dict.get("loss_objectness", torch.tensor(0)).item(),
                mean_iou
            ])

    final_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Saved final model to: {final_path}")


if __name__ == "__main__":
    main()
