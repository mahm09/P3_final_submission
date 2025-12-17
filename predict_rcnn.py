import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision import models
import torch.nn as nn
from PIL import Image, ImageDraw
import random

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "frcnn_output/best_model.pth"
IMG_DIR = "billeddata/val/images"
OUTPUT_DIR = "output_predictions"
NUM_IMAGES = 5
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_model():
    # Load Backbone
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Identity()
    backbone = model.features
    backbone.out_channels = 1280
    
    # Build R-CNN
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    
    return FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, min_size=224, max_size=896)

def main():
    print(f"--- Loading Model from {MODEL_PATH} ---")
    model = get_model()
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("--- Selecting Random Images ---")
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if not all_imgs:
        print("No images found in", IMG_DIR)
        return
        
    selected_imgs = random.sample(all_imgs, min(len(all_imgs), NUM_IMAGES))
    
    for i, img_name in enumerate(selected_imgs):
        img_path = os.path.join(IMG_DIR, img_name)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(DEVICE)
        
        with torch.no_grad():
            prediction = model([img_tensor])[0]
            
        draw = ImageDraw.Draw(img_pil)
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        has_detection = False
        for box, score in zip(boxes, scores):
            if score > 0.50: # Threshold 50%
                has_detection = True
                draw.rectangle(list(box), outline="red", width=3)
                draw.text((box[0], box[1]), f"Gun: {score:.2f}", fill="red")
        
        save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
        img_pil.save(save_path)
        print(f"Saved: {save_path} ({'Gun found' if has_detection else 'No gun'})")

if __name__ == "__main__":
    main()