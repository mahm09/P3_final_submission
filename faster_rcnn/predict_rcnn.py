# ==============================================================================
# DISCLAIMER:
# Part of this evaluation script was written with the help of AI.
# It was reviewed, tested, and modified by the project group to suit 
# project-specific data structures and requirements.
# ==============================================================================

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

# ================= CONFIGURATION =================
# ### Settings for where to look and where to save
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "frcnn_output/best_model.pth"
IMG_DIR = "billeddata/val/images"
OUTPUT_DIR = "output_predictions"
NUM_IMAGES = 5  # How many random images to test
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_model():
    """
    ### Rebuilds the model architecture.
    ### We must define the exact same structure (MobileNetV2 + Faster R-CNN)
    ### that we used during training, otherwise the saved weights won't fit.
    """
    # ### Load Backbone (MobileNetV2)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Identity() # Remove the classification head
    backbone = model.features
    backbone.out_channels = 1280
    
    # ### Configure R-CNN Components
    # ### Anchors: Sizes of boxes to try. RoI Align: Crops features.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    
    return FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, min_size=224, max_size=896)

def main():
    print(f"--- Loading Model from {MODEL_PATH} ---")
    model = get_model()
    
    # ### Load the trained "brain" (weights) into the structure
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    # ### Switch to Eval mode (turns off training features like Dropout)
    model.eval()
    
    print("--- Selecting Random Images ---")
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if not all_imgs:
        print("No images found in", IMG_DIR)
        return
        
    # ### Pick 5 random images to test
    selected_imgs = random.sample(all_imgs, min(len(all_imgs), NUM_IMAGES))
    
    for i, img_name in enumerate(selected_imgs):
        img_path = os.path.join(IMG_DIR, img_name)
        
        # ### Preprocessing: Load Image -> Convert to Tensor -> Send to GPU
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(DEVICE)
        
        # ### Run Inference (No gradient calculation needed)
        with torch.no_grad():
            # ### The model expects a list of tensors, even for one image
            prediction = model([img_tensor])[0]
            
        # ### prepare to draw on the original image
        draw = ImageDraw.Draw(img_pil)
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        has_detection = False
        
        # ### Filter results: Only show boxes with confidence > 50%
        for box, score in zip(boxes, scores):
            if score > 0.50: 
                has_detection = True
                # ### Draw the Box (Red)
                draw.rectangle(list(box), outline="red", width=3)
                # ### Write the Score
                draw.text((box[0], box[1]), f"Gun: {score:.2f}", fill="red")
        
        # ### Save the result
        save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
        img_pil.save(save_path)
        print(f"Saved: {save_path} ({'Gun found' if has_detection else 'No gun'})")

if __name__ == "__main__":
    main()
