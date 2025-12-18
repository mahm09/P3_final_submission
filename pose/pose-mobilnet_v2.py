# ==============================================================================
# DISCLAIMER:
# The boilerplate for this script was written with the help of AI.
# It was reviewed, tested, and modified by the project group to suit 
# project-specific data structures and requirements.
# ==============================================================================

import os
print("Importing Core Libraries...")
import cv2
import numpy as np

print("Importing MediaPipe...")
import mediapipe as mp

print("Setting up Visualization (Headless)...")
import matplotlib
# ### Tell matplotlib not to look for a screen (crucial for servers/headless environments)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("Importing PyTorch...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

print("Imports Complete! Starting device check...")

# ================= CONFIGURATION =================
# ### Paths for input data and where to save the cropped hand images
DATA_ROOT = "billeddata"
OUTPUT_DIR = "output"
HAND_CROPS_DIR = "hand_crops"

# ### MediaPipe & Labeling Settings
HAND_PADDING = 80          # How much extra space (pixels) to crop around the wrist
CONFIDENCE_THRESHOLD = 0.5 # MediaPipe must be 50% sure it's a person
IOMIN_THRESHOLD = 0.3      # If 30% of the hand crop overlaps with a gun box, label it "Gun"

# ### Hyperparameters for MobileNet Training
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = 224           # Standard input size for MobileNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= HELPER FUNCTIONS =================

def yolo_to_bbox(yolo_line, img_width, img_height):
    """
    ### Converter: YOLO format (0.5, 0.5, 0.2, 0.2) -> Pixel format (100, 100, 200, 200)
    """
    parts = yolo_line.strip().split()
    center_x = float(parts[1]) * img_width
    center_y = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height
    
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)
    
    return x_min, y_min, x_max, y_max

def compute_iomin(box1, box2):
    """
    ### Intersection over Minimum Area (IoMin)
    ### Unlike IoU, this checks if one box is mostly *inside* the other.
    ### Useful because a gun bounding box might be smaller than the hand crop.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Divide intersection by the SMALLER of the two boxes
    min_area = min(area1, area2)
    return inter_area / min_area if min_area > 0 else 0.0

# ================= STAGE 1: DATA PREPARATION =================

def extract_hands_from_dataset(images_dir, labels_dir, output_dir, split_name):
    """
    ### The 'Zoom-In' Step:
    ### 1. Finds people using MediaPipe.
    ### 2. Locates their wrists.
    ### 3. Crops the image around the wrist.
    ### 4. Checks if that crop overlaps with a known gun location.
    ### 5. Saves the crop into 'gun' or 'no_gun' folders.
    """
    print(f"\nProcessing {split_name}...")
    
    # Create output folders (e.g., hand_crops/train/gun)
    gun_dir = os.path.join(output_dir, split_name, "gun")
    no_gun_dir = os.path.join(output_dir, split_name, "no_gun")
    os.makedirs(gun_dir, exist_ok=True)
    os.makedirs(no_gun_dir, exist_ok=True)
    
    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=CONFIDENCE_THRESHOLD)
    
    gun_count = 0
    no_gun_count = 0
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_file in tqdm(image_files, desc=f"Extracting {split_name}"):
        # Read image
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Read ground truth labels for guns
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_dir, label_file)
        
        gun_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    gun_box = yolo_to_bbox(line, img_width, img_height)
                    gun_boxes.append(gun_box)
        
        # Find hands with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Keypoints 15 and 16 are Left and Right Wrists
            wrists = [(landmarks[15], "left"), (landmarks[16], "right")]
            
            for wrist, side in wrists:
                if wrist.visibility < 0.5: # Skip if MediaPipe isn't sure
                    continue
                
                # Get wrist position in pixels
                wrist_x = int(wrist.x * img_width)
                wrist_y = int(wrist.y * img_height)
                
                # Create a square box around the wrist
                x_min = max(0, wrist_x - HAND_PADDING)
                y_min = max(0, wrist_y - HAND_PADDING)
                x_max = min(img_width, wrist_x + HAND_PADDING)
                y_max = min(img_height, wrist_y + HAND_PADDING)
                
                hand_box = (x_min, y_min, x_max, y_max)
                hand_crop = image[y_min:y_max, x_min:x_max]
                
                if hand_crop.size == 0:
                    continue
                
                # Check overlap: Does this hand crop touch a gun box?
                has_gun = False
                for gun_box in gun_boxes:
                    if compute_iomin(hand_box, gun_box) >= IOMIN_THRESHOLD:
                        has_gun = True
                        break
                
                # Save the cropped image to the correct folder
                save_name = f"{os.path.splitext(img_file)[0]}_{side}.jpg"
                
                if has_gun:
                    cv2.imwrite(os.path.join(gun_dir, save_name), hand_crop)
                    gun_count += 1
                else:
                    cv2.imwrite(os.path.join(no_gun_dir, save_name), hand_crop)
                    no_gun_count += 1
    
    print(f"{split_name}: {gun_count} guns, {no_gun_count} no-guns")
    return gun_count, no_gun_count

# ================= STAGE 2: DATA LOADING =================

class HandDataset(Dataset):
    """
    ### Standard PyTorch Dataset to load the images we just cropped.
    """
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.samples = []
        
        # Load samples labeled "gun" (Class 1)
        gun_dir = os.path.join(data_dir, "gun")
        if os.path.exists(gun_dir):
            for img_file in os.listdir(gun_dir):
                if img_file.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(gun_dir, img_file), 1))
        
        # Load samples labeled "no_gun" (Class 0)
        no_gun_dir = os.path.join(data_dir, "no_gun")
        if os.path.exists(no_gun_dir):
            for img_file in os.listdir(no_gun_dir):
                if img_file.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(no_gun_dir, img_file), 0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label

# ================= STAGE 3: TRAINING =================

def train_model(train_loader, val_loader):
    print("\nTraining MobileNet classifier...")
    
    # ### Load Pretrained MobileNetV2
    # We use transfer learning: the model already knows how to see shapes/edges
    model = models.mobilenet_v2(pretrained=True)
    
    # ### Modify the last layer (Classifier)
    # MobileNet output is usually 1000 classes. We change it to 2 (Gun / No Gun).
    # We add Dropout (0.3) to randomly turn off neurons during training (prevents overfitting).
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3), 
        nn.Linear(model.last_channel, 2)
    )
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ### Optimizer with L2 Regularization (weight_decay) to keep weights small
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # --- TRAINING LOOP ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"✓ Best model saved! (Val Acc: {best_acc:.2f}%)")
    
    return model, history

# ================= STAGE 4: EVALUATION =================

def evaluate_model(model, test_loader):
    print("\nEvaluating on test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate detailed text report (Precision, Recall, F1)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Gun', 'Gun'], digits=4))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

def plot_results(history, cm):
    """
    ### Generates 3 Graphs: Loss, Accuracy, and Confusion Matrix
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Loss Plot
    plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True)
    
    # 2. Accuracy Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend(); plt.grid(True)
    
    # 3. Confusion Matrix Heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Gun', 'Gun'],
               yticklabels=['No Gun', 'Gun'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results.png'), dpi=300)
    print("\n✓ Results saved to 'output/results.png'")
    # plt.show() # Commented out for headless server

# ================= STAGE 5: INFERENCE =================

def detect_guns_in_image(model, image_path, transform):
    """
    ### Production Simulation:
    ### Takes a raw image, finds hands (MediaPipe), classifies them (MobileNet),
    ### and draws a box if a gun is found.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, []
    
    img_height, img_width = image.shape[:2]
    
    # 1. Run MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=CONFIDENCE_THRESHOLD)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    detections = []
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        wrists = [(landmarks[15], "left"), (landmarks[16], "right")]
        
        for wrist, side in wrists:
            if wrist.visibility < 0.5:
                continue
            
            # 2. Crop the Hand
            wrist_x = int(wrist.x * img_width)
            wrist_y = int(wrist.y * img_height)
            
            x_min = max(0, wrist_x - HAND_PADDING)
            y_min = max(0, wrist_y - HAND_PADDING)
            x_max = min(img_width, wrist_x + HAND_PADDING)
            y_max = min(img_height, wrist_y + HAND_PADDING)
            
            hand_crop = image[y_min:y_max, x_min:x_max]
            if hand_crop.size == 0:
                continue
            
            # 3. Classify the Crop with MobileNet
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_tensor = transform(hand_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(hand_tensor)
                prob = torch.softmax(output, dim=1)
                confidence = prob[0][1].item() # Probability of class 1 (Gun)
                prediction = output.argmax(1).item()
            
            # 4. If Gun detected, add to list
            if prediction == 1:  # gun detected
                detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': confidence
                })
                
                # Draw Red Box on original image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                label = f"Gun: {confidence:.2f}"
                cv2.putText(image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return image, detections

# ================= MAIN EXECUTION =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HAND_CROPS_DIR, exist_ok=True)
    
    print("="*60)
    print("HANDGUN DETECTION - MEDIAPIPE + MOBILENET")
    print("="*60)
    
    # --- Step 1: Preprocess Data ---
    # Walks through the raw dataset and crops thousands of hand images
    print("\n[1/5] Extracting hand regions...")
    extract_hands_from_dataset(f"{DATA_ROOT}/train/images", f"{DATA_ROOT}/train/labels", HAND_CROPS_DIR, "train")
    extract_hands_from_dataset(f"{DATA_ROOT}/val/images", f"{DATA_ROOT}/val/labels", HAND_CROPS_DIR, "val")
    extract_hands_from_dataset(f"{DATA_ROOT}/test/images", f"{DATA_ROOT}/test/labels", HAND_CROPS_DIR, "test")
    
    # --- Step 2: Setup DataLoaders ---
    print("\n[2/5] Creating datasets...")
    
    # Apply Data Augmentation only to Training set to make model robust
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),           
        transforms.RandomRotation(degrees=15),            
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Plain transform for Validation/Test
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = HandDataset(f"{HAND_CROPS_DIR}/train", train_transform)
    val_dataset = HandDataset(f"{HAND_CROPS_DIR}/val", val_transform)
    test_dataset = HandDataset(f"{HAND_CROPS_DIR}/test", val_transform)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 3: Train ---
    print("\n[3/5] Training MobileNet...")
    model, history = train_model(train_loader, val_loader)
    
    # --- Step 4: Evaluate Metrics ---
    print("\n[4/5] Evaluating...")
    cm = evaluate_model(model, test_loader)
    
    # --- Step 5: Visualize ---
    print("\n[5/5] Plotting results...")
    plot_results(history, cm)
    
    # --- Step 6: Test Inference on Real Images ---
    print("\nTesting on sample images...")
    # Takes the first 3 images from the test set to show how it works
    test_images = os.listdir(f"{DATA_ROOT}/test/images")[:3]
    
    for img_file in test_images:
        img_path = os.path.join(f"{DATA_ROOT}/test/images", img_file)
        result_img, detections = detect_guns_in_image(model, img_path, val_transform)
        
        if result_img is not None:
            save_path = os.path.join(OUTPUT_DIR, f"result_{img_file}")
            cv2.imwrite(save_path, result_img)
            print(f"✓ {img_file}: {len(detections)} gun(s) detected")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
