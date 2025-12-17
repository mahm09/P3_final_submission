import os
print("Importing Core Libraries...")
import cv2
import numpy as np

print("Importing MediaPipe...")
import mediapipe as mp

print("Setting up Visualization (Headless)...")
import matplotlib
# THIS IS THE FIX: Tell matplotlib not to look for a screen
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

# Paths
DATA_ROOT = "billeddata"
OUTPUT_DIR = "output"
HAND_CROPS_DIR = "hand_crops"

# MediaPipe settings
HAND_PADDING = 80  # pixels around wrist
CONFIDENCE_THRESHOLD = 0.5
IOMIN_THRESHOLD = 0.3  # overlap threshold for labeling

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#  Convertere fra  YOLO format til pixel coordinates
def yolo_to_bbox(yolo_line, img_width, img_height):
   
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

#hvor stor procentdel  af den mindste box ligger inde i den anden. bruger IOM 
def compute_iomin(box1, box2):
    """Intersection over Minimum area"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(area1, area2)
    
    return inter_area / min_area if min_area > 0 else 0.0

# 
def extract_hands_from_dataset(images_dir, labels_dir, output_dir, split_name):
    """Extract hand regions and label them"""
    print(f"\nProcessing {split_name}...")
    
    # Create output folders
    gun_dir = os.path.join(output_dir, split_name, "gun")
    no_gun_dir = os.path.join(output_dir, split_name, "no_gun")
    os.makedirs(gun_dir, exist_ok=True)
    os.makedirs(no_gun_dir, exist_ok=True)
    
    # MediaPipe sættes op
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
        
        # Read gun labels
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
            
            # Left and right wrists
            wrists = [(landmarks[15], "left"), (landmarks[16], "right")]
            
            for wrist, side in wrists:
                if wrist.visibility < 0.5:
                    continue
                
                # Get wrist position
                wrist_x = int(wrist.x * img_width)
                wrist_y = int(wrist.y * img_height)
                
                # Create box around hand
                x_min = max(0, wrist_x - HAND_PADDING)
                y_min = max(0, wrist_y - HAND_PADDING)
                x_max = min(img_width, wrist_x + HAND_PADDING)
                y_max = min(img_height, wrist_y + HAND_PADDING)
                
                hand_box = (x_min, y_min, x_max, y_max)
                hand_crop = image[y_min:y_max, x_min:x_max]
                
                if hand_crop.size == 0:
                    continue
                
                # Check if hand overlaps with gun
                has_gun = False
                for gun_box in gun_boxes:
                    if compute_iomin(hand_box, gun_box) >= IOMIN_THRESHOLD:
                        has_gun = True
                        break
                
                # Save crop
                save_name = f"{os.path.splitext(img_file)[0]}_{side}.jpg"
                
                if has_gun:
                    cv2.imwrite(os.path.join(gun_dir, save_name), hand_crop)
                    gun_count += 1
                else:
                    cv2.imwrite(os.path.join(no_gun_dir, save_name), hand_crop)
                    no_gun_count += 1
    
    print(f"{split_name}: {gun_count} guns, {no_gun_count} no-guns")
    return gun_count, no_gun_count

# næste step er DATASET CLASS


class HandDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.samples = []
        
        # Load gun samples (label=1)
        gun_dir = os.path.join(data_dir, "gun")
        if os.path.exists(gun_dir):
            for img_file in os.listdir(gun_dir):
                if img_file.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(gun_dir, img_file), 1))
        
        # Load no-gun samples (label=0)
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

# nu kan vi træne vores MOBILENET

def train_model(train_loader, val_loader):
    print("\nTraining MobileNet classifier...")
    
    # Load pretrained MobileNet
    model = models.mobilenet_v2(pretrained=True)
    
    # ✅ ADD DROPOUT (30%) to prevent overfitting
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),  # Dropout layer
        nn.Linear(model.last_channel, 2)
    )
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # ✅ ADD L2 REGULARIZATION (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Track history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training
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
        
        # Validation
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
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"✓ Best model saved! (Val Acc: {best_acc:.2f}%)")
    
    return model, history


# STEP 4: EVALUATE AND VISUALIZE

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
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Gun', 'Gun'], digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

def plot_results(history, cm):
    """Plot training curves and confusion matrix"""
    fig = plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Gun', 'Gun'],
               yticklabels=['No Gun', 'Gun'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results.png'), dpi=300)
    print("\n✓ Results saved to 'output/results.png'")
    plt.show()


# STEP 5: INFERENCE ON NEW IMAGES


def detect_guns_in_image(model, image_path, transform):
    """Detect guns in a single image"""
    image = cv2.imread(image_path)
    if image is None:
        return None, []
    
    img_height, img_width = image.shape[:2]
    
    # MediaPipe pose detection
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
            
            wrist_x = int(wrist.x * img_width)
            wrist_y = int(wrist.y * img_height)
            
            x_min = max(0, wrist_x - HAND_PADDING)
            y_min = max(0, wrist_y - HAND_PADDING)
            x_max = min(img_width, wrist_x + HAND_PADDING)
            y_max = min(img_height, wrist_y + HAND_PADDING)
            
            hand_crop = image[y_min:y_max, x_min:x_max]
            if hand_crop.size == 0:
                continue
            
            # Classify
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_tensor = transform(hand_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(hand_tensor)
                prob = torch.softmax(output, dim=1)
                confidence = prob[0][1].item()
                prediction = output.argmax(1).item()
            
            if prediction == 1:  # gun detected
                detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': confidence
                })
                
                # Draw box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                label = f"Gun: {confidence:.2f}"
                cv2.putText(image, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return image, detections

# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HAND_CROPS_DIR, exist_ok=True)
    
    print("="*60)
    print("HANDGUN DETECTION - MEDIAPIPE + MOBILENET")
    print("="*60)
    
    # Step 1: Extract hand regions
    print("\n[1/5] Extracting hand regions...")
    extract_hands_from_dataset(
        f"{DATA_ROOT}/train/images", 
        f"{DATA_ROOT}/train/labels",
        HAND_CROPS_DIR, "train"
    )
    extract_hands_from_dataset(
        f"{DATA_ROOT}/val/images",
        f"{DATA_ROOT}/val/labels",
        HAND_CROPS_DIR, "val"
    )
    extract_hands_from_dataset(
        f"{DATA_ROOT}/test/images",
        f"{DATA_ROOT}/test/labels",
        HAND_CROPS_DIR, "test"
    )
    
    # Step 2: Create datasets
    print("\n[2/5] Creating datasets...")
    
    # ✅ DATA AUGMENTATION for training set
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),           # Flip left/right
        transforms.RandomRotation(degrees=15),            # Rotate ±15°
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Change lighting
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # NO augmentation for validation/test
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
    
    # Step 3: Train model
    print("\n[3/5] Training MobileNet...")
    model, history = train_model(train_loader, val_loader)
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating...")
    cm = evaluate_model(model, test_loader)
    
    # Step 5: Plot results
    print("\n[5/5] Plotting results...")
    plot_results(history, cm)
    
    # Test on sample images
    print("\nTesting on sample images...")
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