# yolotrain.py
from ultralytics import YOLO
import os
import shutil
import pandas as pd

# ---------------------------
# TRAINING
# ---------------------------
# Load pretrained YOLOv8 small model
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data='datapointer.yaml',
    epochs=100,
    imgsz=896,
    batch=8,
    rect=True,       # preserve aspect ratios
    augment=False,   # disable all augmentation
    mosaic=False,    # disable mosaic
    mixup=False,     # disable mixup
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    project='yolo_results',
    name='run1'
)

# ---------------------------
# LOAD BEST WEIGHTS FOR INFERENCE
# ---------------------------
model = YOLO('yolo_results/run1/weights/best.pt')

# ---------------------------
# PREDICTION ON TEST SET
# ---------------------------
results = model.predict(
    source='./billeddata/test',
    imgsz=896,
    save=True  # automatically saves predicted images
)

# ---------------------------
# The boilerplate below was written with the help of AI. It was reviewed, tested, and modified to suit project-specific needs.
# ---------------------------


# Move predictions to custom folder
output_folder = './yolo_results_test'
os.makedirs(output_folder, exist_ok=True)
default_save_path = './runs/detect/predict'
for f in os.listdir(default_save_path):
    if os.path.isfile(os.path.join(default_save_path, f)):
        shutil.move(os.path.join(default_save_path, f), output_folder)

# Copy first 10 images for quick inspection
example_folder = os.path.join(output_folder, 'examples')
os.makedirs(example_folder, exist_ok=True)
for i, fname in enumerate(os.listdir(output_folder)):
    if i >= 10:
        break
    if os.path.isfile(os.path.join(output_folder, fname)):
        shutil.copy(os.path.join(output_folder, fname), example_folder)

# ---------------------------
# EXPORT NUMERICAL PREDICTIONS (CSV ONLY)
# ---------------------------
all_predictions = []

for res in results:
    boxes = res.boxes.xyxy.tolist()      # bounding box coordinates
    scores = res.boxes.conf.tolist()     # confidence scores
    classes = res.boxes.cls.tolist()     # class IDs
    img_name = os.path.basename(res.path)

    for box, score, cls in zip(boxes, scores, classes):
        all_predictions.append({
            'image': img_name,
            'class': int(cls),
            'confidence': float(score),
            'x_min': float(box[0]),
            'y_min': float(box[1]),
            'x_max': float(box[2]),
            'y_max': float(box[3])
        })

# Save predictions to CSV
csv_path = os.path.join(output_folder, 'predictions.csv')
pd.DataFrame(all_predictions).to_csv(csv_path, index=False)

print(f"All predicted images saved to: {output_folder}")
print(f"Example subset saved to: {example_folder}")
print(f"Numerical predictions saved to CSV: {csv_path}")
