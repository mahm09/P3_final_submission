from ultralytics import YOLO
import os

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())
print()

MODEL_PATH = "best(1).pt"
print(f"Trying to load: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")
print()

try:
    model = YOLO(MODEL_PATH)
    print("✅ SUCCESS! Model loaded!")
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()