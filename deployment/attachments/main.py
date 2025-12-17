import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from typing import List, Dict
import io
from PIL import Image

# Initialize FastAPI
app = FastAPI(
    title="Gun Detection API",
    version="1.0.0",
    description="YOLOv8-based gun detection system"
)

# Load YOLO model
MODEL_PATH = "best(1).pt"
print(f"Loading YOLO model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Gun Detection API",
        "version": "1.0.0",
        "model": "YOLOv8 (best(1).pt)",
        "endpoints": {
            "POST /detect": "Upload image for gun detection",
            "GET /health": "Check server health"
        },
        "status": "ready" if model else "model_not_loaded"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/detect")
async def detect_guns(file: UploadFile = File(...)):
    """
    Detect guns in uploaded image
    
    Returns:
    - detections: list of detected objects with bounding boxes and confidence
    - image_with_boxes: base64 encoded image with drawn boxes
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO detection
        results = model(image)
        
        # Extract detections
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                
                # Draw box on image
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 0, 255), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "detections": detections,
            "num_detections": len(detections),
            "image_with_boxes": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000