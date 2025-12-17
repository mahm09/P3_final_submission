import requests
import base64
import json
import os

# Server URL
SERVER_URL = "http://localhost:8000"

print("="*60)
print("GUN DETECTION API - CLIENT TEST")
print("="*60)

# Test 1: Health check
print("\n1. Checking server health...")
try:
    response = requests.get(f"{SERVER_URL}/health")
    if response.status_code == 200:
        print("✅ Server is healthy!")
        print(f"   Response: {response.json()}")
    else:
        print("❌ Server not healthy!")
        exit()
except Exception as e:
    print(f"❌ Cannot connect to server: {e}")
    print("   Make sure server is running!")
    exit()

# Test 2: Gun detection
print("\n2. Testing gun detection...")
print("   Enter path to image file:")
image_path = input("   > ").strip()

if not os.path.exists(image_path):
    print(f"❌ File not found: {image_path}")
    exit()

print(f"   Uploading {image_path}...")

try:
    # Open and send image
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{SERVER_URL}/detect", files=files)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n✅ Detection successful!")
        print(f"\n   Detections found: {result['num_detections']}")
        
        if result['num_detections'] > 0:
            print("\n   Detection details:")
            for i, det in enumerate(result['detections'], 1):
                print(f"   [{i}] Class: {det['class']}")
                print(f"       Confidence: {det['confidence']:.2%}")
                print(f"       Bbox: {det['bbox']}")
        else:
            print("   No guns detected in image.")
        
        # Save result image
        if result.get('image_with_boxes'):
            output_path = "result_" + os.path.basename(image_path)
            img_data = base64.b64decode(result['image_with_boxes'])
            with open(output_path, 'wb') as f:
                f.write(img_data)
            print(f"\n   Result image saved: {output_path}")
    
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")

except Exception as e:
    print(f"❌ Error during detection: {e}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)