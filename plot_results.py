import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load the CSV file
csv_path = "frcnn_output/training_losses.csv"  # Make sure this matches your folder
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit()

data = pd.read_csv(csv_path)

# 2. Setup the plot
plt.figure(figsize=(12, 5))

# --- PLOT 1: Training Loss ---
plt.subplot(1, 2, 1)
plt.plot(data["epoch"], data["total_loss"], label="Total Loss", color="red", linewidth=2)
# Optional: Plot sub-losses if you want deeper analysis
plt.plot(data["epoch"], data["cls_loss"], label="Class Loss", linestyle="--", alpha=0.7)
plt.plot(data["epoch"], data["box_loss"], label="Box Loss", linestyle="--", alpha=0.7)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.legend()
plt.grid(True, alpha=0.3)

# --- PLOT 2: Validation IoU ---
plt.subplot(1, 2, 2)
plt.plot(data["epoch"], data["val_mean_iou"], label="Mean IoU", color="blue", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("IoU (Intersection over Union)")
plt.title("Validation Accuracy (IoU)")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Save or Show
plt.tight_layout()
plt.savefig("training_graphs.png")
print("Graphs saved to training_graphs.png")
# plt.show() # Uncomment this if running on your laptop