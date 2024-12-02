from ultralytics import YOLO
import torch
from PIL import Image

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: 
# This script is to load the trained model and be able to test individual images.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Load the trained model
model = YOLO("yolov5_trained.pt")  # Load trained model

def test_image():
    # Prompt for image path
    image_path = input("Enter the path to the image file: ")
    try:
        # Load and predict
        results = model.predict(source=image_path, save=True, conf=0.5)
        print(f"Results saved for {image_path}. Check the `runs/predict/` directory.")
    except Exception as e:
        print(f"Error loading image: {e}")

# Example testing
print("Testing the model on an individual image...")
test_image()
