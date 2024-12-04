from ultralytics import YOLO
import torch
import os

if (os.path.exists("yolov5_trained.pt")):
    model = YOLO("yolov5_trained.pt")
else:
    model = YOLO("yolov5nu.pt")

model.info()

ans = input("Do you want to train model? (y/n) ")

if ans == "y":
    model.train(
        data="data.yaml",       # Path to the data.yaml file
        epochs=50,              # Number of epochs
        imgsz=640,              # Image size
        batch=8,                # Batch size
        val=True,               # Validate during training
    )

results = model.val(data="data.yaml", split="test")

print("RESULTS\n", results)

# Save the trained model 
model.save("yolov5_trained.pt")  # This explicitly saves the current weights