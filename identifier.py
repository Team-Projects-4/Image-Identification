from ultralytics import YOLO
import torch

model = YOLO("yolov5nu.pt")

model.info()

model.train(
    data="data.yaml",       # Path to the data.yaml file
    epochs=50,              # Number of epochs
    imgsz=640,              # Image size
    batch=8,                # Batch size
    val=True,               # Validate during training
)

results = model.val(data="data.yaml", split="test")

print("RESULTS\n", results)