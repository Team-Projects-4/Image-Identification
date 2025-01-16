from ultralytics import YOLO
import torch
import os

if (os.path.exists("yolov5_trained.pt")):
    model = YOLO("yolov5_trained.pt")
else:
    model = YOLO("yolov5nu.pt")

model.info()

ans = input("Do you want to train model? (y/n) ")

if ans.lower() == "y":
    model.train(
        data="data.yaml",       # Path to the data.yaml file
        epochs=50,              # Number of epochs
        imgsz=640,              # Image size
        batch=8,                # Batch size
        val=True,               # Validate during training
    )

ans = input("Do you want to test individual image? (y/n) ")
if ans.lower() == "y":
    while True:
        img_path = input("Enter the path to the image: ")   # Input file path
        img_path = img_path.strip('"').strip("'")
        if img_path == "exit":                              # If "exit" is inputted, break and exit
            break

        if os.path.exists(img_path):                        # Check if file path exists
            results = model(img_path, 
                            save=True, 
                            save_crop=True, 
                            save_txt=True, 
                            conf=0.25)                      # Lowering confidence threshold so it makes predictions more often
            print("RESULTS\n", results)  
            print("Results saved for:", img_path)
        else:
            print("Image not found. Try a new path. Type 'exit' to exit.")

# Save the trained model 
model.save("yolov5_trained.pt")  # This explicitly saves the current weights