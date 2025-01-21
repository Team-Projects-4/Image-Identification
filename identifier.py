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

ans = input("Do you want to test images? (y/n) ")
if ans.lower() == "y":
    while True:
        print("Specify file or folder (to test all images in folder)")
        path = input("Enter the path: ")   # Input file path
        path = path.strip('"').strip("'")
        if path == "exit":                              # If "exit" is inputted, break and exit
            break

        if os.path.exists(path):                        # Check if file path exists
            if path.endswith("png") or path.endswith("jpg") or path.endswith("jpeg"): # Individual files
                results = model(path, 
                                save=True, 
                                save_crop=True, 
                                save_txt=True, 
                                conf=0.25)              # Lowering confidence threshold so it makes predictions more often
                print("RESULTS\n", results)  
                print("Results saved for:", path)
            else:                                       # Parsing through folder of images
                for file in os.listdir(path):
                    if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"): # Verify images
                        results = model(file,
                                        save=True,
                                        save_crop=True,
                                        save_txt=True,
                                        conf=0.25)
                        print("RESULTS\n", results)  
                        print("Results saved for:", path)   
        else:
            print("Image not found. Try a new path. Type 'exit' to exit.")

# Save the trained model 
model.save("yolov5_trained.pt")  # This explicitly saves the current weights