from ultralytics import YOLO
import os

# Load model
if (os.path.exists("yolov5_trained.pt")):
    model = YOLO("yolov5_trained.pt")
else:
    model = YOLO("yolov5nu.pt")

model.info()

# Training
ans = input("Do you want to train model? (y/n) ")
if ans.lower() == "y":
    model.train(
        data="data.yaml",       # Path to the data.yaml file
        epochs=50,              # Number of epochs
        imgsz=640,              # Image size
        batch=8,                # Batch size
        val=True,               # Validate during training
    )

# Testing
ans = input("Do you want to test images? (y/n) ")
if ans.lower() == "y":
    while True:
        print("Specify file or folder (to test all images in folder)")
        path = input("Enter the path: ")   # Input file path
        path = path.strip('"').strip("'")
        if path == "exit" or path == 'quit' or path == 'q':                              # If "exit" is inputted, break and exit
            break

        if os.path.exists(path):                        # Check if file path exists
            if os.path.isfile(path):                    # Individual files
                results = model(path, 
                                save=True, 
                                save_crop=True, 
                                save_txt=True, 
                                conf=0.35)              # Lowering confidence threshold so it makes predictions more often
                print("RESULTS\n", results)  
                print("Results saved for:", path)
            else:                                       # Parsing through folder of images
                images = [os.path.join(path, img) for img in os.listdir(path) # Get all images in folder
                        if img.lower().endswith(('png', 'jpg', 'jpeg'))]
                death_star_predictions = []             # Empty list to store death star predictions

                results = model(images,
                                save=True,
                                save_crop=True,
                                save_txt=True,
                                conf=0.35,
                                stream=False)           # Return list, not "generator' (https://docs.ultralytics.com/modes/predict/#__tabbed_1_1)
                
                # Loop through every predicted image and result
                for img_path, result in zip(images, results): 
                    for box in result.boxes:
                        if int(box.cls[0]) == 3:        # Class 3 (Death Star)
                            conf = float(box.conf[0])  
                            death_star_predictions.append((img_path, conf))

                # Sort by confidence
                death_star_predictions.sort(key=lambda x: x[1], reverse=True)

                # Get top 10 images
                top_10 = death_star_predictions[:10]

                print("\nTop 10 Most Confident Death Star Predictions:")
                for img, conf in top_10:
                    print(f"{img}: {conf:.4f}")
        else:
            print("Image not found. Try a new path. Type 'exit' to exit.")

# Save the trained model 
model.save("yolov5_trained.pt")  # This explicitly saves the current weights