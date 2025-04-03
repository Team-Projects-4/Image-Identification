import os
import sys
import shutil
from ultralytics import YOLO

def print_help():
    help_text = """
    Usage: python script.py <folder_path>
    
    Runs YOLO object detection on all images in the specified folder.
    
    Arguments:
      <folder_path>   Path to the folder containing test images
      -h, --help, -?  Show this help message

    Example:
      python script.py /path/to/images
    """
    print(help_text)

def main():
    # Handle help flags
    if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help', '-?'):
        print_help()
        sys.exit(0)
    
    folder_path = sys.argv[1]
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("Error: Invalid folder path. Provide a valid directory containing images.")
        sys.exit(1)
    
    # Load model
    model = YOLO("yolov5_trained.pt") if os.path.exists("yolov5_trained.pt") else YOLO("yolov5nu.pt")
    
    # Get all images in folder
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)
              if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not images:
        print("Error: No valid image files found in the folder.")
        sys.exit(1)
    
    # Run YOLO model on images
    results = model(images, save=True, save_crop=True, save_txt=True, conf=0.35)
    
    death_star_predictions = []

    for img_path, result in zip(images, results):
        for box in result.boxes:
            if int(box.cls[0]) == 3:  # Class 3 (Death Star)
                conf = float(box.conf[0])
                death_star_predictions.append((img_path, conf))
    
    # Sort by confidence score and take top 10
    top_10 = [img for img, _ in sorted(death_star_predictions, key=lambda x: x[1], reverse=True)[:10]]
    destinationDir = "~/repos/TX-Image/imgTX/"
    # Print only file names
    for img in top_10:
        #copy files and print names:
        currentPath = os.path.join(folder_path, os.path.basename(img))
        shutil.copy(currentPath, os.path.join(destinationDir, os.path.basename(img)))
        print(os.path.basename(img))
    
    return top_10  # Return the list of top 10 images

if __name__ == "__main__":
    top_10_images = main()
