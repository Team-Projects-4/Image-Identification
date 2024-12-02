import os
import shutil

# Define the class mappings from folder names to class IDs
class_map = {"Bike": 0, "Car": 1, "Cat": 2, "DeathStar": 3, "Dog": 4, "Flower": 5, "Horse": 6, "Rider": 7}

# Dataset paths
base_path = "datasets"  # Main directory containing train, valid, and test folders
image_folders = ["train", "valid", "test"]  # List of datasets: train, valid, test

# Function to create YOLO format labels
def create_labels(image_folder, label_folder, image_dest_folder):
    # Ensure the label and image destination folders exist
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(image_dest_folder, exist_ok=True)

    # Process each class folder inside the dataset folder
    for class_name, class_id in class_map.items():
        class_folder = os.path.join(base_path, image_folder, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: {class_folder} does not exist.")
            continue

        # Process each image in the class folder
        for image_name in os.listdir(class_folder):
            print(f"Processing file: {image_name}")  # Debugging line to check image names

            # Check if the image file ends with a valid image extension
            if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                # Define the paths for image and label
                image_path = os.path.join(class_folder, image_name)
                label_name = image_name.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
                label_path = os.path.join(label_folder, label_name)

                # Write the YOLO format label for this image
                with open(label_path, "w") as label_file:
                    # For a single class image, we set the bounding box as the whole image (normalized)
                    label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

                # Move the image to the corresponding class folder in the image directory
                class_image_dest_folder = os.path.join(image_dest_folder, class_name)
                os.makedirs(class_image_dest_folder, exist_ok=True)
                shutil.copy(image_path, os.path.join(class_image_dest_folder, image_name))

            else:
                print(f"Skipping non-image file: {image_name}")  # Debugging line for non-image files

# Loop through train, valid, and test datasets
for image_folder in image_folders:
    # Set the paths for labels and images
    label_folder = os.path.join(base_path, image_folder, "labels")
    image_dest_folder = os.path.join(base_path, image_folder, "images")

    # Process the images and labels for this dataset
    create_labels(image_folder, label_folder, image_dest_folder)

print("Label files generated and images processed.")
