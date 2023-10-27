import os
import cv2
import numpy as np

# Input folder containing your images
input_folder = "resized_training"

# Output folder to save the padded images
output_folder = "padded_images"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the target size
max_dimension = 224

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Get the current dimensions of the image
        height, width = image.shape[:2]

        # Determine the scaling factor to resize to max_dimension
        if height > width:
            scale_factor = max_dimension / height
        else:
            scale_factor = max_dimension / width

        # Resize the image while maintaining aspect ratio
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Create a blank canvas with the target size (224x224)
        padded_image = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)

        # Calculate the position to paste the resized image to center it
        y_offset = (max_dimension - new_height) // 2
        x_offset = (max_dimension - new_width) // 2

        # Paste the resized image onto the canvas
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        # Save the padded and resized image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, padded_image)