import os
import shutil

portrait_folder = "portrait_train"
ava_file_path = "AVA.txt"
output_folder = "portrait_train_filtered"

image_ids = set()
for filename in os.listdir(portrait_folder):
    image_id = os.path.splitext(filename)[0]
    image_ids.add(image_id)

with open(ava_file_path, "r") as ava_file:
    for line in ava_file:
        values = line.strip().split()
        image_id = values[1]
        ratings = [int(value) for value in values[7:12]]  # Columns 8 to 12
        
        if image_id in image_ids and all(rating >= 1 for rating in ratings):
            source_path = os.path.join(portrait_folder, f"{image_id}.jpg")
            destination_path = os.path.join(output_folder, f"{image_id}.jpg")
            
            shutil.copy(source_path, destination_path)
            print(f"Copied image {image_id}.jpg with ratings of {ratings}")

print("Filtering and copying process completed.")
