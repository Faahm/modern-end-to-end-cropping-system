import os
import shutil

id_numbers = []
with open("aesthetics_image_lists/portrait_train.jpgl", "r") as file:
    id_numbers = [line.strip() for line in file.readlines()]

source_folder = "images"
destination_folder = "portrait_train"
ids_not_found = []

for id_number in id_numbers:
    source_path = os.path.join(source_folder, f"{id_number}.jpg")
    destination_path = os.path.join(destination_folder, f"{id_number}.jpg")

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied {id_number}.jpg to portrait_train folder.")
    else:
        print(f"Image {id_number}.jpg not found in the images folder.")
        ids_not_found.append(id_number)

print("Copying process completed.")

if ids_not_found:
    print("IDs not found in the images folder:")
    for id_not_found in ids_not_found:
        print(id_not_found)