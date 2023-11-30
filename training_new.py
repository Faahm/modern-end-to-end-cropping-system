import warnings
import os
import cv2
import tensorflow as tf

# Set memory growth before checking for GPUs
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from models import config
from modify_model import new_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

C = config.Config()

rows = open('portrait_data_compiled_resized_padded.csv').read().strip().split("\n")
input_folder = "resized_training_compiled_resized_padded"

data = []
offsets = []
fpp_bbox = []
filenames = []

# Get the training data
num_images_to_use = 10
image_counter = 0
for row in rows:
    # if image_counter >= num_images_to_use:
    #     break
    row = row.split(",")
    filename, x1, x2, y1, y2, top, bottom, left, right, height, width = row
    if (float(x1) < 0 or float(x2) < 0 or float(y1) < 0 or float(y2) < 0
            or int(top) < 0 or int(left) < 0 or int(height) < 0 or int(width) < 0):
        print(f"Found negative values in {filename}. Skipping...")
        continue

    # Get the image path by appending path with filename
    image_path = os.path.join(input_folder, filename)
    if not os.path.exists(image_path):
        print(f"Image file {filename} not found. Skipping...")
        continue

    # Image pre-processing
    img_object = Image.open(image_path)
    # img_object = img_object.resize((224, 224))
    img_object = img_object.convert('RGB')
    image = np.asarray(img_object)

    # fpp_bbox preprocessing
    sr = tf.convert_to_tensor([int(top), int(left), int(height), int(width)])
    sr = tf.cast(sr, dtype='float32')
    sr = sr / 16.0

    filenames.append(filename)
    offsets.append([float(x1), float(x2), float(y1), float(y2)])
    fpp_bbox.append(sr)
    data.append(image)

    # image_counter += 1

# Convert to numpy array and normalize pixel values
data = np.array(data, dtype="float32") / 255.0
fpp_bbox = np.array(fpp_bbox, dtype="float32")
offsets = np.array(offsets, dtype="float32")

# Code to split the data using sklearn
split = train_test_split(data, offsets, fpp_bbox, filenames, test_size=0.20, random_state=42)

(trainImages, testImages) = split[:2]
(trainOffsets, testOffsets) = split[2:4]
(trainFppBbox, testFppBbox) = split[4:6]
(trainFilenames, testFilenames) = split[6:]

# Saving test image filenames
with open("test_images.txt", "w") as f:
    f.write("\n".join(testFilenames))

# Compile the model
print("Compiling the new model...")
opt = SGD(learning_rate=0.0001)
new_model.compile(optimizer=opt, loss=mean_squared_error)

print("Training the new model...")
H = new_model.fit([trainImages, trainFppBbox],
                  trainOffsets,
                  validation_data=([testImages, testFppBbox], testOffsets),
                  batch_size=1,
                  epochs=6,
                  verbose=1,
                  )

train_loss = H.history['loss']
val_loss = H.history['val_loss']
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

# Save the model
print("Saving the weights...")
new_model.save("final_6_compiled.h5")
print("Weights saved.")
