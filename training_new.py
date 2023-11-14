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
from tensorflow.keras.callbacks import LearningRateScheduler


# Suppress warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

C = config.Config()


# def lr_schedule(epoch):
#     initial_lr = 0.0001  # Set your initial learning rate
#     factor = 0.5  # Set the factor by which the learning rate will be reduced
#     drop_every_epochs = 1  # Set the number of epochs after which the learning rate will be reduced
#
#     return initial_lr * (factor ** (epoch // drop_every_epochs))


# lr_scheduler = LearningRateScheduler(lr_schedule)


print("Loading dataset...")
# Initiate csv file and training folder
rows = open('portrait_data.csv').read().strip().split("\n")
training_folder = "resized_training"

data = []
offsets = []
fpp_bbox = []
filenames = []

# Get the training data
for row in rows:
    row = row.split(",")
    filename, x1, x2, y1, y2, top, bottom, left, right, height, width = row

    # Get the image path by appending path with filename
    image_path = os.path.join(training_folder, filename)

    # Check if the image file exists in the folder
    if not os.path.exists(image_path):
        print(f"Image file {filename} not found. Skipping...")
        continue

    # Image pre-processing
    img_object = Image.open(image_path)
    img_object = img_object.convert('RGB')
    image = np.asarray(img_object)

    # Try this for new pre-processing code
    # img_object = cv2.imread(image_path)

    # BBOX preprocessing
    sr = tf.convert_to_tensor([int(top), int(left), int(height), int(width)])
    # print("training sr:", sr)
    sr = tf.cast(sr, dtype='float32')
    sr = sr / 16.0

    filenames.append(filename)
    offsets.append([float(x1), float(x2), float(y1), float(y2)])
    fpp_bbox.append(sr)
    data.append(image)

# Convert to numpy array and normalize pixel values
data = np.array(data, dtype="float32") / 255.0
fpp_bbox = np.array(fpp_bbox, dtype="float32")
offsets = np.array(offsets, dtype="float32")

# Code to split the data using sklearn
split = train_test_split(data, offsets, fpp_bbox, filenames, test_size=0.20, random_state=42)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFppBbox, testFppBbox) = split[4:6]
(trainFilenames, testFilenames) = split[6:]

# train_datagen = ImageDataGenerator()
# val_datagen = ImageDataGenerator()

# train_generator = train_datagen.flow(
#     [trainImages, trainFppBbox],
#     trainTargets,
#     batch_size=1
# )

# val_generator = val_datagen.flow(
#     [testImages, testFppBbox],
#     testTargets,
#     batch_size=1,
#     shuffle=False
# )

# Saving test image filenames
with open("test_images.txt", "w") as f:
    f.write("\n".join(testFilenames))

print("Loading the new model...")
# new_model.summary()

# Compile the model
opt = SGD(learning_rate=0.0001)
new_model.compile(optimizer=opt, loss=mean_squared_error)

trainImagesLength = len(trainImages)
testImagesLength = len(testImages)

print("Training the new model...")
H = new_model.fit([trainImages, trainFppBbox],
                  trainTargets,
                  validation_data=([testImages, testFppBbox], testTargets),
                  batch_size=1,  # Getting error similar with batch_size=1 in the generators
                  epochs=6,
                  verbose=1,
                  # callbacks=[lr_scheduler]
                  )

# H = new_model.fit(train_generator,
#                   batch_size=128,
#                   epochs=6,
#                   validation_data=val_generator,
#                   steps_per_epoch=trainImagesLength,
#                   validation_steps=testImagesLength)

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
new_model.save("test_halp1.h5")
print("Weights saved.")
