import warnings
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from models import config
from models import model as M
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

C = config.Config()

print("Loading dataset...")

rows = open('portrait_data.csv').read().strip().split("\n")

# MAKE SURE TO USE 224 RESIZED IMAGES!!!!!!!
training_folder = "padded_images"

data = []
offsets = []
filenames = []

# Get the training_images data
for row in rows:
    row = row.split(",")
    filename, x1, x2, y1, y2 = row

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

    h1, w1 = image.shape[0], image.shape[1]

    h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16

    data.append(image)
    offsets.append([x1, x2, y1, y2])
    filenames.append(filename)

# Convert to numpy array and normalize pixel values
data = np.array(data, dtype="float32") / 255.0
offsets = np.array(offsets, dtype="float32")

# Code to split the data using sklearn
split = train_test_split(data, offsets, filenames, test_size=0.10, random_state=42)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    trainImages,
    trainTargets,
    batch_size=32
)

val_generator = val_datagen.flow(
    testImages,
    testTargets,
    batch_size=32,
    shuffle=False
)

# Saving test image filenames
with open("test_images.txt", "w") as f:
    f.write("\n".join(testFilenames))

print("Loading model...")
model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='train').BuildModel()
model.load_weights(C.model)

# Freeze decoder layers
start_index = model.layers.index(model.get_layer('block6_conv1'))
end_index = model.layers.index(model.get_layer('segmentation'))

for layer in model.layers[start_index:end_index + 1]:
    layer.trainable = False


# model.summary()

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.0001), loss='mean_squared_error')

trainImagesLength = len(trainImages)
testImagesLength = len(testImages)

# trainImages = tf.convert_to_tensor(trainImages)
# testImages = tf.convert_to_tensor(testImages)
# trainTargets = tf.convert_to_tensor(trainTargets)
# testTargets = tf.convert_to_tensor(testTargets)

# Train the model
print("Training the model...")
# H = model.fit(trainImages, trainTargets, steps_per_epoch=trainImagesLength, validation_steps=testImagesLength,
#               validation_data=(testImages, testTargets),
#               batch_size=None, epochs=6, verbose=1)
H = model.fit(train_generator,
              steps_per_epoch=trainImagesLength,
              validation_data=val_generator,
              validation_steps=testImagesLength,
              epochs=6)

# Save the model
print("Saving the weights...")
model.save("test.h5")
print("Weights saved.")
