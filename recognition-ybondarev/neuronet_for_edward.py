# coding=utf-8
import tensorflow.keras
from PIL import Image
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('our_model.h5')
print(model.metrics_names)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)

path = input("Enter a path to files: ")
image_paths = [os.path.join(path, f) for f in os.listdir(path)]

total = len(image_paths)
mt = 0
oo = 0
eq = 0

for image_path in image_paths:
    # Replace this with the path to your image
    image = Image.open(image_path)
    image = image.convert("RGB")

    # Make sure to resize all images to 150, 150 otherwise they won't fit in the array
    image = image.resize((150, 150))
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict_classes(data)
    #y_classes = prediction.argmax(axis=-1)
    prediction = prediction[0]
    if prediction == 0:
        print("More than 1 subject")
        mt += 1
    elif prediction == 1:
        print("There is only one subject")
        oo += 1
    #else:
        #print("Equal")
        #mt += 1
    #print(y_classes)
print("Total:",total)
print("One subject:",oo)
print("More than one:",mt)
input()

