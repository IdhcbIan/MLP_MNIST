import numpy as np
import tensorflow as tf

def load_and_preprocess_mnist():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize and flatten the images
    # Normalizing the pixel values to 0 or 1 based on a threshold, similar to your Pygame app
    threshold = 128
    train_images_flattened = np.where(train_images > threshold, 1, 0).reshape(train_images.shape[0], -1)
    test_images_flattened = np.where(test_images > threshold, 1, 0).reshape(test_images.shape[0], -1)

    return (train_images_flattened, train_labels), (test_images_flattened, test_labels)

# Usage
(train_images_flattened, train_labels), (test_images_flattened, test_labels) = load_and_preprocess_mnist()


with open('image.txt', 'w') as im:
    list_str = str(train_images_flattened[0].tolist())
    im.write(list_str)

# Example: Print the first flattened image and its label
print("First training image (flattened):", train_images_flattened[200])
print("Label of the first training image:", train_labels[200])
print(f"The data set is {len(train_images_flattened)} long:")
