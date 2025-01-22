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


# Example: Print the first flattened image and its label
#print("First training image (flattened):", train_images_flattened[200])
n = int(input("Enter a number of the index up to 60k: "))
print("Label of the first training image:", train_labels[n])
print("----------------------------------------")
#print(f"The data set is {len(train_images_flattened)} long:")


"""
---------------------------------------------------------------------------------

// Reconstructing //


"""
import pygame
import sys
import re

CELL_SIZE = 35
GRID_SIZE = 28
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def reconstruct(flattened_array):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("28x28 Grid Reconstruction from Input")

    flattened_array

    grid = [flattened_array[i * GRID_SIZE:(i + 1) * GRID_SIZE] for i in range(GRID_SIZE)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = BLACK if grid[row][col] == 1 else WHITE
                pygame.draw.rect(screen, color, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


"""
---------------------------------------------------------------------------------

// Calling the Function //


"""





reconstruct(train_images_flattened[n])
