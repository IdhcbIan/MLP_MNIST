import pygame
import sys

# Constants for the game
CELL_SIZE = 35
GRID_SIZE = 28
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("28x28 Grid Reconstruction from Input")

    # Get input from terminal
    print("Paste the binary string and press enter:")
    flattened_array = input().strip()

    # Filter out non-numeric characters and convert to integers
    filtered_array = ''.join([ch for ch in flattened_array if ch.isdigit()])
    grid = [int(filtered_array[i]) for i in range(len(filtered_array))]

    # Ensure the grid is correctly sized (28x28)
    if len(grid) != GRID_SIZE * GRID_SIZE:
        print("Error: Input must be exactly 784 digits (28x28).")
        return

    # Reshape the filtered array to a 28x28 grid
    grid = [grid[i * GRID_SIZE:(i + 1) * GRID_SIZE] for i in range(GRID_SIZE)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                running = False

        # Drawing the grid reconstructed from the flattened array
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = BLACK if grid[row][col] == 1 else WHITE
                pygame.draw.rect(screen, color, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
