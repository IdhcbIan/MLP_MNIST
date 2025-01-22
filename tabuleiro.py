import pygame
import sys
import torch
import torch.nn as nn  # This is crucial for defining network layers
import torch.optim as optim  # For optimizers like Adam, SGD, etc.
from torch.utils.data import DataLoader, TensorDataset 
#-------// Constants for the game //-----------------------
CELL_SIZE = 35
GRID_SIZE = 28
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#-------// Game Functions //-----------------------


def set_cell_color(grid, position, color):
    x, y = position
    column, row = x // CELL_SIZE, y // CELL_SIZE
    if 0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE:
        grid[row][column] = color

def flatten_grid(grid):
    """Flatten the 2D grid into a 1D list."""
    return [1 if cell == BLACK else 0 for row in grid for cell in row]

def clear_grid(grid):
    """Clear the grid by setting all cells to white."""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            grid[row][col] = WHITE


#-------// NN functions //-----------------------

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 28*28 is the size of the flattened image
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for 10 digits

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Assuming the model class `SimpleNN` is defined in the environment

model = SimpleNN()
model.load_state_dict(torch.load('mnist_model_v2.pth'))
model.eval()  # Set the model to evaluation mode


# Function to predict the digit
def predict_digit(image, model):
    image = torch.tensor(image, dtype=torch.float32)  # Convert to torch tensor
    outputs = model(image.unsqueeze(0))  # Add batch dimension
    _, predicted = torch.max(outputs, 1)
    return predicted.item()
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("28x28 Grid Drawing App - Press Enter to Print")

    # Initialize grid
    grid = [[WHITE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Track the state of the mouse button
    mouse_button_down = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_button_down = True
                set_cell_color(grid, event.pos, BLACK)
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_button_down = False
            elif event.type == pygame.MOUSEMOTION and mouse_button_down:
                set_cell_color(grid, event.pos, BLACK)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    flat_grid = flatten_grid(grid)
                    print(flat_grid)
                    clear_grid(grid)


                    predicted_digit = predict_digit(flat_grid, model)
                    print(f'Predicted Digit: {predicted_digit}')

        # Drawing the grid
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = grid[row][col]
                pygame.draw.rect(screen, color, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()


if __name__ == "__main__":
    main()




