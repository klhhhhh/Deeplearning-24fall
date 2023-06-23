import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Model
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = torch.tanh(self.fc2(h))
        out = out.view(out.size(0), 1, 28, 28)
        return out


# Decide which device we want to run on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the Generator model
G = Generator(z_dim=100)
G.load_state_dict(torch.load('generator.pth'))
G.to(device)
G.eval()

# Set the dimensions of the noise
num_images_per_adjustment = 8
num_adjustments = 3
noise_dim = 100

# Select five indices to adjust
indices_to_adjust = [1, 20, 40, 60, 80]

# Make sure the output directory exists
os.makedirs('output', exist_ok=True)

for index in indices_to_adjust:
    all_images = []

    for adjustment in np.linspace(-2, 2, num_adjustments):
        for _ in range(num_images_per_adjustment):
            # Initialize a noise vector
            noise_vector = torch.randn(noise_dim).to(device)

            # Adjust the value at the current index
            noise_vector[index] += adjustment

            # Generate an image
            with torch.no_grad():
                generated_image = G(noise_vector.unsqueeze(0))

            all_images.append(generated_image)

    # Create a grid from the list of all images
    grid = vutils.make_grid(torch.cat(all_images, 0), nrow=num_images_per_adjustment)

    # Transform the image back to [0-1] range
    grid = 0.5 * grid + 0.5

    # Plot and save the grid
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f'output/adjustment_index_{index}.png')
    plt.close()
