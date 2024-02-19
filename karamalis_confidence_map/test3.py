from confidence_monai import UltrasoundConfidenceMap

import numpy as np

from PIL import Image

img = np.array(Image.open("test.png"))

# Create a confidence map
confidence_map = UltrasoundConfidenceMap(solve_mode="cg")
cm = confidence_map(img)

# Load test cm
test_cm = []
with open("test_matlab.csv", "r") as f:
    for line in f:
        test_cm.append([float(x) for x in line.split(",")])

test_cm = np.array(test_cm)

diff = np.abs(cm - test_cm)

# MSE
mse = np.mean(diff ** 2)
print("MSE: ", mse)

import matplotlib.pyplot as plt

# Find the global minimum and maximum values for consistent color mapping
vmin = min(cm.min(), test_cm.min(), diff.min())
vmax = max(cm.max(), test_cm.max(), diff.max())

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

cmap = 'gray'  # color map

# Python plot
im = axs[0].imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)
axs[0].set_title("Python")

# Matlab plot
axs[1].imshow(test_cm, cmap=cmap, vmin=vmin, vmax=vmax)
axs[1].set_title("Matlab")

# Difference plot
axs[2].imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)
axs[2].set_title("Difference")

# Create a colorbar with the figure, not with the subplots
fig.colorbar(im, ax=axs, orientation='horizontal', fraction=.1)

# Show the plot
plt.show()