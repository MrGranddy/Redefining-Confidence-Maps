import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from confidence_monai import UltrasoundConfidenceMap

files_path = "../data/US_images/3_Spine/"
labels_path = "../data/Labels/3_Spine_labels/"
dst_path = "all_output/"

CM = UltrasoundConfidenceMap(alpha=2.0, beta=90.0, gamma=0.03, sink_mode="all", solve_mode="cg")

PLOT = False
MASK = False
ORG = False

if not PLOT and os.path.exists(dst_path):
    shutil.rmtree(dst_path)
os.makedirs(dst_path, exist_ok=True)

for file_path in os.listdir(files_path):
    if file_path.endswith(".npy"):
        
        label_path = file_path.split("_")[0] + "_segmentation.npy"

        # Load image
        image = np.load(files_path + file_path)
        label = np.load(labels_path + label_path)

        # Cast to float32
        image = image.astype(np.float32)

        # Remove channel dimension
        if len(image.shape) == 3:
            image = image[..., 0]

        if len(label.shape) == 3:
            label = label[..., 0]

        # Make upside down
        image = image[::-1, :]
        label = label[::-1, :]
        
        if not ORG:
            # Compute confidence map
            if MASK:
                cm = CM(image, sink_mask=label)
            else:
                cm = CM(image)
        else:
            cm = image

        if PLOT:

            # Plot image and confidence map
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Ultrasound Image")
            ax[1].imshow(cm, cmap="gray")
            ax[1].set_title("Confidence Map")
            plt.show()

        else:
            # Save confidence map
            cm = (cm * 255).astype(np.uint8)
            cm = Image.fromarray(cm).convert("RGB")
            
            cm.save(dst_path + file_path[:-4] + ".png")