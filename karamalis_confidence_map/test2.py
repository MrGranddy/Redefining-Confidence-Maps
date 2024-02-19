import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

from PIL import Image

from confidence_monai import UltrasoundConfidenceMap

img = nib.load("../data/US_images/1_Leg_LAV_SA.nii.gz")
img = img.get_fdata()

mode = "mid"

dst_path = f"{mode}_output/"
CM = UltrasoundConfidenceMap(alpha=2.0, beta=90.0, gamma=0.03, sink_mode=mode, solve_mode="cg")

ORG = False
if mode == "org":
    ORG = True

PLOT = False

if not PLOT and os.path.exists(dst_path):
    shutil.rmtree(dst_path)
os.makedirs(dst_path, exist_ok=True)

for slice_idx in range(img.shape[2]):

    # Load image
    image = img[..., slice_idx]

    # Cast to float32
    image = image.astype(np.float32) / 255.0

    if not ORG:
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
        log = np.log(cm + 1e-1)
        log = (log - np.min(log)) / (np.max(log) - np.min(log))
        log = (log * 255).astype(np.uint8)

        exp = np.exp(cm)
        exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))
        exp = (exp * 255).astype(np.uint8)

        cm = (cm * 255).astype(np.uint8)

        # Save confidence map
        
        cm = Image.fromarray(cm).convert("RGB")
        cm.save(dst_path + f"{slice_idx}_normal.png")
        
        log = Image.fromarray(log).convert("RGB")
        log.save(dst_path + f"{slice_idx}_log.png")

        exp = Image.fromarray(exp).convert("RGB")
        exp.save(dst_path + f"{slice_idx}_exp.png")