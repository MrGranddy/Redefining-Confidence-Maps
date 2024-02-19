import os
import shutil
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

paths = [
    ("karamalis_confidence_map/org_output/", "org"),
    ("acyclic_directed_graph/acyclic_directed_output/", "acyclic"),
    ("karamalis_confidence_map/all_output/", "all"),
    #("karamalis_confidence_map/mask_output/", "mask"),
    ("karamalis_confidence_map/min_output/", "min"),
    ("karamalis_confidence_map/mid_output/", "mid"),
]

file_names = range(0, 200)
sample_names = random.sample(file_names, 20)

samples_dir = "samples/"

if os.path.exists(samples_dir):
    shutil.rmtree(samples_dir)
os.makedirs(samples_dir, exist_ok=True)

for sample_name in sample_names:

    path_images = []

    for path_idx, (path, name) in enumerate(paths):

        # Load confidence map
        cm = np.array(Image.open(os.path.join(path, f"{sample_name}_normal.png")).convert("L"))
        log = np.array(Image.open(os.path.join(path, f"{sample_name}_log.png")).convert("L"))
        exp = np.array(Image.open(os.path.join(path, f"{sample_name}_exp.png")).convert("L"))

        # Vertical concat
        concat = np.concatenate((cm, log, exp), axis=0)
        
        # Add to list
        path_images.append(concat)

    # Horizontal concat
    concat = np.concatenate(path_images, axis=1)

    # Save
    concat = Image.fromarray(concat).convert("RGB")
    concat.save(os.path.join(samples_dir, f"{sample_name}.png"))
    