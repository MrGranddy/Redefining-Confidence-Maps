import os

import numpy as np

from acyclic_directed_graph.confidence_with_dg import confidence_map as cm_gen

data_main_path = "C:/Users/Bugra/Desktop/masters-thesis/final_data"

# data_names = ["leg", "liver", "spine"]
data_names = ["liver"]
modes = ["right", "partial"]

output_name = "acyclic"

output_path = os.path.join(data_main_path, output_name)

os.makedirs(output_path, exist_ok=True)

for image_name in data_names:

    image = np.load( os.path.join(data_main_path, "original", f"{image_name}", "images.npy") )

    N, H, W = image.shape

    image = image.astype(np.float32) / 255.0

    print(f"Processing {image_name}...")
    images_path = os.path.join(output_path, image_name)
    os.makedirs(images_path, exist_ok=True)

    for mode in modes:

        print(f"Processing {mode}...")
        mode_path = os.path.join(images_path, mode)
        os.makedirs(mode_path, exist_ok=True)

        if mode == "full":
            image_mode = image
        elif mode == "left":
            crop = W // 3
            image_mode = image[:, :, :-crop]
        elif mode == "right":
            crop = W // 3
            image_mode = image[:, :, crop:]
        elif mode == "partial":
            crop = H // 3
            image_mode = image[:, :-crop, :]

        conf_map = np.zeros(image_mode.shape, dtype=np.float32)
        for i in range(N):
            conf_map[i] = cm_gen(image_mode[i])

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{N} slices")
        
        conf_map = (conf_map * 255).astype(np.uint8)

        np.save(os.path.join(mode_path, "conf_map.npy"), conf_map)


