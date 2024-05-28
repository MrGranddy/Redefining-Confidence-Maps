import os
import nibabel as nib
import numpy as np
import time

from acyclic_directed_graph.confidence_with_dg import confidence_map

data_path = "C:/Users/Bugra/Desktop/masters-thesis/final_data/original/spine/images.npy"
dest_path = "test.nii.gz"

# Load the data
data = np.load(data_path)
data = data.astype(np.float32) / 255.0

conf_map = np.zeros(data.shape, dtype=np.float32)

log_interval = 10
total_time = 0

# Calculate the confidence map
for i in range(data.shape[0]):

    start = time.time()
    conf_map[i] = confidence_map(data[i])
    end = time.time()

    total_time += end - start

    if (i + 1) % log_interval == 0:
        print(f"Processed {i+1}/{data.shape[0]}, average time: {total_time / (i + 1)} seconds, total time: {total_time} seconds, slice shape: {data[i].shape}, estimated total time: {(total_time / (i + 1)) * data.shape[0]} seconds")

conf_map = np.swapaxes(conf_map, 0, 2)

# Save the confidence map
conf_map_nii = nib.Nifti1Image(conf_map, np.eye(4))
nib.save(conf_map_nii, dest_path)