import os
import nibabel as nib
import numpy as np
import time

from karamalis_confidence_map.confidence_monai import UltrasoundConfidenceMap

cm = UltrasoundConfidenceMap(sink_mode="min")

data_path = "C:/Users/Bugra/Desktop/masters-thesis/spine_all_sweeps_im.nii.gz"
dest_path = "C:/Users/Bugra/Desktop/masters-thesis/spine_all_sweeps_im_conf_map_min.nii.gz"

# Load the data
data = nib.load(data_path).get_fdata()
data = data.astype(np.float32) / 255.0

data = np.swapaxes(data, 0, 2)

conf_map = np.zeros(data.shape, dtype=np.uint8)

start = time.time()

# Calculate the confidence map
for i in range(data.shape[0]):
    conf_map[i] = (cm(data[i]) * 255).astype(np.uint8)

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{data.shape[0]} slices in {time.time() - start} seconds")
        start = time.time()

conf_map = np.swapaxes(conf_map, 0, 2)

# Save the confidence map
conf_map_nii = nib.Nifti1Image(conf_map, np.eye(4))
nib.save(conf_map_nii, dest_path)