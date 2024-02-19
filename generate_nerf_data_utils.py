import os
import shutil

from PIL import Image

import numpy as np

min_image_arr_val = -0.125
max_image_arr_val = 0.75

min_image_img_val = 0
max_image_img_val = 188

def ts_to_poses(ts):

    if ts.shape[1] == 18:
        ts = ts[:, :16]

    poses = ts.reshape(-1, 4, 4)
    poses = np.transpose(poses, (0, 2, 1))

    return poses


def normalize(arr):
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)

    return arr

def create_ultranerf_package(path, image, ts):

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    poses = ts_to_poses(ts)
    image = normalize(image)
    image = image * (max_image_arr_val - min_image_arr_val) + min_image_arr_val

    np.save(os.path.join(path, "images.npy"), image)
    np.save(os.path.join(path, "poses.npy"), poses)

    os.makedirs(os.path.join(path, "images"))

    image = normalize(image)

    for i in range(image.shape[0]):
        slice = image[i]
        slice = slice * (max_image_img_val - min_image_arr_val) + min_image_img_val
        slice = slice.astype("uint8")

        slice = Image.fromarray(slice)
        slice.save(os.path.join(path, "images", f"{i}.png"))
        

    
def read_ts(ts_path):

    data = []

    with open(ts_path, 'r') as f:
        for line in f:
            data.append( [ float(x.strip()) for x in line.strip().split() ] )

    return np.array(data)

def read_ts_csv(ts_path):

    data = []

    with open(ts_path, 'r') as f:
        for line in f:
            data.append( [ float(x.strip()) for x in line.strip().split(",") ] )

    return np.array(data)