import os
import shutil

from PIL import Image
import numpy as np

import nibabel as nib

from utils import read_tracking, write_tracking

main_path = "C:/Users/Bugra/Desktop/masters-thesis/liver_data"

views = ["l1", "l2", "l3", "r1", "r2", "r3"]

def read_images(view):

    image_names = os.listdir(os.path.join(main_path, view))
    image_names = [name.replace(view, "") for name in image_names]
    image_names = sorted(image_names, key=lambda x: int("0" + x.split(".")[0]))

    images = []
    for name in image_names:
        image = Image.open(os.path.join(main_path, view, view + name))
        images.append(np.array(image))

    return np.array(images)

if os.path.isdir("images"):
    shutil.rmtree("images")
os.mkdir("images")

all_images = []
all_poses = []
cnt = 0

for i, view in enumerate(views):
    images = read_images(view)
    poses = read_tracking(os.path.join(main_path, view + ".ts"))

    for j in range(images.shape[0]):
        image = images[i]
        Image.fromarray(image).save(os.path.join("images", str(cnt) + ".png"))
        cnt += 1

    all_images.append(images)
    all_poses.append(poses)

all_images = np.concatenate(all_images, axis=0)
all_poses = np.concatenate(all_poses, axis=0)

np.save("images.npy", all_images.astype("float32"))
np.save("poses.npy", all_poses)

write_tracking("all_tracking.csv", all_poses)

all_images = np.transpose(all_images, (2, 1, 0))

nim = nib.Nifti1Image(all_images, np.eye(4))
nib.save(nim, "all_images.nii.gz")