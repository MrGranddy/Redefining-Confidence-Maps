import os
import shutil

from PIL import Image
import numpy as np

import nibabel as nib

from scipy.spatial.transform import Rotation as R

from utils import read_tracking, write_tracking

main_path = "C:/Users/Bugra/Desktop/masters-thesis/spine_data"

views = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

all_image_paths = os.listdir(os.path.join(main_path, "raw_images"))

def read_images(view):

    # convert view number into 2 digit format by adding 0 if the number is less than 10
    view_name = "Patient-24-{view}".format(view=str(view).zfill(2))
    image_names = [path for path in all_image_paths if view_name in path]
    image_names = [name.replace(view_name, "") for name in image_names]
    image_names = sorted(image_names, key=lambda x: int("0" + x.split(".")[0]))

    images = []
    for name in image_names:
        image = Image.open(os.path.join(main_path, "raw_images", view_name + name))
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

    poses = read_tracking(os.path.join(main_path, str(view) + ".ts"))

    for j in range(images.shape[0]):
        image = images[i]
        Image.fromarray(image).save(os.path.join("images", str(cnt) + ".png"))
        cnt += 1

    all_images.append(images)
    all_poses.append(poses[:images.shape[0]])

all_images = np.concatenate(all_images, axis=0)
all_poses = np.concatenate(all_poses, axis=0)

# # Create the transformation matrix T
# # Assuming rotation is in degrees, convert to radians for scipy
# rotation_deg = np.array([-179.37, 1.99, -178.07])  # X, Y, Z in degrees
# rotation_rad = np.radians(rotation_deg)
# translation = np.array([-7.71, -1.80, -4.00])  # X, Y, Z

# # Create rotation matrix from Euler angles
# rot_matrix = R.from_euler('xyz', rotation_rad).as_matrix()

# # Create the full 4x4 transformation matrix
# T = np.eye(4)  # Start with an identity matrix
# T[:3, :3] = rot_matrix  # Set rotation
# T[:3, 3] = translation  # Set translation


# for i in range(all_poses.shape[0]):
#     all_poses[i] = np.dot(T, all_poses[i])

np.save("images.npy", all_images.astype("float32"))
np.save("poses.npy", all_poses)

write_tracking("all_tracking.csv", all_poses)

all_images = np.transpose(all_images, (2, 1, 0))

nim = nib.Nifti1Image(all_images, np.eye(4))
nib.save(nim, "all_images.nii.gz")