import argparse
import math
import os

import numpy as np

from PIL import Image

import SimpleITK as sitk
import nibabel as nib

def create_linear_transform(x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation):
    """Creates a linear transformation from the given parameters.

    Args:
        x_translation (float): The x translation.
        y_translation (float): The y translation.
        z_translation (float): The z translation.
        x_rotation (float): The x rotation, in degrees.
        y_rotation (float): The y rotation, in degrees.
        z_rotation (float): The z rotation, in degrees.

    Returns:
        np.ndarray: 4x4 standard homogeneous transformation matrix.
    """

    # Convert degrees to radians.
    x_rotation = math.radians(x_rotation)
    y_rotation = math.radians(y_rotation)
    z_rotation = math.radians(z_rotation)

    # Create the rotation matrices for each axis.
    Rx = np.array([
        [1, 0, 0, 0],
        [0, math.cos(x_rotation), -math.sin(x_rotation), 0],
        [0, math.sin(x_rotation), math.cos(x_rotation), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [math.cos(y_rotation), 0, math.sin(y_rotation), 0],
        [0, 1, 0, 0],
        [-math.sin(y_rotation), 0, math.cos(y_rotation), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [math.cos(z_rotation), -math.sin(z_rotation), 0, 0],
        [math.sin(z_rotation), math.cos(z_rotation), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine the rotations into a single rotation matrix.
    rotation_matrix = Rz @ Ry @ Rx

    # Create the translation matrix.
    translation_matrix = np.array([
        [1, 0, 0, x_translation],
        [0, 1, 0, y_translation],
        [0, 0, 1, z_translation],
        [0, 0, 0, 1]
    ])

    # Combine the rotation and translation matrices.
    return translation_matrix @ rotation_matrix

def apply_transform_to_poses(poses, transform):
    """Applies a transformation matrix to pose matrices.

    Args:
        poses (np.ndarray): Pose matrices of shape (N, 4, 4), where N is the number of images.
        transform (np.ndarray): The 4x4 transformation matrix to apply.

    Returns:
        np.ndarray: Transformed pose matrices.
    """
    transformed_poses = np.array([transform @ pose for pose in poses])

    # Calculate the centroid of all transformed positions (uses the translation components of the pose matrices)
    centroid = np.mean(transformed_poses[:, :3, 3], axis=0)

    # Create a translation matrix to move the centroid to the origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -centroid

    # Apply the translation to center all poses at the origin
    centered_poses = np.array([translation_to_origin @ pose for pose in transformed_poses])

    return centered_poses


transform_template = "Seq_Frame{:04d}_ImageToReferenceTransform"

def poses_from_mha_image(image):
    """Extracts the tracking information from the given MHA image.

    Args:
        image (SimpleITK.Image): The MHA image.

    Returns:
        np.ndarray: The transformation matrices for each slice.
    """

    # Get number of slices
    num_slices = image.GetDepth()

    # Get tracking matrices as string
    transform_lines = [
        image.GetMetaData(transform_template.format(i)) for i in range(num_slices)
    ]

    # Convert to transformation matrices
    transform_matrix = np.stack([
        np.array(list(map(float, line.split()))).reshape(4, 4) for line in transform_lines
    ], axis=0 )

    return transform_matrix


def read_mha_file(file_path):

    # Read the MHA file.
    image = sitk.ReadImage(file_path)

    # Get the image data.
    image_data = sitk.GetArrayFromImage(image)

    # Get the image spacing.
    image_spacing = image.GetSpacing()

    # Get poses
    poses = poses_from_mha_image(image)

    print(f"For file {os.path.basename(file_path)}: Probe width: {image_spacing[0] * image_data.shape[2]}, probe depth: {image_spacing[1] * image_data.shape[1]}")

    return image_data, poses, image_spacing

def main(args):
    
    # Read the MHA file.
    image_data, poses, image_spacing = read_mha_file(args.input)

    # Create correcting transformation
    transform = create_linear_transform(0, 0, 0, 0, 0, 180)

    # Apply the transformation to the poses
    poses = apply_transform_to_poses(poses, transform)

    # Create the output directory.
    os.makedirs(args.output, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save iamges and poses as numpy files
    np.save(os.path.join(args.output, "images.npy"), image_data)
    np.save(os.path.join(args.output, "poses.npy"), poses)

    # Save each slice as a separate PNG image
    for i, slice_data in enumerate(image_data):
        slice_image = Image.fromarray(slice_data)
        slice_image.save(os.path.join(images_dir, f"{i}.png"))

    # Save spacing
    with open(os.path.join(args.output, "spacing.txt"), "w") as f:
        f.write(f"{image_spacing[0]} {image_spacing[1]} {image_spacing[2]}")

    print(f"Saved images and poses to {args.output}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converts a MHA file to an UltraNerf dataset.")
    parser.add_argument("input", type=str, help="The input MHA file.")
    parser.add_argument("output", type=str, help="The output directory.")

    args = parser.parse_args()

    main(args)