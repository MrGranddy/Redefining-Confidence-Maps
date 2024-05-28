import os
import numpy as np

import nibabel as nib

splits = [0, 159, 313, 465, 618, 769, 921, 1067, 1217, 1370, 1534, 1682, 1823, 1975, 2123, 2253, 2389]

def read_labels():
    path = "C:/Users/Bugra/Desktop/masters-thesis/data/spine_labels"
    labels_path = os.path.join(path, "all2.nii")

    label_images = nib.load(labels_path).get_fdata()[..., 0, :]
    label_images = np.transpose(label_images, (2, 1, 0))
    
    return label_images.astype(np.uint8)

def label_to_bone_mask(label):

    bone_mask = label == 1

    return bone_mask

def bone_mask_to_shadow_map(bone_mask):

    shadow_mask = np.zeros(bone_mask.shape, dtype=bool)

    for i in range(bone_mask.shape[0]):
        for j in range(bone_mask.shape[2]):
            line = bone_mask[i, :, j]
            if np.any(line):
                bottom = np.where(line)[0][-1]
                shadow_mask[i, bottom:, j] = 1

    return shadow_mask

def read_image(path):

    image = np.load(path)
    image = image.astype(np.float32) / 255

    return image

def prepare_dataset(confidence_map_path=None):

    mask = label_to_bone_mask(read_labels())
    shadow = bone_mask_to_shadow_map(mask)
    image = read_image("C:/Users/Bugra/Desktop/masters-thesis/final_data/original/spine/images.npy")

    if confidence_map_path:
        confidence_map = read_image(confidence_map_path)
    else:
        confidence_map = None

    return {
        "image": image,
        "confidence_map": confidence_map,
        "shadow": shadow
    }

def dataset_to_view_dataset(dataset):

    views = [i for i in range(1, len(splits))]

    view_dataset = {}

    for i, view in enumerate(views):
    
            start = splits[view - 1]
            end = splits[view]
    
            view_dataset[view] = {
                "image": dataset["image"][start:end],
                "confidence_map": dataset["confidence_map"][start:end] if dataset["confidence_map"] is not None else None,
                "shadow": dataset["shadow"][start:end]
            }

    return view_dataset

def prepare_patch_dataset(dataset, patch_size, num_samples_each_view=1000, seed=None):
    """Given the patch size, create feature vectors of pixels of that patch using confidence and image, the label is the center pixel of the shadow mask.
    Uses num_samples parameter to precalculate the patches to extract from the dataset because number of patches can be very large.
    """

    patch_half_size = patch_size // 2

    # Set the random seed if provided
    if seed:
        np.random.seed(seed)

    patches = []
    labels = []

    for view in dataset:

        image = dataset[view]["image"]
        confidence_map = dataset[view]["confidence_map"]
        shadow = dataset[view]["shadow"]

        num_slices = image.shape[0]
        num_rows = image.shape[1]
        num_cols = image.shape[2]

        sample_slices = np.random.randint(patch_half_size, num_slices - patch_half_size, num_samples_each_view)
        sample_rows = np.random.randint(patch_half_size, num_rows - patch_half_size, num_samples_each_view)
        sample_cols = np.random.randint(patch_half_size, num_cols - patch_half_size, num_samples_each_view)

        for i in range(num_samples_each_view):

            slice = sample_slices[i]
            row = sample_rows[i]
            col = sample_cols[i]

            patch_image = image[slice, row - patch_half_size:row + patch_half_size + 1, col - patch_half_size:col + patch_half_size + 1]
            if confidence_map is not None:
                patch_confidence = confidence_map[slice, row - patch_half_size:row + patch_half_size + 1, col - patch_half_size:col + patch_half_size + 1]
            label = shadow[slice, row, col]

            if confidence_map is not None:
                patches.append(np.concatenate([patch_image.flatten(), patch_confidence.flatten()]))
            else:
                patches.append(patch_image.flatten())

            labels.append(label)

    np.random.seed(None)

    return np.array(patches), np.array(labels)

def calculate_theoretical_number_of_patches(dataset, patch_size):
    """Given the patch size, calculate the theoretical number of patches that can be extracted from the dataset.
    """

    patch_half_size = patch_size // 2

    patches = 0

    for view in dataset:
        image = dataset[view]["image"]

        num_slices = image.shape[0]
        num_rows = image.shape[1]
        num_cols = image.shape[2]

        patches += num_slices * (num_rows - patch_half_size * 2) * (num_cols - patch_half_size * 2)

    return patches

def single_image_to_patches(image, patch_size):
    """Given an image, extract patches of the given size.
    """

    patch_half_size = patch_size // 2

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    patches = []

    for row in range(patch_half_size, num_rows - patch_half_size):
        for col in range(patch_half_size, num_cols - patch_half_size):

            patch = image[row - patch_half_size:row + patch_half_size + 1, col - patch_half_size:col + patch_half_size + 1]
            patches.append(patch.flatten())

    # Sanity check the number of patches
    assert len(patches) == (num_rows - patch_half_size * 2) * (num_cols - patch_half_size * 2)

    pred_image_size = (num_rows - patch_half_size * 2, num_cols - patch_half_size * 2)

    return np.array(patches), pred_image_size

if __name__ == "__main__":

    dataset = prepare_dataset(confidence_map_path="C:/Users/Bugra/Desktop/masters-thesis/final_data/karamalis/spine/full/conf_map.npy")

    images = dataset["image"]
    confidence_maps = dataset["confidence_map"]
    shadows = dataset["shadow"]

    dataset = dataset_to_view_dataset(dataset)

    print(f"Theoretical number of patches: {calculate_theoretical_number_of_patches(dataset, 5)}")

    import matplotlib.pyplot as plt

    random_slice = np.random.randint(0, images.shape[0])

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(images[random_slice], cmap="gray")
    plt.title("Image")

    plt.subplot(132)
    plt.imshow(confidence_maps[random_slice], cmap="gray")
    plt.title("Confidence Map")

    plt.subplot(133)
    plt.imshow(shadows[random_slice], cmap="gray")
    plt.title("Shadow Mask")

    plt.show()
