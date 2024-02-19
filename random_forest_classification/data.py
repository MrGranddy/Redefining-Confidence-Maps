import numpy as np

import nibabel as nib

def read_bone_mask(path):

    labels = nib.load(path).get_fdata()[..., 0, :]
    labels = np.transpose(labels, (2, 1, 0))
    
    bone_mask = labels == 13

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

    image = nib.load(path).get_fdata()
    image = np.transpose(image, (2, 1, 0))
    image = image.astype(np.float32) / 255

    return image

def read_confidence_map(path):

    confidence_map = nib.load(path).get_fdata()
    confidence_map = np.transpose(confidence_map, (2, 1, 0))
    confidence_map = confidence_map.astype(np.float32) / 255

    return confidence_map

image_path_template = "../liver_data/views/{}_im.nii.gz"
label_path_template = "../data/liver_labels/{}.nii"

def prepare_view_dataset(view_name, confidence_map_path_template):
    mask = read_bone_mask(label_path_template.format(view_name))
    shadow = bone_mask_to_shadow_map(mask)
    image = read_image(image_path_template.format(view_name))
    if confidence_map_path_template:
        confidence_map = read_confidence_map(confidence_map_path_template.format(view_name))
    else:
        confidence_map = None

    return {
        "image": image,
        "confidence_map": confidence_map,
        "shadow": shadow
    }

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

if __name__ == "__main__":
    mask = read_bone_mask("../data/liver_labels/l1.nii")
    shadow = bone_mask_to_shadow_map(mask)
    image = read_image("../liver_data/views/l1_im.nii.gz")
    confidence_map = read_confidence_map("../liver_data/views/l1_conf_im.nii.gz")


    import matplotlib.pyplot as plt

    random_slice = np.random.randint(0, mask.shape[0])

    plt.subplot(1, 4, 1)
    plt.imshow(mask[random_slice, :, :])
    plt.title("Bone mask")
    plt.subplot(1, 4, 2)
    plt.imshow(shadow[random_slice, :, :])
    plt.title("Shadow mask")
    plt.subplot(1, 4, 3)
    plt.imshow(image[random_slice, :, :])
    plt.title("Image")
    plt.subplot(1, 4, 4)
    plt.imshow(confidence_map[random_slice, :, :])
    plt.title("Confidence map")
    plt.show()
