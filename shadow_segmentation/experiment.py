import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import directed_hausdorff

import matplotlib.pyplot as plt

from data import prepare_dataset, dataset_to_view_dataset, prepare_patch_dataset, single_image_to_patches

view_order = ["l1", "l2", "l3", "r1", "r2", "r3"]

seed = 42
num_test_slices = 1

view_test_slices = {
    "l1": np.random.randint(0, 200, num_test_slices),
    "l2": np.random.randint(0, 200, num_test_slices),
    "l3": np.random.randint(0, 200, num_test_slices),
    "r1": np.random.randint(0, 200, num_test_slices),
    "r2": np.random.randint(0, 200, num_test_slices),
    "r3": np.random.randint(0, 200, num_test_slices)
}


def evaluate(dataset, method: str):
    """Evaluate the random forest model using cross validation.
    """
    avg_accuracy = 0

    cross_avg_precision = 0
    cross_avg_dice = 0
    cross_avg_hd = 0

    # Cross validation, each view test once
    for view in dataset:
        
        train_views = [v for v in dataset if v != view]
        test_view = view

        train_dataset = {
            view_name: dataset[view_name]
            for view_name in train_views
        }

        test_dataset = {
            view_name: dataset[view_name]
            for view_name in [test_view]
        }

        train_patches, train_labels = prepare_patch_dataset(train_dataset, 5, 10000, seed=seed)
        test_patches, test_labels = prepare_patch_dataset(test_dataset, 5, 1000, seed=seed+1)

        rf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=seed)
        rf.fit(train_patches, train_labels)
        
        accuracy = rf.score(test_patches, test_labels)
        avg_accuracy += accuracy

        test_slices = view_test_slices[test_view]

        avg_precision = 0
        avg_dice = 0
        avg_hd = 0

        for i in range(test_slices.shape[0]):

            random_slice = test_slices[i]

            # Get random image from the test dataset
            test_image = test_dataset[test_view]["image"][random_slice]
            test_image_patches, result_shape = single_image_to_patches(test_image, 5)

            if test_dataset[test_view]["confidence_map"] is not None:
                test_image_confidence_map = test_dataset[test_view]["confidence_map"][random_slice]
                test_confmap_pathces, _ = single_image_to_patches(test_image_confidence_map, 5)
                test_image_patches = np.concatenate([test_image_patches, test_confmap_pathces], axis=-1)

            # Predict the shadow mask
            test_image_predictions = rf.predict(test_image_patches)
            test_image_predictions = test_image_predictions.reshape(result_shape)

            # Get ground truth shadow mask
            test_shadow = test_dataset[test_view]["shadow"][random_slice]
            # Crop the shadow mask to the same size as the prediction
            test_shadow = test_shadow[2:-2, 2:-2]

            # Calculate precision
            tp = np.sum(np.logical_and(test_image_predictions == 1, test_shadow == 1))
            fp = np.sum(np.logical_and(test_image_predictions == 1, test_shadow == 0))
            precision = tp / (tp + fp + 1e-6)

            # Calculate dice coefficient
            dice = 2 * np.sum(np.logical_and(test_image_predictions == 1, test_shadow == 1)) / (np.sum(test_image_predictions) + np.sum(test_shadow) + 1e-6)

            # Calculate Hausdorff distance
            hd1 = directed_hausdorff(test_image_predictions, test_shadow)[0]
            hd2 = directed_hausdorff(test_shadow, test_image_predictions)[0]

            # The Hausdorff distance is the maximum of the two directed Hausdorff distances
            hd = max(hd1, hd2)

            avg_precision += precision
            avg_dice += dice
            avg_hd += hd

            if view == "l1":
                fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                ax[0].imshow(test_image, cmap="gray")
                ax[0].set_title("Image")
                if test_dataset[test_view]["confidence_map"] is not None:
                    ax[1].imshow(test_image_confidence_map, cmap="gray")
                    ax[1].set_title("Confidence map")
                else:
                    ax[1].imshow(np.zeros_like(test_image), cmap="gray")
                    ax[1].set_title("Confidence map")
                ax[2].imshow(test_image_predictions, cmap="gray")
                ax[2].set_title("Prediction")
                ax[3].imshow(test_shadow, cmap="gray")
                ax[3].set_title("Ground truth")
                
                # Add main title with method name
                fig.suptitle(method)

                plt.show()

        avg_precision /= test_slices.shape[0]
        avg_dice /= test_slices.shape[0]
        avg_hd /= test_slices.shape[0]

        cross_avg_precision += avg_precision
        cross_avg_dice += avg_dice
        cross_avg_hd += avg_hd

    avg_accuracy /= len(dataset)

    cross_avg_precision /= len(dataset)
    cross_avg_dice /= len(dataset)
    cross_avg_hd /= len(dataset)


    print(f"Average accuracy: {avg_accuracy:.4f}")
    print(f"Average precision: {cross_avg_precision:.4f}")
    print(f"Average dice coefficient: {cross_avg_dice:.4f}")
    print(f"Average Hausdorff distance: {cross_avg_hd:.4f}")

    return 
    
if __name__ == "__main__":

    images_path = "images.npy"
    masks_path = "masks.npy"
    confidence_maps_path = "confidence_maps.npy"

    images = np.load(images_path)
    masks = np.load(masks_path)
    confidence_maps = np.load(confidence_maps_path)


    # Prepare no confidence dataset
    dataset = prepare_dataset(images, masks, confidence_maps)
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    evaluate(dataset, method="UltraNeRF") # method name is only for display purposes
    print()