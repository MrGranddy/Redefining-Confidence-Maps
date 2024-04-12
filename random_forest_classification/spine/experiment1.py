import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import directed_hausdorff

import matplotlib.pyplot as plt

from data import prepare_dataset, dataset_to_view_dataset, prepare_patch_dataset, single_image_to_patches

splits = [0, 159, 313, 465, 618, 769, 921, 1067, 1217, 1370, 1534, 1682, 1823, 1975, 2123, 2253, 2389]
view_order = [i for i in range(1, len(splits))]

seed = 42
num_test_slices = 1
patch_size = 15

view_test_slices = {
    view: np.random.randint(0, splits[i+1] - splits[i], num_test_slices)
    for i, view in enumerate(view_order)
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

        train_patches, train_labels = prepare_patch_dataset(train_dataset, patch_size, 10000, seed=seed)
        test_patches, test_labels = prepare_patch_dataset(test_dataset, patch_size, 1000, seed=seed+1)

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
            test_image_patches, result_shape = single_image_to_patches(test_image, patch_size)

            if test_dataset[test_view]["confidence_map"] is not None:
                test_image_confidence_map = test_dataset[test_view]["confidence_map"][random_slice]
                test_confmap_pathces, _ = single_image_to_patches(test_image_confidence_map, patch_size)
                test_image_patches = np.concatenate([test_image_patches, test_confmap_pathces], axis=-1)

            # Predict the shadow mask
            test_image_predictions = rf.predict(test_image_patches)
            test_image_predictions = test_image_predictions.reshape(result_shape)

            # Get ground truth shadow mask
            test_shadow = test_dataset[test_view]["shadow"][random_slice]
            # Crop the shadow mask to the same size as the prediction
            test_shadow = test_shadow[patch_size//2:-patch_size//2, patch_size//2:-patch_size//2]

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

            if view == 1:
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

    # Prepare no confidence dataset
    dataset = prepare_dataset()
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    print("No confidence dataset")
    evaluate(dataset, method="no confidence")
    print()


    # Prepare ultranerf dataset
    confidence_map_path_template = "C:/Users/Bugra/Desktop/masters-thesis/final_data/{method}/spine/full/conf_map.npy"

    dataset = prepare_dataset(confidence_map_path=confidence_map_path_template.format(method="ultranerf"))
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    print("Ultranerf dataset")
    evaluate(dataset, method="ultranerf")
    print()


    # Prepare karamalis dataset
    dataset = prepare_dataset(confidence_map_path=confidence_map_path_template.format(method="karamalis"))
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    print("Karamalis dataset")
    evaluate(dataset, method="karamalis")
    print()

    # Prepare acyclic dataset
    dataset = prepare_dataset(confidence_map_path=confidence_map_path_template.format(method="acyclic"))
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    print("Acyclic dataset")
    evaluate(dataset, method="acyclic")
    print()

    # Prepare min dataset
    dataset = prepare_dataset(confidence_map_path=confidence_map_path_template.format(method="min"))
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    print("Min dataset")
    evaluate(dataset, method="min")
    print()

    # Prepare mid dataset
    dataset = prepare_dataset(confidence_map_path=confidence_map_path_template.format(method="mid"))
    dataset = dataset_to_view_dataset(dataset)

    # Evaluate the model
    print("Mid dataset")
    evaluate(dataset, method="mid")
    print()

