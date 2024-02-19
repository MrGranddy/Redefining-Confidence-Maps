import numpy as np

from sklearn.ensemble import RandomForestClassifier

from data import prepare_view_dataset, prepare_patch_dataset, calculate_theoretical_number_of_patches

view_order = ["l1", "l2", "l3", "r1", "r2", "r3"]

seed = 42

def evaluate(dataset):
    """Evaluate the random forest model using cross validation.
    """
    avg_accuracy = 0

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

        train_patches, train_labels = prepare_patch_dataset(train_dataset, 5, 1, seed=seed)
        test_patches, test_labels = prepare_patch_dataset(test_dataset, 5, 100000, seed=seed+1)

        # print("Train patches:", train_patches.shape)
        # print("Train labels:", train_labels.shape)
        # print("Test patches:", test_patches.shape)
        # print("Test labels:", test_labels.shape)

        rf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=seed)
        rf.fit(train_patches, train_labels)
        
        accuracy = rf.score(test_patches, test_labels)
        avg_accuracy += accuracy

        #print("Accuracy:", accuracy)

    print("Average accuracy:", avg_accuracy / len(dataset))

if __name__ == "__main__":

    # Prepare no confidence dataset
    dataset = {
        view_name: prepare_view_dataset(view_name, None)
        for view_name in view_order
    }

    # Evaluate the model
    print("No confidence dataset")
    evaluate(dataset)


    # Prepare ultranerf dataset
    confidence_map_path_template = "../liver_data/views/{}_conf_im.nii.gz"

    dataset = {
        view_name: prepare_view_dataset(view_name, confidence_map_path_template.format(view_name))
        for view_name in view_order
    }

    # Evaluate the model
    print("Ultranerf dataset")
    evaluate(dataset)


    # Prepare karamalis dataset
    confidence_map_path_template = "../liver_data/karamalis_views/{}_conf_im.nii.gz"
    dataset = {
        view_name: prepare_view_dataset(view_name, confidence_map_path_template.format(view_name))
        for view_name in view_order
    }

    # Evaluate the model
    print("Karamalis dataset")
    evaluate(dataset)
    

    # for view in dataset:
    #     random_slice = np.random.randint(0, dataset[view]["image"].shape[0])

    #     plt.subplot(1, 4, 1)
    #     plt.imshow(dataset[view]["image"][random_slice, :, :])
    #     plt.title("Image")
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(dataset[view]["confidence_map"][random_slice, :, :])
    #     plt.title("Confidence map")
    #     plt.subplot(1, 4, 3)
    #     plt.imshow(dataset[view]["shadow"][random_slice, :, :])
    #     plt.title("Shadow mask")
    #     plt.show()