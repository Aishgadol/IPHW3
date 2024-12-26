# Replace with your names and IDs:
# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def clean_gaussian_noise_bilateral(image_gray, radius, std_spatial, std_intensity):
    """
    applies bilateral filtering to reduce gaussian noise while preserving edges

    inputs:
    - image_gray: 2d numpy array, grayscale image
    - radius: window radius (should not be bigger than 5)
    - std_spatial: standard deviation for spatial distances
    - std_intensity: standard deviation for intensity differences

    output:
    - result_image: 2d numpy array, cleaned grayscale image
    """

    # convert to float for accurate calculations
    image_float = image_gray.astype(np.float64)
    rows, cols = image_float.shape

    # pad the image using reflection to handle borders
    padded_float = cv2.copyMakeBorder(
        image_float,
        top=radius,
        bottom=radius,
        left=radius,
        right=radius,
        borderType=cv2.BORDER_REFLECT_101
    )

    # prepare padded output
    padded_clean = np.zeros_like(padded_float)

    # create offset grids from -radius to +radius
    offsets = np.arange(-radius, radius + 1)
    offset_x, offset_y = np.meshgrid(offsets, offsets)

    # precompute spatial gaussian mask (g_s)
    spatial_mask = np.exp(-((offset_x ** 2 + offset_y ** 2) / (2 * (std_spatial ** 2))))

    # iterate over each pixel in the padded image
    padded_rows, padded_cols = padded_float.shape
    for i in range(radius, padded_rows - radius):
        for j in range(radius, padded_cols - radius):
            # extract local window around the current pixel
            local_window = padded_float[i - radius:i + radius + 1,
                           j - radius:j + radius + 1]

            # get the center pixel value
            center_value = padded_float[i, j]

            # compute intensity gaussian mask (g_i)
            intensity_diff = local_window - center_value
            intensity_mask = np.exp(-((intensity_diff ** 2) / (2 * (std_intensity ** 2))))

            # combine spatial and intensity masks
            combined_mask = spatial_mask * intensity_mask

            # normalize the combined mask
            weight_sum = np.sum(combined_mask)
            if weight_sum > 0:
                combined_mask /= weight_sum

            # compute the weighted average
            padded_clean[i, j] = np.sum(combined_mask * local_window)

    # crop the image back to original size
    result_image = padded_clean[radius:radius + rows, radius:radius + cols]

    # clip values to [0, 255] and convert back to uint8
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return result_image


def save_side_by_side(original, cleaned, params, save_path):
    """
    saves a side-by-side comparison of original and cleaned images with parameters

    inputs:
    - original: 2d numpy array, original grayscale image
    - cleaned: 2d numpy array, cleaned grayscale image
    - params: string, parameters used for filtering
    - save_path: string, file path to save the image
    """
    # set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # display original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("original image")
    axes[0].axis('off')

    # display cleaned image
    axes[1].imshow(cleaned, cmap='gray')
    axes[1].set_title("cleaned image")
    axes[1].axis('off')

    # add parameters as a text below the images
    plt.suptitle(f"Parameters: {params}", fontsize=12)

    # adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save the figure
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    # define the path to the original image
    original_image_path = "q2/taj.jpg"

    # read the image in grayscale
    print("Loading image...")
    image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # ensure the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"could not load image from {original_image_path}")
    print("Image loaded successfully.")

    # define parameter ranges
    radius_list = [9]  # radius should not be bigger than 5
    std_spatial_list = [2000]  # example spatial std deviations
    std_intensity_list = [40]  # example intensity std deviations

    # calculate total number of combinations for progress tracking
    total_combinations = len(radius_list) * len(std_spatial_list) * len(std_intensity_list)
    current_combination = 0

    # prepare the results directory
    results_dir = "q2_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory '{results_dir}' for saving results.")
    else:
        print(f"Directory '{results_dir}' already exists. Results will be saved here.")

    print(f"Starting bilateral filtering with {total_combinations} parameter combinations...")

    # loop over all combinations of parameters
    for radius in radius_list:
        for std_spatial in std_spatial_list:
            for std_intensity in std_intensity_list:
                current_combination += 1
                print(f"Processing combination {current_combination}/{total_combinations}: "
                      f"radius={radius}, std_spatial={std_spatial}, std_intensity={std_intensity}")

                # apply bilateral filtering with current parameters
                cleaned_image = clean_gaussian_noise_bilateral(
                    image,
                    radius,
                    std_spatial,
                    std_intensity
                )

                # define the parameters string
                params = f"radius={radius}, std_spatial={std_spatial}, std_intensity={std_intensity}"

                # define the save path with descriptive filename
                save_filename = f"r{radius}_sS{std_spatial}_sI{std_intensity}.png"
                save_path = os.path.join(results_dir, save_filename)

                # save the side-by-side image
                save_side_by_side(image, cleaned_image, params, save_path)

                print(f"Saved result to '{save_path}'")

    print("All parameter combinations processed. All results saved in 'q2_results' folder.")
    print("Script finished successfully.")
