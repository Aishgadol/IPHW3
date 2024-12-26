# Idan Morad, 316451012
# Student_Name2, Student_ID2

import cv2
import numpy as np
import matplotlib.pyplot as plt


def clean_gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):

    #convert image to float64 2d array
    im_float = im.astype(np.float64)
    rows, cols = im_float.shape

    #pad the image using reflection
    padded_float = cv2.copyMakeBorder(im_float, top=radius, bottom=radius,
                                       left=radius, right=radius, borderType=cv2.BORDER_REFLECT_101)

    #padded result image is initialized to 0s
    padded_clean = np.zeros_like(padded_float)

    #create meshgrid to account for all possible distance offsets
    distance = np.arange(-radius, radius + 1)
    dist_x, dist_y = np.meshgrid(distance, distance)

    #calculate spatial mask according to given formula
    spatial_mask = np.exp(-((dist_x ** 2 + dist_y ** 2) / (2 * (stdSpatial ** 2))))

    #go through and filter each pixel in the padded image
    padded_rows, padded_cols = padded_float.shape
    for i in range(radius, padded_rows - radius):
        for j in range(radius, padded_cols - radius):

            window_at_ij = padded_float[i - radius:i+radius+1, j -radius:j+radius+1]
            centeral_value = padded_float[i, j]
            intensity_mask = np.exp(-(((window_at_ij - centeral_value)**2) / (2*(stdIntensity**2))))

            combined_mask = spatial_mask * intensity_mask
            weight_sum = np.sum(combined_mask)
            #this if is important
            if weight_sum > 0:
                combined_mask /= weight_sum

            padded_clean[i, j] = np.sum(combined_mask * window_at_ij)

    #crop back result image to original size
    result_image=padded_clean[radius:radius + rows, radius:radius + cols]
    result_image=np.clip(result_image, 0, 255)
    result_image=result_image.astype(np.uint8)

    return result_image



#read the image, when reading we make sure it's in grayscale
original_image_path = "q2/balls.jpg"
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

#make sure image has loaded
if image is None:
    raise FileNotFoundError(f"could not load image from {original_image_path}")

#input parameters
radius = 7
stdSpatial = 25
stdIntensity = 20

#apply bilateral filtering
clear_image_b = clean_gaussian_noise_bilateral(image, radius, stdSpatial, stdIntensity)

#visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("noisy image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("cleaned image")
plt.imshow(clear_image_b, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
