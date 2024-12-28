import cv2
import matplotlib.pyplot as plt

# path to the image
image_path = "q3/broken.jpg"

# load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# check if the image is loaded
if image is None:
    print(f"error: unable to load image at {image_path}")
else:
    # apply median filter
    median_filtered = cv2.medianBlur(image, 5)

    # apply bilateral filter on median result
    filtered_image = cv2.bilateralFilter(median_filtered, d=7, sigmaColor=30, sigmaSpace=30)

    # save the filtered image
    cv2.imwrite("median_and_bilateral.png", filtered_image)

    # plot original and filtered images
    plt.figure(figsize=(10, 5))

    # original image
    plt.subplot(1, 2, 1)
    plt.title("original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # filtered image
    plt.subplot(1, 2, 2)
    plt.title("median + bilateral")
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
