"""
Image Binarization
Objective: Convert images to binary images using a threshold value.
The program loads a grayscale image, applies a threshold to convert it into a binary image, and saves the output.
Experiment with different threshold values.
Steps:
Load a grayscale image.
Apply a fixed threshold to convert the image to binary.
Experiment with adaptive thresholding methods.
Display and save the binary image."""
import cv2
from sklearn import preprocessing

image = cv2.imread('Image.png')

if image is None:
    print("Error: Could not load image.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flat_gray_image = gray_image.flatten().reshape(-1, 1) / 255.0

    binarizer = preprocessing.Binarizer(threshold=0.5)
    data_binarized = binarizer.fit_transform(flat_gray_image)

    binarized_image = (data_binarized.reshape(gray_image.shape) * 255)

    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binarized Image', binarized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
