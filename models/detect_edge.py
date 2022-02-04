# %%

# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils import paths
from matplotlib import pyplot as plt


# %%

def detect_circle(image_file):
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(image_file)
    plt.imshow(image)
    plt.show()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray = cv2.blur(gray, (3, 3))
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # detect circles in the image
    #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               minDist=120,
                               dp=1,
                               param1=100,
                               param2=30,
                               minRadius=100,
                               maxRadius=200)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(gray, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        plt.imshow(gray)


# %%

image = cv2.imread('../data/raw/dn/DBM_80B_13Jan_DN.jpg')
plt.imshow(image)
detect_circle('../data/raw/dn/DBM_80B_13Jan_DN.jpg')
