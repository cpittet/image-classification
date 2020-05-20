import numpy as np
import cv2


# ================================================================================
# Utils to perform blob detection on an image, resizing images
# ================================================================================


def get_connected_components(img, keep):
    """
    Find the connected components (blobs) in the given img and keep "keep" of them
    :param img: the image (in RGB) to detect the blobs in
    :param keep: the number of blobs to keep, keeping the largest
    :return: image (in RGB) as an array representing the blob with white background elsewhere
    """
    height = img.shape[0]
    width = img.shape[1]
    # Transfer the image into Hue Saturation Value
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    labels = np.empty(height * width, dtype=int)
    labels_equival = np.array(np.array(0))

    # First pass : put the labels on each pixel


def resize(img, width, height):
    """
    Resize the image to fit width x height pixels
    :param img: the image to resize
    :param width: the new width for the image
    :param height: the new height for the image
    :return: the resized image
    """
    return cv2.resize(img, (width, height))


def crop(img, x1, y1, x2, y2):
    """
    Crop the img, keeping only the rectangle defined by (x1, y1) to (x2, y2)
    :param img: the image to crop
    :param x1: the x coordinates of the upper left corner of the rectangle
    :param y1: the y coordinates of the upper left corner of the rectangle
    :param x2: the x coordinate of the lower right corner of the rectangle
    :param y2: the y coordinate of the lower right corner of the rectangle
    :return: the cropped image with size (x2-x1, y2-y1)
    +----->x
    |
    |  image
    y
    """
    return img[y1:y2, x1:x2]

