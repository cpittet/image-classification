import numpy as np
import cv2


# ================================================================================
# Utils to perform blob detection on an image, resizing images
# ================================================================================


def connected_components(img):
    """
    Find the largest connected component (blobs) in the given img and keep "keep" of them
    :param img: the image (in RGB) to detect the blobs in
    :return: image (in RGB) as an array representing the largest blob with white background elsewhere
    """
    height = img.shape[0]
    width = img.shape[1]
    # Transfer the image into Hue Saturation Value
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    labels = np.zeros(height * width, dtype=int)
    labels_equival = {0: {0}}  # dict of sets

    # First pass : put the labels on each pixel
    cur_index = 0
    for j in range(height):
        for i in range(width):
            inside = cur_index + i
            if img_HSV[j, i, 2] >= 250:
                min_label = 0 if i == 0 else labels[inside - 1]

                if j > 0:
                    start = 0 if i == 0 else i - 1
                    end = i+2 if i + 2 <= width else width
                    for k in range(start, end):
                        label_in = labels[k+cur_index - width]
                        if label_in != 0:
                            if min_label == 0:
                                min_label = label_in
                            else:
                                min_label = min_label if min_label <= label_in else label_in

                if min_label != 0:
                    if j > 0:
                        start = 0 if i == 0 else i - 1
                        end = i + 2 if i + 2 <= width else width
                        for k in range(start, end):
                            label_in = labels[k + cur_index - width]
                            if label_in != 0:
                                labels_equival.get(label_in).update(labels_equival.get(min_label))
                                labels_equival.get(min_label).update(labels_equival.get(label_in))
                    if i > 0 and labels[inside - 1] != 0:
                        labels_equival.get(labels[inside-1]).update(labels_equival.get(min_label))
                        labels_equival.get(min_label).update(labels_equival.get(labels[inside-1]))

                    labels[inside] = min_label
                else:
                    nb = len(labels_equival)
                    labels_equival[nb] = {nb}
                    labels[inside] = nb

        cur_index += width

    # Second pass
    count_pixels = np.zeros(len(labels_equival), dtype=int)
    cur_index = 0
    for j in range(height):
        for i in range(width):
            inside = cur_index + i
            labels[inside] = min(labels_equival.get(labels[inside]))
            count_pixels[labels[inside]] += 1

        cur_index += width

    max_label = np.argmax(count_pixels)

    result = np.zeros(img.shape, dtype=np.uint8)
    cur_index = 0
    for j in range(height):
        for i in range(width):
            inside = cur_index + i
            result[j, i] = img[j, i] if labels[inside] == max_label else np.array([0, 0, 255])

        cur_index += width
    return result


def find_rectangle(blob):
    """
    Find the rectangle that fits the only blob in the given image
    :param blob: the image containing the blob
    :return: (x1, y1, x2, y2), the 2 opposite corners of the rectangle (upper left and lower right)
    """
    y2 = 0
    i, j, _ = np.where(blob[:, :] != np.array([255, 255, 255]))

    x1 = np.min(i)
    y1 = np.min(j)
    x2 = np.max(i)
    y2 = np.max(j)
    print('{}, {}, {}, {}'.format(x1, y1, x2, y2))
    return x1, y1, x2, y2


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


def adaptive_threshold(img):
    """
    Separate the source image into 3 channels (HSB). Do an adaptive threshold on each
    channel and then recompose the image
    :param img: the image to do the threshold on
    :return: the thresholded image
    """
    hsb = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_h = hsb[:, :, 0]
    img_s = hsb[:, :, 1]
    img_b = hsb[:, :, 2]

    t_h = cv2.adaptiveThreshold(img_h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 3, 0)
    t_s = cv2.adaptiveThreshold(img_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 3, 0)
    t_b = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 3, 0)

    result = np.stack((t_h, t_s, t_b), axis=-1)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    return result


def threshold_hsb(img, hue_low, hue_high, sat_low, sat_high, brig_low, brig_high):
    """
    Do a HSB threshold on the image with the provided ranges
    :param img: the image to threshold
    :param hue_low: lower bound for the hue
    :param hue_high: higher bound for the hue
    :param sat_low: lower bound for the saturation
    :param sat_high: higher bound for the saturation
    :param brig_low: lower bound for the brightness
    :param brig_high: higher bound for the brightness
    :return: the thresholded image
    """
    # Convert to HSB color space
    hsb = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([hue_low, sat_low, brig_low])
    upper = np.array([hue_high, sat_high, brig_high])

    # Do the thresholding
    mask_low = (hsb[:, :] < lower).any(axis=2)
    hsb[mask_low] = np.array([0, 0, 255])

    mask_high = (hsb[:, :] > upper).any(axis=2)
    hsb[mask_high] = np.array([0, 0, 255])

    hsb = cv2.cvtColor(hsb, cv2.COLOR_HSV2RGB)
    return hsb
