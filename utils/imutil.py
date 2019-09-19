
import logging
import time

import cv2
start = time.time()
import numpy as np

np.set_printoptions(precision=2)
logger = logging.getLogger()


def draw_rect(img, x, y, w, h, color=(0, 40, 255)):
    overlay = img.copy()
    output = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def imread(filename, size=None, rgb=True, flags=cv2.IMREAD_COLOR):
    try:
        image = cv2.imread(filename, flags=flags)
        if rgb and image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if size is not None:
            image = imresize(image, width=size[0], height=size[1])
        return image
    except Exception as e:
        logger.warning("Unable to read image: {}, {}".format(filename, e))
        return None


def imresize(img, width=None, height=None, resize_larger=False):
    try:
        size = np.shape(img)
        h = size[0]
        w = size[1]

        if width is not None and height is not None:
            if not (resize_larger and h <= height and w <= width):
                img = cv2.resize(img, (width, height))
        elif width is not None:
            if not (resize_larger and w <= width):
                wpercent = (width / float(w))
                hsize = int((float(h) * float(wpercent)))
                img = cv2.resize(img, (width, hsize))
        elif height is not None:
            if not (resize_larger and h <= height):
                hpercent = (height / float(h))
                wsize = int((float(w) * float(hpercent)))
                img = cv2.resize(img, (wsize, height))
        return img
    except Exception as e:
        logger.exception(e)
        return None


def draw_rects_dlib(img, rects):
    overlay = img.copy()
    output = img.copy()
    for bb in rects:
        bl = (bb.left(), bb.bottom())  # (x, y)
        tr = (bb.right(), bb.top())  # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color=(0, 255, 255), thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def imcrop(image, bbox):
    """
    :param image:
    :param bbox: [top, left, bottom, right]
    :return:
    """
    print("here", bbox)
    # if all(bbox != numbers.Integral):  # coordinates are normalized
    #     bbox = denormalize_bbox(width=image.shape[1], height=image.shape[0], bbox=bbox)
    top, left, bottom, right = bbox
    crop = image[top:bottom, left:right]
    return crop


def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def normalize_bboxes(image, bboxes):
    """
    :param image:
    :param bboxes: [[top, left ,  bottom, right, optional[score]] or [top, left ,  bottom, right, optional[score]]
    :return:
    """
    temp = bboxes.copy()
    temp.astype(np.float32)
    size = np.shape(image)
    height, width = size[0], size[1]

    if len(temp.shape) == 2:
        if len(temp[0]) == 4:
            temp = np.divide(temp, np.array([height, width, height, width]).astype(np.float))
        elif len(temp[0]) == 5:
            temp = np.divide(temp, np.array([height, width, height, width, 1.0]).astype(np.float))
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return temp
    else:
        if len(temp) == 4:
            temp = np.divide(temp, np.array([height, width, height, width]).astype(np.float))
        elif len(temp) == 5:
            temp = np.divide(temp, np.array([height, width, height, width, 1.0]).astype(np.float))
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return temp


def normalize_coords(image, coords):
    """ Used for facial landmark normalization
    :param image:
    :param coords: [N, [top, left]]
    :return:
    """
    size = np.shape(image)
    height, width = size[0], size[1]
    return _normalize_coords(width=width, height=height, coords=coords)


def _normalize_coords(width, height, coords):
    """ Used for facial landmark normalization
    :param image:
    :param coords: [N, [top, left]]
    :return:
    """
    temp = np.array(coords.copy())

    if len(temp.shape) == 3:
        temp = np.array(list(map(lambda _points: np.divide(_points, [height, width]), coords)))
    else:
        temp = np.divide(temp, [height, width])
    return temp

