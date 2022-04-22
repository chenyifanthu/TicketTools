import cv2
import numpy as np
from skimage.measure import label, regionprops


def get_stamp_area(img, kernel_size_ratio=0.001, area_threshold=0.005):
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get red area
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    imgMask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    lower_red = np.array([160, 43, 46])
    upper_red = np.array([180, 255, 255])
    imgMask2 = cv2.inRange(img_hsv, lower_red, upper_red)
    finalMask = cv2.bitwise_or(imgMask1, imgMask2)

    # remove small regions
    kernel_size = int(max(1, max(h, w)*kernel_size_ratio))
    kernel_clean = np.ones((kernel_size, kernel_size))
    cleanMask = cv2.morphologyEx(finalMask, cv2.MORPH_OPEN, kernel_clean)

    # get bounding box
    kernel_dilate = np.ones((kernel_size*5, kernel_size*5))
    dilateMask = cv2.dilate(cleanMask, kernel_dilate)
    labeled_img = label(dilateMask, background=0, connectivity=2)
    bboxs = []
    for prop in regionprops(labeled_img):
        if prop.bbox_area > area_threshold * h * w:
            bboxs.append(prop.bbox)

    return cleanMask, bboxs

def cut_stamp(img, bboxs, padding_ratio=0.01):
    h, w = img.shape[:2]
    padding = int(padding_ratio * max(h, w))
    stamps = []
    cut_bboxs = []
    for bbox in bboxs:
        min_row, min_col, max_row, max_col = bbox
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(h, max_row + padding)
        max_col = min(w, max_col + padding)

        stamps.append(img[min_row:max_row, min_col:max_col])
        cut_bboxs.append((min_row, min_col, max_row, max_col))

    return stamps, cut_bboxs