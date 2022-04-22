#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :deskew.py
@Description  :
@Time         :2022/04/21 17:39:20
@Author       :Chen Yifan
'''

import cv2
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, rotate


def ImageDeskew(img, 
                sigma: float = 1.0, 
                num_peaks: int = 30, 
                angle_threshold: float = 2.0, 
                offset: float = 0.0):

    """Detect Image rotate angle using hough line transformation and deskew it.

    Args:
        img (ndarray): Image array
        sigma (float, optional): Standard deviation of the Gaussian filter
            used in canny detection. Defaults to 1.0.
        num_peaks (int, optional): Maximum number of peaks used in Hough line 
            transform. Defaults to 30.
        angle_threshold (float, optional): The angle threshold used in hierarchical 
            clustering. Defaults to 2.0.
        offset (float, optional): The angle offset added to the final result. 
            Defaults to 0.0.

    Returns:
        angle (float): Image skew angle. The reference line is horizontal and the 
            reference direction is counter-clockwise direction. Ranging from -90 to 90.
        deskew_img (ndarray): Image after deskewing.
    """

    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Canny detection
    edges = canny(img_gray, sigma=sigma)

    # Hough Line transform
    h, a, d = hough_line(edges)
    _, angles, _ = hough_line_peaks(h, a, d, num_peaks=num_peaks)
    angles = np.rad2deg(angles)

    # Bad condition
    if len(angles) == 0:
        return 0.0, img

    elif len(angles) == 1:
        rot_angle = angles[0]

    else:
        # Hierarchical Clustering
        dists = pdist(angles.reshape(-1, 1), metric=lambda x, y: min(abs(x-y), 180-abs(x-y)))
        Z = linkage(dists, method='centroid')
        clusters = fcluster(Z, t=angle_threshold, criterion="distance")

        # Remove outlier angles
        max_idx = np.argmax(np.bincount(clusters))
        inliers = angles[clusters == max_idx]

        # Angles near +/-90 degrees
        if np.std(inliers) > angle_threshold:
            inliers[inliers < 0] += 180

        rot_angle = np.mean(inliers)

    # Convert angle range to [-90, +90]
    rot_angle = (rot_angle + offset) % 180 - 90

    # Deskew
    deskew_img = rotate(img, rot_angle, resize=True)
    deskew_img = (deskew_img * 255).astype(np.uint8)
    
    return rot_angle, deskew_img