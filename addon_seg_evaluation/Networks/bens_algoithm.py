# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:28:22 2020
@author: Ben Fabry
"""
# this program reads the frames of an avi video file, averages all images,
# and stores the normalized image as a floating point numpy array
# in the same directory as the extracted images, under the name "flatfield.npy"
#
# The program then loops again through all images of the video file,
# identifies cells, extracts the cell shape, fits an ellipse to the cell shape,
# and stores the information on the cell's centroid position, long and short axis,
# angle (orientation) of the long axis, and bounding box widht and height
# in a text file (result_file.txt) in the same directory as the video file.

import numpy as np
from skimage import feature
from skimage.filters import gaussian
from skimage.transform import rescale, resize
from scipy.ndimage import morphology
from skimage.morphology import area_opening
from skimage.measure import label, regionprops
from skimage.transform import resize
import os, sys
import imageio
import json
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import time
import cv2
import scipy
from matplotlib.patches import Ellipse
from skimage.transform import downscale_local_mean

def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im ** 2
    ones = np.ones(im.shape)
    kernel = np.ones((2 * N + 1, 2 * N + 1))
    if 1:
        kernel[0, 0] = 0
        kernel[-1, 0] = 0
        kernel[0, -1] = 0
        kernel[-1, -1] = 0
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    return np.sqrt((s2 - s ** 2 / ns) / ns)

def fill(im_sd, t=0.05):
    from skimage.morphology import flood
    im_sd[0, 0] = 0
    im_sd[0, -1] = 0
    im_sd[-1, 0] = 0
    im_sd[-1, -1] = 0
    mask = flood(im_sd, (0, 0), tolerance=t) | \
           flood(im_sd, (im_sd.shape[0] - 1, 0), tolerance=t) | \
           flood(im_sd, (0, im_sd.shape[1] - 1), tolerance=t) | \
           flood(im_sd, (im_sd.shape[0] - 1, im_sd.shape[1] - 1), tolerance=t)
    return mask


#config = getConfig(configfile)


class Segmentation():

    def __init__(self, pixel_size=None, **kwargs):
        # %% go through every frame and look for cells
        self.struct = morphology.generate_binary_structure(2, 1)  # structural element for binary erosion
        self.pixel_size = pixel_size

        self.r_min = 5  # cells smaller than r_min (in um) will not be analyzed
        self.Amin_pixels = np.pi * (self.r_min / self.pixel_size) ** 2  # minimum region area based on minimum radius
        self.down_scale_factor = 10
        self.edge_exclusion = 10  # in scale after downsampling
        self.count = 0
        self.success = 1

    def segmentation(self, img):
        if len(img.shape) == 3:
            img = img[:, :, 0]
        ellipses = []
        prediction_mask = np.zeros(img.shape)
        h, w = img.shape[0], img.shape[1]
        # flatfield correction
        img = img.astype(float) / np.median(img, axis=1)[:, None]
        #fig = plt.figure();plt.imshow(img),fig.savefig("/home/user/Desktop/out.png")
        im_high = scipy.ndimage.gaussian_laplace(img, sigma=1)  # kind of high-pass filtered image
        im_abs_high = np.abs(im_high)  # for detecting potential cells
        im_r = downscale_local_mean(im_abs_high, (self.down_scale_factor, self.down_scale_factor))
        im_rb = im_r > 0.010
        label_im_rb = label(im_rb)


        # region props are based on the downsampled abs high-pass image, row-column style (first y, then x)
        for region in regionprops(label_im_rb, im_r):

            if (region.max_intensity) > 0.03 and (region.area > self.Amin_pixels / 100):
                im_reg_b = label_im_rb == region.label
                min_row = region.bbox[0] * self.down_scale_factor - self.edge_exclusion
                min_col = region.bbox[1] * self.down_scale_factor - self.edge_exclusion
                max_row = region.bbox[2] * self.down_scale_factor + self.edge_exclusion
                max_col = region.bbox[3] * self.down_scale_factor + self.edge_exclusion

                if min_row > 0 and min_col > 0 and max_row < h and max_col < w:  # do not analyze cells near the edge
                    mask = fill(gaussian(im_abs_high[min_row:max_row, min_col:max_col], 3), 0.01)
                    mask = ~mask
                    mask = morphology.binary_erosion(mask, iterations=7).astype(int)
                    for subregion in regionprops(label(mask)):

                        if subregion.area > self.Amin_pixels:
                            ## Extract the mask
                            coords = subregion.coords ## consider min_row max_row....etc
                            prediction_mask[coords[:,0] + min_row,coords[:,1] + min_col] = 1
                            ## ellipses parameter
                            x_c = subregion.centroid[1]
                            y_c = subregion.centroid[0]
                            ma = subregion.major_axis_length
                            mi = subregion.minor_axis_length

                            # this is to match clickpoints elipse angles checkout test2.py for illustration
                            angle = -region.orientation
                            if angle < 0:
                                angle = np.pi - np.abs(angle)
                            angle *= 180 / np.pi


                            ellipses.append(((y_c + min_row, x_c + min_col), (mi, ma), angle))
        return prediction_mask, ellipses



