'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import itertools

import skimage.io
from skimage.feature import local_binary_pattern
import skimage.color
import skimage.transform
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
import numpy as np


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###

    image_float = img_as_float(im_orig)
    segmentation_mask = felzenszwalb(image_float, scale=scale, sigma=sigma, min_size=min_size)
    segmented_image = np.concatenate((im_orig, segmentation_mask[:, :, np.newaxis]), axis=2)

    return segmented_image


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    intersection_sum = 0
    for h1, h2 in zip(r1["color_hist"], r2["color_hist"]):
        intersection_sum += min(h1, h2)
    return intersection_sum

def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    intersection_sum = 0
    for h1, h2 in zip(r1["texture_hist"], r2["texture_hist"]):
        intersection_sum += min(h1, h2)
    return intersection_sum

def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    return 1.0 - ((r1["size"] + r2["size"]) / imsize)



def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    combined_width = max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])
    combined_height = max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])

    combined_area = combined_width * combined_height

    fill_similarity = 1.0 - ((combined_area - r1["size"] - r2["size"]) / imsize)
    return fill_similarity



def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###
    histograms = [np.histogram(img[:, channel], BINS, (0.0, 1.0))[0] for channel in range(3)]
    hist = np.concatenate(histograms)
    normalized_hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    return normalized_hist



def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    for c in range(3):
        ret[:, :, c] = local_binary_pattern(img[:, :, c], P=8, R=1.0) 
    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###
    histograms = [np.histogram(img[:, channel], BINS, (0.0, 255.0))[0] for channel in range(3)]
    hist = np.concatenate(histograms)
    normalized_hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    return normalized_hist


def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###
    hsv_image = skimage.color.rgb2hsv(img[:, :, :3])
    label_mask = img[:, :, 3].astype(int)
    height, width = label_mask.shape
    print(height,width)
    # Use numpy operations for better performance
    unique_labels = np.unique(label_mask)
    for label in unique_labels:
        mask = (label_mask == label)
        coords = np.where(mask)
        if len(coords[0]) > 0:
            R[label] = {
                "min_y": int(np.min(coords[0])),
                "max_y": int(np.max(coords[0])),
                "min_x": int(np.min(coords[1])),
                "max_x": int(np.max(coords[1])),
                "labels": [label]
            }

    # Compute texture gradients for the whole image once
    texture_gradients = calc_texture_gradient(img[:, :, :3])

    # Calculate histograms and sizes per region
    for label in R.keys():
        mask = (label_mask == label)

        # Extract HSV pixels for this region
        region_hsv_pixels = hsv_image[mask]

        # Extract texture gradients for this region
        region_texture_pixels = texture_gradients[mask]

        R[label]["size"] = np.sum(mask) 
        R[label]["color_hist"] = calc_colour_hist(region_hsv_pixels)
        R[label]["texture_hist"] = calc_texture_hist(region_texture_pixels)
        

    return R



def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    for (label1, region1), (label2, region2) in itertools.combinations(regions.items(), 2):
        if intersect(region1, region2):
            neighbours.append(((label1, region1), (label2, region2)))

    return neighbours



def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])
    rt["size"] = new_size

    rt["color_hist"] = (r1["color_hist"] * r1["size"] + r2["color_hist"] * r2["size"]) / new_size
    rt["texture_hist"] = (r1["texture_hist"] * r1["size"] + r2["texture_hist"] * r2["size"]) / new_size

    rt["labels"] = r1["labels"] + r2["labels"]

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda item: item[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        keys_to_remove = [key for key in S if i in key or j in key]

        # Task 6: Remove old similarities of related regions
        for key in keys_to_remove:
            del S[key]

        # Task 7: Calculate similarities with the new region
        for key in keys_to_remove:
            if key != (i, j):
                other_label = key[1] if key[0] in (i, j) else key[0]
                S[(t, other_label)] = calc_sim(R[t], R[other_label], imsize)

    # Task 8: Generating the final regions from R
    regions = []
    for label, region in R.items():
        bbox = (region['min_x'], region['min_y'],
                region['max_x'] - region['min_x'],
                region['max_y'] - region['min_y'])
        regions.append({
            'rect': bbox,
            'size': region['size'],
            'labels': region['labels']
        })

    return image, regions
