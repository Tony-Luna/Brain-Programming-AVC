# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:21:07 2023

@author: anlun
"""


import numpy as np
from skimage.measure import block_reduce


def pooling_vector(norm_maps):
    # Assuming all images in norm_maps have the same dimensions
    height, width = next(iter(norm_maps.values())).shape

    # We want the output vector size to be 1024
    desired_vector_size = 256

    # Calculate the block size for the height and width
    block_height = height // int(np.sqrt(desired_vector_size))
    block_width = width // int(np.sqrt(desired_vector_size))

    pool_block = (block_height, block_width, 1)
    pool_func = np.sum

    cm_maps = np.stack([norm_maps[key] for key in norm_maps], axis=-1)
    reduced_cm_map = block_reduce(cm_maps, block_size=pool_block, func=pool_func)

    descriptor_vector = reduced_cm_map.flatten()

    return descriptor_vector

def nmax_vector(norm_maps):
    cm_maps = [norm_maps[key] for key in norm_maps]

    descriptor_vector = extract_concatenate_max_values(cm_maps, 50)
    
    return descriptor_vector
    
def extract_concatenate_max_values(arr_list, n):
    max_values_list = []
    
    for arr in arr_list:
        arr_flat = arr.flatten()
        max_values_list.append(arr_flat[np.argpartition(-arr_flat, n)[:n]])
        
    return np.concatenate(max_values_list)