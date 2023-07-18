# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:11:03 2023

@author: anlun
"""

import sys

import os

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Paths to append
sys.path.append(os.path.join(parent_dir, 'Utilities'))

from utilities import linear_normalization, safe_divide

from skimage.transform import resize
from math import floor
import numpy as np

# Define gamma values outside the function
gammas = np.array([0.9, 1.4, 2.7])

# DOC based saliency ----------------------------------------------------------
def doc_activation(maps_list):
    target_shape = maps_list[0].shape
    doc_maps = []

    for img in maps_list:
        img = linear_normalization(resize(img, target_shape))
        
        # Compute difference of contrast (DOC) space
        contrast_space = np.stack([linear_normalization(np.abs(np.power(img, g) - img)) for g in gammas])
        
        # Compute saliency map based on contrast differences
        diff_i = np.max(np.abs(np.diff(contrast_space, axis=1, append=1)), axis=0)
        diff_j = np.max(np.abs(np.diff(contrast_space, axis=2, append=1)), axis=0)
        
        diff_max = np.maximum(diff_i, diff_j)
        
        saliency_map = linear_normalization(diff_max)
        
        doc_maps.append(saliency_map)
        
    # Use np.stack for converting list of arrays into a multidimensional numpy array
    doc_maps = np.stack(doc_maps)
    
    combined_map = linear_normalization(np.max(doc_maps, axis=0))
    
    return combined_map


# Compute the activation of the input map using the Markovian approach shown in 
# the GBVS algorithm -----------------------------------------------------------
def markov_activation(maps_list):
    markov_maps = []
    
    for img in maps_list:
        img = resize(img, (32,32))
        n = img.size
        
        # Compute distance matrix
        ix, iy = np.indices(img.shape)
        ix = ix.reshape(n, 1)
        iy = iy.reshape(n, 1)
        d = (ix - ix.T)**2 + (iy - iy.T)**2
        # Generate weight matrix between nodes based on distance matrix
        sig = (0.15) * np.mean(img.shape)
        Dw = np.exp(-1*d/(2*sig**2))
        
        # Assign a linear index to each node
        linear_map = img.ravel()
        
        # Assign edge weights based on distances between nodes and algtype
        MM = Dw * np.abs(linear_map[:, None] - linear_map)
        
        # Make it a markov matrix (so each column sums to 1)
        # MM /= np.sum(MM, axis=0, keepdims=True)
        MM = safe_divide(MM, np.sum(MM, axis=0, keepdims=True))
        
        # Find the principal eigenvector using matrix_power function
        v = np.ones((n,1), dtype=np.float32)/n
        MM_pow = np.linalg.matrix_power(MM, 5)
        Vo = MM_pow @ v
        # Vo /= np.sum(Vo)
        Vo = safe_divide(Vo, np.sum(Vo))
        
        # Arrange the nodes back into a rectangular map
        activation = Vo.reshape(img.shape)
        
        markov_maps.append(activation)
        
    combined_map = linear_normalization(np.sum(markov_maps, axis=0))
    combined_map = resize(combined_map, maps_list[0].shape)
    
    return combined_map

# Function to compute center-surround activation ------------------------------
def cs_activation(maps_list):
    max_size = maps_list[0].shape
    maps_list = [resize(pyr_map, max_size) for pyr_map in maps_list]
    center_surround_maps = []
    
    j = 1
    pyr_len = len(maps_list)
    
    while(c:=floor((j+pyr_len)/2)) < pyr_len:
        f = floor(j/2)
        center_surround_maps.append(maps_list[c] - maps_list[f])
        
        j = j+1
      
    combined_map = linear_normalization(np.sum(center_surround_maps, axis=0))
    
    return combined_map