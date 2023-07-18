#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:15:16 2023

@author: evovision
\
Description:
    Minimal Working Code that loads the image paths of a dataset passes an 
    image through the avc-doc-p model with predefined functions.
    
"""

import sys
import os
# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Paths to append
sys.path.append(os.path.join(parent_dir, 'Utilities'))
sys.path.append(os.path.join(parent_dir, 'Brain Programming'))

import utilities as utils
from avc_doc import AVC

if __name__ == '__main__':
    # Data loading ------------------------------------------------------------
    dataset_name = 'TJ-Trash Dataset'
    dataset_path = os.path.join(parent_dir, '../Datasets/', dataset_name)

    data_splits=[0.6, 0.2, 0.2]
    train_set, val_set, test_set = utils.load_data(dataset_path, data_splits=data_splits)
    
    
    # Initialize avc model ----------------------------------------------------
    avc_model = AVC()
    
    
    # Individual images visualization -----------------------------------------
    img_id = 150
    img = test_set[img_id]
    label = test_set.targets[img_id]
    
    utils.display_image(img, label)
    
    avc_model.run(img, verbose=1)
