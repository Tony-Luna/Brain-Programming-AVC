#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:27:56 2023

@author: evovision
"""

import sys
import os

import numpy as np
from skimage.color import rgb2hsv
from skimage.transform import pyramid_gaussian

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Paths to append
sys.path.append(os.path.join(parent_dir, 'Evolution'))
sys.path.append(os.path.join(parent_dir, 'Utilities'))

import utilities as bp_utils
import activation_functions as activ_funcs
import description_functions as desc_funcs
import functions_set as bp_func

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from tqdm import tqdm
from joblib import Parallel, delayed

class AVC:
    def __init__(self):
        self.evo_func = {'color': self.color_func, 'orientation': self.ori_func, 
                         'shape': self.shape_func, 'intensity': self.int_func,
                         'normalization': self.norm_func}
        
        self.evo_keys = list(self.evo_func.keys())
        
        self.pool_block = (2,2,1)
        self.pool_func = np.sum
        
        # Define classifier parameters
        classifier = MLPClassifier(hidden_layer_sizes=(512,256,128,64,32,), 
                                  solver='lbfgs', 
                                  batch_size=5, 
                                  learning_rate='adaptive', 
                                  max_iter=100)

        
        
        self.clf = make_pipeline(StandardScaler(), classifier)
        
        # Specific template parameters of AVC-DOC-P Model
        self.activation = activ_funcs.doc_activation
        self.description = desc_funcs.pooling_vector

    def run(self, img, verbose=0):
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        if img.max() > 1:
            img = (img/255).astype(np.float32)
           
        # Generate image pyramid considering a min size of 16
        min_size = min(img.shape[:2])
        max_lvl = int(np.floor(np.log2(min_size/16)))
        img_pyr = list(pyramid_gaussian(img, max_lvl,channel_axis=2))

        
        feat_maps = self.extract_features(img_pyr)
        activations = self.gen_process_maps(self.activation, feat_maps)
        norm_maps = self.gen_process_maps(self.evo_func['normalization'], activations)
      
        descriptor_vector = self.description(norm_maps)
      
        if verbose:
            self.maps = {'features': feat_maps,
                         'activation': activations,
                         'normalization': norm_maps}
            self.display_maps()
      
        return descriptor_vector

    def extract_features(self, img_pyr):
        feat_maps = dict.fromkeys(self.evo_keys[:-1])
        
        for img in img_pyr:
            img_hsv = rgb2hsv(img)
          
            r,g,b = [img[:,:,i] for i in range(img.shape[-1])]
            h,s,v = [img_hsv[:,:,i] for i in range(img_hsv.shape[-1])]
        
          
            for key in feat_maps.keys():
                dim_func = self.evo_func[key]
                
                if feat_maps[key] is None:
                    if callable(dim_func):
                        feat_maps[key] = [dim_func(r,g,b,h,s,v)]
                    else:
                        feat_maps[key] = [dim_func.execute(r=r,g=g,b=b,h=h,s=s,v=v)]
                else:
                    if callable(dim_func):
                        feat_maps[key].append(dim_func(r,g,b,h,s,v))
                    else:
                        feat_maps[key].append(dim_func.execute(r=r,g=g,b=b,h=h,s=s,v=v))
      
        return feat_maps

    def gen_process_maps(self, process_func, maps_dict):
        proc_maps = dict.fromkeys(maps_dict)
        for key in proc_maps.keys():
            if callable(process_func):
                proc_maps[key] = process_func(maps_dict[key])
            else:
                proc_maps[key] = process_func.execute(am=maps_dict[key])
                
        return proc_maps
    
    def display_maps(self):
        for lvl in self.maps:
            print('********{}**********'.format(lvl))
            lvl_maps_dict = self.maps[lvl]
            
            for dim in lvl_maps_dict:
                print('------{}-------'.format(dim))
                dim_maps = lvl_maps_dict[dim]
                
                if isinstance(dim_maps, list):  
                    dim_labels = [str(i) for i in range(len(dim_maps))]
                    bp_utils.display_image_row(dim_maps, dim_labels)    
                else:
                    bp_utils.display_image(dim_maps,'')
        

# Functions to train the classifier of the model -------------------------------
    def train_classifier(self, dataset):
        x_train, y_train = self.process_data(dataset)
        self.clf.fit(x_train, y_train)
        
        # Get prediction and compute balanced accuracy
        y_pred = self.clf.predict(x_train)      
        train_acc = balanced_accuracy_score(y_train, y_pred)
      
        return train_acc
    
    def evaluate_classifier(self, dataset):
        x_eval, y_eval = self.process_data(dataset)
        y_pred = self.clf.predict(x_eval)  # Get predictions
        
        # Compute balanced accuracy
        eval_acc = balanced_accuracy_score(y_eval, y_pred)
    
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_eval, y_pred)
    
        return eval_acc, conf_matrix
  
    
    def process_data(self, dataset):
        n_jobs = -1  # Use all available CPU cores
        x_list = Parallel(n_jobs=n_jobs)(delayed(self.run)(img) for img in tqdm(dataset))
        x_data = np.nan_to_num(np.vstack(x_list).astype(np.float32))
        y_data = np.array(dataset.targets, dtype=int)
        
        return x_data, y_data

# Default functions for evolutionary processes ---------------------------------

    def color_func(self,r,g,b,h,s,v):
        rg_opp = bp_func.ip_oppon(r,g,v)
        rb_opp = bp_func.ip_oppon(r,b,v)
        gb_opp = bp_func.ip_oppon(g,b,v)
      
        out = np.maximum(rg_opp, rb_opp, gb_opp)
        
      
        return bp_utils.linear_normalization(out)
    
    def ori_func(self,r,g,b,h,s,v):
        gabor_7 = bp_func.ip_Gabor7(v)
      
        out = gabor_7
        
      
        return bp_utils.linear_normalization(out)
    
    def shape_func(self,r,g,b,h,s,v):
        skeleton = bp_func.skeletonShp(v)
      
        out = skeleton
      
        return bp_utils.linear_normalization(out)
    
    def int_func(self,r,g,b,h,s,v): 
        return bp_utils.linear_normalization(v)
    
    def norm_func(self,img):
        out = bp_func.kPwr(img, 1.5)
        
        return out