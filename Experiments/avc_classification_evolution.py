#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:37:06 2023

@author: evovision
"""
import sys

import os

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Paths to append
sys.path.append(os.path.join(parent_dir, 'Evolution'))
sys.path.append(os.path.join(parent_dir, 'Utilities'))
sys.path.append(os.path.join(parent_dir, 'Brain Programming'))


import evolutionary_process as evo_proc
import functions_set as bp_func
import utilities as bp_utils
from avc_doc import AVC


"""# Main code"""

if __name__ == '__main__':
    # Data loading ---------------------------------------------------------------
    dataset_name = 'TJ-Trash Dataset'
    experiment_name = 'AVC-DOC Sequential-Trees'
    
    # Data loading ---------------------------------------------------------------
    dataset_path = os.path.join(parent_dir, '../Datasets/', dataset_name)
    data_splits=[0.6, 0.2, 0.2]
    train_set, val_set, test_set = bp_utils.load_data(dataset_path, data_splits=data_splits)
      
    # Save File Paths ------------------------------------------------------------
    results_classes_dir = 'Results/'+ dataset_name +'/' + experiment_name
    results_metrics_dir = results_classes_dir + '/Metrics'
    results_model_dir = results_classes_dir + '/Models'
      
    bp_utils.check_create_dir(results_metrics_dir)
    bp_utils.check_create_dir(results_model_dir)
      
    # Load AVC model -------------------------------------------------------------
    avc_model = AVC()
      
    # Evolutionary parameters ----------------------------------------------------
    pop_size=10
    max_gen=10
    n_experiments = 30
      
    terminal_set_feature = ['r','g','b','h','s','v']
    terminal_set_normal = ['am']
      
    color_function_set = [bp_func.ip_oppon, bp_func.imabs, bp_func.ip_imsubtract, 
                          bp_func.ip_immultiply, bp_func.ip_imdivide, bp_func.ip_imadd, 
                          bp_func.supremum, bp_func.infimum, bp_func.ip_Gauss_15, bp_func.ip_Gauss_05, 
                          bp_func.ip_Gauss_2, bp_func.ip_Gauss_1]
      
    orientation_function_set = [bp_func.imabs, bp_func.ip_imsubtract, bp_func.ip_immultiply, 
                                bp_func.ip_imdivide, bp_func.ip_imadd, bp_func.supremum, bp_func.infimum,
                                bp_func.ip_Sobely, bp_func.ip_Sobelx, bp_func.ip_LoG, bp_func.ip_GaussDy_15, 
                                bp_func.ip_GaussDy_05, bp_func.ip_GaussDy_2, bp_func.ip_GaussDy_1, bp_func.ip_GaussDx_15, bp_func.ip_GaussDx_05, 
                                bp_func.ip_GaussDx_2, bp_func.ip_GaussDx_1, 
                                bp_func.ip_Gauss_15, bp_func.ip_Gauss_05, bp_func.ip_Gauss_2, bp_func.ip_Gauss_1, 
                                bp_func.ip_Gabor7, bp_func.ip_Gabor6, bp_func.ip_Gabor5,
                                bp_func.ip_Gabor4, bp_func.ip_Gabor3, bp_func.ip_Gabor2, bp_func.ip_Gabor1, bp_func.ip_Gabor0]
    
    shape_function_set = [bp_func.topHat, bp_func.skeletonShp, bp_func.perimeterShp, 
                          bp_func.openMph_5, bp_func.erodeSqr_5, bp_func.erodeDsk_5, 
                          bp_func.erodeDmnd_5, bp_func.dilateSqr_5, bp_func.dilateDsk_5, 
                          bp_func.dilateDmnd_5, bp_func.closeMph_5, bp_func.bottomHat_5, 
                          bp_func.hitmissSqr_5, bp_func.hitmissDsk_5, bp_func.hitmissDmnd_5, 
                          bp_func.ip_imsubtract, bp_func.ip_immultiply, bp_func.ip_imdivide, 
                          bp_func.ip_imadd, bp_func.supremum, bp_func.infimum, bp_func.ip_Gauss_15, 
                          bp_func.ip_Gauss_05, bp_func.ip_Gauss_2, bp_func.ip_Gauss_1]
      
    normal_function_set = [bp_func.ip_Exp, bp_func.ip_Logarithm, bp_func.ip_Logarithm10, bp_func.ip_Logarithm2,
                           bp_func.ip_Sqr, bp_func.ip_Sqrt, bp_func.ip_imsubtract,
                           bp_func.ip_immultiply, bp_func.ip_imdivide, bp_func.ip_imadd, bp_func.supremum, bp_func.infimum]
      
    evo_func_set = {'color': color_function_set, 'orientation': orientation_function_set,
                    'shape': shape_function_set, 'normalization': normal_function_set}
      
    evo_terminal_set = {'color': terminal_set_feature, 'orientation': terminal_set_feature, 
                        'shape': terminal_set_feature, 'normalization': terminal_set_normal}
      
    for i in range(1, n_experiments+1):
      metrics_file = results_metrics_dir + '/' + str(i) + '.txt'
      model_file = results_model_dir + '/' + str(i) + '.ind'
      
      for evo_key in evo_func_set.keys():
        evaluator = evo_proc.AVCEvaluator(train_set, val_set, test_set, avc_model, evo_key)
      
        avc_model = evo_proc.evolution_process(evaluator, pop_size=pop_size, max_gen=max_gen, 
                                              function_set=evo_func_set[evo_key], terminal_set=evo_terminal_set[evo_key])
        
      evo_proc.evaluate_best_avc(train_set, val_set, test_set, avc_model, evo_func_set.keys(), metrics_file, model_file)
      
      print('**********************************')