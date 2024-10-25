# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:24:53 2022

@author: owenpaetkau

Steps required here: 
    1) Import CT scan,
    2) import and resample dose,
    3) import and resample structure set,
    4) save existing files as mask, dose image and ct image.
    
"""

import os
import time

import numpy as np
import nrrd

import SimpleITK as sitk
import six

from radiomics import featureextractor, getTestCase

from dicomMethods import *

from ReadDicomMethods import *
from rt_utils import RTStructBuilder
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

sort_data = 'H:/HN_RadDosiomics/2_output/01_sort_dicom/'
single_data = 'H:/HN_RadDosiomics/0_data/single_plans/'
align_images = 'H:/HN_RadDosiomics/2_output/02_align_images/07_AllStructures/'

# Identify organs to import.

soi = ['Pharynx_Constr_F', 'PharynxConst_S_F', 'PharynxConst_M_F',
       'PharynxConst_I_F', 'PharynxCrico']#,
       #'Parotid_RT', 'Parotid_LT']

subset = os.listdir(single_data)
#subset = ['zHNART_032', 'zHNART_035', 'zHNART_038', 'zHNART_052', 'zHNART_056']

for pt in subset[30:]:
    print(45*'-')
    print(f'Analyzing patient {pt}...')
    print(45*'-')
    
    in_dir = os.path.join(single_data, pt)
    out_dir = os.path.join(align_images, pt) 

    plan_list = os.listdir(in_dir)
    
    flag = make_folder(out_dir)

    for plan in plan_list:
        in_wd = f'{in_dir}/{plan}'
        out_wd = f'{out_dir}/{plan}'
        
        flag = make_folder(out_wd)
        
        #if flag == True:
        #    continue
        
        #----------------------------------------------------------------------
        # Import Dose and CT arrays.
        #----------------------------------------------------------------------
        # This is first to allow for the structure set to be looped through
        # if more than one structure of interest needs to be stored as a mask.
        # This file will perform following operations on dose and CT:
            # 1) Import files,
            # 2) Convert to numpy arrays,
            # 3) Resample to same pixel size as CT scan,
            # 4) Reshape to [512, 512, max(num_slices)],
            # 5) Align the dose to ct scan,
            # 6) Save dose and ct files.
            
        # Import and process CT.
        out = f'{out_wd}/ct_scan.npy'
        
        if os.path.exists(out) == False :
            ct, ct_or, ct_res = process_ct_scan(in_wd)
            np.save(out, ct)
            np.save(f'{out_wd}/ct_or.npy', ct_or)
            np.save(f'{out_wd}/ct_res.npy', ct_res)
        else :
            ct_or = np.load(f'{out_wd}/ct_or.npy')
            ct_res = np.load(f'{out_wd}/ct_res.npy')
            ct = np.load(out)
        
        # Import and process dose.
        out = f'{out_wd}/dose_distribution.npy'
        
        if os.path.exists(out) == False:
            dose, dose_or, dose_res = process_dose(in_wd, ct_or)
            np.save(out, dose)     
        else :
            dose = np.load(out)            
            
        #----------------------------------------------------------------------
        # Import Structures as masks of interest
        #----------------------------------------------------------------------
        # Loop through all structures of interest and perform operations:
            # 1) Import structure,
            # 2) Convert to numerical mask,
            # 3) Resample to same pixel size as CT scan,
            # 4) Reshape to [512, 512, max(num_slices)],
            # 5) Crop to dose bounding box,
            # 6) Save mask file.
           
        # Import structure set.
        print('Import structure set...')
        structure_set, flag = load_structures(in_wd, soi)
        
        for ii, organ in enumerate(soi):
            out = f'{out_wd}/mask_{organ}.npy'
            
            if (os.path.exists(out) == False) & (flag[ii] == True):
                print(f'...working on {organ}.')
                mask, mask_origin, mask_res = structure_mask_new(structure_set[ii], ct_or, ct_res, ct.shape)
                #mask, mask_or, mask_res = process_structure(structure_set[ii], ct_or, ct_res, ct.shape)
                np.save(out, mask.astype(int)) 


    

        
