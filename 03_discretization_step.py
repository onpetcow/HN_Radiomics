# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:21:50 2022

@author: owenpaetkau
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
discrete_images = 'H:/HN_RadDosiomics/2_output/03_discretization_step/'

# Identify organs to import.
soi = ['Pharynx_Constr_F', 'PharynxConst_S_F', 'PharynxConst_M_F',
       'PharynxConst_I_F', 'PharynxCrico',
       'Parotid_RT', 'Parotid_LT']

# Isodose Lines.
iso_lines = [70, 80, 90, 95]

subset = os.listdir(single_data)
#subset = ['zHNART_032', 'zHNART_035', 'zHNART_038', 'zHNART_052', 'zHNART_056']

for pt in subset:
    print(45*'-')
    print(f'Analyzing patient {pt}...')
    print(45*'-')
    
    in_dir = os.path.join(align_images, pt) 
    out_dir = os.path.join(discrete_images, pt)

    plan_list = os.listdir(in_dir)
    
    flag = make_folder(out_dir)

    for plan in plan_list:
        in_wd = f'{in_dir}/{plan}'
        out_wd = f'{out_dir}/{plan}'
        
        flag = make_folder(out_wd)
        
        if flag == True:
            continue
        
        # Load CT and dose files.
        ct_path = in_wd + '/ct_scan.npy'
        dose_path = in_wd + '/dose_distribution.npy'
        
        ct = np.load(ct_path)
        dose = np.load(dose_path)  
        
        #------------------------------------------------------------------
        # Perform range reduction, remove air and bone.
        #------------------------------------------------------------------
        bone_lim = 250
        air_lim = -500
        
        for organ in soi:
            mask_path = f'{in_wd}/mask_{organ}.npy'
            out = f'{out_wd}/mask_{organ}.npy'
            
            if os.path.exists(mask_path) == False:
                print(f'...no {organ} mask available.')
                continue
            
            mask = np.load(mask_path)
            mask = np.where((ct > air_lim) & 
                            (ct < bone_lim) & 
                            (mask.astype(bool) == True),
                            True, False)
            
            np.save(out, mask.astype('int32'))      
            
        #------------------------------------------------------------------
        # Create masks for different isodose lines based on dose distribution.
        #------------------------------------------------------------------
        # Assume the prescription is 70 Gy. Isodose lines calculated off this.
        Tx = 70
        
        for iso in iso_lines:
            out = f'{out_wd}/mask_{iso}isodose.npy'
            mask = dose > (iso * Tx / 100)
            
            np.save(out, mask.astype('int32'))
        
        #------------------------------------------------------------------
        # Perform discretization on CT and dose arrays.
        #------------------------------------------------------------------
        # Using fixed bin size method.
        bin_size = 25        
        
        # For CT, window between air and bone HU limits.
        _min, _max = air_lim, bone_lim
        #bin_size = (_max - _min) / bin_num
        bins = np.arange(_min - bin_size, _max, bin_size)
        
        ct = window_image(ct, _min, _max)
        ct = np.digitize(ct, bins) * bin_size + _min
        np.save(f'{out_wd}/ct_scan.npy', ct)
        
        # For dose, round up and force minimum to 0.
        _min, _max = 0, np.ceil(dose.max())
        bin_size = bin_size / 100
        bins = np.arange(_min - bin_size, _max, bin_size)
        
        dose = window_image(dose, _min, _max)
        dose = np.digitize(dose, bins) * bin_size + _min 
        np.save(f'{out_wd}/dose_distribution.npy', dose)

        
        
        

        
        
        