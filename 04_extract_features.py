# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:40:20 2022

@author: owenpaetkau

This file will export radiomics features from given CT and dose scans.
It will import images and mask from result of aligning images.

Should import the .nrrd files, as they are compatible with SimpleITK.
"""

import os
import numpy as np
import pandas as pd
import time
import re

import nrrd
import SimpleITK as sitk
import six

from radiomics import featureextractor, getTestCase
from ReadDicomMethods import *

def csv_exists(wd, plan = None):
    if os.path.isfile(wd):        
        if plan != None : 
            df = pd.read_csv(wd)
            if plan in np.unique(df.plan) :
                return True
            else :
                return False
            
        else :
            return True
    else :
        return False
        

# Define directories for import and export.
align_imgs = 'H:/HN_RadDosiomics/2_output/02_align_images/06_New Structure Method/'
discret_step = 'H:/HN_RadDosiomics/2_output/03_discretization_step/'
ext_feats = 'H:/HN_RadDosiomics/2_output/04_extract_features/'
ext_feats = 'H:/HN_RadDosiomics/2_output/04_extract_features/weekly_wavelet/'

# Import radiomic feature parameters
original = 'H:/HN_RadDosiomics/1_code/rad_original.yaml'
wavelet = 'H:/HN_RadDosiomics/1_code/rad_wavelet.yaml'

# Define the structures of interest.
soi = ['Pharynx_Constr_F', 'PharynxConst_S_F', 'PharynxConst_M_F',
       'PharynxConst_I_F', 'PharynxCrico']

# Isodose Lines.
iso_lines = [70, 80, 90, 95]
iso_names = [str(ii) + 'isodose' for ii in iso_lines]

# Define subset of patients identified.
subset = os.listdir(discret_step)
#subset = ['zHNART_032', 'zHNART_035', 'zHNART_038', 'zHNART_052', 'zHNART_056']

for pt in subset:
    data = []
    print(30*'-')
    print(f'Analyzing patient {pt}...')
    print(30*'-')
    start_pt = time.time()
    
    in_dir = os.path.join(discret_step, pt)
    
    plans = os.listdir(in_dir)
    
    first_plan = 'ORIG. PLAN'
    last_plan = plans[-1]
    plan_list = [first_plan, last_plan] # First and last only.
    
    plan_list = plans # All plans.

    # Determine if file exists and change the plan list if they have already
    # been imported.
    flag = csv_exists(ext_feats + pt + '.csv')
    if flag == True:
        df = pd.read_csv(ext_feats + pt + '.csv')
        import_plans = list(np.unique(df.plan))
        print(f'...patient already exists with plans {import_plans}.')
        
        #data = df.to_dict('records')
        data = df.values.tolist()
        for plan in import_plans:
            plan_list.remove(plan)
        
        # Exit patient loop if there are no new plans to import.
        if len(plan_list) == 0 :
            continue
            
    # Determine week or index number of plan.
    plan_index = [plans.index(plan) for plan in plan_list]
    
    for plan, ii in zip(plan_list, plan_index):
        print(f'...pulling features from {plan}.')
        in_wd = f'{in_dir}/{plan}/'
        
        # Load CT and dose files.
        ct_path = in_wd + 'ct_scan.npy'
        dose_path = in_wd + 'dose_distribution.npy'

        # Find plan number. Started this at pt zHNART_085.
        plan_nums = re.findall(r'\d+', plan)
        if len(plan_nums) == 0:
            week = 0
        else :
            week = int(int(plan_nums[0]) / 5) + 1
        
        # Set-up loop for dosiomic and radiomic features.
        lists = zip([dose_path, ct_path],['dosiomic', 'radiomic'], 
                    #[original, original])
                    [wavelet, wavelet])
        
        for path, name, params in lists:
            
            img = sitk.GetImageFromArray(np.load(path).astype('float32'))
            
            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            
            if name == 'dosiomic':
                mask_list = soi + iso_names
            else : 
                mask_list = soi

            for mask_name in mask_list:
                mask_path = f'{in_wd}mask_{mask_name}.npy'
                mask = sitk.GetImageFromArray(np.load(mask_path).astype('int32'))
                
                print(f'......pulling {name} features from {mask_name}.')
                result = extractor.execute(img, mask, label = 1)
                
                dic = {}
                dic['patient'] = pt
                dic['plan'] = plan
                dic['plan_num'] = ii
                dic['plan_week'] = week
                dic['type'] = name
                dic['structure'] = mask_name
                
                for key, val in six.iteritems(result):
                    if type(val) == np.ndarray :  
                        dic[key] = float(val)
                    else :
                        dic[key] = val                    
   
                data_array = [vals for vals in list(dic.values())]
                name_array = [keys for keys in list(dic.keys())]
                data.append(data_array)
                
                del mask, result
                
            data_df = pd.DataFrame(data, columns = name_array)
            data_df.to_csv(f'{ext_feats}{pt}.csv', header = True, index = False)
    
    end_pt = time.time()
    length_pt = end_pt - start_pt
    print(f'Time for {len(plan_list)} plans with {len(soi)} structures: ') 
    print(f'...{length_pt:.2f} seconds or {length_pt / 60 :.2f} minutes.')


