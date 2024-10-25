# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:21:37 2022

@author: owenpaetkau

This script is to import structure sets, dose files and CT scans from a patient
path. It is able to create masks from the structure sets as well as plot 
features overlapping each other.

This code summarizes work from Fletcher Barrett, Kailyn Stenhouse and Owen Paetkau
to create a set of commands to resample, resize and align the CT, dose and structures
of interest for research purposes.
"""
from dicomMethods import *

# DICOM LIBRARIES
import pydicom
from dicompylercore.dicomparser import DicomParser

# ARRAY DEFINITION LIBRARIES
import numpy as np
import pandas as pd

# PLOTTING LIBRARIES
import matplotlib

# FILE MANAGEMENT LIBRARIES
import glob
import os

# DATA PROCESSING LIBRARIES
from scipy.ndimage import interpolation
from collections import deque


#------------------------------------------------------------------------------
#-------------------- RESAMPLING & ALIGMNENT METHODS --------------------------
#------------------------------------------------------------------------------ 

def get_resolution(path, dicom_type = 'ct'):
    '''
    Get resolution from a CT slice within the given directory. Can also ask for
    resolution of the dose file in the path.

    Parameters
    ----------
    path : string
        Path to the directory of interest.
    dicom_type : string, optional
        Define either 'ct' or 'dose' for the appropriate resolution of interest.
        The default is 'ct'.

    Returns
    -------
    np.array
        Return the resolution in array format [x,y,z].

    '''
    if dicom_type == 'ct':
        file = pydicom.read_file(glob.glob(path + '/CT.*')[0])
    elif dicom_type == 'dose':
        file = pydicom.read_file(glob.glob(path + '/RD.*')[0])
    
    ps = list(file.PixelSpacing) # Pulls out pixel spacing as [X,Y].
    thick = list([file.SliceThickness]) # Pulls out thickness or [Z].
    
    resolution = ps + thick # Forms into list of [X,Y,Z] resolution.
    
    return np.array(resolution)   
    
def resample(image, initial_res, final_res = np.array([1.0, 1.0, 2.0])): 
    '''
    Resampled 3D dose, ct or structure image according to pixel spacing and slice
    thickness to project it into a 1 mm x 1 mm grid on 2 mm slice thickness.

    Parameters
    ----------
    image : numpy.ndarray
        Dose file from pixel.array or imported ct scan after get_pixels_hu.
    initial_res: np.array
        Resolution of ct scan in [x,y,z].
    final_res: np.array
        Goal resolution to resample to. The default is [1.0, 1.0, 2.0].

    Returns
    -------
    resampled_image : numpy.ndarray
        Resampled array in 1 mm x 1 mm x 1 mm grid.

    '''
    
    ratio = np.array(initial_res / final_res, dtype = np.float32)

    # Changed this from 2d interpolation to 3d interpolation.
    resampled_image = interpolation.zoom(image, ratio)
    
    return resampled_image, final_res

def resize(image, new_dim = [750,750,750], crop = [512,512,512]):
    '''
    Resize the dose or ct file to a common size, after resampling and 
    before applying registration shifts. Pads at end of the array
    so the top left pixel remains aligned so shifts are applied correctly.

    Parameters
    ----------
    image : numpy.ndarray
        Input 3D array, typically a dose or ct file.
    new_dim : TYPE, optional
        Temporary image pad size. The default is [600,600,500].
    crop : TYPE, optional
        Final image cropped size. The default is [512,512,300].

    Returns
    -------
    final_image : numpy.ndarray
        Resized dose or ct file.

    '''   

    # Error to find if goal array is too small.
    if any (np.array(image.shape) > new_dim):
        print(f'Error... Dimensions larger than expected, increased new dimension size {np.array(image.shape)}.')
        indices = np.where(np.array(image.shape) > new_dim)[0]
        for index in indices:
            new_dim[index] = np.array(image.shape)[index]

    dim_dif = new_dim - np.array(image.shape)
    
    # Pad step to get to new_dim size.
    pad = ((0,dim_dif[0]),(0,dim_dif[1]),(0,dim_dif[2]))
    temp_image = np.pad(image,pad_width = pad, mode = 'constant', constant_values = 0,)

    # Crop image down to cropped size.
    final_image = temp_image[:crop[0],:crop[1],:crop[2]]
    
    return final_image

def alignment_shift(img, shift, resolution = [1.0, 1.0, 2.0]):
    '''     
    Apply the translational shifts for all three axis. 
    
    This will wrap around if the shift exceeds the size of the array.
    No need to consider pixel_info as they have been resampled into 
    1mm x 1mm x 1mm grids using resample method.   
    
    Need the resize_image applied first. If not, shifts may wrap around.
    
    Other change was to swap Z_shift and X_shift operations. I think this
    works because I swapped the axis before registration, while
    Fletcher did this operation afterwards.

    Parameters
    ----------
    img : numpy.ndarray
        Dose or CT file after being processed by resize_image.
    extra_shift : numpy.ndarray
        Shift from the alignment of dose and ct files, and aligning
        the .ImagePositionPatient to reference.

    Returns
    -------
    numpy.ndarray
        CT or dose file with the appropriate shifts applied.

    '''
    shift = shift / resolution
    
    X_shift =  int(np.round(shift[0]))
    Y_shift =  int(np.round(shift[1]))
    Z_shift =  int(np.round(shift[2]))
    #print (f'Shifted by {X_shift} {Y_shift} {Z_shift}.')
    
    l3 = []
    for k in range(len(img)):
        l1 = []
        for i in range(img[0].shape[0]):
            items = deque(img[k][i])
            items.rotate(Z_shift) # Swapped to Z from previous.
            l1.append(items)
        
        temp = np.array(l1, dtype = np.float32)
        l2 = []
        for j in range(img[0].shape[1]):
            test = np.transpose(temp)
            items = deque(test[j])
            items.rotate(Y_shift)
            l2.append(items)
        
        temp2 = np.array(l2, dtype = np.float32)
        l3.append(np.transpose(temp2))
    
    items = deque(l3)
    items.rotate(X_shift) # Swapped to X from previous.
    return np.array(items, dtype = np.float32)


def new_align(img, shift, resolution = np.array([1.0,1.0,2.0])):
    '''
    Align the image to another image based on a patient-origin based shift.
    Subtract the origin (top left most pixel according to DICOM) of the location
    from the array being shifted. This function will perform the operation.

    Parameters
    ----------
    img : np.ndarray
        CT, dose or structure mask to shift.
    shift : np.array
        Array of difference between current image origin and location
            dif = img_origin - goal_origin
    resolution : np.array, optional
        Resolution of array, used to adjust shift accordingly.
        The default is np.array([1.0,1.0,2.0]).

    Returns
    -------
    np.ndarray
        Shifted image to goal coordinates.
    '''
    
    shift = np.round(shift / resolution).astype(int)
    img = np.roll(img, shift, axis = (0,1,2))
    
    return np.array(img, dtype = np.float32)

def window_image(image, img_min, img_max):
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

def bounding_box(image):
    '''
    Create bounding box around existing pixels within a numpy.array.

    Parameters
    ----------
    image : numpy.ndarray
        Array of any size with a given image.

    Returns
    -------
    bounds : list
        List of paired [min, max] for each dimension of the image.

    '''
    val_exists = np.where(image)
    
    bounds = []
    for ii in range(len(val_exists)):
        p1, p2 = val_exists[ii].min(), val_exists[ii].max()
        bounds.append([p1,p2])
        
    return bounds
    


#------------------------------------------------------------------------------
#-------------------------- CT SCAN METHODS -----------------------------------
#------------------------------------------------------------------------------  
        
def load_slices(path):
    '''
    Load the CT slices from a directory. The directory must contain only
    CT slices!

    Parameters
    ----------
    path : string
        Directory leading to the CT files.

    Returns
    -------
    slices : list
        List of imported ct slices as pydicom format.

    '''
    #slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = [pydicom.read_file(file) for file in glob.glob(path + '/CT.*')]
    
    if len(slices) == 0:
        print(f'No CT DICOM files in the path: {path}')
    
    # Inverted the sorting to match the dose file.
    slices.sort(reverse = True, key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def hu_array(scans):
    '''
    Take in the list of scans from load_scan method and scale them using
    the HU values. Need to do this slice by slice, as there are different
    HU calibration values for each slice in DICOM series.
    
    Natalylun states that the 
    https://groups.google.com/g/comp.soft-sys.matlab/c/4h3mjk6OGJ8

    Parameters
    ----------
    scans : list
        List of Pydicom scans from load_scan method.

    Returns
    -------
    TYPE : numpy.ndarray
        Output an array, scaled with HU units. 
        The shape of output is [Z,X,Y].

    '''
    slices = []
    
    # Convert pixel array to HU for each slice.
    for s in scans:
        # Method 1 - Homemade
        array = s.pixel_array * s.RescaleSlope + s.RescaleIntercept
        
        # Method 2 - PyDICOM
        #array = pydicom.pixel_data_handlers.util.apply_modality_lut(s.pixel_array, s)
        
        # Method 3 - Considering window size. From Natalylun in Google Groups...
        # https://groups.google.com/g/comp.soft-sys.matlab/c/4h3mjk6OGJ8
        #i = s.BitsAllocated
        #w_width = s.WindowWidth
        #w_left = s.WindowCenter - w_width
        
        #array = s.pixel_array * s.RescaleSlope + s.RescaleIntercept
        #array = ((array - w_left) * pow(2, i-1)) / w_width
                
        slices.append(array)
        
    image_array = np.stack(slices)
    
    return np.array(image_array, dtype=np.int16)

def load_ct_scan(path):
    '''
    Complete method to load slices, convert to HU array, and return pixel spacing/thickness.

    Parameters
    ----------
    path : string
        Path to CT DICOM files.

    Returns
    -------
    ct_scan : numpy.array
        Numpy array of HU values corresponding to the ct scan. Returns [x,y,z] array.
    origin : numpy.array
        Array of origin, or top left most pixel, in [x,y,z].  
    resolution : numpy.array
        Array of pixel spacing/thickness in [x,y,z] resolution.

    '''
    ct_slices = load_slices(path)
    ct_scan = np.swapaxes(hu_array(ct_slices),0,-1)
  
    origin = ct_slices[0].ImagePositionPatient
    resolution = get_resolution(path, dicom_type= 'ct')
    
    return ct_scan, origin, resolution

def process_ct_scan(path, res_f=np.array([1.0,1.0,2.0]), _dim = [700,700,600], _crop = [512,512,300]):
    '''
    Process the CT scan from directory. Resample and resize the image to common shape.

    Parameters
    ----------
    path : string
        String of the directory of interest.
    res_f : np.array, optional
        Final resolution to resample to from built in resolution. The default is [1.0,1.0,2.0].
    _dim : list, optional
        Shape of goal dimensions, larger than required. The default is [600,600,600].
    _crop : list, optional
        End shape of interest for the array. The default is [512,512,300].

    Returns
    -------
    ct : np.ndarray
        Array of HU values in CT scan.
    ct_origin : np.array
        Array of origin coordinates - top left most pixel in patient coordinates.
    resolution : np.array
        Array of final resolution values for [x,y,z] of the array.

    '''
    print('Import CT scan...')
    ct, ct_origin, ct_res = load_ct_scan(path)
    
    print('...resize and resample CT scan.')
    ct, resolution = resample(ct, ct_res, res_f)
    ct = resize(ct, new_dim = _dim, crop = _crop)

    return ct, ct_origin, resolution

#------------------------------------------------------------------------------
#--------------------- DOSE DISTRIBUTION METHODS ------------------------------
#------------------------------------------------------------------------------
def load_dose(path):
    '''
    Load all of the dose files in a single path, and sum the dose arrays together.
    Biggest use is when a patient has VMAT arcs in their dose distribution.

    Parameters
    ----------
    path : string
        Path to the dose files.

    Returns
    -------
    dose_array : numpy.ndarray
        Summed dose array after importing all of the dose files.
    origin : numpy.array
        Array coordinates of top left most pixel in patient coordinate space.
    resolution : numpy.array
        Grid size from pixel spacing and thickness using DICOM header information.
    '''
    # Read list of files in folder.
    file_paths = glob.glob(path + '/RD.*')

    # Print error if there are no available dose files to be read.
    if len(file_paths) == 0:
        print(f'There are no dose files in the folder: {path}.')
        return
    
    dose_array = 0
    for ii, file in enumerate(file_paths):
        # Read file and summate the dose distributions of multiple files.
        dose = pydicom.read_file(file)
        dose_array += dose.pixel_array * dose.DoseGridScaling
        
        # Define once the resolution and origin.
        if ii == 0:
            # Define pixel spacing and thickness from dose file.
            dose_ps = dose.PixelSpacing
            dose_thick = dose.GridFrameOffsetVector[1] - dose.GridFrameOffsetVector[0] 
            resolution = np.array([dose_ps[0], dose_ps[1], dose_thick])
            
            origin = np.array(dose.ImagePositionPatient)
            
    dose = np.swapaxes(dose_array, 0, -1)
        
    return dose.astype(np.float32), origin, resolution

def process_dose(path, ct_origin, res_f = np.array([1.0,1.0,2.0]), _dim = [700,600,600], _crop = [512,512,300]):
    '''
    Processing steps to resample, resize and align the dose distribution with
    the CT scan.

    Parameters
    ----------
    path : string
        Path to the folder containing patient DICOM files.
    ct_origin : np.array
        Origin of CT scan from folder - top left most pixel in patient coordinates.
    res_f : np.array, optional
        Final resolution of the dose distribution.
        The default is np.array([1.0,1.0,2.0]).
    _dim : list, optional
        List of dimensions, larger than required, to ensure all data fit.
        The default is [700,600,600].
    _crop : list, optional
        Goal cropped image in [x,y,z]. The default is [512,512,300].

    Returns
    -------
    dose : np.array
        Array of dose distribution.
    dose_origin : np.array
        Origin of dose distribution - top left most pixel in patient coordinates.
    resolution : np.array
        New resolution of the returned dose distribution array.

    '''
    print('Import dose distribution...')
    dose, dose_origin, dose_res = load_dose(path)
    
    print('...resize, resample and shift dose distribution.')
    dose, resolution = resample(dose, dose_res, res_f)
    dose = resize(dose, new_dim = _dim, crop = _crop)
    dose = new_align(dose, dose_origin - ct_origin, resolution)
    
    return dose, dose_origin, resolution

#------------------------------------------------------------------------------
#------------------------ STRUCTURE METHODS -----------------------------------
#------------------------------------------------------------------------------        

def load_structures(path, structure_names_list):
    '''
    Read in the structures listed using the DicomParser method from the 
    package dicompylercore.dicomparser.

    Parameters
    ----------
    path : string
        Path to file with the CT scans, structure set, and dose files.
    stucture_names_list : list
        List of structures as named in the structure set files.

    Returns
    -------
    structure_set : list
        List of structure set dictionary of triangulation point coordinates.

    '''
    structure_file = glob.glob(path + '/RS.*')[0]
    names = DicomParser(structure_file).GetStructures()

    # Loop through list of structures of interest.
    structure_set = list()
    flag = list()
    
    for struct in structure_names_list:
        # Get index of structure of interest.
        index = [ii for ii in list(names.keys()) if names[ii]['name'] == struct]
        
        if len(index) == 0:
            print(f'...organ {struct} does not exist in structure file.')
            flag.append(False)
            structure_set.append(np.nan)
            continue
        
        # Open structure using index, return triangulation point coordinate dictionary.
        structure = DicomParser(structure_file).GetStructureCoordinates(index[0])
        structure_set.append(structure)
        flag.append(True)
        
    # Could maybe add functionality to detect single or multiple structure names.
    return structure_set, flag
    
def structure_mask(structure, resolution = np.array([1.0, 1.0, 2.0])): 
    '''
    This creates a mask from the dictionary from read_structure method.
    The mask is usually in bounding box size to structure set and returns the 
    top left most pixel in patient coordinates for alignment purposes.

    Parameters
    ----------
    structure : dictionary
        Dictionary of structure set slices from read_structure.
    pixel_spacing : list, optional
        Two element list, refering to the x and y resolution or pixel spacing.
        The default is [1.00, 1.00].
    thickness : float, optional
        Float thickness as pulled from CT DICOM info. The default is 2.00.

    Returns
    -------
    mask_array : numpy.ndarray
        Boolean array of structure mask in [x,y,z] format.
    origin : list
        Origin of the top left most pixel in list form.
    resolution : list
        Resolution of the returned array mask.

    '''
    
    #--------------------------------------------------------------------------
    # Convert DicomParser structure dictionary to a np.array of triangulation point coordinates.
    #--------------------------------------------------------------------------
    triang_points = list()
    for key in structure.keys():
        for ii in range(len(structure[key])):
            triang_points += structure[key][ii]['data']
        
    triang_points = np.array(triang_points)
    
    #--------------------------------------------------------------------------
    # Create 2D meshgrid of patient coordinate points.
    #--------------------------------------------------------------------------    
    # Determine x and y number of voxels in grid.
    cols = np.ceil(max(triang_points[:,0]) - min(triang_points[:,0]))
    rows = np.ceil(max(triang_points[:,1]) - min(triang_points[:,1]))
    
    #Gives location of each pixel in patient coordinates
    X = np.arange(0,cols + resolution[1],resolution[1]) + min(triang_points[:,0]) #Want yres, gives space between columns
    Y = np.arange(0,rows + resolution[0],resolution[0]) + min(triang_points[:,1]) #Want xres, gives space between rows
    
    #Return coordinate matrices from coordinate vectors
    xx,yy = np.meshgrid(X,Y)
    grid_points = np.column_stack([np.ndarray.flatten(xx), np.ndarray.flatten(yy)])
    
    # Return top left most pixel to align with ct and dose.
    # Need to verify if it is min(x,y,z) or min(x,y) and max(z) or some other combination.   
    min_coords = [min(grid_points[:,0]), min(grid_points[:,1]), min(triang_points[:,2])]
    max_coords = [max(grid_points[:,0]), max(grid_points[:,1]), max(triang_points[:,2])] 
    
    origin = np.array(min_coords)
    
    #--------------------------------------------------------------------------
    # Loop through slices and convert contour to mask point locations.
    #--------------------------------------------------------------------------
    slices = np.unique(triang_points[:,2])
    voxels = list()
    for s in slices:
        slice_points = triang_points[triang_points[:,2] == s] # Identify points of individual slice.
        
        outline = np.column_stack([slice_points[:,0],slice_points[:,1]])
        path = matplotlib.path.Path(outline)
        
        # Create and stack 2D mask of slice of interest.
        mask_points = grid_points[path.contains_points(grid_points)]
        mask_points = np.column_stack([mask_points, np.ones(len(mask_points)) * s])        
        voxels += list(mask_points)
        
    mask_3d_points = np.array(voxels)
    
    #--------------------------------------------------------------------------
    # Initialize array of appropriate size and apply mask.
    #--------------------------------------------------------------------------
    # Initialize the 3D mask array to fit the mask based on identified resolution.
    mask_size = np.ceil((np.array(max_coords) - np.array(min_coords) + np.array(resolution)) / resolution).astype(int)
    mask_array = np.zeros(mask_size, dtype = bool)
    
    # Normalize the patient coordinates to get array coordinates. This includes normalizing by resolution.
    mask_3d_points = (mask_3d_points - mask_3d_points.min(axis=0))  / np.array(resolution)
    mask_3d_points = mask_3d_points.astype(int)

    # Mapping the mask points to a 3D array.
    for vox in mask_3d_points:
        mask_array[vox[0], vox[1], vox[2]] = True

    return mask_array, origin, resolution

def structure_mask_new(structure, ct_origin, ct_resolution, ct_shape):

    #--------------------------------------------------------------------------
    # Convert DicomParser structure dictionary to a np.array of triangulation point coordinates.
    #--------------------------------------------------------------------------
    triang_points = list()
    for key in structure.keys():
        for ii in range(len(structure[key])):
            triang_points += structure[key][ii]['data']
        
    triang_points = np.array(triang_points)
    
    #--------------------------------------------------------------------------
    # Create 2D meshgrid of patient coordinate points.
    #-------------------------------------------------------------------------- 
    cols, rows = ct_shape[0], ct_shape[1]
    
    #Gives location of each pixel in patient coordinates
    X = np.arange(0, cols, ct_resolution[1]) + ct_origin[0] #Want yres, gives space between columns
    Y = np.arange(0, rows, ct_resolution[0]) + ct_origin[1] #Want xres, gives space between rows

    #Return coordinate matrices from coordinate vectors
    xx,yy = np.meshgrid(X,Y)
    grid_points = np.column_stack([np.ndarray.flatten(xx), np.ndarray.flatten(yy)])
    
    #--------------------------------------------------------------------------
    # Loop through slices and convert contour to mask point locations.
    #--------------------------------------------------------------------------
    slices = np.unique(triang_points[:,2])
    voxels = list()
    for s in slices:
        slice_points = triang_points[triang_points[:,2] == s] # Identify points of individual slice.
        
        outline = np.column_stack([slice_points[:,0],slice_points[:,1]])
        path = matplotlib.path.Path(outline)
        
        # Create and stack 2D mask of slice of interest.
        mask_points = grid_points[path.contains_points(grid_points)]
        mask_points = np.column_stack([mask_points, np.ones(len(mask_points)) * s])        
        voxels += list(mask_points)
        
    mask_3d_points = np.array(voxels)
    
    #--------------------------------------------------------------------------
    # Initialize array of appropriate size and apply mask.
    #--------------------------------------------------------------------------
    # Initialize the 3D mask array to fit the mask based on identified resolution.
    mask_array = np.zeros(ct_shape, dtype = bool)
    
    # Normalize the patient coordinates to get array coordinates. This includes normalizing by resolution.
    mask_3d_points = ((mask_3d_points - np.array(ct_origin)) / np.array(ct_resolution)).astype(int)

    # Mapping the mask points to a 3D array.
    for vox in mask_3d_points:
        mask_array[vox[0], vox[1], vox[2]] = True

    return mask_array, ct_origin, ct_resolution
    
    
def process_structure(structure, ct_origin, res_f = np.array([1.0,1.0,2.0]), _dim = [600,600,600], _crop = [512,512,300]):
    '''
    Combine the full processing steps required to make a mask aligned
    with the CT scan and dose distribution.

    Parameters
    ----------
    structure : dict
        Dictionary of structure set slices from read_structure.
    ct_origin : np.array
        Origin of the CT scan, to allow for shift calculation.
    res_f : np.array, optional
        Final desired resolution of the structure set. 
        The default is np.array([1.0,1.0,2.0]).
    _dim : list, optional
        List of dimensions, larger than required to fill the entire data. 
        The default is [600,600,600].
    _crop : list, optional
        List of cropped dimensions, final size of the array. 
        The default is [512,512,300].

    Returns
    -------
    mask : np.ndarray
        Integer mask of the structure of interest.
    mask_origin : np.array
        Origin of the structure of interest.
    resolution : np.array
        Final resolution of the mask array.

    '''
    print('...create structure mask from triangulation.')
    mask, mask_origin, mask_res = structure_mask(structure, ct_origin, ct_res)


    #print('...resize and shift structure mask.')
    mask = resize(mask, new_dim = _dim, crop = _crop)
    mask = new_align(mask, mask_origin - ct_origin, mask_res)
    
    return mask, mask_origin, mask_res


#------------------------------------------------------------------------------
#--------------------- 3D SCROLLER CLASS & METHOD------------------------------
#------------------------------------------------------------------------------  

class IndexTracker(object):
    def __init__(self, ax, X, Y, axis, first_ind):
        self.ax = ax
        self.X = X
        self.Y = Y
        self.axis = axis
        self.slices, _, _  = X.shape
        self.ind = int(first_ind)
        
        self.im1 = ax.imshow(self.X.take(indices = self.ind, axis = self.axis), cmap="gray")
        self.im2 = ax.imshow(self.Y.take(indices = self.ind, axis = self.axis), cmap="jet", alpha=.5)

        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self): 
        im1_data = self.im1.to_rgba(self.X.take(indices = self.ind, axis = self.axis), alpha=self.im1.get_alpha())
        im2_data = self.im2.to_rgba(self.Y.take(indices = self.ind, axis = self.axis), alpha=self.im2.get_alpha())

        self.im1.set_data(im1_data)
        self.im2.set_data(im2_data)

        self.ax.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()

        
def plot3d(image1, image2, axis = 2, first_ind = None):
    '''
    Overlay two 3D arrays and index along any given axis. If you only want to view
    one image, place the same array in both images.

    Parameters
    ----------
    image1 : numpy.ndarray
        First image - usually ct due to colouring.
    image2 : numpy.ndarray
        Second image - usually dose due to colouring.
    axis : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    tracker : TYPE
        Need to save output so it remains active after command line closes.

    '''
    if axis > (len(image1.shape) - 1):
        axis = (len(image1.shape) - 1)
        print(f'Axis is out of bounds, it has been set to {axis}.')
    
    if first_ind == None:
        first_ind = image1.shape[2] / 2
    
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image1, image2, axis, first_ind)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    return tracker


#------------------------------------------------------------------------------
#------------------------ File Management -------------------------------------
#------------------------------------------------------------------------------ 

def make_folder(path, verbose = True):
    '''
    Create a folder if it does not exist in a given path.

    Parameters
    ----------
    path : string
        Directory of interest.

    '''
    # Check if directory exists in existing path, make folder otherwise.
    if os.path.isdir(path) == False:    
        os.mkdir(path)
        if verbose == True:
            print(f'Created directory: {path}')
        return False
    else :
        if verbose == True:
            print(f'Folder exists: {path}.')
        return True
    
        
def copy_file(path, new_path):
    '''
    Move file from original path to new path if it does not already exist.

    Parameters
    ----------
    path : string
        Original path.
    new_path : string
        Destination path.
    '''
    # Clause to skip if file exists in new_path, 
    # otherwise copy file.
    if os.path.exists(new_path) == False:  
        shutil.copyfile(path, new_path)










        