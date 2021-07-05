"""
Helper functions to read images, depth maps, metadata, and compute poses, and assemble into a single dataset dictionary, with split train and test images.  
"""

import math

import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate
from skimage.transform import resize
from osgeo import gdal

def_dtype=np.float32

# 3d spherical coordinates
trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=def_dtype)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=def_dtype)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=def_dtype)


def pose_spherical(theta, phi, radius):
    """
    
    Camera to world matrix from azimuth (theta) and elevation (phi) angles.
    
    Parameters:
    theta (float): Azimuth angle (rad)
    phi (float): Elevation angle (rad)
    radius (float): Distance from camera to origin of absolute ref. frame
        
    Outputs:
    c2w (array[4, 4]): Camera to world matrix with rotation in upper left 3x3 submatrix and translation on last line 
    
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def read_depth_map(depth_map_path, df = 1):
    """
    
    Open one-band depth map .tif file.
    
    Parameters:
    depth_map_path (string): Path to depth map
    df (int): Downscaling factor for depth map (optional)
    
    Outputs:
    img (array[N,M]): Depth map of size NxM
    """
    data_source = gdal.Open(depth_map_path)
    img = tf.convert_to_tensor(data_source.ReadAsArray())
    if df > 1:
        img = resize(img, (img.shape[0]//df, img.shape[1]//df), anti_aliasing=True)
    return img

def generate_images(image_path, view_indices, downscale_factor=1):
    """
    
    Create a list of downsampled tensors [H, W, bands] from .tif images.
    
    Parameters:
    image_path (string): Path to images
    view_indices (list(string)): List of view indices to be read
    downscale_factor (int): Downscaling factor for depth map (optional)
    
    Outputs:
    imgs (list(array[N,M,3])): List of images of size NxMx3, last dimension is RGB
    
    """
    imgs=[]
    for view_index in view_indices:
        image_name = f"{image_path}_{view_index}.tif"
        data_source = gdal.Open(image_name)
        im = tf.convert_to_tensor(data_source.ReadAsArray(), dtype=def_dtype)
        im = tf.transpose(im, perm=[1,2,0])#gdal puts bands first
        H, W = im.shape[0], im.shape[1]
        im = resize(im, (H//downscale_factor, W//downscale_factor), anti_aliasing=True)
        imgs.append(tf.convert_to_tensor(im, dtype=def_dtype))
    return imgs

def read_image_metadata(arg_dict, sep=' ', ids=None):
    """
    
    Convert metadata file to list of poses (c2w matrices), focals, view and light directions.
    
    Parameters:
    arg_dict (dict): Metadata path, sampling distance and image downscale factor
    sep (char): Separating character in metadata file
    ids (list(string)): List of image IDs to read metadata from
    
    Outputs:
    poses (list(array[4,4])): List of camera to world matrices
    focals (list(float)): List of focal lengths
    view_dirs(list(array[1,2])): List of viewing angles
    light_dirs(list(array[1,2])): List of light angles
    
    """
    poses, focals, view_dirs, light_dirs = [], [], [], []
    with open(f"{arg_dict['data.md.path']}", 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            params = line.split(sep)
            # Image ID
            im_id = params[0]
            if (ids == None) or (im_id in ids): 
                # True image sampling distance
                image_sd = arg_dict['data.image.sd'] * arg_dict['data.image.df']
                # Convert radius to pixel units
                radius = def_dtype(params[1])/image_sd
                # Viewing directions
                az = def_dtype(params[2])
                el = def_dtype(params[3])
                # Light source directions
                az_ls = def_dtype(params[4])
                el_ls = def_dtype(params[5])
                poses.append(pose_spherical(az, -el, radius))
                focals.append(radius)
                view_dirs.append(tf.reshape(tf.convert_to_tensor([az, el], dtype=def_dtype), [1,2]))
                light_dirs.append(tf.reshape(tf.convert_to_tensor([az_ls, el_ls], dtype=def_dtype), [1,2]))
    return poses, focals, view_dirs, light_dirs

def generate_dataset(arg_dict):
    """
    
    Generate data set dictionary by splitting images and metadata into train and test sets with viewing and light parameters.
    
    Parameters: 
    arg_dict (dict): Global configuration variables
    
    Outputs:
    dataset (dict): Image IDs, image data, pose, focal, viewing and light parameters split into train and test sets, and depth map.
    
    """
    ret = {}
    ret['train_id'] = arg_dict['data.train_id']
    ret['test_id'] = arg_dict['data.test_id']
    ret['train_imgs'] = generate_images(arg_dict['data.image.path'], arg_dict['data.train_id'], arg_dict['data.image.df'])
    ret['test_imgs'] = generate_images(arg_dict['data.image.path'], arg_dict['data.test_id'], arg_dict['data.image.df'])
    ret['train_poses'], ret['train_focals'], ret['train_view_dirs'], ret['train_light_dirs'] = read_image_metadata(arg_dict, sep=' ', ids=arg_dict['data.train_id'])
    ret['test_poses'], ret['test_focals'], ret['test_view_dirs'], ret['test_light_dirs'] = read_image_metadata(arg_dict, sep=' ', ids=arg_dict['data.test_id'])
    ret['depth_map'] = read_depth_map(arg_dict['data.depth.path'], df=arg_dict['data.depth.df'])
    return ret
    
