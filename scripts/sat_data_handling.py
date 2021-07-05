#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Functions for preparing images from IEEE TGRS 2019 Data Fusion Competition, Track 3. Requires Orfeo ToolBox (>7.0.0) to be installed, called via the command line interface. 
The images are cropped and rotated to be coherent with the NeRF training, metadata is read from the original IMD files and transferred to a custom metadata file containing image ID, radius, viewing angles and light angles.
"""

import os
import math

import numpy as np
import argparse
from tifffile import imsave
from PIL import Image
from scipy.ndimage import rotate
from skimage.transform import resize
from osgeo import gdal #For reading GEOtiff files 

# Functions for translating utm to lon-lat
import utm_helper

# Global variables
SAT_ALT = 617000.0
SR_IM = 0.3
SR_DSM = 0.5

# metadata reading
def get_value_from_imd(field_name, file_string, dtype=float):
    start = file_string.find(field_name)
    eq = file_string[start:].find("=")
    end = file_string[start+eq:].find(";")
    return dtype(file_string[start+eq+1:start+eq+end])

def read_view_angles(mdpath, zone_name, view_index):
    with open(f"{mdpath}/{zone_name}/{view_index}.IMD","r") as f:
        all_lines = f.read()
        az = np.deg2rad(get_value_from_imd("meanSatAz", all_lines))
        ona = np.deg2rad(get_value_from_imd("meanOffNadirViewAngle", all_lines))
        el = np.deg2rad(get_value_from_imd("meanSatEl", all_lines))
    return az, ona, el

def read_solar_angles(mdpath, zone_name, view_index):
    with open(f"{mdpath}/{zone_name}/{view_index}.IMD","r") as f:
        all_lines = f.read()
        az = np.deg2rad(get_value_from_imd("meanSunAz", all_lines))
        el = np.deg2rad(get_value_from_imd("meanSunEl", all_lines))
    return az, el

def read_gsd(mdpath, zone_name, view_index):
    with open(f"{mdpath}/{zone_name}/{view_index}.IMD","r") as f:
        all_lines = f.read()
        row_gsd = get_value_from_imd("meanProductRowGSD", all_lines)
        col_gsd = get_value_from_imd("meanProductColGSD", all_lines)
        gsd = get_value_from_imd("meanProductGSD", all_lines)
    return row_gsd, col_gsd, gsd

def read_corner_coordinates(mdpath, zone_name, view_index):
    with open(f"{mdpath}/{zone_name}/{view_index}.IMD","r") as f:
        all_lines = f.read()
        UL = (get_value_from_imd("ULLat", all_lines), 
              get_value_from_imd("ULLon", all_lines))
        LL = (get_value_from_imd("LLLat", all_lines),
              get_value_from_imd("LLLon", all_lines))
        LR = (get_value_from_imd("LRLat", all_lines),
              get_value_from_imd("LRLon", all_lines))
        UR = (get_value_from_imd("URLat", all_lines),
              get_value_from_imd("URLon", all_lines))
        return UL, LL, LR, UR

def az_lat_lon(UL, LL):
    return math.atan((LL[0]-UL[0])/(LL[1]-UL[1]))
    
def read_az_lat_lon(zone_name, view_index):
    UL, LL, LR, UR = read_corner_coordinates(zone_name, view_index)
    return az_lat_lon(UL, LL)

def read_az_lat_lon_ds(data_source):
    UL, LL, LR, UR = read_corner_coordinates_tif(data_source)
    return az_lat_lon(UL, LL)

# Image reading
def open_image(impath, zone_name, area_index, view_index):
    return gdal.Open(f"{impath}/{zone_name}_{area_index}_0{view_index}_RGB_crop.tif")

# Digital Surface Model (DSM) reading
def open_dsm(dsmpath, zone_name, area_index):
    return gdal.Open(f"{dsmpath}/{zone_name}_{area_index}_DSM.tif")

def read_dsm_utm(dsmpath, zone_name, area_index):
    with open(f"{dsmpath}/{zone_name}_{area_index}_DSM.txt", 'r') as f:
        lines = f.readlines()
    lines = [str(line[:-1]) for line in lines]
    easting = lines[0]
    northing = lines[1]
    side = lines[2]
    res = lines[3]
    return easting, northing, side, res

def read_dsm_coords(dsmpath, zone_name, area_index):
    e, n, side, res = read_dsm_utm(dsmpath, zone_name, area_index)
    e, n, side, res = float(e), float(n), int(float(side)), float(res)
    ul, lr, center = utm_helper.utm2lonlat(e, n, side, res)
    return ul, lr, center, side, res

# Image manipulation
def central_area(h, w, arr):
    H, W = arr.shape[0], arr.shape[1]
    return arr[int(H/2 - h/2) : int(H/2 + h/2), int(W/2 - w/2) : int(W/2 + w/2), :]

def extract_central_area_rot(h, w, im, ang):
#     W, H = arr.shape
    ang_pos=ang%(np.pi/2)
    h2=h*np.cos(ang_pos)+w*np.sin(ang_pos)
    w2=w*np.cos(ang_pos)+h*np.sin(ang_pos)
    big_area = central_area(h2, w2, im)
    big_area = rotate(big_area, np.rad2deg(ang))
    big_area = np.clip(big_area,0,1)
    return central_area(h, w, big_area)

def read_corner_coordinates_tif(data_source):
    if data_source is not None:
        md = data_source.GetMetadata()
        all_coords = md['NITF_IGEOLO']
        UL = (float(all_coords[0:6])/10000,-float(all_coords[8:14])/10000)
        UR = (float(all_coords[15:21])/10000,-float(all_coords[23:29])/10000)
        LR = (float(all_coords[30:36])/10000,-float(all_coords[38:44])/10000)
        LL = (float(all_coords[45:51])/10000,-float(all_coords[53:59])/10000)                
        return UL, UR, LR, LL
    else:
        return ""

def generate_sat_train_images(impath, mdpath, zone_name, H, W, area_index, view_indices, downscale_factor, rot=True):
    """Create a data set of downsampled, cropped, forward facing images from north-aligned images"""
    imgs=[]
    for view_index in view_indices:
        data_source = open_image(impath, zone_name, area_index, view_index)
        az, _, _ = read_view_angles(mdpath, zone_name, view_index)
        im = data_source.ReadAsArray()
        # nbands, W, H = im.shape[0], im.shape[1], im.shape[2]
        im = np.array(im/255.0)
        im = np.transpose(im, axes=(1,2,0))#gdal puts bands first       
        #Some of the images are rotated by 90 degrees and are not north-aligned
        az2 = read_az_lat_lon_ds(data_source)
        if np.rad2deg(az2) > 80.0:
            im = extract_central_area_rot(int(H*np.sqrt(2)),int(W*np.sqrt(2)),im, az2)
        else:
            im = central_area(int(H*np.sqrt(2)),int(W*np.sqrt(2)),im)
        if rot:
            # Rotate by 180 + azimuth value to get forward facing images
            im = extract_central_area_rot(H, W, im, np.pi+az)
        else:
            im = extract_central_area_rot(H, W, im, 0.0)
        im = resize(im, (H//downscale_factor, W//downscale_factor), anti_aliasing=True)
        imgs.append(im)
    return imgs

def generate_dsm(dsmpath, zone_name, area_index, df, SR_im, SR_dsm):
    data_source = open_dsm(dsmpath, zone_name, area_index)
    img = data_source.ReadAsArray()
    # H = H_image * 0.5 / 0.3 (same for W)
    #img = central_area(int(H*SR_dsm/SR_im), int(W*SR_dsm/SR_im), img[...,np.newaxis])[...,0]
    img = resize(img, (img.shape[0]//df, img.shape[1]//df), anti_aliasing=True)
    return img

# Dataset generation
def write_sat_train_metadata(mdpath, destpath, zone_name, area_index, view_indices, sat_alt_pix, df):
    """Focal + view angles + solar angles for a set of view indices"""
    lines = ["ID Focal Az El Azs Els"]
    for view_index in view_indices:
        az, ona, el = read_view_angles(mdpath, zone_name, view_index)
        az_sun, el_sun = read_solar_angles(mdpath, zone_name, view_index)
        # Distance between satellite and captured area in pixel units
        radius = sat_alt_pix/np.sin(el)
        lines.append(f"{view_index} {radius} {az} {el} {az_sun} {el_sun}")
    with open(f"{destpath}/{zone_name}_{area_index}_df{df}_md.txt", 'w') as f:
        f.writelines("\n".join(lines))

        
def write_image_set(path, indices, imgs):
    for index, img in zip(indices, imgs):
        imsave(f"{path}{index}.tif", img)
    
def write_sat_data_set(zone_name, H, W, area_index, view_indices, impath, dsmpath, mdpath, destpath, df_im=4, df_dsm=1, SR_im=SR_IM, SR_dsm=SR_DSM):
    imgs = generate_sat_train_images(impath, mdpath, zone_name, H, W, area_index, view_indices, df_im, rot=True)
    dsm = generate_dsm(dsmpath, zone_name, area_index, df_dsm, SR_im, SR_dsm*df_dsm)
    write_sat_train_metadata(mdpath, destpath, zone_name, area_index, view_indices, SAT_ALT*df_im, df_im)
    write_image_set(f"{destpath}/{zone_name}_{area_index}_df{df_im}_", view_indices, imgs)
    write_image_set(f"{destpath}/{zone_name}_{area_index}_df{df_dsm}_dsm", [""], [dsm])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cropped, downscaled, forward facing images and DSM for one area')
    parser.add_argument('--zone', type=str)
    parser.add_argument('--area', type=str)
    parser.add_argument('--dfim', type=int)
    parser.add_argument('--dfdsm', type=int)
    parser.add_argument('--impath', type=str)
    parser.add_argument('--dsmpath', type=str)
    parser.add_argument('--mdpath', type=str)
    parser.add_argument('--destpath', type=str)
    args = parser.parse_args()
    all_image_names = os.listdir("data/Train-Track3-RGB-1/Track3-RGB-1/")
    selected_images = [f for f in all_image_names if (f[0:3]==args.zone and f[4:7]==args.area)]
    view_indices = [f[8:11] for f in selected_images]
    SR_im = 0.3
    ul, lr, center, dsm_side, dsm_res = read_dsm_coords(args.dsmpath, args.zone, args.area)
    H = dsm_side*dsm_res/SR_im
    H = (H//2)*2 #Close even number 
    W = H
    print(f"Extracting images of size {H}x{W}")
    ext_radius = int(H/np.sqrt(2)+H/10)
    for view_index in view_indices:
        input_image_name = f"{args.impath}/{args.zone}_{args.area}_{view_index}_RGB.tif"
        output_image_name = f"{args.destpath}/{args.zone}_{args.area}_{view_index}_RGB_crop.tif"
        os.system(f"otbcli_ExtractROI -in {input_image_name} -out {output_image_name} -mode radius -mode.radius.r {ext_radius} -mode.radius.cx {center[1]}  -mode.radius.cy {center[0]} -mode.radius.unitc lonlat;")
    view_indices= [i[1:] for i in view_indices]
    write_sat_data_set(args.zone, H, W, args.area, view_indices, impath=args.destpath, dsmpath=args.dsmpath,  mdpath=args.mdpath, destpath=args.destpath, df_im=args.dfim, df_dsm=args.dfdsm, SR_im=SR_im, SR_dsm=dsm_res)

    
