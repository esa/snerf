#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S-NeRF plotting script. Visualize outputs of a trained S-NeRF model. Uses the same configuration file as the trainer script (snerf/train.py).  
Usage: python plots.py --config <config_file>
"""

import os
import imageio
from base64 import b64encode

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_handling import pose_spherical, generate_dataset
from render import render_image, calculate_rescale_factor, render_dataset
from train import compute_image_scores, config_parser, score_overview, test_model
from models import load_model

def_dtype = np.float32

def plot_images(ids, images, view_dirs, light_dirs):
    """
    
    Plot dataset of images with viewing and light directions in the titles.
    
    Parameters:
    ids (list(string)): Image IDs
    images (list(Tensor[H,W,3]): RGB images
    view_dirs (list(Tensor[1,2]): Viewing directions (rad)
    light_dirs (list(Tensor[1,2]): Lighting directions (rad)
    
    """
    H = int(np.ceil(len(ids)/3.0))
    fig, ax = plt.subplots(H,3,figsize=(15,30))
    ax=ax.ravel()
    for k, (idd, image, view_dir, light_dir) in enumerate(zip(ids, images, view_dirs, light_dirs)):   
        az, el = view_dir[0, 0], view_dir[0, 1]
        az_light, el_light = light_dir[0, 0], light_dir[0, 1]
        m_ax = ax[k]
        m_ax.imshow(image)
        m_ax.axis("off")
        m_ax.set_title((f"ID {idd} az, el (deg)\n" 
                        f"View ({np.rad2deg(az):.1f}, {np.rad2deg(el):.1f})\n"
                        f"Light ({np.rad2deg(az_light):.1f}, {np.rad2deg(el_light):.1f})")) 
        
def plot_view_light_directions(view_dirs, light_dirs):
    """
    
    2D Scatter plot of viewing and light directions
    
    Parameters:
    view_dirs (list(Tensor[1,2]): Viewing directions (rad)
    light_dirs (list(Tensor[1,2]): Lighting directions (rad)
    
    """
    fig=plt.figure(figsize=(10, 10))
    ax=plt.gca()
    view_dirs = tf.concat(view_dirs, axis=0)
    light_dirs = tf.concat(light_dirs, axis=0)
    ax.scatter(np.rad2deg(view_dirs[:,0]),np.rad2deg(view_dirs[:,1]), label="View angles")
    ax.scatter(np.rad2deg(light_dirs[:,0]),np.rad2deg(light_dirs[:,1]), label="Light angles")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(f'Distribution of view and light angles')
    ax.legend()
    
def plot_depth_map(depth_map):
    """
    
    Plot depth map with colorbar. 
    
    Parameters:
    depth_map ((Tensor[H,W])
    
    """
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.imshow(depth_map)
    plt.title('Depth map')
    plt.colorbar()
    
def plot_results(dataset, focals, dataset_rend, path=None):
    """
    
    Plot rendered images, disparity maps, and other outputs if available in the rendered dataset.
    Shows the image quality scores in the title. 
    
    Parameters:
    dataset (list(Tensor[H,W,3])): Reference images to compute scores
    focals (list(float)): Focals to compute disparity from depth maps
    dataset_rend (list(dict)): Rendered outputs
    path (string): Path to save figure.
    
    """
    fig, ax = plt.subplots(len(dataset),5,figsize=(15,len(dataset)*4))
    for i, (img, focal, ret_dict) in enumerate(zip(dataset, focals, dataset_rend)):
        train_psnr, train_ssim = compute_image_scores(img, ret_dict['rgb'])
        ax[i,0].imshow(img, interpolation="none")
        ax[i,0].set_title('Reference image')
        
        ax[i,1].imshow(ret_dict['rgb'], interpolation="none")
        ax[i,1].set_title(f'Rendered PSNR ={train_psnr:.3}\n SSIM = {train_ssim:.3}')
        
        ax[i,2].imshow(focal-ret_dict['depth'], interpolation="none")
        ax[i,2].set_title('Disparity map')
        if 'ret_sun' in ret_dict:
            ax[i,3].imshow(np.clip(ret_dict['ret_sun'], 0., 1.), vmin=0, vmax=1, interpolation="none", cmap='gray')
            ax[i,3].set_title('Solar visibility')
        if 'sky' in ret_dict:
            ax[i,3].imshow(np.clip(ret_dict['sky'], 0., 1.), vmin=0, vmax=1, interpolation="none")
            ax[i,3].set_title('Incoming light color')
        if 'no_shadow' in ret_dict:
            ax[i,4].set_title('Shadow removal')
            ax[i,4].imshow(ret_dict['no_shadow'], interpolation="none")
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    
def plot_light_angle_inter(model, arg_dict, hwf, light_start, light_end, view_angle, nplots=10, rets=['rgb, depth'], path=None):
    """
    
    Plot interpolation between two lighting conditions.
    
    Parameters:
    model (dict): Model with embeddings and dimensions (models.py).
    arg_dict (dict): Global configuration variables.
    hwf (int, int, float): height, width and focal length of desired images.
    light_start (Tensor[1,2]): Starting angles for interpolation.
    light_end (Tensor[1,2]): End angles for interpolation.
    view_angle (Tensor[1,2]): Constant viewing angle for interpolation plot.
    n_plots (int): Number of points in the interpolation.
    rets (list(string)): Desired outputs to be visualized, should always include rgb and depth.
    path (string): Path to save plot.
    
    """
    azs_li = np.linspace(np.deg2rad(light_start[0]), np.deg2rad(light_end[0]), nplots)
    els_li = np.linspace(np.deg2rad(light_start[1]), np.deg2rad(light_end[1]), nplots)

    fig, ax = plt.subplots(nplots,len(rets),figsize=(15,len(rets)*nplots))
    for i in range(nplots):
        az, el = view_angle
        H, W, radius = hwf
        pose = pose_spherical(az, -el, radius)
        light_dir=tf.reshape(tf.convert_to_tensor([azs_li[i], els_li[i]], dtype=def_dtype), [1,2])
        view_dir=tf.reshape(tf.convert_to_tensor([az, el], dtype=def_dtype), [1,2])
        ret_dict = render_image(model, arg_dict, hwf, pose, 1.0, light_dir, view_dir, rets=rets)
        ax[i,0].imshow(ret_dict['rgb'], interpolation="none")
        ax[i,0].set_title("Rendered")
        ax[i,1].imshow(radius-ret_dict['depth'], interpolation="none")
        ax[i,1].set_title("Disparity")
        if 'sky' in rets:
            ax[i,2].imshow(np.clip(ret_dict['sky'], 0, 1), interpolation="none", vmin=0, vmax=1)
            ax[i,2].set_title("Incoming light color")
        if 'ret_shadow' in rets:
            ax[i,2].imshow(np.clip(ret_dict['ret_shadow'], 0, 1), interpolation="none", cmap='gray', vmin=0, vmax=1)
            ax[i,2].set_title("Solar visibility")
        if 'no_shadow' in rets:
            ax[i,3].imshow(ret_dict['no_shadow'], interpolation="none", vmin=0, vmax=1)
            ax[i,3].set_title("Shadow removal")
    if path == None:
        plt.show()
    else:
        plt.savefig(path)

def render_video(frames, path):
    """
    
    Render a video from a list of frames.
    
    Parameters:
    frames (list(array[H,W,3]): RGB frames.
    path (string): Path to save video.
    
    """
    imageio.mimwrite(path, frames, fps=20, quality=7)
    mp4_f = open(path,'rb').read()
    data_url_f = "data:video/mp4;base64," + b64encode(mp4_f).decode()
    return data_url_f

def train_data_video(dataset, path):
    """
    
    Render video of training images in data set sorted by azimuth.
    
    Parameters:
    dataset (dict): as defined in generate_dataset (data_handling.py)
    path (string): Path to save video.
    
    """
    N_images = len(dataset['train_imgs'])
    azs = np.array([dataset['train_view_dirs'][i][0,0] for i in range(len(dataset['train_id']))])
    imgs_a = np.stack(dataset['train_imgs'], axis = 3)
    imgs_sorted = imgs_a[...,np.argsort(azs)]
    ds_frames = []
    for i in range(120):
        im = imgs_sorted[...,int((i/120)*N_images)]
        im = (255*np.clip(im,0,1)).astype(np.uint8)
        ds_frames.append(im)
    data_url_ds=render_video(ds_frames,f'{path}ds_frames.mp4')
    return data_url_ds

def min_max_normalize(a):
    """Normalize an array between 0 and 1 according to the global min and max values in the array."""
    return (a-np.min(a))/(np.max(a)-np.min(a))

def render_flyover_video(path, model, arg_dict, hwf, light_start, light_end, rets):
    """
    
    Render video combining smooth changes in viewing and lighting angles.
    The flyover is divided into 6 sequences.
    1. Azimuth change from 0 to 180.
    2. Elevation change from 60 to 90 (nadir). 
    3. Solar angle change from light_start to light_end.
    4. Elevation change from 90 to 60. 
    5. Azimuth change from 180 to 360.
    6. Solar angle change from light_end to light_start. 
    
    Parameters:
    path (string): Path to save video.
    model (dict): Model with embeddings and dimensions (models.py).
    arg_dict (dict): Global configuration variables.
    hwf (int, int, float): height, width and focal length of desired images.
    light_start (Tensor[1,2]): Starting angles for interpolation.
    light_end (Tensor[1,2]): End angles for interpolation.
    rets (list(string)): Desired outputs to be visualized, should always include rgb and depth.
    
    """
    frames = {ret: [] for ret in rets}
    nfr=[15, 6, 10, 6, 15, 10]
    nfr = [i*3 for i in nfr]
    nframes = np.sum(np.array(nfr))
    azs = np.deg2rad(np.concatenate([np.linspace(0.0, 180.0, nfr[0]),
                                     180.0*np.ones((nfr[1] + nfr[2] + nfr[3])),
                                     np.linspace(180.0, 360.0, nfr[4]),
                                     360.0*np.ones((nfr[5]))
                                    ]))
    els = np.deg2rad(np.concatenate([60.0*np.ones((nfr[0])),
                                     np.linspace(60.0, 90.0, nfr[1]),
                                     90.0*np.ones((nfr[2])),
                                     np.linspace(90.0, 60.0, nfr[3]),
                                     60.0*np.ones((nfr[4]+nfr[5]))
                                    ]))
    az_light0, el_light0 = [np.deg2rad(x) for x in light_start]
    az_light1, el_light1 = [np.deg2rad(x) for x in light_end]

    azs_light = np.concatenate([az_light0*np.ones((nfr[0] + nfr[1])), 
                              np.linspace(az_light0, az_light1, nfr[2]),
                              az_light1*np.ones((nfr[3] + nfr[4])),
                              np.linspace(az_light1, az_light0, nfr[5])
                             ])
    els_light = np.concatenate([el_light0*np.ones((nfr[0] + nfr[1])), 
                              np.linspace(el_light0, el_light1, nfr[2]),
                              el_light1*np.ones((nfr[3] + nfr[4])),
                              np.linspace(el_light1, el_light0, nfr[5])
                             ])
    H, W, radius = hwf
    for i in range(nframes):
        pose = pose_spherical(azs[i], -els[i], radius)
        focal = radius
        view_dir = tf.reshape(tf.convert_to_tensor([azs[i], els[i]], dtype=def_dtype), [1,2])
        light_dir = tf.reshape(tf.convert_to_tensor([azs_light[i], els_light[i]], dtype=def_dtype), [1,2])
        ret_dict = render_image(model, arg_dict, hwf, pose, 1.0, light_dirs=light_dir, view_dirs=view_dir, rets=rets)
        for ret in rets:
            frames[ret].append((255*min_max_normalize(ret_dict[ret].numpy())).astype(np.uint8))
    return [render_video(frames[ret], f'{path}flyover_{ret}.mp4') for ret in rets]

def parse_train_loss(arg_dict):
    """
    
    Parse the training loss file that has been produced by train.py.
    
    Parameters: 
    arg_dict (dict): Contains the path to outputs, and tells if a shadow loss is also present.
    
    Ouptuts:
    its (list(int)): List of iterations where loss was printed.
    loss_out (list(float, float)): loss values, optionally including shadow loss.
    
    """
    with open(f"{arg_dict['out.path']}train_loss.txt", 'r') as f:
        lines = f.readlines()
    train_loss = [line.split(' ') for line in lines]
    its = [int(l[0]) for l in train_loss]
    rgb_loss = [float(l[1]) for l in train_loss]
    loss_out = [rgb_loss]
    if arg_dict['train.shad']:
        shad_loss = [float(l[2]) for l in train_loss]
        loss_out.append(shad_loss)
    return its, loss_out

def plot_train_loss(loss, path):
    """
    
    Visualize training losses. 
    
    Parameters: 
    loss (list(int), list(float, float)): Ouputs of parse_train_loss.
    path (string): Path to save plot.
    
    """
    its, loss_out = loss[0], loss[1]
    plt.figure(figsize=(10, 10))
    rgb_loss = loss_out[0]
    plt.plot(its, rgb_loss, c='r')
    if len(loss_out) == 2:
        shad_loss = loss_out[1]
        plt.plot(its, shad_loss, c='g')
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    
def parse_scores(arg_dict):
    """
    
    Parse scores computed during training.

    Parameters: 
    arg_dict (dict): Contains path to score file.
    
    Ouputs:
    its (list(int)): Iterations where scores were computed.
    tx_xx (list(float)): PSNR, SSIM of full train and test images with standard deviations.
    alt (float): Altitude error on depth map.
    
    """
    with open(f"{arg_dict['out.path']}scores.txt", 'r') as f:
        lines = f.readlines()
    train_loss = [line.split(',') for line in lines[1:]]
    its = [int(l[0]) for l in train_loss]
    tr_p = [float(l[1]) for l in train_loss]
    tr_ps = [float(l[2]) for l in train_loss]
    tr_s = [float(l[3]) for l in train_loss]
    tr_ss = [float(l[4]) for l in train_loss]
    te_p = [float(l[5]) for l in train_loss]
    te_ps = [float(l[6]) for l in train_loss]
    te_s = [float(l[7]) for l in train_loss]
    te_ss = [float(l[8]) for l in train_loss]
    alt = [float(l[9]) for l in train_loss]
    return its, (tr_p, tr_ps), (tr_s, tr_ss), (te_p, te_ps), (te_s, te_ss), alt

def plot_scores(its, tr_p, tr_s, te_p, te_s, alt, path):
    """
    
    Plot scores computed during training.

    Parameters: 
    its, tr_p, tr_s, te_p, te_s, alt: Outputs of parse_scores.
    path (string): Path to save plot.
    
    Ouputs:
    its (list(int)): Iterations where scores were computed.
    tx_xx (list(float)): PSNR, SSIM of full train and test images with standard deviations.
    alt (float): Altitude error on depth map.
    
    """
    plt.figure(figsize=(5, 10))
    plt.subplot(311)
    plt.errorbar(its, tr_p[0], yerr=tr_p[1], c='r', label='Train')
    plt.errorbar(its, te_p[0], yerr=te_p[1], c='b', label='Test')
    plt.ylabel('PSNR')
    plt.title('PSNR during training')
    plt.legend()
    plt.subplot(312)
    plt.errorbar(its, tr_s[0], yerr=tr_s[1], c='r', label='Train')
    plt.errorbar(its, te_s[0], yerr=te_s[1], c='b', label='Test')
    plt.ylabel('SSIM')
    plt.title('SSIM during training')
    plt.legend()
    plt.subplot(313)
    plt.errorbar(its, alt, c='purple', label='Alt. err')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Altitude error during training')
    plt.legend()
    if path == None:
        plt.show()
    else:
        plt.savefig(path)

def plot_model(model, path):
    """Plot the S-NeRF model weights."""
    fig, ax = plt.subplots(len(model.layers), figsize=(5,4*len(model.layers)))
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) > 0:
            ax[i].imshow(weights[0])
    plt.savefig(path)
    
def render_vertical_depth_comparison(model, arg_dict, dsm, path=None):
    """
    
    Plot Digital Surface Map comparison and difference, with RMSE in title. 

    Parameters:
    model (dict): Model with embeddings and dimensions (models.py).
    arg_dict (dict): Global configuration variables.
    dsm (Tensor[N, M]): Ground truth DSM
    path (string): Path to save plot.
    
    """
    SR = 0.5 * arg_dict['data.depth.df']
    radius = 617000.0/SR
    arg_dict_temp = arg_dict.copy()
    arg_dict_temp['data.image.sd'] = SR
    arg_dict_temp['data.image.df'] = 1
    az, el = np.pi, np.pi/2
    pose = pose_spherical(az, -el, radius)
    hwf = dsm.shape[0], dsm.shape[1], radius
    light_dir=tf.reshape(tf.convert_to_tensor([np.deg2rad(100), np.deg2rad(80)], dtype=def_dtype), [1,2])
    view_dir=tf.reshape(tf.convert_to_tensor([az, el], dtype=def_dtype), [1,2])
    ret_dict = render_image(model, arg_dict_temp, hwf, pose, 1.0, light_dir, view_dir, rets=['rgb','depth'])
    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.imshow(ret_dict['rgb'])
    plt.title('Rendered RGB')
    plt.subplot(222)
    disp = radius*SR - ret_dict['depth'] * SR
    plt.imshow(disp, vmin=np.min(dsm), vmax=np.max(dsm))
    m_e = tf.sqrt(tf.reduce_mean(tf.square(dsm-disp)))
    plt.title(f"Altitude rendering\n"
              f"RMSE : {m_e:.4} m")
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(dsm)
    plt.title('Ground truth altitude')
    plt.colorbar()
    plt.subplot(224)
    a_max = max(arg_dict['rend.mode.alt.max'],  -arg_dict['rend.mode.alt.min'])
    plt.imshow(disp-dsm, cmap = 'rainbow', vmin=-a_max, vmax=a_max)
    plt.title('Difference between estimated surface altitude and lidar DSM')
    plt.colorbar()
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
                                               
if __name__ == "__main__":
    # Read and setup config parameters
    parser = config_parser()
    args = parser.parse_args()
    arg_dict = vars(args)
    print(arg_dict)
    
    # Setup GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = arg_dict['gpu']
    
    # Load model and training scores
    model = load_model(arg_dict['out.path']+'model.npy', arg_dict)

    # Create dataset of test and train images + metadata
    dataset = generate_dataset(arg_dict)
    
    # Compute rescale factor if not given
    if arg_dict['rend.rescale'] is None:
        arg_dict['rend.rescale'] = calculate_rescale_factor(dataset)

    # Setup outputs according to  config parameters
    rets = ['rgb', 'depth', 'sky' if arg_dict['model.outs.sky'] else 'ret_sun', 'no_shadow' if (arg_dict['model.outs.shad'] or arg_dict['shad.direct']) else '']
        
    # Render dataset
    dataset_rend = render_dataset(dataset, model, rets, arg_dict)

    # Final score values
    scores = [(arg_dict['train.n_epoch'], test_model(model, dataset, dataset_rend, arg_dict))]
    score_overview(scores, None, path=arg_dict['out.path']+'final_')

    # Plot training images
    plot_results(dataset['train_imgs'], dataset['train_focals'], dataset_rend['train_rend'], path=arg_dict['out.path']+'train.png')

    # Plot test images
    plot_results(dataset['test_imgs'], dataset['test_focals'], dataset_rend['test_rend'], path=arg_dict['out.path']+'test.png')

    # Plot model
    plot_model(model['model'], path=arg_dict['out.path']+'model.png')

    # Plot light source interpolation
    hwf = [dataset['train_imgs'][0].shape[0], dataset['train_imgs'][0].shape[1], 617000.0/0.3/arg_dict['data.image.df']]
    light_start = arg_dict['train.shad.custom.bounds.start']
    light_end = arg_dict['train.shad.custom.bounds.end']
    view_angle=(np.pi, np.pi/2)
    plot_light_angle_inter(model, arg_dict, hwf, light_start, light_end, view_angle, nplots=5, rets=rets, path=arg_dict['out.path']+'inter.png')

    # Plot training losses
    its, loss_out = parse_train_loss(arg_dict)
    its_test, tr_p, tr_s, te_p, te_s, alt = parse_scores(arg_dict)

    # Plot training scores
    plot_train_loss((its, loss_out), path=arg_dict['out.path']+'train_loss.png')
    plot_scores(its_test, tr_p, tr_s, te_p, te_s, alt, path=arg_dict['out.path']+'scores.png')

    # Plot Nadir comparison
    render_vertical_depth_comparison(model, arg_dict, dataset['depth_map'], path=arg_dict['out.path']+'nadir.png')
