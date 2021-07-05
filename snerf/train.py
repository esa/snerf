#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S-NeRF Training script, contains definition of config file parameters, functions to train and test a S-NeRF model.
Usage: python train.py --config <config_file>
"""

import os
import time
import math

import pprint
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import configargparse 

import data_handling
import render
import models

def_dtype=np.float32
COMPLETENESS_THRESHOLD=1.0 #In meters. Completeness is the ratio of pixels in the estimated DEM where the error is lower than the threshold.

def config_parser():
    parser = configargparse.ArgumentParser(description='Train and test a Shadow Neural Radiance Field on a set of posed, forward-facing images. Produces the model weights and a summary of the evaluation on the test set.')
    parser.add_argument('--config', is_config_file=True, help='Config file path.')
    # Dataset arguments
    parser.add_argument('--data.image.path', type=str, default='./data/', help='Path that contains all images.')
    parser.add_argument('--data.image.df', type=int, default=1, help='Image downsample factor.')
    parser.add_argument('--data.image.sd', type=np.float32, default=1.0, help='Image sampling distance.')
    parser.add_argument('--data.depth.path', type=str, default='./data/depth.tif', help='Depth map path.')
    parser.add_argument('--data.depth.df', type=int, default=1, help='Depth map downsample factor.')
    parser.add_argument('--data.md.path', type=str, default='./data/md.txt', help='Metadata file path.')
    parser.add_argument('--data.train_id', type=str, nargs='+', help='ID of train images.')
    parser.add_argument('--data.test_id', type=str, nargs='+', help='ID of test images.')
    # Model arguments
    parser.add_argument('--model.ins.light', type=bool, default=False, help='Use light directions as network inputs.')
    parser.add_argument('--model.ins.views', type=bool, default=False, help='Use view directions as inputs.')
    parser.add_argument('--model.outs.shad', type=bool, default=False, help='Directional light source visibility function as network output.')
    parser.add_argument('--model.outs.sky', type=bool, default=False, help='Diffuse light color as network output.')
    parser.add_argument('--model.act', type=str, default='relu', help='Neuron activation function [relu, sin].')
    parser.add_argument('--model.act.sin.w0', type=np.float32, default=30.0, help='Initial wavelength for SIREN.')
    parser.add_argument('--model.sigma.depth', type=int, default=8, help='Number of fully-connected layers for sigma function.')
    parser.add_argument('--model.sigma.width', type=int, default=256, help='Width of layers for sigma function.')
    parser.add_argument('--model.sigma.skips', type=int, nargs='+', default=[], help='Skip connections.')
    parser.add_argument('--model.c.depth', type=int, default=1, help='Number of fully-connected layers for color function.')
    parser.add_argument('--model.c.width', type=int, default=128, help='Width of layers for color function.')
    parser.add_argument('--model.shad.depth', type=int, default=4, help='Number of fully-connected layers for shadow function.')
    parser.add_argument('--model.shad.width', type=int, default=128, help='Width of layers for shadow function.')
    parser.add_argument('--model.emb.pos', type=int, default=0, help='Length of on-axis positional encoding. 0 to disable.')
    parser.add_argument('--model.emb.dir', type=int, default=0, help='Length of on-axis directional encoding. 0 to disable.')
    
    # Rendering arguments
    parser.add_argument('--rend.nsamples', type=int, default=64, help='Number of samples for coarse rendering.')
    parser.add_argument('--rend.nimportance', type=int, default=64, help='Number of samples for fine rendering, 0 to disable.')
    parser.add_argument('--rend.mode', type=str, default='nf', help='Rendering mode : near-far or altitude sampling [nf, alt].')
    parser.add_argument('--rend.mode.nf.near', type=np.float32, default=3.0, help='Near point (px).')
    parser.add_argument('--rend.mode.nf.far', type=np.float32, default=10.0, help='Far point (px).')
    parser.add_argument('--rend.mode.alt.max', type=np.float32, default=30.0, help='Max alt (px).')
    parser.add_argument('--rend.mode.alt.min', type=np.float32, default=-30.0, help='Min alt(px).')
    parser.add_argument('--rend.unzoom', type=bool, default=False, help='Special unzoom mode for off-nadir EO images.')
    parser.add_argument('--rend.rescale', type=np.float32, default=None, help='Largest scene extent in pixel units. Calculated based on image sizes if not provided.')
    
    # Training arguments
    parser.add_argument('--train.n_epoch', type=int, default=200, help='Number of iterations for training.')
    parser.add_argument('--train.n_rand', type=int, default=1024, help='Number of random rays at each iteration.')
    parser.add_argument('--train.lr.init', type=np.float32, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--train.lr.decay', type=np.float32, default=0.2, help='Learning rate decay over entire training.')
    parser.add_argument('--train.noise.sigma', type=np.float32, default=10.0, help='Standard deviation of sigma pre-activation noise.')
    parser.add_argument('--train.noise.shad', type=np.float32, default=1.0, help='Standard deviation of shadow function pre-activation noise.')
    parser.add_argument('--train.shad', type=bool, default=False, help='Use solar correction rays.')
    parser.add_argument('--train.shad.lambda', type=np.float32, default=0.1, help='Weight of solar correction loss.')
    parser.add_argument('--train.shad.df', type=int, default=1, help='Downsample factor of solar correction rays compared to image rays, 1 to sample at same resolution.')
    parser.add_argument('--train.shad.custom', type=str, default='none', help='Type of custon solar correction rays [linear, rectangle].')
    parser.add_argument('--train.shad.custom.bounds.start', type=np.float32, nargs='+', default=[160.0, 40.0], help='Start point (az, el) in deg.')
    parser.add_argument('--train.shad.custom.bounds.end', type=np.float32, nargs='+', default=[100.0, 80.0], help='End point (az, el) in deg.')
    parser.add_argument('--train.shad.custom.bounds.samp', type=int, nargs='+', default=[10, 1], help='Sampling scheme for solar correction rays. If linear, 1st dimension is number of samples.')
    # Output arguments
    parser.add_argument('--out.iplot', type=int, default=0, help='Frequency of test evaluation for output, 0 to disable.')
    parser.add_argument('--out.path', type=str, default='./results/', help='Path to save outputs.')

    #Hardware options
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use.')
    
    return parser

def read_config(path):
    """Read config file from a path and return the corresponding argument dictionary"""
    parser = config_parser()
    args = parser.parse_args(f'--config {path}')
    arg_dict = vars(args)
    return arg_dict

def init_exp_decay_adam(init_lr, N_iters, decay):   
    """
    
    Intialize Adam optimizer with learning rate (lr) following an exponential decay.
    The leaning rate of init_lr*decay is reached after N_iters.
    
    Parameters:
    init_lr (float): Initial learning rate.
    N_iters (int): Number of iterations for decay.
    decay (float): Decay factor. 
    
    Outputs:
    optimizer: Keras optimizer
    
    """
    lrate = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, N_iters, decay_rate=decay)
    optimizer = tf.keras.optimizers.Adam(lrate)
    return optimizer

def train_model(model, optimizer, N_iterations, arg_dict, train_rays, sc_train_rays=None, decrease_noise=True, eval_dataset=None):
    """
    
    Main training function, optimize a S-NeRF model to a set of pre-extracted training rays.
    
    Parameters:
    model (dict): Initial model with embeddings and dimensions (models.py).
    optimizer (keras.optimizer): optimizer with learning rate schedule (output of init_exp_decay_adam).
    N_iterations (int): Number of epochs to train the model. Each epoch corresponds to one batch of rays.
    arg_dict (dict): Global config variables.
    train_rays (dict): Pre-extracted training rays from generate_train_rays (render.py).
    sc_train_rays (dict): Pre-extracted solar correction rays (render.py)
    decrease_noise (bool): Activate to linearly decrease the strength of the regularization noise.
    eval_dataset (dict): Dataset of test images to evalue the test scores during training. 
    
    Outputs:
    model (dict): Trained model. 
    loss_log (list(string)): Log of the different components of the loss function during training.
    scores (list(...)): Log of test scores achieved during training.
    
    """
    N_rand = arg_dict['train.n_rand']
    N_train_rays = train_rays['rays_o'].shape[0]
    i_batch = 0
    raw_noise_std_init = tf.convert_to_tensor((arg_dict["train.noise.sigma"], arg_dict["train.noise.shad"]), dtype=def_dtype)
    raw_noise_std = raw_noise_std_init
    loss_log = [] # For logging the training losses
    scores = [] # For logging the train + test scores
    # Shuffle train rays
    train_rays = render.shuffle_rays(train_rays)
    if sc_train_rays is not None:
        i_batch_sc = 0
        N_sc_train_rays = sc_train_rays['rays_o'].shape[0]
        sc_train_rays = render.shuffle_rays(sc_train_rays)
        lambda_sc = arg_dict["train.shad.lambda"]
        # No importance sampling for shadow correction : all samples in first pass
        sc_arg_dict = arg_dict.copy()
        sc_arg_dict['rend.nsamples'] = arg_dict['rend.nsamples'] + arg_dict['rend.nimportance']
        sc_arg_dict['rend.nimportance'] = 0
    print("Begin training")
    # Extract gradient variables from the network
    grad_vars = model['model'].trainable_variables
    for i in range(N_iterations):
        if i_batch >= N_train_rays:
            # Once all rays have been sampled shuffle and reset batch index
            train_rays = render.shuffle_rays(train_rays)
            i_batch = 0
        # Extract N_rand rays from the batch
        train_ray_batch = render.get_ray_batch(train_rays, i_batch, i_batch+N_rand)
        i_batch+=N_rand        
        if sc_train_rays is not None:
            if i_batch_sc > N_sc_train_rays:
                i_batch_sc = 0
                sc_train_rays = render.shuffle_rays(sc_train_rays)
            sc_train_ray_batch = render.get_ray_batch(sc_train_rays, i_batch_sc, i_batch_sc+N_rand)
            i_batch_sc+=N_rand
        if decrease_noise:
            raw_noise_std = raw_noise_std_init*(1-i/N_iterations)
        # Render train ray batch and sc train batch
        with tf.GradientTape() as tape:
            ret_dict_c = render.render_rays(model, arg_dict, train_ray_batch, rand=True, raw_noise_std=raw_noise_std, rets=['rgb'])
            # Compute rgb loss
            rgb_loss = tf.reduce_mean(tf.square(ret_dict_c['rgb'] - train_ray_batch['values']))
            loss = rgb_loss
            if sc_train_rays is not None:
                # Render shadow correction rays without perturbation on opacity
                ret_dict_sc = render.render_rays(model, sc_arg_dict, sc_train_ray_batch, rand=True,
                                                       raw_noise_std=(0.0, raw_noise_std[1]), rets=['ret_sun', 'ret_shadow_loss'])
                # Compute shadow loss
                s_loss = (tf.reduce_mean(ret_dict_sc['ret_shadow_loss']) + tf.reduce_mean(1.0-ret_dict_sc['ret_sun']))*lambda_sc
                loss += s_loss
        # Propagate gradients
        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))
        # Log loss values
        loss_log.append(f"{i} {rgb_loss} {s_loss if sc_train_rays is not None else ''}\n")
        if (i < 10) or (i % 25 == 0):
            rgb_psnr = -10. * tf.math.log(rgb_loss) / tf.math.log(10.)
            print(f"{i} {rgb_psnr} {s_loss if sc_train_rays is not None else ''}")
        if (eval_dataset is not None) and (arg_dict['out.iplot'] > 0) and (i % arg_dict['out.iplot'] == 0):
            dataset_rend = render.render_dataset(eval_dataset, model, ['rgb'], arg_dict)
            train, test, alt = test_model(model, eval_dataset, dataset_rend, arg_dict)
            scores.append((i, (train, test, alt)))
            print(f"Test {i}")
            print(f"{test} {alt}")
    return model, loss_log, scores

def compute_image_scores(ref_img, rend_img):
    """
    
    Evaluate the two image quality metrics between a rendered image and a reference image.
    1. Peak Signal to Noise Ratio (PSNR) 
    2. Structural SIMilarity (SSIM) fromm scikit-image.
    
    Parameters:
    ref_img (array[H, W, 3]): Reference image.
    rend_img  (array[H, W, 3]): Rendered image.
    
    Outputs:
    psnr (float): PSNR value
    ssim (float): SSIM value 
    
    """
    mse = tf.reduce_mean(tf.square(rend_img - ref_img))
    psnr = -10. * tf.math.log(mse) / tf.math.log(10.)
    struct_sim = ssim(rend_img.numpy(), ref_img.numpy(), data_range=1, multichannel=True)
    return [psnr, struct_sim]

def render_dsm(model, arg_dict, dsm):
    """
    
    Render the depth map from nadir angle and subtract from orbital radius to extract Digital Surface Model.
    
    Parameters:
    model (dict): Model with embeddings and dimensions (models.py).
    arg_dict (dict): Global configuration variables.
    dsm (Tensor[N, M]): Ground truth DSM
    
    Outputs:
    dsm_predict (Tensor[N, M]): Predicted DSM
    
    """
    SR = 0.5 * arg_dict['data.depth.df']
    radius = 617000.0/SR
    arg_dict_temp = arg_dict.copy()
    arg_dict_temp['data.image.sd'] = SR
    arg_dict_temp['data.image.df'] = 1
    az, el = np.pi, np.pi/2
    pose = data_handling.pose_spherical(az, -el, radius)
    hwf = dsm.shape[0], dsm.shape[1], radius
    light_dir=tf.reshape(tf.convert_to_tensor([np.deg2rad(100), np.deg2rad(80)], dtype=def_dtype), [1,2])
    view_dir=tf.reshape(tf.convert_to_tensor([az, el], dtype=def_dtype), [1,2])
    ret_dict = render.render_image(model, arg_dict_temp, hwf, pose, 1.0, light_dir, view_dir, rets=['depth'])
    dsm_predict = (radius - ret_dict['depth'])*SR
    return dsm_predict
  
def test_model(model, dataset, dataset_rend, arg_dict):
    """
    
    Evaluate the image scores of a set of rendered images (PSNR, SSIM), and compute the altitude scores.
    1. Global Mean Average Error
    2. Completeness : ratio of pixels with error < 1m (between 0 and 1).
    3. Accuracy : Mean Average Error of those pixels with an error < 1m.
    
    Parameters:
    model (dict): Model with embeddings and dimensions (models.py).
    dataset (dict): Training and test images (generate_dataset from data_handling.py).
    dataset_rend (dict): Rendered training and test images (render_dataset from render.py)
    arg_dict (dict): Global configuration variables.
    
    Outputs:
    train_scores: Train scores, see compute_image_scores
    test_scores: Test scores, see compute_image_scores
    alt_scores: Altitude scores (MAE, Completeness, Accuracy).
    
    """
    train_scores, test_scores, alt_scores=[], [], []
    for i in range(len(dataset['train_imgs'])):
        train_scores.append(compute_image_scores(dataset['train_imgs'][i], dataset_rend['train_rend'][i]['rgb']))
    for i in range(len(dataset['test_imgs'])):
        test_scores.append(compute_image_scores(dataset['test_imgs'][i], dataset_rend['test_rend'][i]['rgb']))
    # Evaluate loss from one given depth map
    dsm = render_dsm(model, arg_dict, dataset['depth_map'])
    alt_abs_diff = tf.abs(dataset['depth_map'] - dsm)
    alt_mae = tf.reduce_mean(alt_abs_diff)
    alt_comp = tf.reduce_mean(tf.where(alt_abs_diff < COMPLETENESS_THRESHOLD, 1.0, 0.0))
    alt_acc = tf.reduce_mean(tf.gather_nd(alt_abs_diff, tf.where(alt_abs_diff < COMPLETENESS_THRESHOLD)))
    alt_scores.append(alt_mae)
    alt_scores.append(alt_comp)
    alt_scores.append(alt_acc)
    return np.array(train_scores), np.array(test_scores), np.array(alt_scores)

def score_overview(score_list, train_loss, path=None):
    """
    
    Write the overview of losses and scores computed during training to two text files
    1. scores.txt for image and altitude quality during training.
    2. train_loss.txt for different loss functions during training.
    
    Parameters:
    score_list: List of scores, see test_model.
    train_loss (list(string)): List of losses memorized during training.
    path (string): Path to save scores
    
    Outputs:
    psnr (float): PSNR value
    ssim (float): SSIM value 
    
    """
    lines=["It, Train_PSNR, Train_PSNR_std, Train_SSIM, Train_SSIM_std, Test_PSNR, Test_PSNR_std, Test_SSIM, Test_SSIM_std, Alt_MAE, Alt_Comp, Alt_Acc\n"]
    for i, scores in score_list:
        tr_p = np.mean(scores[0][:,0]), np.std(scores[0][:,0])
        tr_s = np.mean(scores[0][:,1]), np.std(scores[0][:,1])
        te_p = np.mean(scores[1][:,0]), np.std(scores[1][:,0])
        te_s = np.mean(scores[1][:,1]), np.std(scores[1][:,1])

        lines.append(f"{i},"+",".join([f"{x[0]:.4},{x[1]:.3}" for x in [tr_p, tr_s, te_p, te_s]])+f",{scores[2][0]},{scores[2][1]},{scores[2][2]}\n")
    if path is not None:
        with open(f"{path}scores.txt", 'w') as f:
            f.writelines(lines)
        if train_loss is not None:
            with open(f"{path}train_loss.txt", 'w') as f:
                f.writelines(train_loss)
    return lines

def prepare_train_rays(dataset, arg_dict):
    """
    
    Prepare the training and solar correction rays for a given dataset. 
    
    Parameters:
    dataset (dict): Dataset of training images, see generate_dataset (data_handling.py).
    arg_dict (dict): Config variables containing shading options.
    
    Outputs:
    train_rays (dict): Training rays, see generate_train_rays (render.py).
    sc_train_rays (dict): Solar correction rays (render.py). 
    
    """
    train_rays = render.generate_train_rays(dataset, arg_dict)
    if arg_dict['train.shad']:
        sc_train_rays = render.generate_train_light_correction_rays(dataset, arg_dict)
        if arg_dict['train.shad.custom'] in ['linear', 'rectangle']:
            custom_sc_rays = render.generate_custom_light_correction_rays(dataset, arg_dict)
            sc_train_rays = render.concat_rays(sc_train_rays, custom_sc_rays)
    else:
        sc_train_rays=None
    return train_rays, sc_train_rays

if __name__ == "__main__":
    # Parse input arguments to arg_dict
    parser = config_parser()
    args = parser.parse_args()
    arg_dict = vars(args)
    pprint.pprint(arg_dict)
    
    # Setup GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = arg_dict['gpu']
    
    # Create dataset of test and train images + metadata
    dataset = data_handling.generate_dataset(arg_dict)

    # Setup output folder
    if not os.path.exists(os.path.dirname(arg_dict['out.path'])):
        os.makedirs(os.path.dirname(arg_dict['out.path']))    

    # Compute rescale factor if not given
    if arg_dict['rend.rescale'] is None:
        arg_dict['rend.rescale'] = render.calculate_rescale_factor(dataset)
        
    # Model initialization
    model = models.generate_model(arg_dict)
    print(model['model'].summary())
    
    # Early ray generation
    print("Generating rays")
    train_rays, sc_train_rays= prepare_train_rays(dataset, arg_dict)
    
    # Train model
    optimizer = init_exp_decay_adam(arg_dict['train.lr.init'], arg_dict['train.n_epoch'], arg_dict['train.lr.decay'])                 
    model, loss, scores = train_model(model, optimizer, arg_dict['train.n_epoch'], arg_dict, train_rays, sc_train_rays=sc_train_rays,  decrease_noise=True, eval_dataset=dataset)
    
    # Write model and training scores
    models.save_model(arg_dict['out.path'], model)
    dataset_rend = render.render_dataset(dataset, model, ['rgb'], arg_dict)
    train, test, alt = test_model(model, dataset, dataset_rend, arg_dict)
    scores.append((arg_dict['train.n_epoch'], (train, test, alt)))
    score_overview(scores, loss, arg_dict['out.path'])
