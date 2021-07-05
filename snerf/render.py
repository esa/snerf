"""
S-NeRF rendering functions, contains generation of train rays, solar correction rays, ray sampling, ray rendering (alpha-compositing), and rendering a set of images. 
"""

import tensorflow as tf
import numpy as np

import data_handling
import models

def_dtype=np.float32

###### Ray generation
def get_rays_zoom(H, W, focal, c2w, factor):
    """
    
    Generate rays with a zoom factor.
    
    Generates a gridded set of rays following a desired grid width, grid height, and camera parameters.
  
    Parameters:
    H (int): Height ray grid
    W (int): Width of ray grid
    focal (float): Focal length of camera
    c2w (Tensor[4,4]): Camera to world matrix
    factor (float): Zoom factor (smaller than 1 = zoom)
    
    Returns:
    rays_o (Tensor[H*W, 3]): 3D origin of rays in absolute ref. frame 
    rays_d (Tensor[H*W, 3]): 3D direction of rays in absolute ref. frame
    
    """
    # Ray origins in camera reference frame
    i, j = tf.meshgrid(tf.linspace(0.0, W*factor, int(W)), tf.linspace(0.0, H*factor, int(H)), indexing='xy')
    # Ray directions in camera reference frame
    dirs = tf.stack([(i-W*factor*.5)/focal, -(j-H*factor*.5)/focal, -tf.ones_like(i)], -1)
    # Transform rays to absolute ref. frame using c2w matrix
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))
    return tf.reshape(rays_o, shape=[-1,3]), tf.reshape(rays_d, shape=[-1,3])

def generate_train_rays(dataset, arg_dict):    
    """
    Generate complete ray set based on training images in a dataset.
    
    A ray is a 3d origin, a 3d direction, a 3d color value, and optionally 2d light and 2d view directions.
    
    Parameters:
    dataset (dict): dataset as specified in 'data_handling.generate_dataset'
    arg_dict (dict): config variables
    
    Returns:
    all_train(dict): set of all rays
    
    """
    
    use_view_dirs = arg_dict['model.ins.views']
    use_light_dirs = arg_dict['model.ins.light']
    el_adj = arg_dict['rend.unzoom']
    all_rays_o, all_rays_d, all_values = [], [], []
    if use_view_dirs:
        all_view_dirs = []
    if use_light_dirs:
        all_light_dirs = []

    for view_i in range(len(dataset['train_imgs'])):
        H, W, nbands = dataset['train_imgs'][view_i].shape
        focals, poses = dataset['train_focals'], dataset['train_poses']
        if el_adj:
            el = dataset['train_view_dirs'][view_i][0,1]
            # Special case to handle satellite images, unzoom by 1/sin(elevation) to compensate
            rays_o, rays_d = get_rays_zoom(H, W, focals[view_i], poses[view_i], 1/np.sin(el))
        else:
            rays_o, rays_d = get_rays_zoom(H, W, focals[view_i], poses[view_i], 1.0)
        rays_o = tf.reshape(rays_o, [H, W, 3])
        rays_d = tf.reshape(rays_d, [H, W, 3])
        values = dataset['train_imgs'][view_i]
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_values.append(values)
        if use_view_dirs:
            all_view_dirs.append(tf.ones([rays_o.shape[0],rays_o.shape[1],1])@dataset['train_view_dirs'][view_i])
        if use_light_dirs:
            all_light_dirs.append(tf.ones([rays_o.shape[0],rays_o.shape[1],1])@dataset['train_light_dirs'][view_i])

    all_values = tf.reshape(tf.convert_to_tensor(all_values), [-1, nbands])
    all_rays_o = tf.reshape(tf.convert_to_tensor(all_rays_o), [-1, 3])
    all_rays_d = tf.reshape(tf.convert_to_tensor(all_rays_d), [-1, 3])
    
    all_train = {'rays_o':all_rays_o, 'rays_d':all_rays_d, 'values':all_values}
    if use_view_dirs:
        all_train['view_dirs'] = tf.reshape(tf.convert_to_tensor(all_view_dirs), [-1, 2])
    if use_light_dirs:
        all_train['light_dirs'] = tf.reshape(tf.convert_to_tensor(all_light_dirs), [-1, 2])
    return all_train

def generate_train_light_correction_rays(dataset, arg_dict):
    """
    
    Generate ray set for solar correction based on the solar angles from the training data set.
    
    Here no values are required, but light directions are necessary for solar correction.
    
    Parameters:
    dataset (dict): dataset as specified in 'data_handling.generate_dataset'
    arg_dict (dict): config variables
    
    Returns:
    all_train(dict): origin, direction, light direction of solar correction rays.
    
    """
    light_df = arg_dict['train.shad.df']
    el_adj = arg_dict['rend.unzoom']
    sc_rays_o, sc_rays_d, sc_light_dirs=[], [], []
    # On all training image angles
    for view_i, img in enumerate(dataset['train_imgs']):
        H, W, _ = img.shape
        H_sc = H//light_df
        W_sc = W//light_df
        light_angles = dataset['train_light_dirs'][view_i]
        az, el = light_angles[0,0], light_angles[0,1]
        focal_sc = dataset['train_focals'][view_i]/light_df
        pose_sc = data_handling.pose_spherical(az, -el, focal_sc)
        if el_adj:
            el_view = dataset['train_view_dirs'][view_i][0,1]
            rays_o, rays_d = get_rays_zoom(H_sc, W_sc, focal_sc, pose_sc, 1/np.sin(el_view))
        else:          
            rays_o, rays_d = get_rays_zoom(H_sc, W_sc, focal_sc, pose_sc, 1.0)
        rays_o = tf.reshape(rays_o, [H_sc, W_sc, 3])
        rays_d = tf.reshape(rays_d, [H_sc, W_sc, 3])
        sc_rays_o.append(rays_o)
        sc_rays_d.append(rays_d)
        sc_light_dirs.append(tf.ones([rays_o.shape[0],rays_o.shape[1],1])@light_angles)
        
    sc_rays_o = tf.reshape(tf.convert_to_tensor(sc_rays_o), [-1,3])
    sc_rays_d = tf.reshape(tf.convert_to_tensor(sc_rays_d), [-1,3])
    sc_light_dirs=tf.reshape(tf.convert_to_tensor(sc_light_dirs), [-1,2])
    return {'rays_o':sc_rays_o, 'rays_d':sc_rays_d, 'light_dirs':sc_light_dirs}
        
def generate_custom_light_correction_rays(dataset, arg_dict):
    """
    
    Generate ray set for solar correction based on custom angles.
    Two modes are possible. 
    1. 'linear' for angles equally spread between a start and end direction.
    2. 'rectangle' for angles on a rectangluar grid, using start and end as corners.
    
    Parameters:
    dataset (dict): dataset as specified in 'data_handling.generate_dataset'
    arg_dict (dict): config variables
    
    Returns:
    all_train(dict): set of custom solar_correction rays.
    
    """
    zoom_factor=1.0
    base_i=0
    light_df = arg_dict['train.shad.df']
    mode=arg_dict['train.shad.custom']
    bounds_start = arg_dict['train.shad.custom.bounds.start'] 
    bounds_end = arg_dict['train.shad.custom.bounds.end']
    # Convert to radian
    bounds = np.deg2rad(np.array([bounds_start, bounds_end]))
    n_ray_images = arg_dict['train.shad.custom.bounds.samp']
    if mode == 'linear':
        # Linear interpolation from bounds[0] to bounds[1]
        n_ray_images = n_ray_images[0]
        azs = tf.cast(tf.linspace(bounds[0,0], bounds[1,0], n_ray_images), dtype=def_dtype)
        els = tf.cast(tf.linspace(bounds[0,1], bounds[1,1], n_ray_images), dtype=def_dtype)
    if mode == 'rectangle':
        # Sample regularly on the rectangle defined by bounds, following the sampling pattern given
        azs = tf.cast(tf.linspace(bounds[0,0], bounds[1,0], n_ray_images[0]), dtype=def_dtype)
        els = tf.cast(tf.linspace(bounds[0,1], bounds[1,1], n_ray_images[1]), dtype=def_dtype)
        n_ray_images = n_ray_images[0]*n_ray_images[1]
        azs, els = tf.meshgrid(azs, els)
    azs = tf.reshape(azs, [-1])
    els = tf.reshape(els, [-1])
    H, W, _ = dataset['train_imgs'][base_i].shape
    H_sc = H//light_df
    W_sc = W//light_df
    focal_sc = dataset['train_focals'][base_i]/light_df
    
    sc_rays_o, sc_rays_d, sc_light_dirs=[], [], []
    for i in range(n_ray_images):
        az, el = azs[i], els[i]
        sc_view_dir = tf.reshape(tf.convert_to_tensor([az, el], dtype=def_dtype), [1,2])
        pose_sc = data_handling.pose_spherical(az, -el, focal_sc)
        rays_o, rays_d = get_rays_zoom(H_sc, W_sc, focal_sc, pose_sc, zoom_factor)
        rays_o = tf.reshape(rays_o, [H_sc, W_sc, 3])
        rays_d = tf.reshape(rays_d, [H_sc, W_sc, 3])
#         rays_o = array_to_patches(rays_o, bc_size)
        sc_rays_o.append(rays_o)
        sc_rays_d.append(rays_d)
        sc_light_dirs.append(tf.ones([rays_o.shape[0],rays_o.shape[1],1])@sc_view_dir)

    sc_rays_o = tf.reshape(tf.convert_to_tensor(sc_rays_o), [-1,3])
    sc_rays_d = tf.reshape(tf.convert_to_tensor(sc_rays_d), [-1,3])
    sc_light_dirs = tf.reshape(tf.convert_to_tensor(sc_light_dirs), [-1,2])

    return {'rays_o':sc_rays_o, 'rays_d':sc_rays_d, 'light_dirs':sc_light_dirs}

def shuffle_rays(rays):
    """
    
    Randomly shuffle rays using tf.random.shuffle.
    
    Parameters: 
    rays (dict): set of rays to shuffle
    
    Returns: 
    rays (dict): shuffled rays
    
    """
    rand_indices = tf.random.shuffle(tf.convert_to_tensor(range(rays['rays_o'].shape[0]), dtype=tf.int32))
    return {k: tf.gather(v, rand_indices) for k, v in iter(rays.items())}
    
def get_ray_batch(rays, start, end):
    """Retrieve subset of rays from start index to end index"""
    sub_rays = {k: v[start:end,...] for k, v in iter(rays.items())}
    return sub_rays

def concat_rays(rays1, rays2):
    """Group two sets of rays"""
    return {k: tf.concat([v, rays2[k]], axis=0) for k, v in iter(rays1.items())}

#### Sampling
def sample_alt(rays_o, rays_d, bounds, N_samples):
    """
    
    Sample along the rays, using the altitude bounds.
    
    Parameters:
    rays_o (Tensor[H*W, 3]): rays origin.
    rays_d (Tensor[H*W, 3]): rays direction.
    bounds (float, float): minimum and maximum altitude.
    N_samples (int): number of samples along the rays.
    
    Returns:
    z_vals (Tensor[H*W, N_samples]): depth values along the rays.
    
    """
    alt_min, alt_max = def_dtype(bounds[0]), def_dtype(bounds[1])
    alt_vals = tf.linspace(alt_max, alt_min, N_samples)
    z_vals = (alt_vals - rays_o[...,None,2])/rays_d[...,None,2]
    return z_vals
    
def sample_nf(bounds, N_samples):
    """
    
    Sample using the near far distance.
    
    Parameters:
    bounds (float, float): near and far distance.
    N_samples (int): number of samples.
    
    Returns:
    z_vals (Tensor[N_samples]): depth values.
    
    """
    near, far = bounds[0], bounds[1]
    z_vals = tf.linspace(near, far, N_samples)
    return z_vals

def uniform_bin_sampling(z_vals):
    """
    
    Randomly perturb a tensor of depth values of multiple rays.
    
    Each sample is pushed between its current position and the position of the next sample by a random amount, to achieve a continuous integration over a large number of iterations.
    
    Parameters:
    z_vals (Tensor[N_rays, N_samples]): Initial depth values.
    
    Returns:
    z_vals (Tensor[N_rays, N_samples]): Perturbed depth values.
 
    """
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = tf.concat([mids, z_vals[..., -1:]], -1)
    lower = tf.concat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = tf.random.uniform(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand
    return z_vals

def sample_pdf(bins, weights, N_samples, det=False):
    """
    
    Sample new integration depths from a discrete probability density function given by bins and weights.
    
    Parameters:
    bins (Tensor[N_b]): Depth values.
    weights (Tensor[N_b]): Importance of each bin.
    N_samples (int): number of samples along the ray.
    det (bool) : random sampling in the pdf
    
    Returns:
    samples (Tensor[N_samples]): New samples.
 
    
    """
    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    return samples

def resample_importance(z_vals, weights, N_importance, render_mode='nf'):
    """
    
    Importance sampling.

    Obtain additional integration samples to evaluate based on the alpha-compositing weights.
    
    Parameters:
    z_vals (Tensor[N_samples]): Initial sample depths.
    weights (Tensor[N_samples]): Alpha-compositing weights.
    N_importance (int): Number of importance samples along the ray
    
    Returns:
 
    z_vals (Tensor[N_samples+N_importance]): all depth values (initial + importance).
    
    """
    
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    if render_mode == 'nf':
        # Sample according to the probability density function
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, True)
    elif render_mode == 'alt':
        # Add first and last points to fix edge cases where weights is 1 only on the first or last element
        # Not the case for near-far rendering where the final distance is "infinity"
        z_vals_mid = tf.concat([z_vals[...,0:1], z_vals_mid, z_vals[...,-1:]], axis=-1)
        # Sample according to the probability density function
        z_samples = sample_pdf(z_vals_mid, weights, N_importance, True)
    else:
        print("Unrecognized rendering mode")
        return
    z_samples = tf.stop_gradient(z_samples)
    # Obtain all points to evaluate color, density at.
    z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
    return z_vals


#### Rendering
def render_rays(model, arg_dict, ray_batch, rets=['rgb'], rand=False, raw_noise_std=(0.0, 0.0), rescale_factor=None, chunk=1024*256):
    """
    
    Main rendering function.

    Render outputs of a set of rays, used both for training and for inference. 
    
    Parameters:
    model (dict): TF model, dimensions and embedding functions as defined in models.generate_model
    arg_dict (dict): Global configuration variables
    ray_batch (dict): Target rays for rendering
    rets (list(string)): rendering outputs (see pts2outputs)
    rand (bool): Randomly perturb sample positions. Set to True for training and False for inference.
    raw_noise_std (float, float): Resp. strength of noise on opacity and on shadow outputs.
    rescale_factor (float): Global rescale factor for dataset.
    chunk (int): Chunking parameter for memory management, reduce if overflow (slower execution).

    Returns:
    ret_dict (dict): desired outputs according to 'rets' parameter
    
    """
    N_samples = arg_dict["rend.nsamples"]
    N_importance = arg_dict["rend.nimportance"]
    fine_render = (N_importance > 0)
    if rescale_factor is None:
        rescale_factor = arg_dict["rend.rescale"]
    # True sampling distance is original sampling distance multiplied by added image downsample factor
    image_sd = arg_dict['data.image.sd'] * arg_dict['data.image.df']
    # Convert bounds from original units to pixel units
    if arg_dict['rend.mode'] == 'alt':
        bounds = [arg_dict['rend.mode.alt.min']/image_sd, arg_dict['rend.mode.alt.max']/image_sd]
    elif arg_dict['rend.mode'] == 'nf':
        bounds = [arg_dict['rend.mode.nf.near']/image_sd, arg_dict['rend.mode.nf.far']/image_sd]
    
    view_dirs = ray_batch['view_dirs'] if 'view_dirs' in ray_batch.keys() else None
    light_dirs = ray_batch['light_dirs'] if 'light_dirs' in ray_batch.keys() else None
    
    # Sample z values (distance along ray from ray_o)
    if arg_dict["rend.mode"] == 'alt':
        z_vals = sample_alt(ray_batch['rays_o'], ray_batch['rays_d'], bounds, N_samples)
    elif arg_dict["rend.mode"] == 'nf':
        z_vals = sample_nf(bounds, N_samples)
    else:
        print("Unrecognized rendering mode " + arg_dict["rend.mode"])
        return
    
    # Randomly perturb z values
    if rand:
        z_vals = uniform_bin_sampling(z_vals)
    
    # Calculate x,y,z position of query points in absolute ref frame
    pts = ray_batch['rays_o'][...,None,:] + ray_batch['rays_d'][...,None,:] * z_vals[...,:,None]
    pts_flat = tf.reshape(pts, [-1,3])
    
    # Perform the rendering
    if fine_render:
        # First pass : coarse rendering to estimate weights
        ret_dict_coarse = pts2outputs(model, pts_flat, z_vals, N_samples, light_dirs, view_dirs, norm=rescale_factor, raw_noise_std=raw_noise_std, mode=arg_dict["rend.mode"], rets=['weights', 'z_vals'], chunk=chunk)
        # Obtain new sample positions
        z_vals = resample_importance(ret_dict_coarse['z_vals'], ret_dict_coarse['weights'], N_importance, render_mode=arg_dict["rend.mode"])
        pts = ray_batch['rays_o'][..., None, :] + ray_batch['rays_d'][..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        pts_flat = tf.reshape(pts, [-1,3])
        # Final pass on all samples
    ret_dict = pts2outputs(model, pts_flat, z_vals, N_samples + N_importance, light_dirs, view_dirs,  norm=rescale_factor, raw_noise_std=raw_noise_std, mode=arg_dict["rend.mode"], rets=rets, chunk=chunk)
    return ret_dict

def pts2raw(pts_flat, network_fn, embed_fn_pos):
    """
    
    Convert points to raw (unactivated) network outputs. These correspond to the density and 3D color (radiance), and optionally solar visiblity and sky color. 
    
    Parameters:
    pts_flat (Tensor[N_pts, 3 + L_embed_dir]): 3D Points to apply the network.
    network_fn (keras.model): Function of the neural network.
    embed_fn_pos (function): positional embedding function as defined in models.py
    
    Returns:
    raw (Tensor[N_pts, N_outputs]: model outputs
    
    """
    # Positional encoding
    if embed_fn_pos is not None:
        pts_flat = tf.concat([embed_fn_pos(pts_flat[:,:3]), pts_flat[:,3:]], axis=-1)
    # Apply network function
    raw = network_fn(pts_flat)
    return raw

def dir_encoding(pts_flat, view_dirs, light_dirs, N_samples, embed_fn_dir):
    """
    
    Add directional encoding to a set of points.
    
    Parameters:
    pts_flat (Tensor[N_rays*N_samples, 3]): 3D Points to apply the encoding.
    view_dirs (Tensor[N_rays, 2]): Viewing directions.
    light_dirs (Tensor[N_rays, 2]): Lighting directions.
    N_samples (int): Number of samples along each ray.
    embed_fn_dir (function): directional embedding function as defined in models.py    
    
    Returns:
    pts_flat (Tensor[N_rays*N_samples, 3 + L_embed_dir]): Points with directional embedding.
    
    """
    if view_dirs is not None:
        view_dirs = tf.broadcast_to(view_dirs[...,None], [view_dirs.shape[0], view_dirs.shape[1] , N_samples])
        view_dirs = tf.transpose(view_dirs, perm=[0,2,1])
        v_dirs_flat = tf.reshape(view_dirs, [-1,2])
        v_dirs_flat = embed_fn_dir(v_dirs_flat)
        pts_flat = tf.concat([pts_flat, v_dirs_flat], axis=-1)
    if light_dirs is not None:
        light_dirs = tf.broadcast_to(light_dirs[...,None], [light_dirs.shape[0], light_dirs.shape[1] , N_samples])
        light_dirs = tf.transpose(light_dirs, perm=[0,2,1])
        s_dirs_flat = tf.reshape(light_dirs, [-1,2])
        s_dirs_flat = embed_fn_dir(s_dirs_flat)
        pts_flat = tf.concat([pts_flat, s_dirs_flat], axis=-1)
    return pts_flat

def batchify(fn, chunk=1024*256):
    """
    
    Batching function to reduce memory footprint during ray rendering.
    
    Parameters:
    fn (function): Function to be applied to inputs.
    
    Returns:
    fn_batch (function): Function that will apply fn to inputs in batches.
    
    """
    return lambda inputs : tf.concat([fn(inputs[i:i+chunk,:]) for i in range(0, inputs.shape[0], chunk)], 0)
    
def pts2outputs(model, pts_flat, z_vals, N_samples, light_dirs=None, view_dirs=None, norm=1.0, raw_noise_std=(1.0, 1.0), mode='nf', rets=['rgb'], chunk=1024*256):
    """
    
    Compute outputs of a set of points.
    
    Performs alpha-compositing for a set of target points. Runs the network inference and shading model. Outputs are only generated if requested in the rets parameter.
    
    Parameters:
    model (dict): TF model, dimensions and embedding functions as defined in models.generate_model
    pts_flat (Tensor[N_rays*N_samples, 3]): Points to compute outputs of alpha-compositing
    z_vals (Tensor[N_rays*N_samples]): Target depths for rendering
    N_samples (int): Number of samples along each ray.
    view_dirs (Tensor[N_rays, 2]): Viewing directions.
    light_dirs (Tensor[N_rays, 2]): Lighting directions.
    norm (float): Rescale factor given by calculate_rescale_factor
    raw_noise_std (float, float): Resp. strength of noise on opacity and on shadow outputs.
    mode (string): 'nf' or 'alt_max' depending on the sampling mode.
    rets (list(string)): Requested rendering outputs.  
    chunk (int): Chunking parameter for memory management, reduce if overflow (slower execution).
    
    Returns:
    ret_dict (dict): Dictionary of outputs with the following key-value associations.
    rgb (Tensor[N_rays, 3]): integrated (and shaded) RGB
    depth (Tensor[N_rays]): estimated surface depth
    weights (Tensor[N_rays, N_samples]): alpha-compositing weights
    trans (Tensor[N_rays]): accumulated transparency (final trans value)
    acc (Tensor[N_rays]): sum of alpha-compositng weights
    z_vals (Tensor[N_rays*N_samples]): depth values used for rendering
    no_shadow (Tensor[N_rays, 3]): albedo rendering (no shadows)
    sky_only (Tensor[N_rays, 3]): only sky illumination (no direct light)
    ret_sun (Tensor[N_rays]): integrated solar visibility function
    ret_shadow_loss (Tensor[N_rays]): Shadow loss between solar visibility and transparency along solar ray
    sky (Tensor[N_rays, 3]): visualize irradiance at surface
    
    """
    
    # Normalize so that the largest extent fits within a -1, 1 cube
    if norm != 1.0:
        pts_flat = pts_flat/norm
    
    # Extract model parameters
    network_fn = model["model"]
    embed_fns = model["emb"]
    model_dims = model["dim"]
    n_in = np.sum(np.array(model_dims["in"]))
    n_out = np.sum(np.array(model_dims["out"]))

    pts_flat = dir_encoding(pts_flat, view_dirs, light_dirs, N_samples, embed_fns[1])

    # Convert points to raw values with batching for memory
    raw = batchify(lambda pts: pts2raw(pts, network_fn, embed_fns[0]))(pts_flat)
    raw = tf.reshape(raw, [-1 , N_samples, n_out])
    # Add noise to the opacity prediciton to regularlize 
    noise, noise_s = 0., 0.
    if raw_noise_std[0] > 0:
        noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std[0]
    if (raw_noise_std[1] > 0) and (model_dims["out"][2] != 0):
        noise_s = tf.random.normal(raw[..., 4].shape) * raw_noise_std[1]
                
    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3] + noise)
    rgb = tf.math.sigmoid(raw[...,:3]) 

    # Do volume rendering
    # Rescale distances w.r.t rescale factor
    dists = (z_vals[..., 1:] - z_vals[..., :-1])/norm
    if mode == 'alt':
        #Replicate last distance as distance for last point
        dists = tf.concat([dists, dists[...,-2:-1]], axis = -1)
        alpha = 1.-tf.exp(-sigma_a * dists)
        trans = tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * trans
        #Replace last weight with sum of others to always get sum of weights = 1
        last_weight = tf.convert_to_tensor(1.0-tf.reduce_sum(weights[...,:-1], axis=-1))
        last_weight = tf.reshape(last_weight, [-1,1])
        weights = tf.concat([weights[...,:-1], last_weight], axis=-1)
    elif mode == 'nf':
        # The 'distance' from the last integration time is infinity
        dists = tf.concat([dists, tf.broadcast_to([1e10], dists[..., :1].shape)], axis = -1)
        alpha = 1.-tf.exp(-sigma_a * dists)
        trans = tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * trans
    # Shading options
    if (model_dims["out"][2] == 0) and (model_dims["out"][3] == 0):
        #No shading
        integrated_rgb = tf.reduce_sum(weights[...,None] * rgb, -2)
    if (model_dims["out"][2] == 1) and (model_dims["out"][3] == 0):
        #Sun light only
        s = tf.nn.sigmoid(raw[...,4] + noise_s)
        integrated_rgb = tf.reduce_sum(weights[...,None] * rgb * s, -2)
    if (model_dims["out"][2] == 0) and (model_dims["out"][3] == 3):
        #Sky light only
        sky = tf.math.sigmoid(raw[...,4:7])
        integrated_rgb = tf.reduce_sum(weights[...,None] * rgb * sky, -2)
    if (model_dims["out"][2] == 1) and (model_dims["out"][3] == 3):
        #Sun + Sky
        s = tf.nn.sigmoid(raw[...,4] + noise_s)
        sky = tf.math.sigmoid(raw[...,5:8])
        li = s[...,None] + sky*(1.0-s[...,None])
        integrated_rgb = tf.reduce_sum(weights[...,None] * rgb * li, -2)

    # Return dictionary
    ret_dict={}
    if 'rgb' in rets:
        ret_dict['rgb'] = integrated_rgb
    if 'depth' in rets:
        ret_dict['depth'] = tf.reduce_sum(weights * z_vals, -1)
    if 'weights' in rets:
        ret_dict['weights'] = weights
    if 'trans' in rets:
        ret_dict['trans'] = trans[:, -1]
    if 'acc' in rets:
        ret_dict['acc'] = tf.reduce_sum(weights, -1)
    if 'z_vals' in rets:
        ret_dict['z_vals'] = z_vals
    if 'no_shadow' in rets:
        ret_dict['no_shadow'] = tf.reduce_sum(weights[...,None] * rgb, -2)
    if 'sky_only' in rets:
        ret_dict['sky_only'] = tf.reduce_sum(weights[...,None] * rgb * sky, -2)
    if 'ret_sun' in rets:
        ret_dict['ret_sun'] = tf.reduce_sum(tf.stop_gradient(weights[...,None]) * s[...,np.newaxis], -2)
    if 'ret_shadow_loss' in rets:
        ret_dict['ret_shadow_loss']= tf.reduce_mean(tf.math.square(tf.stop_gradient(trans[..., None] + weights[...,None]) - s[...,None]), -2)
    if 'sky' in rets:
        ret_dict['sky'] = tf.reduce_sum(weights[...,None]*(s[...,None] + sky*(1.0-s[...,None])),-2)
    return ret_dict

def render_image(model, arg_dict, hwf, pose, zoom_factor, light_dirs, view_dirs, rets=['rgb']):
    """
    
    Compute an output image for one pose configuration.
    
    Similar to render_rays but for an image, the outputs are defined in the same way. 
    
    Parameters:
    model (dict): TF model, dimensions and embedding functions as defined in models.generate_model
    arg_dict (dict): Global configuration variables
    hwf (int, int, float): H, W and Focal of requested image.
    zoom_factor: Optional zoom factor 
    light_dirs (Tensor[1, 2]): Lighting direction.
    view_dirs (Tensor[1, 2]): Viewing direction.
    rets (list(string)): Requested rendering outputs.  
    
    Returns:
    ret_dict (dict): Dictionary of outputs reshaped to [H, W, x] where x can be 3 or 1 depending on which outputs is requested.
    
    """
    H, W, focal = hwf
    el = view_dirs[0,1]
    ray_batch = {}
    # Handle special unzoom mode for satellite images
    if arg_dict['rend.unzoom']:
        ray_batch['rays_o'], ray_batch['rays_d'] = get_rays_zoom(H, W, focal, pose, zoom_factor/np.sin(el))
    else:
        ray_batch['rays_o'], ray_batch['rays_d'] = get_rays_zoom(H, W, focal, pose, zoom_factor)
    # Constant view and light directions across the scene
    if arg_dict['model.ins.views']:
        ray_batch['view_dirs'] = tf.ones([H*W,1])@view_dirs
    if arg_dict['model.ins.light']:
        ray_batch['light_dirs'] = tf.ones([H*W,1])@light_dirs
        ray_batch['light_vectors'] = tf.broadcast_to(data_handling.pose_spherical(light_dirs[0, 0], -light_dirs[0, 1], 1.0)[:3,2], [H*W,3])
    # Rescale factor for image size H, W
    diag = semi_diagonal(H, W)
    # Ray rendering
    ret_dict = render_rays(model, arg_dict, ray_batch, rets=rets, rand=False, raw_noise_std=(0.0, 0.0), rescale_factor=diag, chunk=1024*128)
    ## Reshape results to [H, W, ...]
    if 'rgb' in rets:
        ret_dict['rgb'] = tf.reshape(tf.clip_by_value(ret_dict['rgb'], 0.0, 1.0) ,[H, W, 3])
    if 'depth' in rets:
        ret_dict['depth'] = tf.reshape(ret_dict['depth'], [H, W])
    if 'acc' in rets:
        ret_dict['acc'] = tf.reshape(ret_dict['acc'], [H, W])
    if 'no_shadow' in rets:
        ret_dict['no_shadow'] = tf.reshape(tf.clip_by_value(ret_dict['no_shadow'], 0.0, 1.0), [H, W, 3])
    if 'sky_only' in rets:
        ret_dict['sky_only'] = tf.reshape(ret_dict['sky_only'], [H, W, 3])
    if 'ret_sun' in rets:
        ret_dict['ret_sun'] = tf.reshape(ret_dict['ret_sun'],[H, W])
    if 'ret_shadow_loss' in rets:
        ret_dict['ret_shadow_loss'] = tf.reshape(ret_dict['ret_shadow_loss'],[H, W])
    if 'sky' in rets:
        ret_dict['sky'] =  tf.reshape(tf.clip_by_value(ret_dict['sky'], 0.0, 1.0),[H, W, 3])
    return ret_dict    
     
def render_dataset(dataset, model, rets, arg_dict, zoom_factor=1.0):
    """
    
    Compute outputs for the whole dataset.
    
    Similar to render_image but for a dataset, the outputs are defined in the same way. 
    
    Parameters:
    dataset (dict): 
    model (dict): TF model, dimensions and embedding functions as defined in models.generate_model
    rets (list(string)): Requested rendering outputs.  
    arg_dict (dict): Global configuration variables
    zoom_factor: Optional zoom factor 
    
    Returns:
    rendered_images (dict): Two separate lists of train and test rendering dicts (see render_image).
    
    """
    
    rendered_images={'train_rend':[], 'test_rend':[]}
    # Train images
    train_imgs = dataset['train_imgs']
    poses, focals, view_dirs, light_dirs = dataset['train_poses'], dataset['train_focals'], dataset['train_view_dirs'], dataset['train_light_dirs']
    for img, pose, focal, view_dir, light_dir in zip(train_imgs, poses, focals, view_dirs, light_dirs):
        hwf = img.shape[0], img.shape[1], focal
        ret_dict = render_image(model, arg_dict, hwf, pose, zoom_factor, light_dir, view_dir, rets=rets)
        rendered_images['train_rend'].append(ret_dict)
    
    # Test images
    test_imgs = dataset['test_imgs']
    poses, focals, view_dirs, light_dirs = dataset['test_poses'], dataset['test_focals'], dataset['test_view_dirs'], dataset['test_light_dirs']
    for img, pose, focal, view_dir, light_dir in zip(test_imgs, poses, focals, view_dirs, light_dirs):
        hwf = img.shape[0], img.shape[1], focal
        ret_dict = render_image(model, arg_dict, hwf, pose, zoom_factor, light_dir, view_dir, rets=rets)
        rendered_images['test_rend'].append(ret_dict)
    return rendered_images

def semi_diagonal(H, W):
    """Semi-diagonal of a rectangle"""
    return np.sqrt((H/2)**2 + (W/2)**2)

def calculate_rescale_factor(dataset):
    """Compute the rescale factor, used to ensure that the scene stays within the [-1, 1] bounds for the network."""
    H, W, _ = dataset['train_imgs'][0].shape
    return semi_diagonal(H, W)

