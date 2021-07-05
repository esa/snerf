"""
Definition of a general S-NeRF model. The shadow module can be disabled altogether, or set to only model direct light. Activation can be standard Rectified Linear Units (ReLU, as in Mildenhall et. al. 2020) or sine (Sinusiodal Implicit Representation Networks, SIREN, taken from tf_siren package). The depth and width of each sub-network (opacity, color, shadow) is also parametrizable.    
"""

import tensorflow as tf
import numpy as np

from tf_siren import SinusodialRepresentationDense
from tf_siren import SIRENModel

# Positional encoding
def posenc(x, L_embed):
    """
    
    Basic positional encoding with log-sampling of frequencies. Note that the same function can be used for directional encoding.
    
    Parameters:
    x (Tensor[N]): variables to apply positional encoding 
    L_embed (int): length of positional encoding
    
    Outputs:
    y (Tensor[N, L_embed*2+1]): positional encoding
    
    """
    rets = [x]
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2.**i * x))
    return tf.concat(rets, -1)

def posenc_no_x(x, L_embed):
    """Basic positional encoding with log-sampling of frequencies without the initial vector itself. See posenc."""
    rets = []
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2.**i * x))
    return tf.concat(rets, -1)

# Layers
def dense_siren(W, w0, ker='siren_uniform'):
    """
    
    Dense layer of sine activated neurons, based on the library tf_siren.
    
    Parameters:
    W (int): Width of layer.
    w0 (float): Base wavelength of sine neurons.
    ker (string): Initializer, set to 'siren_first_uniform' for first layer special initialization
    
    """
    return SinusodialRepresentationDense(W, activation='sine', w0=w0, use_bias=True, kernel_initializer=ker)

def dense_no_act(W):
    """Dense layer of width W, with no activation function"""
    return tf.keras.layers.Dense(W, activation=None)

def dense_relu(W):
    """Dense layer width W with ReLU activation function"""
    return tf.keras.layers.Dense(W, activation=tf.keras.layers.ReLU())

def generate_model(arg_dict):
    """
    
    Create and initialize a S-NeRF model according to the model parameters in the config file.
    
    Parameters:
    arg_dict (dict): Contains model size, activation functions, inputs and outputs.
    
    Ouputs:
    model (dict): Model, encodings and dimensions for ease of use.
        model (function): keras model 
        emb (function, function): positional and directional encodings
        dim (dict): input and output sizes of different inputs and outputs of the model 
        
    
    """
    siren_model = (arg_dict['model.act'] == 'sin')
    relu_model = (arg_dict['model.act'] == 'relu')
    if not (siren_model or relu_model):
        print("Unrecognized activation function")
        return None
    # Compute size of input and output dimensions of the network
    input_ch = 3
    input_ch_views = 2 if arg_dict['model.ins.views'] else 0
    input_ch_light = 2 if arg_dict['model.ins.light'] else 0
    
    output_ch_sigma = 1
    output_ch_rgb = 3
    output_ch_sh = 1 if arg_dict['model.outs.shad'] else 0
    output_ch_sky = 3 if arg_dict['model.outs.sky'] else 0
    
    model_dims = {'in':[input_ch, input_ch_views, input_ch_light],
                  'out':[output_ch_sigma, output_ch_rgb, output_ch_sh, output_ch_sky]}
    
    # Positional encoding
    if (arg_dict['model.emb.pos'] == 0) or siren_model:
        embed_fn_pos=(lambda x: tf.identity(x))
    else:
        embed_fn_pos=(lambda x: posenc(x, arg_dict['model.emb.pos']))
        input_ch += input_ch*2*arg_dict['model.emb.pos']
        
    # Directional encoding
    if (arg_dict['model.emb.dir'] == 0):
        embed_fn_dir=(lambda x: tf.identity(x))
    else:
        embed_fn_dir=(lambda x: posenc_no_x(x, arg_dict['model.emb.dir']))
        input_ch_views = input_ch_views*2*arg_dict['model.emb.dir']
        input_ch_light = input_ch_light*2*arg_dict['model.emb.dir']

    # Setup input layer
    inputs = tf.keras.Input(shape=(input_ch + input_ch_light + input_ch_views))
    inputs_pts, inputs_light, inputs_views = tf.split(inputs, [input_ch, input_ch_light, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    if input_ch_light > 0:
        inputs_light.set_shape([None, input_ch_light])
    if input_ch_views > 0:
        inputs_views.set_shape([None, input_ch_views])
    # Initial layer
    W = arg_dict['model.sigma.width']
    if siren_model:
        D_sigma = arg_dict['model.sigma.depth'] - 1
        layer_fn = lambda x : dense_siren(x, w0=arg_dict['model.act.sin.w0'])
        init_layer_fn = lambda x : dense_siren(x, w0=arg_dict['model.act.sin.w0'], ker='siren_first_uniform')
        outputs = init_layer_fn(W)(inputs_pts)
    if relu_model:
        D_sigma = arg_dict['model.sigma.depth']
        layer_fn = dense_relu
        outputs = inputs_pts
    
    # All other sigma layers
    for i in range(D_sigma):
        outputs = layer_fn(W)(outputs)
        if i in arg_dict['model.sigma.skips']:
            outputs = tf.concat([inputs_pts, outputs], -1)
    
    bottleneck = dense_no_act(W)(outputs)
    alpha_out = dense_no_act(1)(bottleneck)
    
    # Color layers
    outputs_rgb = bottleneck
    if input_ch_views > 0:
        outputs_rgb = tf.concat([outputs_rgb, inputs_views])
    for i in range(arg_dict['model.c.depth']):
        outputs_rgb = layer_fn(arg_dict['model.c.width'])(outputs_rgb)
    outputs_rgb = dense_no_act(3)(outputs_rgb)
    
    # Standard NeRF outputs
    outputs = tf.concat([outputs_rgb, alpha_out], -1)
    
    # Shadow function layers
    if arg_dict['model.outs.shad']:
        W_shad = arg_dict['model.shad.width']
        if siren_model:
            inputs_light_init = init_layer_fn(W_shad)(inputs_light)
        else:
            inputs_light_init = inputs_light
        outputs_shad = tf.concat([bottleneck, inputs_light_init], -1)
        for i in range(arg_dict['model.shad.depth']):
            outputs_shad = dense_relu(W_shad)(outputs_shad)
        outputs_shad = dense_no_act(1)(outputs_shad)
        outputs = tf.concat([outputs, outputs_shad], -1)
    
    # Sky color
    if arg_dict['model.outs.sky']:
        outputs_light = dense_relu(arg_dict['model.c.width'])(inputs_light)
        outputs_light = dense_no_act(3)(outputs_light)
        outputs = tf.concat([outputs, outputs_light], -1)
       
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return {'model' : model, 
            'emb' : (embed_fn_pos, embed_fn_dir),
            'dim' : model_dims}

def save_model(path, model):
    """Save model weights to path"""
    np.save(f"{path}model.npy", model['model'].get_weights())

def load_model(path, arg_dict):
    """Initialize model and load weights from path"""
    model = generate_model(arg_dict)
    model['model'].set_weights(np.load(path, allow_pickle=True))
    return model
