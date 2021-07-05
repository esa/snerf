# Shadow Neural Radiance Fields
This project shows the application of Shadow Neural Radiance Fields (S-NeRF) on Very High Spatial Resolution RGB imagery from WorldView-3, based on data from the IEEE TGRS Data Fusion Competition 2019. This code was used to produce the results in the paper by Derksen, Dawa, and Dario Izzo. **Shadow Neural Radiance Fields for Multi-view Satellite Photogrammetry**[^1]. This is the result of a joint research collaboration between the Advanced Concepts Team (ESTEC) and the φ-lab (ESRIN)

[^1]: arXiv preprint arXiv:2104.09877 (2021).

# Installation
The code is heavily based on TensorFlow-2.2.0, but also makes use of matplotlib, scikit-image, and gdal for image utilities. The conda environment required to run the code is contained in the `snerf_env.yml` file. The code is intended for use on a single GPU.

# Contents
This repository contains:
1. The source code of the project in the `snerf` folder, including training and plotting scripts `train.py` and `plots.py`
2. A demonstration Jupyter notebook, to reproduce some of the results shown in the paper and youtube video `snerf/snerf_demo.ipynb` based on a pre-trained model.
3. The data that was used to generate the results in the `data` folder.
4. Pre-trained models in the `models` folder, four areas in Jacksonville were selected for this paper : 004, 068, 214 and 260.

# Usage
Training a S-NeRF requires a configuration file defining the model parameters, training procudure, shading model, and logging capabilities. The description of these parameters can be found in `snerf/train.py`. The training is run via the training script `train.py` as follows (replace "XXX" with the area index 004, 068, 214 or 260).

```
python train.py --config ../configs/XXX_config.txt
```

This will produce a file called `model.npy` as well as the scores and loss logs in the outputs folder (as specified in the configuration). Once finished, the various outputs and scores are plotted with `snerf/plots.py`. 

```
python plots.py --config ../configs/XXX_config.txt
```

# Acknowledgements
The code is largely based on the Tensorflow implementation of the original NeRF authors, thanks to Mildenhall et. al. : https://github.com/bmild/nerf. 
The code for SIREN networks with the special initialization procedure was taken from https://github.com/titu1994/tf_SIREN.
Thank you to Dario Izzo, Marcus Maertens, Anne Mergy, Pablo Gomez and Gurvan Lecuyer from Advanced Concepts Team, and Bertrand Le Saux from φ-lab for collaboration on this project. 
