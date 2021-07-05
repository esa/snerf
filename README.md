# Shadow Neural Radiance Fields
This project shows the application of Shadow Neural Radiance Fields (S-NeRF) to Very High Spatial Resolution RGB imagery from WorldView-3. This code was used for the paper called **Shadow Neural Radiance Fields for Multi-view Satellite Photogrammetry** presented at CVPR 2021 - Workshop on Earth Vision. This is the result of a joint research collaboration between the Advanced Concepts Team (ESTEC) and the φ-lab (ESRIN). The paper is available (open access) [here](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.html). This repository is intended as a means to reproduce the results shown in the paper, and to stiumulate further research in this direction.

# Installation
The code is heavily based on TensorFlow-2.2.0, but also makes use of matplotlib, scikit-image, and gdal for image utilities. The conda environment required to run the code is contained in the `snerf_env.yml` file. The code is intended for use on a single CUDA-enabled GPU. 

# Contents
This repository contains:
1. The source code of the project in the `snerf` folder, including training and plotting scripts `train.py` and `plots.py`.
2. A demonstration Jupyter notebook, to reproduce some of the results shown in the paper, `snerf/snerf_demo.ipynb` based on a pre-trained model.
3. The data that was used to generate the results shown in the paper, in the `data` folder.
4. Pre-trained models in the `models` folder, four areas in Jacksonville were selected for this study : 004, 068, 214 and 260. S-NeRF requires a unique model to be trained for each area.

# Data
The original images were kindly collected and provided in open access by the IEEE GRSS organization, for the [Data Fusion Competition 2019](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019).
For the paper, only four scenes over Jacksonville were used. The original scenes have been cropped and rotated using the `sat_data_handling.py` script, located in the `scripts` folder. This script pre-processes the DFC2019 data and is provided for reference. The pre-processed images are available for download, at DOI 10.5281/zenodo.5070039. After decompression the images should be placed in the `data/` folder (e.g. `data/068/JAX_068_010_RGB_crop.tif`). If placed in a different directory than `data/`, the configuration files in `config/` should be adapted to point to the appropriate location.

# Usage
Training a S-NeRF requires a configuration file defining the model parameters, training procudure, shading model, and logging parameters. The description of the configuration parameters can be found in `snerf/train.py`. Training is run via the training script `train.py` as follows (replace "XXX" with the area index 004, 068, 214 or 260).

```
python train.py --config ../configs/XXX_config.txt
```

This will produce a file called `model.npy` as well as the scores and loss logs in the `outputs` folder (as specified in the configuration). Once finished, the various outputs and scores are plotted with `snerf/plots.py`. 

```
python plots.py --config ../configs/XXX_config.txt
```

# Acknowledgements
Thank you to Dario Izzo, Marcus Maertens, Anne Mergy, Pablo Gomez and Gurvan Lecuyer from Advanced Concepts Team, and Bertrand Le Saux from φ-lab for collaboration on this project. 

The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest.

The code is based on the Tensorflow implementation of the authors of Neural Radiance Fields, https://github.com/bmild/nerf (distributed under MIT Licence), thanks to Ben Mildenhall, Daniel Duckworth, Matthew Tancik for their ground-breaking work.

The code for SIREN networks with the special initialization procedure is from https://github.com/titu1994/tf_SIREN, (distributed under MIT Licence), thanks to Somshubra Majumdar and other contributors.
