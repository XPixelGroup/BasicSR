# Code Framework
The overall code framework is shown in the following figure. It mainly consists of four parts - `Config`, `Data`, `Model` and `Network`.

<p align="center">
   <img src="https://github.com/xinntao/public_figures/blob/master/BasicSR/code_framework.png" height="450">
</p>

Let us take the train commad `python train.py -opt options/train/train_esrgan.json` for example. A sequence of actions will be done after this command. 

- [`train.py`](https://github.com/xinntao/BasicSR/blob/master/codes/train.py) is called. 
- Reads the configuration (a json file) in [`options/train/train_esrgan.json`](https://github.com/xinntao/BasicSR/blob/master/codes/options/train/train_esrgan.json), including the configurations for data loader, network, loss, training strategies and etc. The json file is processed by [`options/options.py`](https://github.com/xinntao/BasicSR/blob/master/codes/options/options.py).
- Creates the train and validation data loader. The data loader is constructed in [`data/__init__.py`](https://github.com/xinntao/BasicSR/blob/master/codes/data/__init__.py) according to different data modes.
- Creates the model (is constructed in [`models/__init__.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/__init__.py) according to differnt model types). A model mainly consists of two parts - [network structure] and [model defination, e.g., loss definition, optimization and etc]. The network is constructed in [`models/network.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py) and the detailed structures are in [`models/modules`](https://github.com/xinntao/BasicSR/tree/master/codes/models/modules).
- Start to train the model. Other actions like logging, saving intermediate models, validation, updating learning rate and etc are also done during the training.  

Moreover, there are utils and userful scripts. A detailed description is provided as follows.


## Table of Contents
1. [Config](#config)
1. [Data](#data)
1. [Model](#model)
1. [Network](#network)
1. [Utils](#utils)
1. [Scripts](#scripts)

## Config
#### [`options/`](https://github.com/xinntao/BasicSR/tree/master/codes/options) Configure the options for data loader, network structure, model, training strategies and etc.

- `json` file is used to configure options and [`options/options.py`](https://github.com/xinntao/BasicSR/blob/master/codes/options/options.py) will convert the json file to python dict.
- `json` file uses `null` for `None`; and supports `//` comments, i.e., in each line, contents after the `//` will be ignored. 
- Supports `debug` mode, i.e, model name start with `debug_` will trigger the debug mode.
- The configuration file and descriptions can be found in [`options`](https://github.com/xinntao/BasicSR/tree/master/codes/options).

## Data
#### [`data/`](https://github.com/xinntao/BasicSR/tree/master/codes/data) A data loader to provide data for training, validation and testing.

- A separate data loader module. You can modify/create data loader to meet your own needs.
- Uses `cv2` package to do image processing, which provides rich operations.
- Supports reading files from image folder or `lmdb` file. For faster IO during training, recommand to create `lmdb` dataset first. More details including lmdb format, creation and usage can be found in our [lmdb wiki](https://github.com/xinntao/BasicSR/wiki/Faster-IO-speed).
- [`data/util.py`](https://github.com/xinntao/BasicSR/blob/master/codes/data/util.py) provides useful tools. For example, the `MATLAB bicubic` operation; rgb<-->ycbcr as MATLAB. We also provide [MATLAB bicubic imresize wiki](https://github.com/xinntao/BasicSR/wiki/MATLAB-bicubic-imresize) and [Color conversion in SR wiki](https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR).
- Now, we convert the images to format NCHW, [0,1], RGB, torch float tensor.

## Model
#### [`models/`](https://github.com/xinntao/BasicSR/tree/master/codes/models) Construct different models for training and testing.

- A model mainly consists of two parts - [network structure] and [model defination, e.g., loss definition, optimization and etc]. The network description is in the [Network part](#network).
- Based on the [`base_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/base_model.py), we define different models, e.g., [`SR_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SR_model.py), [`SRGAN_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SRGAN_model.py), [`SRRaGAN_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SRRaGAN_model.py) and [`SFTGAN_ACD_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SFTGAN_ACD_model.py).

## Network
#### [`models/modules/`](https://github.com/xinntao/BasicSR/tree/master/codes/models/modules) Construct different network architectures.

- The network is constructed in [`models/network.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py) and the detailed structures are in [`models/modules`](https://github.com/xinntao/BasicSR/tree/master/codes/models/modules).
- We provide some useful blocks in [`block.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/block.py) and it is flexible to construct your network structures with these pre-defined blocks.
- You can also easily write your own network architecture in a seperate file like [`sft_arch.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/sft_arch.py). 

## Utils
#### [`utils/`](https://github.com/xinntao/BasicSR/tree/master/codes/utils) Provide useful utilities.

- [logger.py](https://github.com/xinntao/BasicSR/blob/master/codes/utils/logger.py) provides logging service during training and testing.
- Support to use [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) to visualize and compare training loss, validation PSNR and etc. Installationand usage can be found [here](https://github.com/xinntao/BasicSR/tree/master/codes/utils).
- [progress_bar.py](https://github.com/xinntao/BasicSR/blob/master/codes/utils/progress_bar.py) provides a progress bar which can print the progress. 

## Scripts
#### [`scripts/`](https://github.com/xinntao/BasicSR/tree/master/codes/scripts) Privide useful scripts.
Details can be found [here](https://github.com/xinntao/BasicSR/tree/master/codes/scripts).
