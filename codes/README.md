Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing). Put them in `experiments/pretrained_models` folder.

Data preparation can be found in [`codes/data`](https://github.com/xinntao/BasicSR/tree/master/codes/data).

## Test for SR and SRGAN model
1. modify the configuration file in `options/test/test.json`
1. run command: `python3 test.py -opt options/test/test.json`

## Test for SFTGAN
1. obtain segmentation probability maps: `python3 test_seg.py`
1. run command: `python3 test_sftgan.py`

We have provided two versions of SFTGAN, one is converted from torch model; the other is training by pytorch. The pytorch training of SFTGAN is a bit different from that of torch.

## Training for SR
1. prepare the data: HR images OR HR-LR image pairs
1. modify the configuration file in `options/train/SR.json`
1. run command: `python3 train.py -opt options/train/SR.json`

## Training for SRGAN
**Pretraining is important**. A pretrained SR model is used to initialize the parameters.

1. prepare dataset: HR images OR HR-LR image pairs
1. prerapre the pretrained model. You can use the `SRResNet_bicx4_in3nf64nb16.pth` as the pretrained model. 
1. modify the configuration file in `options/train/SRGAN.json`
1. run command: `python3 train.py -opt options/train/SRGAN.json`

## SFTGAN 
*Pretraining is also very important*. We use a pretrained SRGAN model (trained on DIV2K) to initialize the SFTGAN model.

1. first prepare the segmentation probability maps for training data: run [`test_seg.py`](https://github.com/xinntao/BasicSR/blob/master/codes/test_seg.py). A pretrained segmentation model for 7 outdoor categories is provided in [Google Drive](https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing). We use [Xiaoxiao Li's codes](https://github.com/lxx1991/caffe_mpi) to train our segmentation model and turn it to pytorch model.
1. put the images and segmentation probability maps in a folder as described in [`codes/data`](https://github.com/xinntao/BasicSR/tree/master/codes/data).
1. transfer the pretrained model parameters to the SFTGAN model. 
    1. first train with `debug` mode and obtain a saved model.
    1. run [`transfer_params_sft.py`](https://github.com/xinntao/BasicSR/blob/master/codes/scripts/transfer_params_sft.py) to initialize the model.
    1. an initialized model has been provided in [Google Drive](https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing) named `sft_net_ini.pth`.
1. modify the configuration file in `options/train/SFTGAN.json`
1. run command: `python3 train.py -opt options/train/SFTGAN.json`
