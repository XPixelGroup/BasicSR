# BasicSR

BasicSR mainly contains 3 parts:

1. general SR models
1. [SRGAN model](https://arxiv.org/abs/1609.04802)
1. [SFTGAN model](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)

Now it supports 1 and 2.

### Table of Contents
1. [Introduction](#introduction)

### Introduction

1. Dependencies

python3 and pytorch 0.3.1

1. Commands

- train: `python3 train.py -opt options/train/SRResNet.json`
- test: `python3 test.py -opt options/test/test.json`

1. Codes descriptions

Please see [wiki pages](https://github.com/xinntao/BasicSR.wiki.git), which contains
- data
- options (all configuration descriptions)

1. How to train
    1. prepare your data (it's better to test whether the data is ok using `test_dataloader`)
    1. modify the corresponding training json file in `options/train/xxx.json`
    1. train the model with the command `python3 train.py -opt options/train/SRResNet.json`
1. How to test
    1. prepare your data and pretrained model.
    1. modify the corresponding testing json file in `options/test/test.json`
    1. test the model with the command `python3 test_LRinput.py -opt options/test/test.json`






