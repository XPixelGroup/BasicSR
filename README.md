# BasicSR

BasicSR repo mainly contains 3 parts:

1. general SR model [contemporary SR CNN models]
1. SRGAN model with [vanilla | lsgan | wgan-gp] GAN type.
1. SFTGAN model

### Table of Contents
1. [Introduction](#introduction)
2. [Code Structures](#code-structures)
1. [TODO List](#todo-list)

### Introduction

1. commands

* train: `python3 train.py -opt options/train/SRResNet.json`
* test: `python3 test.py -opt options/test/test.json`

### Code Structures

1. data

    use cv2 to process images.

* LRHR_pair (also for HR images only and down-sampling on-the-fly)

2. models


3. options

    use JSON file to load options.

    JSON files supports `//` comments

4. utils


### TODO list

- [ ] test code.
- [ ] on-the-fly down-sampling supports matlab bicubic.
- [ ] tensorboard logger
- [ ] SRGAN models
- [ ] unpair SRGAN dataloader
- [ ] support Y channel training and testing (the save_image function and metric may not work now.)
- [ ] multi-GPU training (test whether the current codes are OK.)
