There are three kinds of datasets: training dataset, validation dataset, and testing dataset. Usually, we do not explicitly distinguish between the validation and testing datasets in image/video restoration. So we use the validation/testing dataset in our description. <br/>
We recommend to use [LMDB](https://lmdb.readthedocs.io/en/release/) (Lightning Memory-Mapped Database) formats for the training datasets, and directly read images (using image folder) during validation/testing. So there is no need to prepare LMDB files for evaluation/testing datasets.

---
We organize the training datasets in LMDB format for **faster training IO speed**. If you do not want to use LMDB, you can also use the **image folder**.<br/>
Besides the standard LMDB folder, we add an extra `meta_info.pkl` file to record the **meta information** of the dataset, such as the dataset name, keys and resolution of each image in the dataset.

Take the DIV2K dataset in LMDB for example, the folder structure and meta information are as follows:
#### folder structure
```
- DIV2K800_sub.lmdb
|--- data.mdb
|--- lock.mdb
|--- meta_info.pkl
```
#### meta information in `meta_info.pkl`
`meta_info.pkl` is a python-pickled dict.

|     Key    |                           Value                           |
|:----------:|:---------------------------------------------------------:|
|    name    |                `DIV2K800_sub_GT`                    |
|    keys    |   [ `0001_s001`, `0001_s002`, ..., `0800_s040` ] |
| resolution |                       [ `3_480_480` ]                    |

If all the images in the LMDB file have the same resolution, only one copy of `resolution` is stored. Otherwise, each key has its corresponding `resolution`.

----

## Table of Contents
1. [Prepare DIV2K](#prepare-div2k)
1. [Common Image SR Datasets](#common-image-sr-datasets)
1. [Prepare Vimeo90K](#prepare-vimeo90k)
1. [Prepare REDS](#prepare-reds)

The following shows how to prepare the datasets in detail.<br/>
It is recommended to symlink the dataset root to `datasets`. If your folder structure is different, you may need to change the corresponding paths in config files.

## Prepare DIV2K
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is a widely-used dataset in image super-resolution. In many research works, a MATLAB bicubic downsampling kernel is assumed. It may not be practical because the MATLAB bicubic downsampling kernel is not a good approximation for the implicit degradation kernels in real-world scenarios. And there is another topic named **blind restoration** that deals with this gap.

We provide a demo script for DIV2K X4 datasets preparation.
```
cd codes/data_scripts
bash prepare_DIV2K_x4_dataset.sh
```
The specific steps are  as follows:

**Step 1**: Download the GT images and corresponding LR images from the [official DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).<br/>
Here are shortcuts for the download links:

| Name | links (training) | links (validation)|
|:----------:|:----------:|:----------:|
|Ground-Truth|[DIV2K_train_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)|[DIV2K_valid_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)|
|LRx2 (MATLAB bicubic)|[DIV2K_train_LR_bicubic_X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)|[DIV2K_valid_LR_bicubic_X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip)|
|LRx3 (MATLAB bicubic)|[DIV2K_train_LR_bicubic_X3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip)|[DIV2K_valid_LR_bicubic_X3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip)|
|LRx4 (MATLAB bicubic)|[DIV2K_train_LR_bicubic_X4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip)|[DIV2K_valid_LR_bicubic_X4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip)|
|LRx8 (MATLAB bicubic)|[DIV2K_train_LR_bicubic_X8](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip)|[DIV2K_valid_LR_bicubic_X8](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip)|

**Step 2**: Rename the downloaded LR images to have the same name as those of GT.<br/> Run the script `data_scripts/rename.py`. Remember to modify the folder path.

**Step 3 (optional)**: Generate low-resolution counterparts. <br/>If you have downloaded the LR datasets, skip this step. Otherwise, you can use the script `data_scripts/generate_mod_LR_bic.m` or `data_scripts/generate_mod_LR_bic.py` to generate LR images. Make sure the LR and GT pairs have the same name.

**Step 4**: Crop to sub-images. <br/>DIV2K has 2K resolution (e.g., 2048 × 1080) images but the training patches are usually very small (e.g., 128x128). So there is a waste if reading the whole image but only using a very small part of it. In order to accelerate the IO speed during training, we crop the 2K resolution images to sub-images (here, we crop to 480x480 sub-images). You can skip this step if your have a high IO speed.<br/>
Note that the size of sub-images is different from the training patch size (`GT_size`) defined in the config file. Specifically, the sub-images with 480x480 are stored in the LMDB files. The dataloader will further randomly crop the sub-images to `GT_size x GT_size` patches for training. <br/>
Use the script `data_scripts/extract_subimages.py` with `mode = 'pair'`. Remember to modify the following configurations if you have different settings:
```
GT_folder = '../../datasets/DIV2K/DIV2K800'
LR_folder = '../../datasets/DIV2K/DIV2K800_bicLRx4'
save_GT_folder = '../../datasets/DIV2K/DIV2K800_sub'
save_LR_folder = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4'
scale_ratio = 4
```
**Step 5**: Create LMDB files. <br/>You need to run the script `data_scripts/create_lmdb.py` separately for GT and LR images.<br/>

**Step 6**: Test the dataloader with the script `data_scripts/test_dataloader.py`.

This procedure is also applied to other datasets, such as 291 images, or your custom datasets.
```
@InProceedings{Agustsson_2017_CVPR_Workshops,
 author = {Agustsson, Eirikur and Timofte, Radu},
 title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {July},
 year = {2017}
}
```
## Common Image SR Datasets
We provide a list of common image super-resolution datasets. You can download the images from the official website or Google Drive or Baidu Drive.

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Baidu Drive</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="9"><a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray LR images without the ground-truth</sub></td>
  </tr>

  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a> (800 train and 100 validation)</sub></td>
    <td rowspan="3"><a href="https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing">Google Drive</a></td>
    <td rowspan="3"><a href="https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
  </tr>

  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scences</sub></td>
  </tr>

  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

## Prepare Vimeo90K
The description of the Vimeo90K can be found in [Open-VideoRestoration](https://xinntao.github.io/open-videorestoration/rst_src/datasets_sr.html#vimeo90k) and [the official webpage](http://toflow.csail.mit.edu/).<br/>

**Step 1**: Download the dataset<br/>
Download the [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip). This is the Ground-Truth (GT). There is a `sep_trainlist.txt` file recording the training samples in the download zip file.

**Step 2**: Generate the low-resolution images<br/>
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.

**Step 3**: Create LMDB files<br/>
Use the script `data_scripts/create_lmdb.py` to generate the lmdb files separately for GT and LR images. You need to modify the configurations in the script:
1) For GT
```
dataset = 'vimeo90K'
mode = 'GT'
```
2) For LR
```
dataset = 'vimeo90K'
mode = 'LR'
```

**Step 4**: Test the dataloader with the script `data_scripts/test_dataloader.py`.

```
@Article{xue2017video,
  author    = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  title     = {Video enhancement with task-oriented flow},
  journal   = {International Journal of Computer Vision},
  year      = {2017}
}
```

## Prepare REDS
We re-group the REDS training and validation sets as follows:

| name | from | total number |
|:----------:|:----------:|:----------:|
| REDS training | the original training (except 4 clips) and validation sets | 266 clips |
| REDS4 testing | 000, 011, 015 and 020 clips from the *original training set* | 4 clips |

The description of the REDS dataset can be found in [Open-VideoRestoration](https://xinntao.github.io/open-videorestoration/rst_src/datasets_sr.html#reds) and the [official website](https://seungjunnah.github.io/Datasets/reds.html).

**Step 1**: Download the datasets<br/>
You can download the REDS datasets from the [official website](https://seungjunnah.github.io/Datasets/reds.html). The download links are also sorted as follows:

| track | links (training) | links (validation)|links (testing)|
|:----------:|:----------:|:----------:|:----------:|
| Ground-truth| [train_sharp - part1](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_sharp_part1.zip), [part2](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_sharp_part2.zip), [part3](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_sharp_part3.zip) |[val_sharp](https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/REDS_validation_set/val_sharp.zip) | Not Available |- |
| SR-clean | [train_sharp_bicubic](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_sharp_bicubic.zip) | [val_sharp_bicubic](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/val_sharp_bicubic.zip) |[test_sharp_bicubic](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/test_sharp_bicubic.zip) |
| SR-blur)  | [train_blur_bicubic](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_bicubic.zip) | [val_blur_bicubic](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/val_blur_bicubic.zip) |[test_blur_bicubic](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/test_blur_bicubic.zip) |
| Deblurring  | [train_blur - part1](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_part1.zip),  [part2](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_part2.zip), [part3](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_part3.zip) | [val_blur](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/val_blur.zip) |[test_blur](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/test_blur.zip) |
| Deblurring - Compression  | [train_blur_comp - part1](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_comp_part1.zip),  [part2](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_comp_part2.zip), [part3](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/train_blur_comp_part3.zip) | [val_blur_comp](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/val_blur_comp.zip) |[test_blur_comp](https://data.vision.ee.ethz.ch/timofter/NTIRE19video/test_blur_comp.zip) |

**Step 2**: Re-group the datasets<br/>
We rename the clips in the original validation set, starting from 240 ... It can be accomplished by `data_scripts/regroup_REDS.py`.
Note that the REDS4 will be excluded in the data loader, so there is no need to remove the REDS4 explicitly.

**Step 3**: Create LMDB files<br/>
Use the script `data_scripts/create_lmdb.py` to generate the lmdb files separately for GT and LR frames. You need to modify the configurations in the script:
1) For GT (train_sharp)
```
dataset = 'REDS'
mode = 'train_sharp'
```
2) For LR (train_sharp_bicubic)
```
dataset = 'REDS'
mode = 'train_sharp_bicubic'
```
**Step 4**: Test the dataloader with the script `data_scripts/test_dataloader.py`.

```
@InProceedings{nah2019reds,
  author    = {Nah, Seungjun and Baik, Sungyong and Hong, Seokil and Moon, Gyeongsik and Son, Sanghyun and Timofte, Radu and Lee, Kyoung Mu},
  title     = {NTIRE 2019 challenges on video deblurring and super-resolution: Dataset and study},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2019}
}
```
