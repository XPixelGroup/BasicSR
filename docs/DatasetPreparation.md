# Dataset Preparation

[English](DatasetPreparation.md) **|** [简体中文](DatasetPreparation_CN.md)

#### Contents

1. [Data Storage Format](#Data-Storage-Format)
    1. [How to Use](#How-to-Use)
    1. [How to Implement](#How-to-Implement)
    1. [LMDB Description](#LMDB-Description)
    1. [Data Pre-fetcher](#Data-Pre-fetcher)
1. [Image Super-Resolution](#Image-Super-Resolution)
    1. [DIV2K](#DIV2K)
    1. [Common Image SR Datasets](#Common-Image-SR-Datasets)
1. [Video Super-Resolution](#Video-Super-Resolution)
    1. [REDS](#REDS)
    1. [Vimeo90K](#Vimeo90K)
1. [StylgeGAN2](#StyleGAN2)
    1. [FFHQ](#FFHQ)

## Data Storage Format

At present, there are three types of data storage formats supported:

1. Store in `hard disk` directly in the format of images / video frames.
1. Make [LMDB](https://lmdb.readthedocs.io/en/release/), which could accelerate the IO and decompression speed during training.
1. [memcached](https://memcached.org/) or [CEPH](https://ceph.io/) are also supported, if they are installed (usually on clusters).

#### How to Use

At present, we can modify the configuration yaml file to support different data storage formats. Taking [PairedImageDataset](../basicsr/data/paired_image_dataset.py) as an example, we can modify the yaml file according to different requirements.

1. Directly read disk data.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    io_backend:
      type: disk
    ```

1. Use LMDB.
We need to make LMDB before using it. Please refer to [LMDB description](#LMDB-Description). Note that we add meta information to the original LMDB, and the specific binary contents are also different. Therefore, LMDB from other sources can not be used directly.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```

1. Use Memcached
Your machine/clusters mush support memcached before using it. The configuration file should be modified accordingly.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K_train_LR_bicubicX4_sub
    io_backend:
      type: memcached
      server_list_cfg: /mnt/lustre/share/memcached_client/server_list.conf
      client_cfg: /mnt/lustre/share/memcached_client/client.conf
      sys_path: /mnt/lustre/share/pymc/py3
    ```

#### How to Implement

The implementation is to call the elegant fileclient design in [mmcv](https://github.com/open-mmlab/mmcv). In order to be compatible with BasicSR, we have made some changes to the interface (mainly to adapt to LMDB). See [file_client.py](../basicsr/utils/file_client.py) for details.

When we implement our own dataloader, we can easily call the interfaces to support different data storage forms. Please refer to [PairedImageDataset](../basicsr/data/paired_image_dataset.py) for more details.

#### LMDB Description

During training, we use LMDB to speed up the IO and CPU decompression. (During testing, usually the data is limited and it is generally not necessary to use LMDB). The acceleration depends on the configurations of the machine, and the following factors will affect the speed:

1. Some machines will clean cache regularly, and LMDB depends on the cache mechanism. Therefore, if the data fails to be cached, you need to check it. After the command `free -h`, the cache occupied by LMDB will be recorded under the `buff/cache` entry.
1. Whether the memory of the machine is large enough to put the whole LMDB data in. If not, it will affect the speed due to the need to constantly update the cache.
1. If you cache the LMDB dataset for the first time, it may affect the training speed. So before training, you can enter the LMDB dataset directory and cache the data by: ` cat data.mdb > /dev/nul`.

In addition to the standard LMDB file (data.mdb and lock.mdb), we also add `meta_info.txt` to record additional information.
Here is an example:

**Folder Structure**

```txt
DIV2K_train_HR_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**meta information**

`meta_info.txt`, We use txt file to record for readability. The contents are:

```txt
0001_s001.png (480,480,3) 1
0001_s002.png (480,480,3) 1
0001_s003.png (480,480,3) 1
0001_s004.png (480,480,3) 1
...
```

Each line records an image with three fields, which indicate:

- Image name (with suffix): 0001_s001.png
- Image size: (480, 480,3) represents a 480x480x3 image
- Other parameters (BasicSR uses cv2 compression level for PNG): In restoration tasks, we usually use PNG format, so `1` represents the PNG compression level `CV_IMWRITE_PNG_COMPRESSION` is 1. It can be an integer in [0, 9]. A larger value indicates stronger compression, that is, smaller storage space and longer compression time.

**Binary Content**

For convenience, the binary content stored in LMDB dataset is encoded image by cv2: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. You can control the compression level by `compress_level`, balancing storage space and the speed of reading (including decompression).

**How to Make LMDB**
We provide a script to make LMDB. Before running the script, we need to modify the corresponding parameters accordingly. At present, we support DIV2K, REDS and Vimeo90K datasets; other datasets can also be made in a similar way.<br>
 `python scripts/create_lmdb.py`

#### Data Pre-fetcher

Apar from using LMDB for speed up, we could use data per-fetcher. Please refer to [prefetch_dataloader](../basicsr/data/prefetch_dataloader.py) for implementation.<br>
It can be achieved by setting `prefetch_mode` in the configuration file. Currently, it provided three modes:

1. None. It does not use data pre-fetcher by default. If you have already use LMDB or the IO is OK, you can set it to None.

    ```yml
    prefetch_mode: ~
    ```

1. `prefetch_mode: cuda`. Use CUDA prefetcher. Please see [NVIDIA/apex](https://github.com/NVIDIA/apex/issues/304#) for more details. It will occupy more GPU memory. Note that in the mode. you must also set `pin_memory=True`.

    ```yml
    prefetch_mode: cuda
    pin_memory: true
    ```

1. `prefetch_mode: cpu`. Use CPU prefetcher, please see [IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#) for more details. (In my tests, this mode does not accelerate)

    ```yml
    prefetch_mode: cpu
    num_prefetch_queue: 1  # 1 by default
    ```

## Image Super-Resolution

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### DIV2K

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is a widely-used dataset in image super-resolution. In many research works, a MATLAB bicubic downsampling kernel is assumed. It may not be practical because the MATLAB bicubic downsampling kernel is not a good approximation for the implicit degradation kernels in real-world scenarios. And there is another topic named *blind restoration* that deals with this gap.

**Preparation Steps**

1. Download the datasets from the [official DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).<br>
1. Crop to sub-images: DIV2K has 2K resolution (e.g., 2048 × 1080) images but the training patches are usually small (e.g., 128x128 or 192x192). So there is a waste if reading the whole image but only using a very small part of it. In order to accelerate the IO speed during training, we crop the 2K resolution images to sub-images (here, we crop to 480x480 sub-images). <br>
Note that the size of sub-images is different from the training patch size (`gt_size`) defined in the config file. Specifically, the cropped sub-images with 480x480 are stored. The dataloader will further randomly crop the sub-images to `GT_size x GT_size` patches for training. <br/>
    Run the script [extract_subimages.py](../scripts/extract_subimages.py):

    ```python
    python scripts/extract_subimages.py
    ```

    Remember to modify the paths and configurations if you have different settings.
1. [Optional] Create LMDB files. Please refer to [LMDB Description](#LMDB-Description). `python scripts/create_lmdb.py`. Use the `create_lmdb_for_div2k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_paired_image_dataset.py`.
Remember to modify the paths and configurations accordingly.
1. [Optional] If you want to use meta_info_file, you may need to run `python scripts/generate_meta_info.py` to generate the meta_info_file.

### Common Image SR Datasets

We provide a list of common image super-resolution datasets.

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
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
    <td><sub>10 gray low-resolution images without the ground-truth</sub></td>
  </tr>

  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a> (800 train and 100 validation)</sub></td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">official website</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
    <td><a href="https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar">official website</a></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
    <td>-</a></td>
  </tr>

  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a> / <a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scenes</sub></td>
  </tr>

  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

## Video Super-Resolution

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### REDS

[Official website](https://seungjunnah.github.io/Datasets/reds.html).<br>
We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. And we  rename the validation clips from 240 to 269.

**Validation Partition**

The official validation partition and that used in EDVR for competition are different:

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

All the left clips are used for training. Note that it it not required to explicitly separate the training and validation datasets; and the dataloader does that.

**Preparation Steps**

1. Download the datasets from the [official website](https://seungjunnah.github.io/Datasets/reds.html).
1. Regroup the training and validation datasets: `python scripts/regroup_reds_dataset.py`
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/create_lmdb.py`. Use the `create_lmdb_for_reds` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_reds_dataset.py`.
Remember to modify the paths and configurations accordingly.

### Vimeo90K

[Official webpage](http://toflow.csail.mit.edu/)

1. Download the dataset: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).This is the Ground-Truth (GT). There is a `sep_trainlist.txt` file listing the training samples in the download zip file.
1. Generate the low-resolution images (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/create_lmdb.py`. Use the `create_lmdb_for_vimeo90k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_vimeo90k_dataset.py`.
Remember to modify the paths and configurations accordingly.

## StyleGAN2

### FFHQ

Training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset).

1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
1. Extract tfrecords to images or LMDBs. (TensorFlow is required to read tfrecords). For each resolution, we will create images folder or LMDB files separately.

    ```bash
    python scripts/extract_images_from_tfrecords.py
    ```
