# 数据准备

[English](DatasetPreparation.md) **|** [简体中文](DatasetPreparation_CN.md)

#### 目录

1. [数据存储形式](#数据存储形式)
    1. [如何使用](#如何使用)
    1. [如何实现](#如何实现)
    1. [LMDB具体说明](#LMDB具体说明)
    1. [预读取数据](#预读取数据)
1. [图像数据](#图像数据)
    1. [DIV2K](#DIV2K)
    1. [其他常见图像超分数据集](#其他常见图像超分数据集)
1. [视频帧数据](#视频帧数据)
    1. [REDS](#REDS)
    1. [Vimeo90K](#Vimeo90K)
1. [StylgeGAN2](#StyleGAN2)
    1. [FFHQ](#FFHQ)

## 数据存储形式

目前支持的数据存储形式有以下三种:

1. 直接以图像/视频帧的格式存放在硬盘
2. 制作成 [LMDB](https://lmdb.readthedocs.io/en/release/). 训练数据使用这种形式, 一般会加快读取速度.
3. 若是支持 [Memcached](https://memcached.org/), 则可以使用. 它们一般应用在集群上.

#### 如何使用

目前, 我们可以通过 configuration yaml 文件方便的修改. 以支持DIV2K的 [PairedImageDataset](../basicsr/data/paired_image_dataset.py) 为例, 根据不同的要求修改yaml文件:

1. 直接读取硬盘数据

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    io_backend:
      type: disk
    ```

1. 使用LMDB.
在使用前需要先制作LMDB, 参见 [LMDB具体说明](#LMDB具体说明), 注意我们在原有的 LMDB 上, 新增加了 meta 信息, 而且具体保存二进制内容也不同, 因此其他来源的LMDB并不能直接拿过来使用.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```

1. 使用Memcached
机器/集群需要支持 Memcached. 具体的配置文件根据实际的 Memcached 需要进行修改:

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

#### 如何实现

实现是调用了[MMCV](https://github.com/open-mmlab/mmcv) 优雅的 FileClient 设计. 为了兼容 BasicSR, 我们对接口做了一些改动 (主要是为了适应LMDB), 参见 [file_client.py](../basicsr/utils/file_client.py).

在实现我们自己的 dataloader 的时候, 可以方便地调用接口, 以实现对不同数据存储形式的支持, 具体可以参考 [PairedImageDataset](../basicsr/data/paired_image_dataset.py) 的写法.

#### LMDB具体说明

我们在训练的时候使用 LMDB 存储形式可以加快IO和CPU解压缩的速度 (测试的时候数据较少, 一般就没有太必要使用 LMDB). 其具体的加速要根据机器的配置来, 以下几个因素会影响:

1. 有的机器设置了定时清理缓存, 而 LMDB 依赖于缓存. 因此若一直缓存不进去, 则需要检查一下. 一般 `free -h` 命令下, LMDB 占用的缓存会记录在 `buff/cache` 条目下面
1. 机器的内存是否足够大, 能够把整个 LMDB 数据都放进去. 如果不是, 则它由于需要不断更换缓存, 会影响速度
1. 若是第一次缓存 LMDB 数据集, 可能会影响训练速度. 可以在训练前, 进入 LMDB 数据集目录, 把数据先缓存进去: `cat data.mdb > /dev/nul`

除了标准的 LMDB 文件 (data.mdb 和 lock.mdb) 外, 我们还增加了 `meta_info.txt` 来记录额外的信息.
下面用一个例子来说明:

**文件结构**

```txt
DIV2K_train_HR_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**meta信息**

`meta_info.txt`, 我们采用txt来记录, 是为了可读性. 其里面的内容为:

```txt
0001_s001.png (480,480,3) 1
0001_s002.png (480,480,3) 1
0001_s003.png (480,480,3) 1
0001_s004.png (480,480,3) 1
...
```

每一行记录了一张图片, 有三个字段, 分别表示:

- 图像名称 (带后缀): 0001_s001.png
- 图像大小: (480,480,3) 表示是480x480x3的图像
- 其他参数 (BasicSR里面使用了 cv2 压缩 png 程度): 因为在复原任务中, 我们通常使用 png 来存储, 所以这个 1 表示 png 的压缩程度 `CV_IMWRITE_PNG_COMPRESSION` 是 1. 它可以取值为[0, 9]的整数, 更大的值表示更强的压缩, 即更小的储存空间和更长的压缩时间.

**二进制内容**

为了方便, 我们在 LMDB 数据集中存储的二进制内容是 cv2 encode过的 image: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. 可以通过 `compress_level` 控制压缩程度, 平衡存储空间和读取(包括解压缩)的速度.

**如何制作**

我们提供了脚本来制作. 在运行脚本前, 需要根据需求修改相应的参数. 目前支持 DIV2K, REDS 和 Vimeo90K 数据集; 其他数据集可仿照进行制作. <br>
 `python scripts/data_preparation/create_lmdb.py`

#### 预读取数据

除了使用LMDB来加速外, 还可以采用预读取数据来加速, 实现参见 [prefetch_dataloader](../basicsr/data/prefetch_dataloader.py).<br>
这个可以通过配置文件中的 `prefetch_mode` 来指定. 目前提供了三种模式:

1. None. 默认不使用. 如果使用了 LMDB 或者 IO 不成问题, 则可不使用

    ```yml
    prefetch_mode: ~
    ```

1. `prefetch_mode: cuda`. 使用 CUDA prefetcher, 具体介绍参见 [NVIDIA/apex](https://github.com/NVIDIA/apex/issues/304#). 它会多占用一些GPU显存. 注意: 这个模式下, 一定要设置 `pin_memory=True`

    ```yml
    prefetch_mode: cuda
    pin_memory: true
    ```

1. `prefetch_mode: cpu`. 使用 CPU prefetcher, 具体介绍参见 [IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#). (目前测试，这个加速不明显)

    ```yml
    prefetch_mode: cpu
    num_prefetch_queue: 1  # 1 by default
    ```

## 图像数据

推荐把数据通过 `ln -s xxx yyy` 软链到`BasicSR/datasets`下. 如果你的文件结构不同, 需要相应地修改configuration yaml文件的路径.

### DIV2K

DIV2K 数据集被广泛使用在图像复原的任务中.

**数据准备步骤**

1. 从[官网](https://data.vision.ee.ethz.ch/cvl/DIV2K)下载数据.
1. Crop to sub-images: 因为 DIV2K 数据集是 2K 分辨率的 (比如: 2048x1080), 而我们在训练的时候往往并不要那么大 (常见的是 128x128 或者 192x192 的训练patch). 因此我们可以先把2K的图片裁剪成有overlap的 480x480 的子图像块. 然后再由 dataloader 从这个 480x480 的子图像块中随机crop出 128x128 或者 192x192 的训练patch.<br>
    运行脚本 [extract_subimages.py](../scripts/data_preparation/extract_subimages.py):

    ```python
    python scripts/data_preparation/extract_subimages.py
    ```

    使用之前可能需要修改文件里面的路径和配置参数.
    **注意**: sub-image 的尺寸和训练patch的尺寸 (`gt_size`) 是不同的. 我们先把2K分辨率的图像 crop 成 sub-images (往往是 480x480), 然后存储起来. 在训练的时候, dataloader会读取这些sub-images, 然后进一步随机裁剪成 `gt_size` x `gt_size`的大小.
1. [可选] 若需要使用 LMDB, 则需要制作 LMDB, 参考 [LMDB具体说明](#LMDB具体说明).  `python scripts/data_preparation/create_lmdb.py`, 注意选择`create_lmdb_for_div2k`函数, 并需要修改函数相应的配置和路径.
1. 测试: `tests/test_paired_image_dataset.py`, 注意修改函数相应的配置和路径.
1. [可选] 若需要使用 meta_info_file, 运行 `python scripts/data_preparation/generate_meta_info.py` 来生成 meta_info_file.

### 其他常见图像超分数据集

我们提供了常见图像超分数据集的列表.

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
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
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

## 视频帧数据

推荐把数据通过 `ln -s xxx yyy` 软链到`BasicSR/datasets`下. 如果你的文件结构不同, 需要相应地修改configuration yaml文件的路径.

### REDS

[官网](https://seungjunnah.github.io/Datasets/reds.html) <br>
我们重新整合了 training 和 validation 数据到一个文件夹中: 训练集合原来有240个clip (序号从000到239), 我们把validation clips重命名, 从240到269.

**Validation的划分**

官方的validation划分和EDVR的划分不同 (当时为了比赛的设置):

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

余下的clips拿来做训练集合. 注意: 我们不需要显式地分开训练和验证集合, dataloader会做这件事.

**数据准备步骤**

1. 从[官网](https://seungjunnah.github.io/Datasets/reds.html)下载数据
1. 整合 training 和 validation 数据: `python scripts/data_preparation/regroup_reds_dataset.py`
1. [可选] 若需要使用 LMDB, 则需要制作 LMDB, 参考 [LMDB具体说明](#LMDB具体说明).  `python scripts/data_preparation/create_lmdb.py`, 注意选择`create_lmdb_for_reds`函数, 并需要修改函数相应的配置和路径.
1. 测试: `python tests/test_reds_dataset.py`, 注意修改函数相应的配置和路径.

### Vimeo90K

[官网](http://toflow.csail.mit.edu/)

**数据准备步骤**

1. 下载数据: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip). 这些是Ground-Truth. 里面有`sep_trainlist.txt`文件来区分训练数据.
1. 生成低分辨率图片. (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [可选] 若需要使用 LMDB, 则需要制作 LMDB, 参考 [LMDB具体说明](#LMDB具体说明).  `python scripts/data_preparation/create_lmdb.py`, 注意选择`create_lmdb_for_vimeo90k`函数, 并需要修改函数相应的配置和路径.
1. 测试: `python tests/test_vimeo90k_dataset.py`, 注意修改函数相应的配置和路径.

## StyleGAN2

### FFHQ

训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset).

1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
1. 从 tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords). 我们对每一个分辨率的人脸都单独创建文件夹或者LMDB文件.

    ```bash
    python scripts/data_preparation/extract_images_from_tfrecords.py
    ```
