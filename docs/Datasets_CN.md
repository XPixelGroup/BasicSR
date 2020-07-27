# 数据准备
[English](Datasets.md) | [简体中文](Datasets_CN.md)

#### 目录
1. [数据存储形式](#数据存储形式)
    1. [如何使用](#如何使用)
    1. [如何实现](#如何实现)
    1. [LMDB具体说明](#LMDB具体说明)
1. [图像数据](#图像数据)
1. [视频帧数据](#视频帧数据)

## 数据存储形式
目前支持的数据存储形式有以下三种:
1. 直接以图像/视频的格式存放在硬盘
2. 制作成 [LMDB](https://lmdb.readthedocs.io/en/release/). 训练数据使用这种形式, 一般会加快读取速度.
3. 若是支持 [Memcached](https://memcached.org/) 或 [Ceph](https://ceph.io/), 则可以使用. 它们一般应用在集群上.

#### 如何使用
目前, 我们可以通过 configuation yaml 文件方便的修改. 以支持DIV2K的 [PairedImageDataset](../basicsr/data/paired_image_dataset.py) 为例, 根据不同的要求修改yaml文件:
1. 直接读取硬盘数据
    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    io_backend:
      type: disk
    ```
1. 使用LMDB.
在使用前需要先制作LMDB, 参见 [LMDB具体说明](#LMDB具体说明), 注意我们在原有的 LDMB 上, 新增加了特有的 meta 信息, 因此其他来源的LMDB并不能直接拿过来使用.
    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```
1. 使用Memecached
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
实现是调用了[MMCV](https://github.com/open-mmlab/mmcv) 优雅的 FileClient 设计. 为了使用 BasicSR 的设计, 我们对接口做了一些接口 (主要是为了适应LMDB), 参见 [file_client.py](../basicsr/utils/file_client.py).

在实现我们自己的 dataloader 的时候, 可以方便的调用接口, 以实现对不同数据存储形式的支持, 具体可以参考 [PairedImageDataset](../basicsr/data/paired_image_dataset.py).

#### LMDB具体说明
我们在训练的时候使用 LMDB 存储形式可以加快IO和CPU解压缩的速度 (测试的时候数据较少, 一般就没有太必要使用 LMDB). 其具体的加速要根据机器的配置来, 以下几个因素会影响:
1. 有的机器设置了定时清理缓存, 而 LMDB 依赖于缓存. 因此若一直缓存不进去, 则需要检查以下. 一般 `free -h` 命令下, LMDB 占用的缓存会记录在 `buff/cache` 条目下面
1. 机器的内存是否足够大, 把整个 LMDB 数据都放进去. 如果不是, 则它会不会更换缓存, 影响速度
1. 若是第一次缓存 LMDB 数据集, 可能会影响训练速度. 可以在训练前, 进入 LMDB 数据集, 把数据先缓存进去: `cat data.mdb > /dev/nul`

除了标准的 LMDB 文件 (data.mdb 和 lock.mdb) 外, 我们还增加了 `meta_info.txt` 来记录额外的信息.
下面用一个例子来说明:

**文件结构**
```
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
- 图像名字 (带后缀): 0001_s001.png
- 图像大小: (480,480,3) 表示是480x480x3的图像.
- 其他参数 (BasicSR里面是使用 cv2 压缩 png 程度): 因为在复原任务中, 我们通常使用 png 来存储, 所以这个 1 表示 png 的压缩程度`CV_IMWRITE_PNG_COMPRESSION `是 1. 它取值为[0, 9]的整数, 更大的值表示更强的压缩, 即更小的储存空间和更长的压缩时间.

**二进制内容**
为了方便, 我们在 LMDB 数据集中存储的二进制内容是 cv2 encode过的 image: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. 可以通过 `compress_level` 控制压缩程度, 平衡存储空间和读取(解压缩)速度.

## 图像数据
推荐把其他数据通过 `ln -s xxx yyy` 软链到`BasicSR/datasets`下. 如果你的文件结构不同, 需要修改configuration yaml文件的对应路径.

#### DIV2K
DIV2K 数据集被广泛使用在图像复原的任务中.

1. 从[官网](https://data.vision.ee.ethz.ch/cvl/DIV2K)下载数据.
1. Crop to sub-images: 因为 DIV2K 数据集是 2K 分辨率的 (比如: 2048x1080)的, 而我们在训练的时候往往并不要那么大 (常见的是 128x128 或者 192x192 的训练patch). 因此我们可以可以先把2K的图片裁剪成有交叠的 480x480 的子图像块. 然后再由 dataloader 从这个 480x480 的子图像块中随机crop出 128x128 或者 192x192 的训练patch.<br>
    运行脚本 [extract_subimages.py](../scripts/extract_subimages.py):
    ```
    python scripts/extract_subimages.py
    ```
    使用之前可能需要修改文件里面的路径和配置参数.
1.
#### 其他常见图像超分数据集

## 视频帧数据

### REDS

### Vimeo90K




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
