# HiFaceGAN 使用说明

[English](HiFaceGAN.md) **|** [简体中文](HiFaceGAN_CN.md) | [原工程地址](https://github.com/Lotayou/Face-Renovation)

## 如何测试 HiFaceGAN
1. 制作测试数据，当前预训练模型仅支持512分辨率。对于学术数据集，需将低质人像与参考人像缩放至512(cv2.INTER_CUBIC)，并沿上下拼接，形成512x1024x3的图片。真实数据无参考图像，则不用拼接。将所有数据存放在同一文件夹内，记为`xxx`.
2. 下载模型，[百度网盘](https://pan.baidu.com/s/1lp-mj5LaTfNxAxrn4QOcSA)提取码：rwzh 

P.S.google上传暂时不可用，需xinntao协助，并将解压后的文件夹放在指定路径下（记为`yyy`）。

3. 修改配置文件，目前配置文件放在`/backup/lingbo/projects/BasicSR/basicsr/models/archs/hifacegan_options.py`，分`TrainOptions`与`TestOptions`两个类，其中以下几项需手动指定：

    ```
        dataroot = xxx
        checkpoints_dir = yyy
        name = <欲执行的测试>（默认为generic）
    ```

4. 运行以下脚本测试，结果默认在 `results/HiFaceGAN`内，其中generic通用模式下的测试效果已提供。
    >  python inference_hifacegan.py

## 开发历史

#### v0.1: 
- 添加HiFaceGAN网络架构定义及配置文件，提供测试脚本，真实测试样例及修复预览效果。


## 后续工作计划
- 解决inference文件夹内调用无法`import basicsr`的问题。
- 迁移训练代码
- 数据集支持lmdb格式
- 提供DDP训练支持（预计2021.10以后）
