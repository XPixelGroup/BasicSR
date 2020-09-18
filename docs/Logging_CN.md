# Logging日志

[English](Logging.md) **|** [简体中文](Logging_CN.md)

#### 目录

1. [文本屏幕日志](#文本屏幕日志)
1. [Tensorboard日志](#Tensorboard日志)
1. [Wandb日志](#Wandb日志)

## 文本屏幕日志

将日志信息同时输出到文件和屏幕. 文件位置一般为`experiments/exp_name/train_exp_name_timestamp.txt`.

## Tensorboard日志

- 开启. 在 yml 配置文件中设置 `use_tb_logger: true`:

    ```yml
    logger:
      use_tb_logger: true
    ```

- 文件位置: `tb_logger/exp_name`
- 在浏览器中查看:

    ```bash
    tensorboard --logdir tb_logger --port 5500 --bind_all
    ```

## Wandb日志

[wandb](https://www.wandb.com/) 类似tensorboard的云端版本, 可以在浏览器方便地查看模型训练的过程和曲线. 我们目前只是把tensorboard的内容同步到wandb上, 因此要使用wandb, 必须打开tensorboard logger.

配置文件如下:

```yml
logger:
  # 是否使用tensorboard logger
  use_tb_logger: true
  # 是否使用wandb logger, 目前wandb只是同步tensorboard的内容, 因此要使用wandb, 必须也同时使用tensorboard
  wandb:
    # wandb的project. 默认是 None, 即不使用wandb.
    # 这里使用了 basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # 如果是resume, 可以输入上次的wandb id, 则log可以接起来
    resume_id: ~
```

**[wandb训练曲线样例](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="280">
</a></p>
