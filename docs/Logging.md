# Logging

[English](Logging.md) **|** [简体中文](Logging_CN.md)

#### Contents

1. [Text Logger](#Text-Logger)
1. [Tensorboard Logger](#Tensorboard-Logger)
1. [Wandb Logger](#Wandb-Logger)

## Text Logger

Print the log to both the text file and screen. The text file usually locates in `experiments/exp_name/train_exp_name_timestamp.txt`.

## Tensorboard Logger

- Use Tensorboard logger. Set `use_tb_logger: true` in the yml configuration file:

    ```yml
    logger:
      use_tb_logger: true
    ```

- File location: `tb_logger/exp_name`
- View in the browser:

    ```bash
    tensorboard --logdir tb_logger --port 5500 --bind_all
    ```

## Wandb Logger

[wandb](https://www.wandb.com/) can be viewed as a cloud version of tensorboard. One can easily view training processes and curves in wandb. Currently, we only sync the tensorboard log to wandb. So we should also turn on tensorboard when using wandb.

Configuration file:

```yml
ogger:
  # Whether to tensorboard logger
  use_tb_logger: true
  # Whether to use wandb logger. Currently, wandb only sync the tensorboard log. So we should also turn on tensorboard when using wandb
  wandb:
    # wandb project name. Default is None, that is not using wandb.
    # Here, we use the basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # If resuming, wandb id could automatically link previous logs
    resume_id: ~
```

**[Examples of training curves in wandb](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="280">
</a></p>
