# Utils

## Tensorboard Logger (tb_logger)

[tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) is a nice visualization tool for visualizing/comparing training loss, validation PSNR and etc.

You can turn it on/off in json option file with the key: `use_tb_logger`.

### Install
1. `pip install tensorflow` - Maybe it is the easiest way to install tensorboard, though we will install tensorflow at the same time.
1. `pip install tensorboard_logger` - install [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

### Run
1. In terminal: `tensorboard --logdir xxx/xxx`.
1. Open TensorBoard UI at http://localhost:6006 in your browser
