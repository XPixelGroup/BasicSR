# BasicSR

BasicSR is an open source image and video super-resolution toolbox based on PyTorch.
BasicSR is based on our previous projects: [BasicSR](https://github.com/xinntao/BasicSR), [ESRGAN](https://github.com/xinntao/ESRGAN), and [EDVR](https://github.com/xinntao/EDVR).


## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

Please run the following cmds to install basicsr.
```bash
python setup.py develop
pip install requirements.txt
```

## Dataset Preparation
Please refer to [DATASETS.md](docs/DATASETS.md) for more details.

## Training and Testing
Please see [Training and Testing](docs/TrainingTesting.md) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [Model Zoo](docs/ModelZoo.md).


**Python code style**<br/>
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style. We use [flake8](http://flake8.pycqa.org/en/latest/) as the linter and [yapf](https://github.com/google/yapf) as the formatter. Please upgrade to the latest yapf (>=0.27.0) and refer to the [yapf configuration](.style.yapf) and [flake8 configuration](.flake8).

> Before you create a PR, make sure that your code lints and is formatted by yapf.

## License
This project is released under the Apache 2.0 license.
