# Datasets

[English](Datasets.md) **|** [简体中文](Datasets_CN.md)

## Supported Datasets

| Class         | Task   |Train/Test | Description       |
| :------------- | :----------:| :----------:    | :----------:   |
| [PairedImageDataset](../basicsr/data/paired_image_dataset.py) | Image Super-Resolution | Train|Support paired data |
| [SingleImageDataset](../basicsr/data/single_image_dataset.py) | Image Super-Resolution | Test|Only read low quality images, used in tests without Ground-Truth|
| [REDSDataset](../basicsr/data/reds_dataset.py) | Video Super-Resolution | Train|REDS training dataset |
| [Vimeo90KDataset](../basicsr/data/vimeo90k_dataset.py) | Video Super-Resolution |Train| Vimeo90K training dataset|
| [VideoTestDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution | Test|Base video test dataset, supporting Vid4, REDS testing datasets|
| [VideoTestVimeo90KDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution |Test| Inherit `VideoTestDataset`, Vimeo90K testing dataset|
| [VideoTestDUFDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution |Test| Inherit `VideoTestDataset`, testing dataset for method DUF, supporting Vid4 dataset|
| [FFHQDataset](../basicsr/data/ffhq_dataset.py) | Face Generation |Train| FFHQ training dataset|

1. Common transformations and functions are in [transforms.py](../basicsr/data/transforms.py) and [util.py](../basicsr/data/util.py), respectively
