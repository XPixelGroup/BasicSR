# 代码库的设计和约定

[English](DesignConvention.md) **|** [简体中文](DesignConvention_CN.md)

#### 目录

1. [整体框架](#整体框架)
1. [特性](#特性)
    1. [动态实例化](#动态实例化)
1. [约定](#约定)

## 整体框架

整个 `BasicSR` 框架可以分为以下几个部分 —— 数据 (Data), 模型 (Model), 配置文件 (Options/Configs) 和训练过程.<br>
当我们修改或定义新的方法时, 也往往是从以上几个方面进行修改/添加的.<br>
下图概括了整体的框架.

![overall_structure](../assets/overall_structure.png)

## 特性

### 动态实例化

(Dynamic Instantiation)<br>

当我们新写了类 (Class) 或 函数 时, 可直接在配置文件中使用. 程序会根据配置文件的类名 或 函数名, 自动查找并实例化. 这个过程称为 动态实例化 (Dynamic Instantiation).

具体而言, 我们是通过 `importlib` 和 `getattr` 来实现的. 以data为例, 我们在[`data/__init__.py`](../basicsr/data/__init__.py) 中是如下做的:

1. 扫描所有以`_dataset.py`为结尾的文件 (这是约定)
1. 把这些文件中的 类 或 函数 通过 importlib 都 import 进来
1. 根据配置文件中的名称, 通过`getattr`实例化

```python
# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'basicsr.data.{file_name}')
    for file_name in dataset_filenames
]

...

# dynamic instantiation
for module in _dataset_modules:
    dataset_cls = getattr(module, dataset_type, None)
    if dataset_cls is not None:
        break
```

我们对以下模块使用了类似的技巧, 在使用的时候需要注意文件后缀名称的约定:

| Module         | File Suffix     | Example        |
| :------------- | :----------:    | :----------:   |
| Data           | `_dataset.py`   | `data/paired_image_dataset.py` |
| Model          | `_model.py`     | `basicsr/models/sr_model.py` |
| Archs          | `_arch.py`      | `basicsr/archs/srresnet_arch.py`|

注意:

1. 上面的文件后缀只用在需要的文件中, 其他文件命名尽量避免使用以上的后缀
1. 注意 类名 或 函数名 不能重复

另外对 `losses` 和 `metrics`, 我们也使用了 `importlib` 和 `getattr`, 但是和上面不一样的是, 对于losses和metrics, 由于文件数量比较少, 改动也少, 因此我们不采用扫描文件的方式, 而是在新增加类/函数后, 需要在相应的 `__init__.py` 中增加类/函数名称.

| Module         | Path     | Modify `__init__.py`        |
| :------------- | :----------:    | :----------:   |
| Losses           | `basicsr/models/losses`   | [`basicsr/models/losses/__init__.py`](../basicsr/models/losses/__init__.py) |
| Metrics          | `basicsr/metrics`     | [`basicsr/metrics/__init__.py`](../basicsr/metrics/__init__.py)|

## 约定

1. 动态实例化, 以下模块文件后缀名有要求, 否则不能做到自动实例化.

    | Module         | File Suffix     | Example        |
    | :------------- | :----------:    | :----------:   |
    | Data           | `_dataset.py`   | `data/paired_image_dataset.py` |
    | Model          | `_model.py`     | `basicsr/models/sr_model.py` |
    | Archs          | `_arch.py`      | `basicsr/archs/srresnet_arch.py`|

1. 在Log的时候, loss项使用`l_`开头, 这样在 tensorboard 显示的时候, 所有loss会被组织到一起. 比如在 [basicsr/models/srgan_model.py](../basicsr/models/srgan_model.py)中, 使用了`l_g_pix`, `l_g_percep`, `l_g_gan`等. 在[basicsr/utils/logger.py](../basicsr/utils/logger.py), 他们会被组织到一起:

    ```python
    if k.startswith('l_'):
        self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
    else:
        self.tb_logger.add_scalar(k, v, current_iter)
    ```
