# BasicSR docs

This folder includes:

- Auto-generated API in [*basicsr.readthedocs.io*](https://basicsr.readthedocs.io/en/latest/#)
- Other documents about BasicSR

## 中文文档

我们提供了更完整的 BasicSR 中文解读文档 PDF，你所需要的内容可以在相应的章节中找到。

文档的最新版可以从 [BasicSR-docs/releases](https://github.com/XPixelGroup/BasicSR-docs/releases) 下载。

欢迎大家一起来帮助查找文档中的错误，完善文档。

## 如何在本地自动生成 API docs

```bash
cd docs
make html
```

## 规范

rst 语法参考: https://3vshej.cn/rstSyntax/

着重几个点：

```rst
- 空行
- :file:`file`, :func:`func`, :class:`class`, :math:`\gamma`
- **粗体**，*斜体*
- ``Paper: title``
- Reference: link
```

Examples:

```python
class SPyNetTOF(nn.Module):
    """SPyNet architecture for TOF.

    Note that this implementation is specifically for TOFlow. Please use :file:`spynet_arch.py` for general use.
    They differ in the following aspects:

    1. The basic modules here contain BatchNorm.
    2. Normalization and denormalization are not done here, as they are done in TOFlow.

    ``Paper: Optical Flow Estimation using a Spatial Pyramid Network``

    Reference: https://github.com/Coldog2333/pytoflow

    Args:
        load_path (str): Path for pretrained SPyNet. Default: None.
    """
```
