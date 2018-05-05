# Pretrained models

Here are some pretrained models.

## SRResNet (EDSR)

Through experiments, we found that

- no batch normalization
- Residual block style: Conv-ReLU-Conv

are the best network setting.

### MATLAB bicubic down-sampling

| Model | Scale | Channel | DIV2K | Set5 | Set14 | BSD100 | Urban100 |
| SRResNet_bicx2_in3nf64nb16 | 2 | RGB | | | | |