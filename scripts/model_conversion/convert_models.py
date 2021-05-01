import torch


def convert_edvr():
    ori_net = torch.load('experiments/pretrained_models/EDVR_REDS_SR_M.pth')
    crt_net = torch.load('xxx/net_g_8.pth')
    save_path = './edvr_medium_x4_reds_sr_official.pth'

    # for k, v in ori_net.items():
    #     print(k)

    # print('*****')
    # for k, v in crt_net.items():
    #     print(k)

    for crt_k, crt_v in crt_net.items():
        # deblur hr in
        if 'predeblur.stride_conv_hr1' in crt_k:
            ori_k = crt_k.replace('predeblur.stride_conv_hr1',
                                  'pre_deblur.conv_first_2')
        elif 'predeblur.stride_conv_hr2' in crt_k:
            ori_k = crt_k.replace('predeblur.stride_conv_hr2',
                                  'pre_deblur.conv_first_3')
        elif 'predeblur.conv_first' in crt_k:
            ori_k = crt_k.replace('predeblur.conv_first',
                                  'pre_deblur.conv_first_1')

        # predeblur module
        # elif 'predeblur.conv_first' in crt_k:
        #     ori_k = crt_k.replace('predeblur.conv_first',
        #                           'pre_deblur.conv_first')
        elif 'predeblur.stride_conv_l2' in crt_k:
            ori_k = crt_k.replace('predeblur.stride_conv_l2',
                                  'pre_deblur.deblur_L2_conv')
        elif 'predeblur.stride_conv_l3' in crt_k:
            ori_k = crt_k.replace('predeblur.stride_conv_l3',
                                  'pre_deblur.deblur_L3_conv')
        elif 'predeblur.resblock_l3' in crt_k:
            ori_k = crt_k.replace('predeblur.resblock_l3',
                                  'pre_deblur.RB_L3_1')
        elif 'predeblur.resblock_l2' in crt_k:
            ori_k = crt_k.replace('predeblur.resblock_l', 'pre_deblur.RB_L')
        elif 'predeblur.resblock_l1' in crt_k:
            a, b, c, d, e = crt_k.split('.')
            ori_k = f'pre_deblur.RB_L1_{int(c)+1}.{d}.{e}'

        elif 'conv_l2' in crt_k:
            ori_k = crt_k.replace('conv_l2_', 'fea_L2_conv')

        elif 'conv_l3' in crt_k:
            ori_k = crt_k.replace('conv_l3_', 'fea_L3_conv')

        elif 'pcd_align.dcn_pack' in crt_k:
            idx = crt_k.split('.l')[1].split('.')[0]
            name = crt_k.split('.l')[1].split('.')[1]
            ori_k = f'pcd_align.L{idx}_dcnpack.{name}'
            if 'conv_offset' in crt_k:
                name = name.replace('conv_offset', 'conv_offset_mask')
                weight_bias = crt_k.split('.l')[1].split('.')[2]
                ori_k = f'pcd_align.L{idx}_dcnpack.{name}.{weight_bias}'
        elif 'pcd_align.offset_conv' in crt_k:
            a, b, c, d = crt_k.split('.')
            idx = b.split('conv')[1]
            level = c.split('l')[1]
            ori_k = f'pcd_align.L{level}_offset_conv{idx}.{d}'
        elif 'pcd_align.feat_conv' in crt_k:
            a, b, c, d = crt_k.split('.')
            level = c.split('l')[1]
            ori_k = f'pcd_align.L{level}_fea_conv.{d}'
        elif 'pcd_align.cas_dcnpack' in crt_k:
            ori_k = crt_k.replace('conv_offset', 'conv_offset_mask')
        elif ('conv_first' in crt_k or 'feature_extraction' in crt_k
              or 'pcd_align.cas_offset' in crt_k or 'upconv' in crt_k
              or 'conv_last' in crt_k or 'conv_1x1' in crt_k):
            ori_k = crt_k

        elif 'temporal_attn1' in crt_k:
            ori_k = crt_k.replace('fusion.temporal_attn1', 'tsa_fusion.tAtt_2')
        elif 'temporal_attn2' in crt_k:
            ori_k = crt_k.replace('fusion.temporal_attn2', 'tsa_fusion.tAtt_1')
        elif 'fusion.feat_fusion' in crt_k:
            ori_k = crt_k.replace('fusion.feat_fusion',
                                  'tsa_fusion.fea_fusion')
        elif 'fusion.spatial_attn_add' in crt_k:
            ori_k = crt_k.replace('fusion.spatial_attn_add',
                                  'tsa_fusion.sAtt_add_')
        elif 'fusion.spatial_attn_l' in crt_k:
            ori_k = crt_k.replace('fusion.spatial_attn_l', 'tsa_fusion.sAtt_L')
        elif 'fusion.spatial_attn' in crt_k:
            ori_k = crt_k.replace('fusion.spatial_attn', 'tsa_fusion.sAtt_')

        elif 'reconstruction' in crt_k:
            ori_k = crt_k.replace('reconstruction', 'recon_trunk')
        elif 'conv_hr' in crt_k:
            ori_k = crt_k.replace('conv_hr', 'HRconv')

        # for model woTSA
        elif 'fusion' in crt_k:
            ori_k = crt_k.replace('fusion', 'tsa_fusion')

        else:
            print('unprocess key', crt_k)

        # print(ori_k)
        crt_net[crt_k] = ori_net[ori_k]
        ori_k = None

    torch.save(crt_net, save_path)


def convert_edsr(ori_net_path, crt_net_path, save_path, num_block=32):
    """Convert EDSR models in https://github.com/thstkdgus35/EDSR-PyTorch.

    It supports converting x2, x3 and x4 models.

    Args:
        ori_net_path (str): Original network path.
        crt_net_path (str): Current network path.
        save_path (str): The path to save the converted model.
        num_block (int): Number of blocks. Default: 16.
    """
    ori_net = torch.load(ori_net_path)
    crt_net = torch.load(crt_net_path)

    for crt_k, crt_v in crt_net.items():
        if 'conv_first' in crt_k:
            ori_k = crt_k.replace('conv_first', 'head.0')
            crt_net[crt_k] = ori_net[ori_k]
        elif 'conv_after_body' in crt_k:
            ori_k = crt_k.replace('conv_after_body', f'body.{num_block}')
        elif 'body' in crt_k:
            ori_k = crt_k.replace('conv1', 'body.0').replace('conv2', 'body.2')
        elif 'upsample.0' in crt_k:
            ori_k = crt_k.replace('upsample.0', 'tail.0.0')
        elif 'upsample.2' in crt_k:
            ori_k = crt_k.replace('upsample.2', 'tail.0.2')
        elif 'conv_last' in crt_k:
            ori_k = crt_k.replace('conv_last', 'tail.1')
        else:
            print('unprocess key', crt_k)

        crt_net[crt_k] = ori_net[ori_k]

    torch.save(crt_net, save_path)


def convert_rcan_model():
    ori_net = torch.load('RCAN_model_best.pt')
    crt_net = torch.load(
        'experiments/201_RCANx4_scratch_DIV2K_rand0/models/net_g_5000.pth')

    # for ori_k, ori_v in ori_net.items():
    #     print(ori_k)
    for crt_k, crt_v in crt_net.items():
        # print(crt_k)
        if 'conv_first' in crt_k:
            ori_k = crt_k.replace('conv_first', 'head.0')
            crt_net[crt_k] = ori_net[ori_k]
        elif 'conv_after_body' in crt_k:
            ori_k = crt_k.replace('conv_after_body', 'body.10')
        elif 'upsample.0' in crt_k:
            ori_k = crt_k.replace('upsample.0', 'tail.0.0')
        elif 'upsample.2' in crt_k:
            ori_k = crt_k.replace('upsample.2', 'tail.0.2')
        elif 'conv_last' in crt_k:
            ori_k = crt_k.replace('conv_last', 'tail.1')

        elif 'attention' in crt_k:
            a, ai, b, bi, c, ci, d, di, e = crt_k.split('.')
            ori_k = f'body.{ai}.body.{bi}.body.{ci}.conv_du.{int(di)-1}.{e}'
        elif 'rcab' in crt_k:
            a, ai, b, bi, c, ci, d = crt_k.split('.')
            ori_k = f'body.{ai}.body.{bi}.body.{ci}.{d}'
        elif 'body' in crt_k:
            ori_k = crt_k.replace('conv.', 'body.20.')
        else:
            print('unprocess key', crt_k)

        crt_net[crt_k] = ori_net[ori_k]

    torch.save(crt_net, 'RCAN_model_best.pth')


def convert_esrgan_model():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    rrdb = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32)
    crt_net = rrdb.state_dict()
    # for k, v in crt_net.items():
    #     print(k)

    ori_net = torch.load('experiments/pretrained_models/RRDB_ESRGAN_x4.pth')

    # for k, v in ori_net.items():
    #     print(k)

    for crt_k, crt_v in crt_net.items():
        if 'rdb' in crt_k:
            ori_k = crt_k.replace('rdb', 'RDB').replace('body', 'RRDB_trunk')
        elif 'conv_body' in crt_k:
            ori_k = crt_k.replace('conv_body', 'trunk_conv')
        elif 'conv_up' in crt_k:
            ori_k = crt_k.replace('conv_up', 'upconv')
        elif 'conv_hr' in crt_k:
            ori_k = crt_k.replace('conv_hr', 'HRconv')
        else:
            ori_k = crt_k
            print(crt_k)
        crt_net[crt_k] = ori_net[ori_k]
    torch.save(
        crt_net,
        'experiments/pretrained_models/ESRGAN_x4_SR_DF2KOST_official.pth')


def convert_duf_model():
    from basicsr.archs.duf_arch import DUF
    scale = 2
    duf = DUF(scale=scale, num_layer=16, adapt_official_weights=True)
    crt_net = duf.state_dict()
    # for k, v in crt_net.items():
    #     print(k)

    ori_net = torch.load(
        'experiments/pretrained_models/old_DUF_x2_16L_official.pth')
    # print('******')
    # for k, v in ori_net.items():
    #     print(k)
    '''
    for crt_k, crt_v in crt_net.items():
        if 'conv3d1' in crt_k:
            ori_k = crt_k.replace('conv3d1', 'conv3d_1')
        elif 'conv3d2' in crt_k:
            ori_k = crt_k.replace('conv3d2', 'conv3d_2')
        elif 'dense_block1.dense_blocks' in crt_k:
            # dense_block1.dense_blocks.0.0.weight
            a, b, c, d, e = crt_k.split('.')
            # dense_block_1.dense_blocks.0.weight
            ori_k = f'dense_block_1.dense_blocks.{int(c) * 6 + int(d)}.{e}'

        elif 'dense_block2.temporal_reduce1.0' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.0',
                                  'dense_block_2.bn3d_1')
        elif 'dense_block2.temporal_reduce1.2' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.2',
                                  'dense_block_2.conv3d_1')
        elif 'dense_block2.temporal_reduce1.3' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.3',
                                  'dense_block_2.bn3d_2')
        elif 'dense_block2.temporal_reduce1.5' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.5',
                                  'dense_block_2.conv3d_2')

        elif 'dense_block2.temporal_reduce2.0' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.0',
                                  'dense_block_2.bn3d_3')
        elif 'dense_block2.temporal_reduce2.2' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.2',
                                  'dense_block_2.conv3d_3')
        elif 'dense_block2.temporal_reduce2.3' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.3',
                                  'dense_block_2.bn3d_4')
        elif 'dense_block2.temporal_reduce2.5' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.5',
                                  'dense_block_2.conv3d_4')

        elif 'dense_block2.temporal_reduce3.0' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.0',
                                  'dense_block_2.bn3d_5')
        elif 'dense_block2.temporal_reduce3.2' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.2',
                                  'dense_block_2.conv3d_5')
        elif 'dense_block2.temporal_reduce3.3' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.3',
                                  'dense_block_2.bn3d_6')
        elif 'dense_block2.temporal_reduce3.5' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.5',
                                  'dense_block_2.conv3d_6')

        elif 'bn3d2' in crt_k:
            ori_k = crt_k.replace('bn3d2', 'bn3d_2')
        else:
            ori_k = crt_k
            print(crt_k)

        crt_net[crt_k] = ori_net[ori_k]
    '''
    # for 16 layers
    for crt_k, crt_v in crt_net.items():
        if 'conv3d1' in crt_k:
            ori_k = crt_k.replace('conv3d1', 'conv3d_1')
        elif 'conv3d2' in crt_k:
            ori_k = crt_k.replace('conv3d2', 'conv3d_2')

        elif 'dense_block1.dense_blocks.0.0' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.0.0',
                                  'dense_block_1.bn3d_1')
        elif 'dense_block1.dense_blocks.0.2' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.0.2',
                                  'dense_block_1.conv3d_1')
        elif 'dense_block1.dense_blocks.0.3' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.0.3',
                                  'dense_block_1.bn3d_2')
        elif 'dense_block1.dense_blocks.0.5' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.0.5',
                                  'dense_block_1.conv3d_2')

        elif 'dense_block1.dense_blocks.1.0' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.1.0',
                                  'dense_block_1.bn3d_3')
        elif 'dense_block1.dense_blocks.1.2' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.1.2',
                                  'dense_block_1.conv3d_3')
        elif 'dense_block1.dense_blocks.1.3' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.1.3',
                                  'dense_block_1.bn3d_4')
        elif 'dense_block1.dense_blocks.1.5' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.1.5',
                                  'dense_block_1.conv3d_4')

        elif 'dense_block1.dense_blocks.2.0' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.2.0',
                                  'dense_block_1.bn3d_5')
        elif 'dense_block1.dense_blocks.2.2' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.2.2',
                                  'dense_block_1.conv3d_5')
        elif 'dense_block1.dense_blocks.2.3' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.2.3',
                                  'dense_block_1.bn3d_6')
        elif 'dense_block1.dense_blocks.2.5' in crt_k:
            ori_k = crt_k.replace('dense_block1.dense_blocks.2.5',
                                  'dense_block_1.conv3d_6')

        elif 'dense_block2.temporal_reduce1.0' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.0',
                                  'dense_block_2.bn3d_1')
        elif 'dense_block2.temporal_reduce1.2' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.2',
                                  'dense_block_2.conv3d_1')
        elif 'dense_block2.temporal_reduce1.3' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.3',
                                  'dense_block_2.bn3d_2')
        elif 'dense_block2.temporal_reduce1.5' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce1.5',
                                  'dense_block_2.conv3d_2')

        elif 'dense_block2.temporal_reduce2.0' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.0',
                                  'dense_block_2.bn3d_3')
        elif 'dense_block2.temporal_reduce2.2' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.2',
                                  'dense_block_2.conv3d_3')
        elif 'dense_block2.temporal_reduce2.3' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.3',
                                  'dense_block_2.bn3d_4')
        elif 'dense_block2.temporal_reduce2.5' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce2.5',
                                  'dense_block_2.conv3d_4')

        elif 'dense_block2.temporal_reduce3.0' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.0',
                                  'dense_block_2.bn3d_5')
        elif 'dense_block2.temporal_reduce3.2' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.2',
                                  'dense_block_2.conv3d_5')
        elif 'dense_block2.temporal_reduce3.3' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.3',
                                  'dense_block_2.bn3d_6')
        elif 'dense_block2.temporal_reduce3.5' in crt_k:
            ori_k = crt_k.replace('dense_block2.temporal_reduce3.5',
                                  'dense_block_2.conv3d_6')

        elif 'bn3d2' in crt_k:
            ori_k = crt_k.replace('bn3d2', 'bn3d_2')
        else:
            ori_k = crt_k
            print(crt_k)

        crt_net[crt_k] = ori_net[ori_k]

    x = crt_net['conv3d_r2.weight'].clone()
    x1 = x[::3, ...]
    x2 = x[1::3, ...]
    x3 = x[2::3, ...]
    crt_net['conv3d_r2.weight'][:scale**2, ...] = x1
    crt_net['conv3d_r2.weight'][scale**2:2 * (scale**2), ...] = x2
    crt_net['conv3d_r2.weight'][2 * (scale**2):, ...] = x3

    x = crt_net['conv3d_r2.bias'].clone()
    x1 = x[::3, ...]
    x2 = x[1::3, ...]
    x3 = x[2::3, ...]
    crt_net['conv3d_r2.bias'][:scale**2, ...] = x1
    crt_net['conv3d_r2.bias'][scale**2:2 * (scale**2), ...] = x2
    crt_net['conv3d_r2.bias'][2 * (scale**2):, ...] = x3
    torch.save(crt_net,
               'experiments/pretrained_models/DUF_x2_16L_official.pth')


if __name__ == '__main__':
    # convert EDSR models
    # ori_net_path = 'path to original model'
    # crt_net_path = 'path to current model'
    # save_path = 'save path'
    # convert_edsr(ori_net_path, crt_net_path, save_path, num_block=32)

    convert_duf_model()
