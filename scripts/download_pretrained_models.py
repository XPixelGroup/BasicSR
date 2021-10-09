import argparse
import os
from os import path as osp

from basicsr.utils.download_util import download_file_from_google_drive


def download_pretrained_models(method, file_ids):
    save_path_root = f'./experiments/pretrained_models/{method}'
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(f'{file_name} already exist. Do you want to cover it? Y/N\n')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accepts Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            download_file_from_google_drive(file_id, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'method',
        type=str,
        help=("Options: 'ESRGAN', 'EDVR', 'StyleGAN', 'EDSR', 'DUF', 'DFDNet', 'dlib', 'TOF', 'flownet', 'BasicVSR'. "
              "Set to 'all' to download all the models."))
    args = parser.parse_args()

    file_ids = {
        'ESRGAN': {
            'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth':  # file name
            '1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT',  # file id
            'ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth': '1swaV5iBMFfg-DL6ZyiARztbhutDCWXMM'
        },
        'EDVR': {
            'EDVR_L_x4_SR_REDS_official-9f5f5039.pth': '127KXEjlCwfoPC1aXyDkluNwr9elwyHNb',
            'EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth': '1aVR3lkX6ItCphNLcT7F5bbbC484h4Qqy',
            'EDVR_M_woTSA_x4_SR_REDS_official-1edf645c.pth': '1C_WdN-NyNj-P7SOB5xIVuHl4EBOwd-Ny',
            'EDVR_M_x4_SR_REDS_official-32075921.pth': '1dd6aFj-5w2v08VJTq5mS9OFsD-wALYD6',
            'EDVR_L_x4_SRblur_REDS_official-983d7b8e.pth': '1GZz_87ybR8eAAY3X2HWwI3L6ny7-5Yvl',
            'EDVR_L_deblur_REDS_official-ca46bd8c.pth': '1_ma2tgHscZtkIY2tEJkVdU-UP8bnqBRE',
            'EDVR_L_deblurcomp_REDS_official-0e988e5c.pth': '1fEoSeLFnHSBbIs95Au2W197p8e4ws4DW'
        },
        'StyleGAN': {
            'stylegan2_ffhq_config_f_1024_official-3ab41b38.pth': '1qtdsT1FrvKQsFiW3OqOcIb-VS55TVy1g',
            'stylegan2_ffhq_config_f_1024_discriminator_official-a386354a.pth': '1nPqCxm8TkDU3IvXdHCzPUxlBwR5Pd78G',
            'stylegan2_cat_config_f_256_official-0a9173ad.pth': '1gfJkX6XO5pJ2J8LyMdvUgGldz7xwWpBJ',
            'stylegan2_cat_config_f_256_discriminator_official-2c97fd08.pth': '1hy5FEQQl28XvfqpiWvSBd8YnIzsyDRb7',
            'stylegan2_church_config_f_256_official-44ba63bf.pth': '1FCQMZXeOKZyl-xYKbl1Y_x2--rFl-1N_',
            'stylegan2_church_config_f_256_discriminator_official-20cd675b.pth':  # noqa: E501
            '1BS9ODHkUkhfTGFVfR6alCMGtr9nGm9ox',
            'stylegan2_car_config_f_512_official-e8fcab4f.pth': '14jS-nWNTguDSd1kTIX-tBHp2WdvK7hva',
            'stylegan2_car_config_f_512_discriminator_official-5008e3d1.pth': '1UxkAzZ0zvw4KzBVOUpShCivsdXBS8Zi2',
            'stylegan2_horse_config_f_256_official-26d57fee.pth': '12QsZ-mrO8_4gC0UrO15Jb3ykcQ88HxFx',
            'stylegan2_horse_config_f_256_discriminator_official-be6c4c33.pth': '1me4ybSib72xA9ZxmzKsHDtP-eNCKw_X4'
        },
        'EDSR': {
            'EDSR_Mx2_f64b16_DIV2K_official-3ba7b086.pth': '1mREMGVDymId3NzIc2u90sl_X4-pb4ZcV',
            'EDSR_Mx3_f64b16_DIV2K_official-6908f88a.pth': '1EriqQqlIiRyPbrYGBbwr_FZzvb3iwqz5',
            'EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth': '1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn',
            'EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth': '15257lZCRZ0V6F9LzTyZFYbbPrqNjKyMU',
            'EDSR_Lx3_f256b32_DIV2K_official-3660f70d.pth': '18q_D434sLG_rAZeHGonAX8dkqjoyZ2su',
            'EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth': '1GCi30YYCzgMCcgheGWGusP9aWKOAy5vl'
        },
        'DUF': {
            'DUF_x2_16L_official-39537cb9.pth': '1e91cEZOlUUk35keK9EnuK0F54QegnUKo',
            'DUF_x3_16L_official-34ce53ec.pth': '1XN6aQj20esM7i0hxTbfiZr_SL8i4PZ76',
            'DUF_x4_16L_official-bf8f0cfa.pth': '1V_h9U1CZgLSHTv1ky2M3lvuH-hK5hw_J',
            'DUF_x4_28L_official-cbada450.pth': '1M8w0AMBJW65MYYD-_8_be0cSH_SHhDQ4',
            'DUF_x4_52L_official-483d2c78.pth': '1GcmEWNr7mjTygi-QCOVgQWOo5OCNbh_T'
        },
        'TOF': {
            'tof_x4_vimeo90k_official-32c9e01f.pth': '1TgQiXXsvkTBFrQ1D0eKPgL10tQGu0gKb'
        },
        'DFDNet': {
            'DFDNet_dict_512-f79685f0.pth': '1iH00oMsoN_1OJaEQw3zP7_wqiAYMnY79',
            'DFDNet_official-d1fa5650.pth': '1u6Sgcp8gVoy4uVTrOJKD3y9RuqH2JBAe'
        },
        'dlib': {
            'mmod_human_face_detector-4cb19393.dat': '1FUM-hcoxNzFCOpCWbAUStBBMiU4uIGIL',
            'shape_predictor_5_face_landmarks-c4b1e980.dat': '1PNPSmFjmbuuUDd5Mg5LDxyk7tu7TQv2F',
            'shape_predictor_68_face_landmarks-fbdc2cb8.dat': '1IneH-O-gNkG0SQpNCplwxtOAtRCkG2ni'
        },
        'flownet': {
            'spynet_sintel_final-3d2a1287.pth': '1VZz1cikwTRVX7zXoD247DB7n5Tj_LQpF'
        },
        'BasicVSR': {
            'BasicVSR_REDS4-543c8261.pth': '1wLWdz18lWf9Z7lomHPkdySZ-_GV2920p',
            'BasicVSR_Vimeo90K_BDx4-e9bf46eb.pth': '1baaf4RSpzs_zcDAF_s2CyArrGvLgmXxW',
            'BasicVSR_Vimeo90K_BIx4-2a29695a.pth': '1ykIu1jv5wo95Kca2TjlieJFxeV4VVfHP',
            'EDVR_REDS_pretrained_for_IconVSR-f62a2f1e.pth': '1ShfwddugTmT3_kB8VL6KpCMrIpEO5sBi',
            'EDVR_Vimeo90K_pretrained_for_IconVSR-ee48ee92.pth': '16vR262NDVyVv5Q49xp2Sb-Llu05f63tt',
            'IconVSR_REDS-aaa5367f.pth': '1b8ir754uIAFUSJ8YW_cmPzqer19AR7Hz',
            'IconVSR_Vimeo90K_BDx4-cfcb7e00.pth': '13lp55s-YTd-fApx8tTy24bbHsNIGXdAH',
            'IconVSR_Vimeo90K_BIx4-35fec07c.pth': '1lWUB36ERjFbAspr-8UsopJ6xwOuWjh2g'
        }
    }

    if args.method == 'all':
        for method in file_ids.keys():
            download_pretrained_models(method, file_ids[method])
    else:
        download_pretrained_models(args.method, file_ids[args.method])
