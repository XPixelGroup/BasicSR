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
            user_response = input(
                f'{file_name} already exist. Do you want to cover it? Y/N\n')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accpets Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            download_file_from_google_drive(file_id, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'method',
        type=str,
        help=(
            "Options: 'ESRGAN', 'EDVR', 'StyleGAN', 'EDSR', 'DUF', 'DFDNet', "
            "'dlib'. Set to 'all' if you want to download all the models."))
    args = parser.parse_args()

    file_ids = {
        'ESRGAN': {
            'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth':  # file name
            '1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT',  # file id
            'ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth':
            '1swaV5iBMFfg-DL6ZyiARztbhutDCWXMM'
        },
        'EDVR': {
            'EDVR_L_x4_SR_REDS_official-9f5f5039.pth':
            '127KXEjlCwfoPC1aXyDkluNwr9elwyHNb',
            'EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth':
            '1aVR3lkX6ItCphNLcT7F5bbbC484h4Qqy',
            'EDVR_M_woTSA_x4_SR_REDS_official-1edf645c.pth':
            '1C_WdN-NyNj-P7SOB5xIVuHl4EBOwd-Ny',
            'EDVR_M_x4_SR_REDS_official-32075921.pth':
            '1dd6aFj-5w2v08VJTq5mS9OFsD-wALYD6',
            'EDVR_L_x4_SRblur_REDS_official-983d7b8e.pth':
            '1GZz_87ybR8eAAY3X2HWwI3L6ny7-5Yvl',
            'EDVR_L_deblur_REDS_official-ca46bd8c.pth':
            '1_ma2tgHscZtkIY2tEJkVdU-UP8bnqBRE',
            'EDVR_L_deblurcomp_REDS_official-0e988e5c.pth':
            '1fEoSeLFnHSBbIs95Au2W197p8e4ws4DW'
        },
        'StyleGAN': {
            'stylegan2_ffhq_config_f_1024_official-b09c3668.pth':
            '163PfuVSYKh4vhkYkfEaufw84CiF4pvWG',
            'stylegan2_ffhq_config_f_1024_discriminator_official-806ddc5e.pth':
            '1wyOdcJnMtAT_fEwXYJObee7hcLzI8usT',
            'stylegan2_cat_config_f_256_official-b82c74e3.pth':
            '1dGUvw8FLch50FEDAgAa6st1AXGnjduc7',
            'stylegan2_cat_config_f_256_discriminator_official-f6f5ed5c.pth':
            '19wuj7Ztg56QtwEs01-p_LjQeoz6G11kF',
            'stylegan2_church_config_f_256_official-12725a53.pth':
            '1Rcpguh4t833wHlFrWz9UuqFcSYERyd2d',
            'stylegan2_church_config_f_256_discriminator_official-feba65b0.pth':  # noqa: E501
            '1ImOfFUOwKqDDKZCxxM4VUdPQCc-j85Z9',
            'stylegan2_car_config_f_512_official-32c42d4e.pth':
            '1FviBGvzORv4T3w0c3m7BaIfLNeEd0dC8',
            'stylegan2_car_config_f_512_discriminator_official-31f302ab.pth':
            '1hlZ7M2GrK6cDFd2FIYazPxOZXTUfudB3',
            'stylegan2_horse_config_f_256_official-d3d97ebc.pth':
            '1LV4OR22tJN19HHfGk0e7dVqMhjD0APRm',
            'stylegan2_horse_config_f_256_discriminator_official-efc5e50e.pth':
            '1T8xbI-Tz8EeSg3gCmQBNqGjLP5l3Mv84'
        },
        'EDSR': {
            'EDSR_Mx2_f64b16_DIV2K_official-3ba7b086.pth':
            '1mREMGVDymId3NzIc2u90sl_X4-pb4ZcV',
            'EDSR_Mx3_f64b16_DIV2K_official-6908f88a.pth':
            '1EriqQqlIiRyPbrYGBbwr_FZzvb3iwqz5',
            'EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth':
            '1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn',
            'EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth':
            '15257lZCRZ0V6F9LzTyZFYbbPrqNjKyMU',
            'EDSR_Lx3_f256b32_DIV2K_official-3660f70d.pth':
            '18q_D434sLG_rAZeHGonAX8dkqjoyZ2su',
            'EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth':
            '1GCi30YYCzgMCcgheGWGusP9aWKOAy5vl'
        },
        'DUF': {
            'DUF_x2_16L_official-39537cb9.pth':
            '1e91cEZOlUUk35keK9EnuK0F54QegnUKo',
            'DUF_x3_16L_official-34ce53ec.pth':
            '1XN6aQj20esM7i0hxTbfiZr_SL8i4PZ76',
            'DUF_x4_16L_official-bf8f0cfa.pth':
            '1V_h9U1CZgLSHTv1ky2M3lvuH-hK5hw_J',
            'DUF_x4_28L_official-cbada450.pth':
            '1M8w0AMBJW65MYYD-_8_be0cSH_SHhDQ4',
            'DUF_x4_52L_official-483d2c78.pth':
            '1GcmEWNr7mjTygi-QCOVgQWOo5OCNbh_T'
        },
        'DFDNet': {
            'DFDNet_dict_512-f79685f0.pth':
            '1iH00oMsoN_1OJaEQw3zP7_wqiAYMnY79',
            'DFDNet_official-d1fa5650.pth':
            '1u6Sgcp8gVoy4uVTrOJKD3y9RuqH2JBAe'
        },
        'dlib': {
            'mmod_human_face_detector-4cb19393.dat':
            '1FUM-hcoxNzFCOpCWbAUStBBMiU4uIGIL',
            'shape_predictor_5_face_landmarks-c4b1e980.dat':
            '1PNPSmFjmbuuUDd5Mg5LDxyk7tu7TQv2F',
            'shape_predictor_68_face_landmarks-fbdc2cb8.dat':
            '1IneH-O-gNkG0SQpNCplwxtOAtRCkG2ni'
        }
    }

    if args.method == 'all':
        for method in file_ids.keys():
            download_pretrained_models(method, file_ids[method])
    else:
        download_pretrained_models(args.method, file_ids[args.method])
