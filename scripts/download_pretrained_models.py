import mmcv
from os import path as osp

from basicsr.utils.download import download_file_from_google_drive


def download_pretrained_models(file_ids):
    save_path_root = './experiments/pretrained_models'
    mmcv.mkdir_or_exist(save_path_root)

    for file_name, file_id in file_ids.items():
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(
                f'{file_name} already exist. Do you want to cover it? Y/N')
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
    # file_ids is a dict: file name --> file id.
    file_ids = {
        'EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth':
        '1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn',
        'stylegan2_ffhq_config_f_official-36991f37.pth':
        '1KF6r14qyL05EfjigLSvU-RBBYU4ATWqF',
    }
    download_pretrained_models(file_ids)
