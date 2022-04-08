import argparse
import glob
import os
from os import path as osp

from basicsr.utils.download_util import download_file_from_google_drive


def download_dataset(dataset, file_ids):
    save_path_root = './datasets/'
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

        # unzip
        if save_path.endswith('.zip'):
            extracted_path = save_path.replace('.zip', '')
            print(f'Extract {save_path} to {extracted_path}')
            import zipfile
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)

            file_name = file_name.replace('.zip', '')
            subfolder = osp.join(extracted_path, file_name)
            if osp.isdir(subfolder):
                print(f'Move {subfolder} to {extracted_path}')
                import shutil
                for path in glob.glob(osp.join(subfolder, '*')):
                    shutil.move(path, extracted_path)
                shutil.rmtree(subfolder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset',
        type=str,
        help=("Options: 'Set5', 'Set14'. "
              "Set to 'all' if you want to download all the dataset."))
    args = parser.parse_args()

    file_ids = {
        'Set5': {
            'Set5.zip':  # file name
            '1RtyIeUFTyW8u7oa4z7a0lSzT3T1FwZE9',  # file id
        },
        'Set14': {
            'Set14.zip': '1vsw07sV8wGrRQ8UARe2fO5jjgy9QJy_E',
        }
    }

    if args.dataset == 'all':
        for dataset in file_ids.keys():
            download_dataset(dataset, file_ids[dataset])
    else:
        download_dataset(args.dataset, file_ids[args.dataset])
