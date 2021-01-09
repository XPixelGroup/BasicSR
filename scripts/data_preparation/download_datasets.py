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
            user_response = input(
                '%s already exist. Do you want to cover it? Y/N\n' % file_name)
            if user_response.lower() == 'y':
                print('Covering %s to %s' % (file_name, save_path))
                download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print('Skipping %s' % file_name)
            else:
                raise ValueError('Wrong input. Only accpets Y/N.')
        else:
            print('Downloading %s to %s' % (file_name, save_path))
            download_file_from_google_drive(file_id, save_path)

        # unzip
        if save_path.endswith('.zip'):
            extracted_path = save_path.replace('.zip', '')
            print('Extract %s to %s' % (save_path, extracted_path))
            import zipfile
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)

            file_name = file_name.replace('.zip', '')
            subfolder = osp.join(extracted_path, file_name)
            if osp.isdir(subfolder):
                print('Move %s to %s' % (subfolder, extracted_path))
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
