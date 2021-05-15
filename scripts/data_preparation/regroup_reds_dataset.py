import glob
import os


def regroup_reds_dataset(train_path, val_path):
    """Regroup original REDS datasets.

    We merge train and validation data into one folder, and separate the
    validation clips in reds_dataset.py.
    There are 240 training clips (starting from 0 to 239),
    so we name the validation clip index starting from 240 to 269 (total 30
    validation clips).

    Args:
        train_path (str): Path to the train folder.
        val_path (str): Path to the validation folder.
    """
    # move the validation data to the train folder
    val_folders = glob.glob(os.path.join(val_path, '*'))
    for folder in val_folders:
        new_folder_idx = int(folder.split('/')[-1]) + 240
        os.system(f'cp -r {folder} {os.path.join(train_path, str(new_folder_idx))}')


if __name__ == '__main__':
    # train_sharp
    train_path = 'datasets/REDS/train_sharp'
    val_path = 'datasets/REDS/val_sharp'
    regroup_reds_dataset(train_path, val_path)

    # train_sharp_bicubic
    train_path = 'datasets/REDS/train_sharp_bicubic/X4'
    val_path = 'datasets/REDS/val_sharp_bicubic/X4'
    regroup_reds_dataset(train_path, val_path)
