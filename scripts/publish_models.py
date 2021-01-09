import glob
import subprocess
import torch
from os import path as osp
from torch.serialization import _is_zipfile, _open_file_like


def update_sha(paths):
    print('# Update sha ...')
    for idx, path in enumerate(paths):
        print('{:03d}: Processing {}'.format(idx + 1, path))
        net = torch.load(path, map_location=torch.device('cpu'))
        basename = osp.basename(path)
        if 'params' not in net and 'params_ema' not in net:
            raise ValueError('Please check! Model ' + basename +
                             ' does not have "params"/"params_ema" key.')
        else:
            if '-' in basename:
                # check whether the sha is the latest
                old_sha = basename.split('-')[1].split('.')[0]
                new_sha = subprocess.check_output(['sha256sum',
                                                   path]).decode()[:8]
                if old_sha != new_sha:
                    final_file = path.split('-')[0] + '-' + new_sha + '.pth'
                    print('\tSave from %s to %s' % (path, final_file))
                    subprocess.Popen(['mv', path, final_file])
            else:
                sha = subprocess.check_output(['sha256sum', path]).decode()[:8]
                final_file = path.split('.pth')[0] + '-' + sha + '.pth'
                print('\tSave from %s to %s' % (path, final_file))
                subprocess.Popen(['mv', path, final_file])


def convert_to_backward_compatible_models(paths):
    """Convert to backward compatible pth files.

    PyTorch 1.6 uses a updated version of torch.save. In order to be compatible
    with previous PyTorch version, save it with
    _use_new_zipfile_serialization=False.
    """
    print('# Convert to backward compatible pth files ...')
    for idx, path in enumerate(paths):
        print('{:03d}: Processing {}'.format(idx + 1, path))
        flag_need_conversion = False
        with _open_file_like(path, 'rb') as opened_file:
            if _is_zipfile(opened_file):
                flag_need_conversion = True

        if flag_need_conversion:
            net = torch.load(path, map_location=torch.device('cpu'))
            print('\tConverting to compatible pth file...')
            torch.save(net, path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    paths = glob.glob('experiments/pretrained_models/*.pth') + glob.glob(
        'experiments/pretrained_models/**/*.pth')
    convert_to_backward_compatible_models(paths)
    update_sha(paths)
