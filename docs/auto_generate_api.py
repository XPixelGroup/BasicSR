import os
from os import path as osp


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


# specifically generate a fake __init__.py for scripts folder to generate docs
with open('../scripts/__init__.py', 'w') as f:
    pass

module_name_list = ['basicsr', 'scripts']
for module_name in module_name_list:
    cur_dir = osp.abspath(osp.dirname(__file__))
    output_dir = osp.join(cur_dir, 'api')
    module_dir = osp.join(osp.dirname(cur_dir), module_name)
    os.makedirs(output_dir, exist_ok=True)

    api_content = f'{module_name} API\n=========================\n'
    submodule_name_list = []

    for path in sorted(scandir(module_dir, suffix='.py', recursive=True)):
        if path in ['__init__.py', 'version.py']:
            continue

        path = f'{module_name}.' + path.replace('\\', '/').replace('/', '.').replace('.py', '.rst')

        # create .rst file
        output_rst = osp.join(output_dir, path)
        with open(output_rst, 'w') as f:
            # write contents
            title = path.replace('.rst', '').replace('_', '\\_')
            content = f'{title}\n===================================================================================\n'
            automodule = path.replace('.rst', '')
            content += f'\n.. automodule:: {automodule}'
            content += r'''
    :members:
    :undoc-members:
    :show-inheritance:
    '''
            f.write(content)

        # add to api.rst
        submodule_name = path.split('.')[1]
        if submodule_name not in submodule_name_list:
            submodule_name_list.append(submodule_name)
            api_content += f'\n\n{module_name}.{submodule_name}\n-----------------------------------------------------'
            api_content += r'''
.. toctree::
    :maxdepth: 4

'''

        api_content += f'    {automodule}\n'

    # write to api.rst
    with open(os.path.join(output_dir, f'api_{module_name}.rst'), 'w') as f:
        f.write(api_content)
