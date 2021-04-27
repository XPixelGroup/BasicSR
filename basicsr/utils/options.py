import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

'''
    [Lotayou] 20210426: Convert between yaml and object option types
'''

class Options(object):
    """
        A base option class with smart print options
    """
    
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
    
    def __repr__(self, indent_level=1):
        ''' 
            A replica of dict2str with [] -> {}
            Recursively append attributes from base class
        '''
        
        msg = '\n'
        for k, v in self.__dict__.items():
            if isinstance(v, Options):
                msg += ' ' * (indent_level * 2) + k + ':{'
                msg += v.__repr__(indent_level + 1)
                msg += ' ' * (indent_level * 2) + '}\n'
            else:
                msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
        return msg
            
def dict2object(dic):
    """ 
       [Lotayou] 20210426: HiFaceGAN compatible helper function
       Convert a dict (read from yaml) to an object class
       to call an element with opt.xxx instead of opt['xxx']
       
    """
    opt = Options()   # Use built-in Python object
    for k, v in dic.items():
        if isinstance(v, dict):
            opt.__dict__[k] = dict2objectclass(v)
        else:
            opt.__dict__[k] = v
    return opt


def object2dict(opt):
    """ 
       [Lotayou] 20210426: HiFaceGAN compatible helper function
       Convert an object class to a dict to support opt['xxx'].
       
       Note the recursive case, vars() ain't gonna work
    """
    dic = dict()
    for k, v in opt.__dict__.items():
        if isinstance(v, Options):
            dic[k] = objectclass2dict(v)
        else:
            dic[k] = v
    return dic

if __name__ == '__main__':

    a = {
        'name': 'test',
        'num': 10.0,
        'bool': True,
        'recursive_dict': {
            'name': 'test',
            'num': 10.0,
            'bool': True,
            'recursive_dict_2': {
                'name': 'test',
                'num': 10.0,
                'bool': True,                
            }
        }
    }

    b = dict2objects(a)
    print(b)
    print(b.recursive_dict.recursive_dict_2)
    a = object2dict(b)
    print(a)