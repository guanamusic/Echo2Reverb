import os
import glob
import argparse

import torch


def show_message(text, verbose=True, end='\n', rank=0):
    if verbose and (rank == 0): print(text, end=end)


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_filelist(filelist_path):
    with open(filelist_path, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist


def latest_checkpoint_path(dir_path, regex="checkpoint_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_latest_checkpoint(logdir, model, optimizer=None):
    latest_model_path = latest_checkpoint_path(logdir, regex="checkpoint_*.pt")
    print(f'Latest checkpoint: {latest_model_path}')
    d = torch.load(
        latest_model_path,
        map_location=lambda loc, storage: loc
    )
    epoch = d['epoch']
    valid_incompatible_unexp_keys = [
        'nn.reverberation_generator.export_misc'
    ]
    d['model'] = {
        key: value for key, value in d['model'].items() if key not in valid_incompatible_unexp_keys
    }
    model.load_state_dict(d['model'], strict=False)
    if not isinstance(optimizer, type(None)):
        optimizer.load_state_dict(d['optimizer'])
    return model, optimizer, epoch


class ConfigWrapper(object):
    """
    Wrapper dict class to avoid annoying key dict indexing like:
    `config.sample_rate` instead of `config["sample_rate"]`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigWrapper(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict_type(self):
        return {
            key: (value if not isinstance(value, ConfigWrapper) else value.to_dict_type())
            for key, value in dict(**self).items()
        }

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()