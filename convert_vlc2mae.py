import os
import torch
from collections import OrderedDict


def convert_vanilla_mae():
    src_path = '/data/liangkeg/vlc_checkpoints/vlc_largeset.ckpt'
    dst_root = '/data/liangkeg/vlc_checkpoints/'

    vlc_model = torch.load(src_path)
    vlc_state_dict = vlc_model['state_dict']

    mae_model = {}
    mae_state_dict = OrderedDict()
    for k,v in vlc_state_dict.items():
        if k.startswith('transformer.'):
            mae_state_dict[k[12:]] = vlc_state_dict[k]

    mae_model['model'] = mae_state_dict
    torch.save(mae_model, os.path.join(dst_root, 'vanilla_mae_model.ckpt'))

def convert_token_mae():
    src_path = '/data/liangkeg/vlc_checkpoints/vlc_largeset.ckpt'
    dst_root = '/data/liangkeg/vlc_checkpoints/'

    vlc_model = torch.load(src_path)
    vlc_state_dict = vlc_model['state_dict']

    mae_model = {}
    mae_state_dict = OrderedDict()
    for k,v in vlc_state_dict.items():
        if k.startswith('transformer.'):
            mae_state_dict[k[12:]] = vlc_state_dict[k]

    mae_model['model'] = mae_state_dict
    torch.save(mae_model, os.path.join(dst_root, 'vanilla_mae_model.ckpt'))


if __name__ == '__main__':
    convert_vanilla_mae()
