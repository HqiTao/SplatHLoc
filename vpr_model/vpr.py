import torch
from .mixvpr import MixVPRModel

def get_trained_mixvpr():
    vpr_model = MixVPRModel(agg_config={'in_channels' : 1024,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : 1024,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' :4})

    state_dict = torch.load('vpr_model/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
    vpr_model.load_state_dict(state_dict)
    return vpr_model
