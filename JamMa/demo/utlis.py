import torch
from torch import nn
from src.jamma.jamma import JamMa as JamMa_
from src.jamma.backbone import CovNextV2_nano
from loguru import logger


class JamMa(nn.Module):
    def __init__(self, pretrained, config) -> None:
        super().__init__()
        self.backbone = CovNextV2_nano()
        self.matcher = JamMa_(config)

        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained}\' as pretrained checkpoint")

    def forward(self, data):
        self.backbone(data)
        return self.matcher(data)


cfg = {
    'coarse': {
        'd_model': 256,
    },
    'fine': {
        'd_model': 64,
        'dsmax_temperature': 0.1,
        'thr': 0.1,
        'inference': True
    },
    'match_coarse': {
        'thr': 0.2,
        'use_sm': True,
        'border_rm': 2,
        'dsmax_temperature': 0.1,
        'inference': True
    },
    'fine_window_size': 5,
    'resolution': [8, 2]
}
