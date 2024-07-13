import torch
from torch.nn import ReLU

from forehead_search import config


if __name__ == '__main__':
    unet = torch.load('output/focal_unet.pth').to(config.Device)
    
