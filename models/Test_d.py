# -*- coding: utf-8 -*-
#
# Developed by Yuzhen Mao

import torch
import numpy as np
from .Res3D import generate_model

class up(torch.nn.Module):
    def __init__(self):
        super(up, self).__init__()
        
        self.layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 196, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(196),
            torch.nn.ReLU()
        )

    def forward(self, x):
        y = self.layer(x)
        return y


class Trans_Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Trans_Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.up = up()


    def forward(self, voxal_features):
        resnet3d = generate_model(34).cuda()
        voxal_features = voxal_features.contiguous()

        voxal_features = voxal_features[np.newaxis, :]
        gen_volume = voxal_features.permute(1, 0, 2, 3, 4).contiguous()
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        gen_volume = resnet3d(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 2, 2, 2])

        gen_volume = self.up(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 196, 4, 4, 4])

        return gen_volume
