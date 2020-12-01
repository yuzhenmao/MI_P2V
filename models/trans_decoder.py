# -*- coding: utf-8 -*-
#
# Developed by Yuzhen Mao

import torch
import pdb


class Trans_Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Trans_Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1568, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(1568),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, volumns):
        voxal_features = torch.unsqueeze(volumns, 1).contiguous()
        # print(voxal_features.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        
        voxal_features = self.layer1(voxal_features)
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        voxal_features = self.layer2(voxal_features)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
        voxal_features = self.layer3(voxal_features)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        voxal_features = self.layer4(voxal_features)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
        voxal_features = self.layer5(voxal_features)
        # pdb.set_trace()
        # print(gen_volume.size())   # torch.Size([batch_size, 1568, 2, 2, 2])

        return voxal_features
