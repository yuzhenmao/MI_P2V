# -*- coding: utf-8 -*-
#
# Developed by Yuzhen Mao

import torch


class FullConnected(torch.nn.Module):
    def __init__(self, cfg):
        super(FullConnected, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(256*7*7, 1024),
            torch.nn.BatchNorm1d(num_features = 1024),
            torch.nn.ReLU(inplace=True),

            # torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(num_features = 512),
            torch.nn.ReLU(inplace=True),

            # torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 128),
        )

    def forward(self, image_features):
        image_features = torch.squeeze(image_features, dim=1)
        image_features = image_features.contiguous()

        image_features = image_features.view(-1, 256 * 7 * 7)
        image_vectors = self.fc_net(image_features)

        # print(image_vectors.size())      # torch.Size([batch_size, 128])
        return image_vectors
