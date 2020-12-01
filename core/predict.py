# -*- coding: utf-8 -*-
#
# Developed by Yuzhen Mao

import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter
from tensorboardX import SummaryWriter 

import csv
import cv2


def predict_net(cfg,
             img,
             epoch_idx=-1,
             test_data_loader=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    test_writer = SummaryWriter('./tensorboard')

    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    trans = utils.data_transforms.Compose(
    [
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    
    rendering_images = []
    rendering_image = cv2.imread(img, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    rendering_images.append(rendering_image)
    rendering_images = np.asarray(rendering_images)
    rendering_images = trans(rendering_images)
    rendering_images = rendering_images[np.newaxis, :]

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    with torch.no_grad():
        # Get data from data loader
        rendering_images = utils.helpers.var_or_cuda(rendering_images)

        # Test the encoder, decoder, refiner and merger
        image_features = encoder(rendering_images)
        raw_features, generated_volume = decoder(image_features)

        if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)

        if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
            generated_volume = refiner(generated_volume)

        # Append generated volumes to TensorBoard
        if test_writer:
            # Volume Visualization
            rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
            data = generated_volume.cpu().numpy().__ge__(0.5).reshape(1,32*32*32)
            np.save('out.npy', data)

            # test_writer.add_image('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx)
            # rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
            # test_writer.add_image('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx)
            
    test_writer.close()
    return None
