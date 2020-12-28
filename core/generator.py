# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import logging
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.plane_loaders

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter

import pdb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils.binvox_rw
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import cv2
from multiprocessing import Pool


def gen_video_1(ii,item,iou,data,i):
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.voxels(data, facecolors='blue', edgecolor="k", shade=False)
    ax.view_init(elev=10., azim=ii)
    plt.axis('off')
    ii = (ii-90)/30
    ax.text(16, 16, 36, "IoU: %.4f"%iou, verticalalignment="top",horizontalalignment="center")
    plt.savefig("./result/%d/%s/movie%d.png" % (i, item, ii))
    plt.close()

def gen_video(params):
    return gen_video_1(params[0], params[1], params[2], params[3], params[4])


def gen_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    dataset_loader = utils.plane_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.plane_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                    batch_size=1,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=False)

    # # Set up data loader
    # train_dataset_loader = utils.plane_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    # val_dataset_loader = utils.plane_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    # train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
    #     utils.plane_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
    #                                                 batch_size=50,
    #                                                 num_workers=cfg.CONST.NUM_WORKER,
    #                                                 pin_memory=True,
    #                                                 shuffle=True,
    #                                                 drop_last=True)
    # val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
    #     utils.plane_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
    #                                               batch_size=1,
    #                                               num_workers=cfg.CONST.NUM_WORKER,
    #                                               pin_memory=True,
    #                                               shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    mi_encoder = Encoder(cfg)
    mi_decoder = Decoder(cfg)
    mi_refiner = Refiner(cfg)
    mi_merger = Merger(cfg)
    logging.debug('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.debug('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    logging.debug('Parameters in Refiner: %d.' % (utils.helpers.count_parameters(refiner)))
    logging.debug('Parameters in Merger: %d.' % (utils.helpers.count_parameters(merger)))

    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    refiner.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)
    mi_encoder.apply(utils.helpers.init_weights)
    mi_decoder.apply(utils.helpers.init_weights)
    mi_refiner.apply(utils.helpers.init_weights)
    mi_merger.apply(utils.helpers.init_weights)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
        mi_encoder = torch.nn.DataParallel(mi_encoder).cuda()
        mi_decoder = torch.nn.DataParallel(mi_decoder).cuda()
        mi_refiner = torch.nn.DataParallel(mi_refiner).cuda()
        mi_merger = torch.nn.DataParallel(mi_merger).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))

    if 'MI_WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % (cfg.CONST.MI_WEIGHTS))
        checkpoint = torch.load(cfg.CONST.MI_WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        mi_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        mi_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_REFINER:
            mi_refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            mi_merger.load_state_dict(checkpoint['merger_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))

    # switch models to training mode
    encoder.eval()
    decoder.eval()
    merger.eval()
    refiner.eval()

    directory = 'result'

    if not os.path.exists(directory):
        os.makedirs(directory)

    with torch.no_grad():
        batch_end_time = time()
        n_batches = len(test_data_loader)
        IoUs = []
        for sample_idx, (taxonomy_id, sample_name, rendering_images, 
                        ground_truth_volumes) in enumerate(test_data_loader):

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            mi_image_features = mi_encoder(rendering_images)
            mi_raw_features, mi_generated_volume = mi_decoder(mi_image_features)

            if cfg.NETWORK.USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
                mi_generated_volume = mi_merger(mi_raw_features, mi_generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
                mi_generated_volume = torch.mean(mi_generated_volume, dim=1)

            if cfg.NETWORK.USE_REFINER:
                generated_volume = refiner(generated_volume)
                mi_generated_volume = mi_refiner(mi_generated_volume)

            # pdb.set_trace()
            # Volume Visualization
            # rendering_views = utils.helpers.get_volume_views(generated_volume[i].cpu().numpy())
            i = sample_idx

            directory = './result/%d/'%i
            if not os.path.exists(directory):
                os.makedirs(directory)

            data1 = ground_truth_volumes.cpu().numpy().__ge__(0.5).reshape([32,32,32])
            # path = './result/' + str(i) + '_gd.npy'
            # np.save(path, data1)

            data2 = generated_volume.cpu().numpy().__ge__(0.5).reshape([32,32,32])
            # path = './result/' + str(i) + '_out.npy'
            # np.save(path, data2)

            data3 = mi_generated_volume.cpu().numpy().__ge__(0.5).reshape([32,32,32])
            # path = './result/' + str(i) + '_miout.npy'
            # np.save(path, data3)

            img = rendering_images.cpu().numpy().reshape([3,224,224]).transpose(1, 2, 0)
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('./result/%d/img.png'%i)
            plt.close()
            # path = './result/' + str(i) + '_img.npy'
            # np.save(path, img)

            overlap1 = np.logical_and(data1, data2)
            union1 = np.logical_or(data1, data2)
            iou = np.sum(overlap1) / np.sum(union1)

            overlap2 = np.logical_and(data1, data3)
            union2 = np.logical_or(data1, data3)
            mi_iou = np.sum(overlap2) / np.sum(union2)

            # IoUs.append((iou, mi_iou))
            for item, iou, data in [('gd',1.0, data1), ('base',iou, data2), ('mi',mi_iou, data3)]:
                directory = './result/%d/%s'%(i,item)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                params = []
                for views in range(90,450,30):
                    params.append((views,item,iou,data,i))
                pool = Pool(12)
                pool.map(gen_video, params)
                pool.close()
                pool.join()

            for item in ['gd', 'base', 'mi']:
                path = './result/%d/%s/'%(i,item)
                ffmpeg_cmd = 'ffmpeg -framerate 3 -i '+path+'movie%d.png -r 15 '+path+'output.mp4'
                os.system(ffmpeg_cmd)

            ffmpeg_cmd = 'ffmpeg -i ./result/%d/img.png -i ./result/%d/gd/output.mp4 -i ./result/%d/mi/output.mp4 -i ./result/%d/base/output.mp4 -filter_complex hstack=4 ./result/%d_output.mp4'%(i,i,i,i,i)
            os.system(ffmpeg_cmd)

            # pdb.set_trace()

            rm_cmd = 'rm -r ./result/%d'%i
            os.system(rm_cmd)

            logging.info('Image %d IoU: [%.4f] MI-IoU: [%.4f]' % (i, iou, mi_iou))

