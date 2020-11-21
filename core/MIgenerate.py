# -*- coding: utf-8 -*-
#
# Developed by Yuzhen Mao

import os
import logging
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.Trans_decoder import Trans_Decoder
from models.MIFC import FullConnected
from models.MILoss import MILoss
from utils.average_meter import AverageMeter

import numpy as np
import pdb

import matplotlib
import matplotlib.pyplot as plt


def MI_train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    trans_decoder = Trans_Decoder(cfg)
    fc = FullConnected(cfg)
    logging.debug('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.debug('Parameters in Trans_Decoder: %d.' % (utils.helpers.count_parameters(trans_decoder)))
    logging.debug('Parameters in FullConnect: %d.' % (utils.helpers.count_parameters(fc)))

    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    trans_decoder.apply(utils.helpers.init_weights)
    fc.apply(utils.helpers.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=0.00001,
                                          betas=cfg.TRAIN.BETAS)
        trans_decoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, trans_decoder.parameters()),
                                          lr=0.0025,
                                          betas=cfg.TRAIN.BETAS)
        fc_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, fc.parameters()),
                                          lr=0.0025,
                                          betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=0.00001,
                                         momentum=cfg.TRAIN.MOMENTUM)
        trans_decoder_solver = torch.optim.SGD(trans_decoder.parameters(),
                                         lr=0.0025,
                                         momentum=cfg.TRAIN.MOMENTUM)
        fc_solver = torch.optim.SGD(fc.parameters(),
                                         lr=0.0025,
                                         momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=[200, 230],
                                                                gamma=0.5)
    trans_decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(trans_decoder_solver,
                                                                milestones=[20, 40, 60, 80, 100, 120, 200],
                                                                gamma=0.5)
    fc_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(fc_solver,
                                                                milestones=[20, 40, 60, 80, 100, 120, 200],
                                                                gamma=0.5)
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        trans_decoder = torch.nn.DataParallel(trans_decoder).cuda()
        fc = torch.nn.DataParallel(fc).cuda()

    # Set up loss functions
    mi_loss = MILoss()
    # mi_loss = torch.nn.CosineEmbeddingLoss(reduction = "mean")

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        trans_decoder.load_state_dict(checkpoint['trans_decoder_state_dict'])
        fc.load_state_dict(checkpoint['fc_state_dict'])
        # decoder.load_state_dict(checkpoint['decoder_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))
    
    plt.ioff()
    fig = plt.figure()
    train_loss_over_epochs = []

    # Training loop
    for epoch_idx in range(init_epoch, 250):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # switch models to training mode
        encoder.train()
        trans_decoder.train()
        fc.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).requires_grad_()
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes).requires_grad_()

            image_features = encoder(rendering_images)
            voxal_features = trans_decoder(ground_truth_volumes)
            
            # Full connected layers
            image_vectors = fc(image_features)
            voxal_vectors = fc(voxal_features)

            # data = voxal_vectors.cpu().detach().numpy().reshape(64,128)
            # np.save('test.npy', data)

            # Get Loss
            loss = mi_loss(image_vectors, voxal_vectors)

            # Gradient decent
            encoder.zero_grad()
            trans_decoder.zero_grad()
            fc.zero_grad()
            
            loss.backward()
            # print(rendering_images.grad)
            # print(ground_truth_volumes.grad)
            if epoch_idx > 169:
                encoder_solver.step()
            trans_decoder_solver.step()
            fc_solver.step()

            # Append loss to average metrics
            losses.update(loss.item())

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) -Loss = %.4f lr = %f' %
                (epoch_idx + 1, 250, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                 -1*loss.item(), trans_decoder_lr_scheduler.state_dict()['_last_lr'][0]))

        # Adjust learning rate
        encoder_lr_scheduler.step()
        trans_decoder_lr_scheduler.step()
        fc_lr_scheduler.step()

        train_loss_over_epochs.append(-1*losses.avg)

        # Tick / tock
        epoch_end_time = time()
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) -Loss = %.4f' %
                     (epoch_idx + 1, 250, epoch_end_time - epoch_start_time, -1*losses.avg))

        # # Validate the training models
        # iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder, decoder, refiner, merger)

        # Save weights to file
        if (epoch_idx + 1) % 10 == 0:
            file_name = 'mi-checkpoint-epoch-%03d.pth' % (epoch_idx + 1)

            output_path = os.path.join(cfg.DIR.OUT_PATH, file_name)
            if not os.path.exists(cfg.DIR.OUT_PATH):
                os.makedirs(cfg.DIR.OUT_PATH)

            checkpoint = {
                'epoch_idx': epoch_idx + 1,
                'encoder_state_dict':encoder.state_dict(),
                'trans_decoder_state_dict': trans_decoder.state_dict(),
                'fc_state_dict':fc.state_dict()
            }

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    plt.ylabel('Train loss')
    plt.plot(np.arange(250), train_loss_over_epochs, 'k-')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(250, dtype=int, step=50))
    plt.grid(True)
    plt.savefig("plot.png")
    plt.close(fig)

