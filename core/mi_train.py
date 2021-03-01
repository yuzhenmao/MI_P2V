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
from models.Trans_decoder import Trans_Decoder
from models.MIFC import FullConnected
from models.MILoss import MILoss

import numpy as np
import pdb

import matplotlib
import matplotlib.pyplot as plt


def mi_train_net(cfg):
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
    train_dataset_loader = utils.plane_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.plane_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.plane_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.plane_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    logging.debug('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.debug('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    logging.debug('Parameters in Refiner: %d.' % (utils.helpers.count_parameters(refiner)))
    logging.debug('Parameters in Merger: %d.' % (utils.helpers.count_parameters(merger)))
    # Set up mi-loss network
    mi_encoder = Encoder(cfg)
    mi_trans_decoder = Trans_Decoder(cfg)
    # mi_fc = FullConnected(cfg)
    logging.debug('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(mi_encoder)))
    logging.debug('Parameters in Trans_Decoder: %d.' % (utils.helpers.count_parameters(mi_trans_decoder)))
    # logging.debug('Parameters in FullConnect: %d.' % (utils.helpers.count_parameters(mi_fc)))

    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    refiner.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)
    mi_encoder.apply(utils.helpers.init_weights)
    mi_trans_decoder.apply(utils.helpers.init_weights)
    # mi_fc.apply(utils.helpers.init_weights)

    # Set up generate_solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(),
                                         lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(merger.parameters(),
                                        lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                        momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up MILoss_solver
    if cfg.TRAIN.POLICY == 'adam':
        mi_encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, mi_encoder.parameters()),
                                          lr=1e-3,
                                          betas=cfg.TRAIN.BETAS)
        mi_trans_decoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, mi_trans_decoder.parameters()),
                                          lr=1e-3,
                                          betas=cfg.TRAIN.BETAS)
        # mi_fc_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, mi_fc.parameters()),
        #                                   lr=1e-3,
        #                                   betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        mi_encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, mi_encoder.parameters()),
                                         lr=1e-1,
                                         momentum=cfg.TRAIN.MOMENTUM)
        mi_trans_decoder_solver = torch.optim.SGD(mi_trans_decoder.parameters(),
                                         lr=1e-1,
                                         momentum=cfg.TRAIN.MOMENTUM)
        # mi_fc_solver = torch.optim.SGD(mi_fc.parameters(),
        #                                  lr=1e-1,
        #                                  momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                               milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=cfg.TRAIN.GAMMA)
    mi_encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(mi_encoder_solver,
                                                                milestones=[150],
                                                                gamma=cfg.TRAIN.GAMMA)
    mi_trans_decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(mi_trans_decoder_solver,
                                                                milestones=[150],
                                                                gamma=cfg.TRAIN.GAMMA)
    # mi_fc_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(mi_fc_solver,
    #                                                             milestones=[150],
    #                                                             gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
        mi_encoder = torch.nn.DataParallel(mi_encoder).cuda()
        mi_trans_decoder = torch.nn.DataParallel(mi_trans_decoder).cuda()
        # mi_fc = torch.nn.DataParallel(mi_fc).cuda()

    # Set up loss functions
    # bce_loss = torch.nn.BCELoss()
    mi_loss = MILoss()

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
        mi_encoder.load_state_dict(checkpoint['mi_encoder_state_dict'])
        mi_trans_decoder.load_state_dict(checkpoint['mi_trans_decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    enc_train_loss_over_epochs = []
    ref_train_loss_over_epochs = []
    mi_enc_loss_over_epochs = []
    mi_gd_loss_over_epochs = []


    # Training loop
    for epoch_idx in range(init_epoch, 150):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()
        mi_encoder_losses = AverageMeter()
        mi_refiner_losses = AverageMeter()
        mi_losses = AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()
        mi_encoder.train()
        mi_trans_decoder.train()
        # mi_fc.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes, weight) in enumerate(train_data_loader):

            # for name, param in mi_trans_decoder.named_parameters():
            #     param.requires_grad = True
            # for name, param in mi_encoder.named_parameters():
            #     param.requires_grad = True
            # for name, param in mi_fc.named_parameters():
            #     param.requires_grad = True
            
            # Measure data time
            data_time.update(time() - batch_end_time)
            # pdb.set_trace()

            weight = weight.reshape(16, 32, 32, 32)
            rendering_images = rendering_images.reshape(16, 1, 3, 224, 224)
            # pdb.set_trace()
            ground_truth_volumes = torch.cat((ground_truth_volumes[0].expand(8,32,32,32), ground_truth_volumes[1].expand(8,32,32,32)), 0)
            # pdb.set_trace()
            # Get data from data loader
            # rendering_images = utils.helpers.var_or_cuda(rendering_images)
            # ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

            # im_vectors = mi_fc(mi_encoder(rendering_images))
            # gdvx_vecotrs = mi_fc(mi_trans_decoder(ground_truth_volumes))

            # mi_gd_loss = mi_loss(im_vectors, gdvx_vecotrs)

            # mi_encoder.zero_grad()
            # mi_trans_decoder.zero_grad()
            # mi_fc.zero_grad()

            # mi_gd_loss.backward()

            # mi_encoder_solver.step()
            # mi_trans_decoder_solver.step()
            # mi_fc_solver.step()

            # for name, param in mi_trans_decoder.named_parameters():
            #     param.requires_grad = False
            # for name, param in mi_encoder.named_parameters():
            #     param.requires_grad = False
            # for name, param in mi_fc.named_parameters():
            #     param.requires_grad = False

            # Train the encoder, decoder, refiner, and merger
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
            weight = utils.helpers.var_or_cuda(weight)
            weighted_bce_loss = torch.nn.BCELoss(weight=weight)

            image_features = encoder(rendering_images)
            raw_features, generated_volumes = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)
            encoder_loss = weighted_bce_loss(generated_volumes, ground_truth_volumes) * 10

            # im_vectors = mi_fc(mi_encoder(rendering_images))
            # vx_vectors = mi_fc(mi_trans_decoder(generated_volumes))
            im_vectors = mi_encoder(rendering_images)
            vx_vectors = mi_trans_decoder(generated_volumes)
            im_vectors = torch.squeeze(im_vectors, dim=1).contiguous().view(-1, 256 * 7 * 7)
            vx_vectors = torch.squeeze(vx_vectors, dim=1).contiguous().view(-1, 256 * 7 * 7)
            # pdb.set_trace()
            mi_encoder_loss = mi_loss(im_vectors, vx_vectors)

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                refiner_loss = weighted_bce_loss(generated_volumes, ground_truth_volumes) * 10
                # vx_vectors = mi_fc(mi_trans_decoder(generated_volumes))
                vx_vectors = mi_trans_decoder(generated_volumes)
                vx_vectors = torch.squeeze(vx_vectors, dim=1).contiguous().view(-1, 256 * 7 * 7)
                mi_refiner_loss = mi_loss(im_vectors, vx_vectors)
            else:
                refiner_loss = encoder_loss
                mi_refiner_loss = mi_encoder_loss

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()
            mi_encoder.zero_grad()
            mi_trans_decoder.zero_grad()
            # mi_fc.zero_grad()

            # mi_enc_loss = mi_encoder_loss.detach()
            # mi_ref_loss = mi_refiner_loss.detach()

            encoder_loss = encoder_loss + 0.01*mi_encoder_loss
            refiner_loss = refiner_loss + 0.01*mi_refiner_loss

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                # pdb.set_trace()
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            else:
                encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()
            mi_encoder_solver.step()
            mi_trans_decoder_solver.step()
            # mi_fc_solver.step()


            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            mi_encoder_losses.update(mi_encoder_loss.item())
            mi_refiner_losses.update(mi_refiner_loss.item())
            # mi_losses.update(mi_gd_loss.item())

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f -MIEDLoss = %.4f' %
                (epoch_idx + 1, 150, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                 encoder_loss.item(), -1*mi_encoder_loss.item()))

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        enc_train_loss_over_epochs.append(encoder_losses.avg)
        ref_train_loss_over_epochs.append(refiner_losses.avg)
        mi_enc_loss_over_epochs.append(mi_encoder_losses.avg)
        mi_gd_loss_over_epochs.append(mi_losses.avg)

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f -MIEDLoss = %.4f' %
                     (epoch_idx + 1, 150, epoch_end_time - epoch_start_time, encoder_losses.avg,
                      -1*mi_encoder_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            logging.info('Epoch [%d/%d] Update #RenderingViews to %d' %
                         (epoch_idx + 2, 150, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer, encoder, decoder, refiner, merger)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            file_name = 'plane_mirestrain_checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'plane_mirestrain_checkpoint-best.pth'

            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            if not os.path.exists(cfg.DIR.CHECKPOINTS):
                os.makedirs(cfg.DIR.CHECKPOINTS)

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'mi_encoder_state_dict': mi_encoder.state_dict(),
                'mi_trans_decoder_state_dict': mi_trans_decoder.state_dict(),
            }
            if cfg.NETWORK.USE_REFINER:
                checkpoint['refiner_state_dict'] = refiner.state_dict()
            if cfg.NETWORK.USE_MERGER:
                checkpoint['merger_state_dict'] = merger.state_dict()

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()

    plt.ioff()
    fig = plt.figure()

    plt.ylabel('Train loss')
    plt.plot(np.arange(150), enc_train_loss_over_epochs, 'b-', np.arange(150), mi_enc_loss_over_epochs, 'r-')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(150, dtype=int, step=10))
    plt.grid(True)
    plt.savefig("train_mi_plot.png")
    plt.close(fig)
