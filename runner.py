#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import numpy as np
import os
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net
from core.predict import predict_net
from core.MIgenerate import MI_train_net
from core.mi_train import mi_train_net
from core.generator import gen_net


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHS, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--mi_weights', dest='mi_weights', help='Initialize network from the mutual information weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--pred', dest='pred', help='predict neural networks', action='store_true')
    parser.add_argument('--gen', dest='gen', help='generate results', action='store_true')
    parser.add_argument('--mcmi', dest='mcmi', help='use mutual information', action='store_true')
    parser.add_argument('--mitrain', dest='mitrain', help='use mutual information', action='store_true')
    parser.add_argument('--img', dest='img', help='input img', default=None)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHS = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True
    if args.mi_weights is not None:
        cfg.CONST.MI_WEIGHTS = args.mi_weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test:
        if not args.pred and not args.mcmi and not args.mitrain and not args.gen:
            train_net(cfg)
        elif args.mcmi:
            MI_train_net(cfg)
        elif args.mitrain:
            mi_train_net(cfg)
        elif args.gen:
            gen_net(cfg)
        else:
            if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
                predict_net(cfg, args.img)
            else:
                logging.error('Please specify the file path of checkpoint.')
                sys.exit(2)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            test_net(cfg)
        else:
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
