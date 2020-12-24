import cv2
import json
import numpy as np
import logging
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset
import pandas as pd
from pytorch3d.renderer.cameras import look_at_view_transform
from config import cfg
from enum import Enum, unique

import utils.binvox_rw


def Rotate_Weight(volume, azim, elev, dist):
    volume = volume.transpose(2,1,0)
    R, T = look_at_view_transform(dist=dist, elev=90-elev, azim=azim)
    R, T = R.numpy(), T.numpy()
    R, T = np.asmatrix(R), np.asmatrix(T)
    M = np.zeros([4,4])
    M[0:3, 0:3] = R
    M[0:3,3] = T
    x,y,z = np.nonzero(volume)
    voxles = np.ones([len(x),4])
    voxles[:,0] = x-16
    voxles[:,1] = y-16
    voxles[:,2] = z-16
    after = np.matmul(voxles,M)
    after = np.matrix.round(after[:,0:3])
    after = after.astype(np.int32)
    after = np.transpose(after)
    mask = np.zeros(volume.shape)
    after = np.clip(after, -16, 15)
    mask[after[2]+16,after[0]+16,after[1]+16] = 1
    face = []
    for i in range(32):
        for j in range(32):
            if np.max(mask[:,i,j]) == 1:
                k = np.where(mask[:,i,j]==1)[0][0]
                face.append([i-16,j-16,k-16])               
    index = []            
    for i in range(len(after[0])):
        if list(after[:,i]) in face:
            index.append(i)
    weight = np.ones([32,32,32])*0.88
    voxles_n = (voxles+16).astype(np.int32)
    for i in index:
        z, y, x, _ = voxles_n[i]
        weight[x,y,z] = 1

    return weight


with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
    dataset_taxonomy = json.loads(file.read())
files = []
taxonomy = dataset_taxonomy[0]
taxonomy_folder_name = taxonomy['taxonomy_id']
logging.info('Collecting files of Taxonomy[ID=%s, Name=%s]' %
                (taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
samples = []
samples = taxonomy['train']

metadata_path_template = cfg.DATASETS.SHAPENET.METADATA_PATH
volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH
weight_path_template = cfg.DATASETS.SHAPENET.WEIGHT_PATH
rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH

files_of_taxonomy = []
for sample_idx, sample_name in enumerate(samples):
    print('%d / %d.' % (sample_idx, len(samples)))
    # Get file path of volumes
    volume_file_path = volume_path_template % (taxonomy_folder_name, sample_name)
    metadata_path = metadata_path_template % (taxonomy_folder_name, sample_name)
    weight_file_path = weight_path_template % (taxonomy_folder_name, sample_name, 0)
    img_file_path = rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
    with open(volume_file_path, 'rb') as f:
        volume = utils.binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)
    metadata = pd.read_csv(metadata_path, sep=' ', names=['azimuth', 'elevation', 'in-plane rotation', 'distance', 'the field of view'])

    img_folder = os.path.dirname(img_file_path)
    total_views = int(metadata.iloc[0]['the field of view'])-1
    rendering_image_indexes = range(total_views)
    for image_idx in rendering_image_indexes:
        weight_file_path = weight_path_template % (taxonomy_folder_name, sample_name, image_idx)
        angels = [metadata.iloc[image_idx]['azimuth'], metadata.iloc[image_idx]['elevation'], metadata.iloc[image_idx]['distance']]
        weight = Rotate_Weight(volume, angels[0], angels[1], angels[2])
        weight = weight.reshape(1,32*32*32)
        np.save(weight_file_path, weight)
