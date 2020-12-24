# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

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

from enum import Enum, unique

import utils.binvox_rw


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']
        weights_path = self.file_list[idx]['weights']

        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = []
            weights = []
            for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering):
                selected_rendering_image_paths.append(rendering_image_paths[i])
                weights.append(weights_path[i])
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                logging.error('It seems that there is something wrong with the image file %s' % (image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)
                if self.dataset_type == DatasetType.TRAIN:
                    weight = weights[0]
                    volume = np.multiply(volume, weight)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume

    def Rotate_Weight(self, volume, azim, elev, dist):
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
        weight = np.ones([32,32,32])*0.5
        voxles_n = (voxles+16).astype(np.int32)
        for i in index:
            z, y, x, _ = voxles_n[i]
            weight[x,y,z] = 1

        return weight


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.metadata_path_template = cfg.DATASETS.SHAPENET.METADATA_PATH
        self.weight_path_template = cfg.DATASETS.SHAPENET.WEIGHT_PATH
        self.volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        taxonomy = self.dataset_taxonomy[0]
        taxonomy_folder_name = taxonomy['taxonomy_id']
        logging.info('Collecting files of Taxonomy[ID=%s, Name=%s]' %
                        (taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
        samples = []
        if dataset_type == DatasetType.TRAIN:
            samples = taxonomy['train']
        elif dataset_type == DatasetType.TEST:
            samples = taxonomy['test']
        elif dataset_type == DatasetType.VAL:
            samples = taxonomy['val']

        files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, samples))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return ShapeNetDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            metadata_path = self.metadata_path_template % (taxonomy_folder_name, sample_name)
            weight_file_path = self.weight_path_template % (taxonomy_folder_name, sample_name, 0)
            metadata = pd.read_csv(metadata_path, sep=' ', names=['azimuth', 'elevation', 'in-plane rotation', 'distance', 'the field of view'])
            img_folder = os.path.dirname(img_file_path)
            total_views = int(metadata.iloc[0]['the field of view'])-1
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            weights_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                weight_file_path = self.weight_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_path.append(img_file_path)
                weights_file_path.append(weight_file_path)

            

            if len(rendering_images_file_path) == 0:
                logging.warn('Ignore sample %s/%s since image files not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
                'metadata': metadata_path,
                'weights': weights_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of Things3DDataLoader Class Definition = /////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    # 'Pascal3D': Pascal3dDataLoader,
    # 'Pix3D': Pix3dDataLoader,
    # 'Things3D': Things3DDataLoader
}  # yapf: disable
