#! /usr/bin/env python3

import pathlib
from PIL import Image
import numpy as np


# objectseg class color map (RGB) — matches objectseg.yaml
COLOR_MAP = {
    0: (61,  93, 255),   # background
    1: (255,  28, 145),  # man-made_structures
    2: (255,  61,  61),  # vulnerable_living
    3: (255, 190,  61),  # vehicle
    4: (  0, 100,   0),  # natural_obstacles
}
NUM_CLASSES = len(COLOR_MAP)
IGNORE_INDEX = 255  # label pixels whose color matches no class

DATASETS = ['Goose', 'OFFSED', 'Rellis_3D', 'CARLA_Mining']


class LoadDataObjectSeg:
    def __init__(self, labels_filepath, images_filepath, dataset: str):

        self.dataset = dataset
        if self.dataset not in DATASETS:
            raise ValueError('Dataset type is not correctly specified')

        self.labels = sorted(
            [f for f in pathlib.Path(labels_filepath).glob('*.png')])
        self.images = sorted(
            [f for f in pathlib.Path(images_filepath).glob('*.png')])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        if self.num_images != self.num_labels:
            raise ValueError(
                'Number of images and ground truth labels are mismatched')

        if self.num_images == 0:
            raise ValueError('No images found - check the root path')

        if self.num_labels == 0:
            raise ValueError('No ground truth masks found - check the root path')

        self.train_images = []
        self.train_labels = []
        self.val_images   = []
        self.val_labels   = []

        self.num_train_samples = 0
        self.num_val_samples   = 0

        for count in range(self.num_images):
            if (count + 1) % 10 == 0:
                self.val_images.append(str(self.images[count]))
                self.val_labels.append(str(self.labels[count]))
                self.num_val_samples += 1
            else:
                self.train_images.append(str(self.images[count]))
                self.train_labels.append(str(self.labels[count]))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    def createGroundTruth(self, input_label):
        input_np = np.array(input_label.convert('RGB'))
        row, col, _ = input_np.shape
        num_pixels  = row * col

        # Pack RGB into a single uint32 key per pixel, then map colors to
        # class ids in one pass (pixels with unknown colors -> IGNORE_INDEX).
        key = (input_np[..., 0].astype(np.uint32) << 16) | \
              (input_np[..., 1].astype(np.uint32) << 8) | \
              input_np[..., 2].astype(np.uint32)
        index_mask = np.full((row, col), IGNORE_INDEX, dtype=np.uint8)
        for class_id, (r, g, b) in COLOR_MAP.items():
            index_mask[key == ((r << 16) | (g << 8) | b)] = class_id

        epsilon = 5120
        counts = np.bincount(index_mask.ravel(), minlength=256)[:NUM_CLASSES]
        class_weights = [num_pixels / (counts[c] + epsilon) for c in range(NUM_CLASSES)]

        # gt[0] = (H,W) uint8 class index mask (IGNORE_INDEX = no class)
        return [index_mask], class_weights

    def getItemTrain(self, index):
        image = Image.open(self.train_images[index]).convert('RGB')
        label = Image.open(self.train_labels[index])
        gt, class_weights = self.createGroundTruth(label)
        return np.array(image), gt, class_weights

    def getItemTrainPath(self, index):
        return self.train_images[index], self.train_labels[index]

    def getItemVal(self, index):
        image = Image.open(self.val_images[index]).convert('RGB')
        label = Image.open(self.val_labels[index])
        gt, class_weights = self.createGroundTruth(label)
        return np.array(image), gt, class_weights

    def getItemValPath(self, index):
        return self.val_images[index], self.val_labels[index]
