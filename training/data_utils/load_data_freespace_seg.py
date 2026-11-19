#! /usr/bin/env python3

import pathlib
from PIL import Image
import numpy as np


DATASETS = ['Rellis_3D', 'Goose', 'ORFD', 'OFFSED', 'Yamaha_CMU'] #, 'CARLA_Mining', 'Fortysix']

class LoadDataFreespaceSeg:
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

        if (self.num_images != self.num_labels):
            raise ValueError(
                'Number of images and ground truth labels are mismatched')

        if (self.num_images == 0):
            raise ValueError('No images found - check the root path')

        if (self.num_labels == 0):
            raise ValueError(
                'No ground truth masks found - check the root path')

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.num_train_samples = 0
        self.num_val_samples = 0

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
        # Colourmaps for classes
        non_drivable_colour = (61, 93, 255)
        drivable_colour = (0, 255, 220)

        # Convert input PIL image to NumPy array
        input_np = np.array(input_label)
        row, col, _ = input_np.shape
        num_pixels = row*col

        # Initialize visualization and masks
        vis = np.array(input_np, dtype=np.uint8)
        ground_truth_non_drivable = np.zeros((row, col), dtype=np.uint8)
        ground_truth_drivable = np.zeros((row, col), dtype=np.uint8)

        # Define boolean masks for classes
        non_drivable_mask = np.isin(input_np.reshape(-1, 3), non_drivable_colour).all(axis=1).reshape(row, col)
        drivable_mask = np.isin(input_np.reshape(-1, 3), drivable_colour).all(axis=1).reshape(row, col)

        # Apply masks
        ground_truth_non_drivable[non_drivable_mask] = 255
        ground_truth_drivable[drivable_mask] = 255

        # Compute class frequencies
        non_drivable_class_freq = np.sum(non_drivable_mask)
        drivable_class_freq = np.sum(drivable_mask)

        # Calculate class weights for loss function
        epsilon = 5120
        class_weights = [
            num_pixels / (non_drivable_class_freq + epsilon),
            num_pixels / (drivable_class_freq + epsilon)
        ]

        # Getting ground truth data
        ground_truth = [
            vis,
            ground_truth_non_drivable,
            ground_truth_drivable
        ]

        return ground_truth, class_weights

    def getItemTrain(self, index):
        self.train_image = Image.open(str(self.train_images[index])).convert("RGB")
        self.train_label = Image.open(str(self.train_labels[index]))
        self.train_ground_truth, self.train_class_weights = \
            self.createGroundTruth(self.train_label)

        return np.array(self.train_image), self.train_ground_truth, \
            self.train_class_weights

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])

    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index])).convert("RGB")
        self.val_label = Image.open(str(self.val_labels[index]))
        self.val_ground_truth, self.val_class_weights = \
            self.createGroundTruth(self.val_label)

        return np.array(self.val_image), self.val_ground_truth, \
            self.val_class_weights

    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])


# ---------------------------------------------------------------------------
# Fast torch Dataset (used with DataLoader worker processes)
# ---------------------------------------------------------------------------

import cv2
import torch
from torch.utils.data import Dataset

from data_utils.augmentations import Augmentations
from data_utils.radial_contour_dists import RadialContourDists

# Label colours (RGB). The masks contain only these two colours, so the red
# channel alone identifies the class (0 -> drivable, 61 -> non-drivable).
_DRIVABLE_R = 0
_NON_DRIVABLE_R = 61
_EPSILON = 5120


def worker_init_fn(_worker_id):
    # cv2 spawns its own thread pool; with many DataLoader workers this
    # oversubscribes the CPU.
    cv2.setNumThreads(1)


class FreespaceSegTorchDataset(Dataset):
    """Wraps a LoadDataFreespaceSeg split.

    train: returns (image uint8 (3,H,W), target uint8 (H,W) with 1 = drivable,
                    class_weights float32 (2,))
    val:   returns (image uint8 (3,H,W), masks uint8 (2,H,W) [non_drivable,
                    drivable], class_weights)

    contour=True (FreespaceContour training): the second element is instead the
    int64 ray labels (37,) computed from the augmented drivable mask, for both
    train and val.
    """

    def __init__(self, loader: LoadDataFreespaceSeg, is_train: bool, contour: bool = False):
        self.is_train = is_train
        self.contour = contour
        if is_train:
            self.images, self.labels = loader.train_images, loader.train_labels
        else:
            self.images, self.labels = loader.val_images, loader.val_labels
        self.aug = None  # built lazily, once per worker process

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.aug is None:
            self.aug = Augmentations(is_train=self.is_train, data_type='SEGMENTATION')

        image = cv2.cvtColor(cv2.imread(self.images[index], cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB)
        red = cv2.imread(self.labels[index], cv2.IMREAD_COLOR)[..., 2]  # BGR -> R

        drivable = red == _DRIVABLE_R
        non_drivable = red == _NON_DRIVABLE_R
        num_pixels = red.size
        class_weights = torch.tensor(
            [num_pixels / (non_drivable.sum() + _EPSILON),
             num_pixels / (drivable.sum() + _EPSILON)], dtype=torch.float32)

        image, (non_drivable, drivable) = self.aug.applyTransformSeg(
            image=image,
            ground_truth=[non_drivable.astype(np.uint8), drivable.astype(np.uint8)])

        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        if self.contour:
            rays = RadialContourDists(drivable, num_rays=37, ray_slice_dist=10,
                                      max_search_dist=460).get_contour_dists()
            return image, torch.tensor(rays, dtype=torch.long), class_weights
        if self.is_train:
            # Original pipeline: argmax over (non_drivable, drivable) channels,
            # i.e. drivable wins only where its mask is set.
            target = torch.from_numpy(np.ascontiguousarray(drivable))
        else:
            target = torch.from_numpy(np.stack([non_drivable, drivable]))
        return image, target, class_weights
