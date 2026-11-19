#! /usr/bin/env python3

import torch
import numpy as np
from tqdm import tqdm
import random
from argparse import ArgumentParser
import os
import sys
_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [p for p in (_MODELS, os.path.dirname(_MODELS)) if p not in sys.path]
from data_utils.load_data_object_seg import LoadDataObjectSeg, DATASETS
from training.resume_util import save_last, load_resume
from training.ObjectSeg.object_seg_trainer import ObjectSegTrainer, augment_sample

NUM_WORKERS = min(12, os.cpu_count() or 1)


class _SampleListDataset(torch.utils.data.Dataset):
    """Loads + augments samples in a precomputed order inside DataLoader workers
    (PNG decoding / label conversion / augmentation are CPU-bound)."""
    def __init__(self, datasets, sample_list, is_train):
        self.datasets = datasets
        self.sample_list = sample_list
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, i):
        name, index = self.sample_list[i]
        if self.is_train:
            image, gt, class_weights = self.datasets[name].getItemTrain(index)
        else:
            image, gt, class_weights = self.datasets[name].getItemVal(index)
        image, index_mask = augment_sample(image, gt, is_train=self.is_train)
        return image, index_mask, class_weights


def _identity(sample):
    return sample


def make_loader(datasets, sample_list, is_train):
    return torch.utils.data.DataLoader(
        _SampleListDataset(datasets, sample_list, is_train),
        batch_size=None, shuffle=False, num_workers=NUM_WORKERS,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        collate_fn=_identity)


def batch_size_for_epoch(epoch: int) -> int:
    if epoch < 10:  return 16
    if epoch < 30:  return 8
    if epoch < 60:  return 4
    if epoch < 80:  return 2
    return 1


def main():

    parser = ArgumentParser()
    parser.add_argument('-s', '--model_save_root_path', dest='model_save_root_path',
                        default=os.path.join(_MODELS, 'checkpoints', 'ObjectSeg'),
                        help='root path where pytorch checkpoint files should be saved')
    parser.add_argument('-m', '--pretrained_checkpoint_path', dest='pretrained_checkpoint_path',
                        default='',
                        help='path to SceneSeg weights file for pre-trained backbone')
    parser.add_argument('-c', '--checkpoint_path', dest='checkpoint_path', default='',
                        help='path to ObjectSeg weights file for resuming training')
    parser.add_argument('-r', '--root', dest='root',
                        required=True, help='root path to folder where dataset directories live')
    parser.add_argument('-d', '--datasets', dest='datasets', nargs='+', default=None,
                        help=f'datasets to use (default: all). Choices: {DATASETS}')
    parser.add_argument('-l', '--load_from_save', action='store_true',
                        help='resume training from an ObjectSeg checkpoint')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=100, type=int,
                        help='number of training epochs (default: 100)')
    parser.add_argument('-a', '--start_epoch', dest='start_epoch', default=0, type=int,
                        help='starting epoch (default: 0)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001, type=float,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--resume', action='store_true',
                        help='continue from last_objectseg.pth / last_objectseg.state.pt in the save '
                             'directory (weights, optimizer, epoch; run ends at the original last epoch)')
    args = parser.parse_args()

    os.makedirs(args.model_save_root_path, exist_ok=True)

    active_datasets = args.datasets if args.datasets else DATASETS

    datasets_info = {
        name: {
            'labels': os.path.join(args.root, name, name, 'gt_masks/'),
            'images': os.path.join(args.root, name, name, 'images/')
        }
        for name in active_datasets
    }

    datasets = {}
    for name in active_datasets:
        try:
            datasets[name] = LoadDataObjectSeg(
                datasets_info[name]['labels'],
                datasets_info[name]['images'],
                name)
        except Exception as e:
            print(f'Warning: skipping dataset "{name}": {e}')

    if not datasets:
        raise RuntimeError('No datasets loaded — check --root path and dataset directories.')

    counts = {name: datasets[name].getItemCount() for name in datasets}
    total_train_samples = sum(v[0] for v in counts.values())
    total_val_samples   = sum(v[1] for v in counts.values())
    print(total_train_samples, ': total training samples')
    print(total_val_samples,   ': total validation samples')

    if args.resume:
        trainer = ObjectSegTrainer(is_pretrained=True, learning_rate=args.learning_rate)
    elif args.load_from_save:
        trainer = ObjectSegTrainer(checkpoint_path=args.checkpoint_path,
                                   is_pretrained=True,
                                   learning_rate=args.learning_rate)
    else:
        trainer = ObjectSegTrainer(pretrained_checkpoint_path=args.pretrained_checkpoint_path,
                                   learning_rate=args.learning_rate)

    trainer.zero_grad()

    start_epoch = args.start_epoch
    end_epoch = start_epoch + args.num_epochs
    if args.resume:
        start_epoch, end_epoch = load_resume(trainer, args.model_save_root_path, 'objectseg')

    for epoch in range(start_epoch, end_epoch):

        train_counts = {name: 0 for name in datasets}
        completed    = {name: False for name in datasets}
        data_list    = list(datasets.keys())
        random.shuffle(data_list)
        data_list_count = 0
        batch_size = batch_size_for_epoch(epoch)
        losses = []

        # Precompute this epoch's (dataset, index) order (same round-robin
        # schedule as before), then load/augment it in parallel workers.
        sample_list = []
        for count in range(total_train_samples):
            for name in list(data_list):
                if train_counts[name] == counts[name][0] and not completed[name]:
                    completed[name] = True
                    data_list.remove(name)

            if data_list_count >= len(data_list):
                data_list_count = 0

            dataset_name = data_list[data_list_count]
            sample_list.append((dataset_name, train_counts[dataset_name]))
            train_counts[dataset_name] += 1
            data_list_count += 1

        train_loader = make_loader(datasets, sample_list, is_train=True)

        for count, (image, index_mask, class_weights) in enumerate(tqdm(
                train_loader, total=total_train_samples,
                desc=f'Training Epoch {epoch + 1} / {end_epoch}')):

            log_count = count + total_train_samples * epoch

            trainer.set_augmented_data(image, index_mask, class_weights)
            trainer.load_data(is_train=True)
            trainer.run_model()
            losses.append(trainer.get_loss())
            trainer.loss_backward()

            if (count + 1) % batch_size == 0 or (count + 1) == total_train_samples:
                trainer.run_optimizer()

            if (count + 1) % 250 == 0 or (count + 1) == total_train_samples:
                trainer.log_loss(log_count, loss=np.mean(losses))
                losses.clear()

            if (count + 1) % 1000 == 0 or (count + 1) == total_train_samples:
                trainer.save_visualization(log_count)

            if (count + 1) % 8000 == 0 or (count + 1) == total_train_samples:

                model_save_path = os.path.join(
                    args.model_save_root_path,
                    f'iter_{count + total_train_samples * epoch}'
                    f'_epoch_{epoch}_step_{count}.pth')
                trainer.save_model(model_save_path)

                # Always keep the newest checkpoint + training state (for --resume)
                save_last(trainer, args.model_save_root_path, 'objectseg', epoch, end_epoch,
                          epoch_finished=((count + 1) == total_train_samples))

                print('Validating')
                trainer.set_eval_mode()

                running_iou_full     = 0.0
                running_iou_per_class = [0.0] * 5

                val_list = [(name, i) for name in datasets for i in range(counts[name][1])]
                with torch.no_grad():
                    for image_val, index_mask_val, _ in tqdm(
                            make_loader(datasets, val_list, is_train=False),
                            total=total_val_samples, desc='Validating'):
                        trainer.set_augmented_val_data(image_val, index_mask_val)
                        trainer.load_data(is_train=False)
                        iou_full, iou_per_class = trainer.calc_IoU_val()
                        running_iou_full += iou_full
                        for i in range(5):
                            running_iou_per_class[i] += iou_per_class[i]

                mIoU_full     = running_iou_full / total_val_samples
                mIoU_per_class = [v / total_val_samples for v in running_iou_per_class]
                trainer.log_IoU(mIoU_full, mIoU_per_class, log_count)

                trainer.set_train_mode()

    trainer.cleanup()


if __name__ == '__main__':
    main()
