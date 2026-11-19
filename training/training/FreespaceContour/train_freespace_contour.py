#! /usr/bin/env python3

import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import sys
_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [p for p in (_MODELS, os.path.dirname(_MODELS)) if p not in sys.path]
from torch.utils.data import ConcatDataset, DataLoader
from data_utils.load_data_freespace_seg import (LoadDataFreespaceSeg, DATASETS,
                                                FreespaceSegTorchDataset, worker_init_fn)
from training.resume_util import save_last, load_resume
from training.FreespaceContour.freespace_contour_trainer import FreespaceContourTrainer


def main():

    parser = ArgumentParser()
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path",
                        default=os.path.join(_MODELS, 'checkpoints', 'FreespaceContour'),
                        help="root path where pytorch checkpoint file should be saved")
    parser.add_argument("-m", "--pretrained_checkpoint_path", dest="pretrained_checkpoint_path",
                        default='', help="path to SceneSeg weights file for pre-trained backbone")
    parser.add_argument("-c", "--checkpoint_path", dest="checkpoint_path",
                        help="path to FreespaceContour weights file for training from saved checkpoint")
    parser.add_argument("-r", "--root", dest="root",
                        required=True, help="root path to folder where training data is stored")
    parser.add_argument("-t", "--test_images_save_root_path", dest="test_images_save_root_path",
                        help="root path where test images are stored")
    parser.add_argument('-l', "--load_from_save", action='store_true',
                        help="flag for whether model is being loaded from a FreespaceContour checkpoint file")
    parser.add_argument("-e", "--num_epochs", dest="num_epochs", default=100, type=int,
                        help="number of training epochs")
    parser.add_argument("-a", "--start_epoch", dest="start_epoch", default=0, type=int,
                        help="starting epoch for training")
    parser.add_argument("--learning_rate", dest="learning_rate", default=0.0001, type=float,
                        help="learning rate for training")
    parser.add_argument("--num_workers", dest="num_workers", default=min(12, os.cpu_count() or 1),
                        type=int, help="DataLoader worker processes")
    parser.add_argument("--micro_batch_size", dest="micro_batch_size", default=8, type=int,
                        help="images per forward/backward pass; the effective batch size "
                             "from the epoch schedule is reached by gradient accumulation")
    parser.add_argument("--resume", action="store_true",
                        help="continue from last_freespacecontour.pth / last_freespacecontour.state.pt in the save "
                             "directory (weights, optimizer, epoch; run ends at the original last epoch)")
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path
    os.makedirs(model_save_root_path, exist_ok=True)

    # Load from checkpoint
    load_from_checkpoint = False
    if args.load_from_save:
        load_from_checkpoint = True

    datasets_info = {name: {"labels": os.path.join(root, name, name, "gt_masks/"),
                            "images": os.path.join(root, name, name, "images/")}
                             for name in DATASETS}
    datasets = {name: LoadDataFreespaceSeg(datasets_info[name]["labels"],
                                        datasets_info[name]["images"],
                                        name) for name in DATASETS}
    counts = {name: datasets[name].getItemCount() for name in datasets}

    total_train_samples = sum(v[0] for v in counts.values())
    total_val_samples = sum(v[1] for v in counts.values())
    print(total_train_samples, ": total training samples")
    print(total_val_samples, ": total validation samples")

    # Pre-trained model checkpoint path
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    checkpoint_path = args.checkpoint_path

    # Trainer Class
    if args.resume:
        trainer = FreespaceContourTrainer(is_pretrained=True, learning_rate=args.learning_rate)
    elif not load_from_checkpoint:
        trainer = FreespaceContourTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path,
                                   learning_rate=args.learning_rate)
    else:
        trainer = FreespaceContourTrainer(checkpoint_path=checkpoint_path,
                                   is_pretrained=True,
                                   learning_rate=args.learning_rate)

    trainer.zero_grad()

    # Total training epochs
    num_epochs = args.num_epochs
    start_epoch = args.start_epoch
    end_epoch = start_epoch + num_epochs
    if args.resume:
        start_epoch, end_epoch = load_resume(trainer, model_save_root_path, 'freespacecontour')

    # Datasets -> torch DataLoaders (decode/augment/contour labels in worker processes)
    train_set = ConcatDataset([FreespaceSegTorchDataset(datasets[n], True, contour=True) for n in DATASETS])
    val_set = ConcatDataset([FreespaceSegTorchDataset(datasets[n], False, contour=True) for n in DATASETS])
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         worker_init_fn=worker_init_fn,
                         persistent_workers=args.num_workers > 0,
                         prefetch_factor=4 if args.num_workers > 0 else None)
    val_workers = min(4, args.num_workers)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False,
                            **dict(loader_kwargs, num_workers=val_workers,
                                   persistent_workers=val_workers > 0,
                                   prefetch_factor=4 if val_workers > 0 else None))

    def crossed(prev, cur, every):
        return cur // every > prev // every

    # Epochs
    for epoch in range(start_epoch, end_epoch):

        # Effective batch size (gradient accumulation over micro-batches)
        batch_size = batch_size_for_epoch_100(epoch)
        micro_batch = min(batch_size, args.micro_batch_size)
        train_loader = DataLoader(train_set, batch_size=micro_batch, shuffle=True,
                                  drop_last=False, **loader_kwargs)
        losses = []
        seen = 0
        since_step = 0

        pbar = tqdm(total=total_train_samples,
                    desc=f"Training Epoch {epoch + 1} / {end_epoch}")
        for images, contours, _ in train_loader:
            n = images.shape[0]
            prev = seen
            seen += n
            since_step += n
            log_count = seen + total_train_samples*epoch

            # Forward + gradient accumulation (loss stays on GPU, no sync)
            losses.append(trainer.train_batch_contour(images, contours,
                                                      loss_scale=n / batch_size))

            # Simulating batch size through gradient accumulation
            if since_step >= batch_size or seen == total_train_samples:
                trainer.run_optimizer()
                since_step = 0

            pbar.update(n)

            # Logging loss to Tensor Board every 250 samples
            if crossed(prev, seen, 250) or seen == total_train_samples:
                mean_loss = torch.stack(losses).mean().item()
                trainer.log_loss(log_count, loss=mean_loss, contour_loss=mean_loss)
                losses.clear()

            # Logging Image to Tensor Board every 1000 samples
            if crossed(prev, seen, 1000) or seen == total_train_samples:
                trainer.save_visualization_batch(log_count)

            # Save model and run validation on entire validation
            # dataset after 8000 samples / end of epoch
            if crossed(prev, seen, 8000) or seen == total_train_samples:

                # Save Model
                model_save_path = os.path.join(model_save_root_path, 'iter_' + \
                    str(seen - 1 + total_train_samples*epoch) \
                    + '_epoch_' + str(epoch) + '_step_' + \
                    str(seen - 1) + '.pth')

                trainer.save_model(model_save_path)
                # Always keep the newest checkpoint + training state (for --resume)
                save_last(trainer, model_save_root_path, 'freespacecontour', epoch, end_epoch,
                          epoch_finished=(seen == total_train_samples))

                # Validate
                print('Validating')
                trainer.set_eval_mode()

                sum_accuracy = sum_mae = 0.0
                for images_val, contours_val, _ in tqdm(val_loader, desc="Validating"):
                    accuracy, mae = trainer.validate_batch_contour(images_val, contours_val)
                    sum_accuracy += accuracy.sum().item()
                    sum_mae += mae.sum().item()

                trainer.log_contour_metrics(sum_accuracy/total_val_samples,
                                            sum_mae/total_val_samples, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
        pbar.close()

    trainer.cleanup()

def batch_size_for_epoch_100(epoch: int) -> int:
    if epoch < 10:   return 16
    if epoch < 30:   return 8
    if epoch < 60:   return 4
    if epoch < 80:   return 2
    return 1

if __name__ == '__main__':
    main()
