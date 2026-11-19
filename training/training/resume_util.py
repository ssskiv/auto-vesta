"""Shared "last checkpoint + resume" helpers for the train_*.py scripts.

Two files are kept in the save directory, both replaced atomically:

  last_<name>.pth        plain model state_dict (what the inference classes load)
  last_<name>.state.pt   training state: optimizer, epoch bookkeeping

Checkpoints are written at the scripts' existing save points (every 8000
samples and at the end of each epoch). On resume, an epoch that was interrupted
mid-way is restarted from its beginning; an epoch that finished is skipped.
"""

import os
import tempfile

import torch


def _atomic_save(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix='.tmp')
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def last_paths(save_root, name):
    return (os.path.join(save_root, f'last_{name}.pth'),
            os.path.join(save_root, f'last_{name}.state.pt'))


def save_last(trainer, save_root, name, epoch, end_epoch, epoch_finished):
    """Write last_<name>.pth and last_<name>.state.pt.

    epoch          epoch that is being trained / just finished
    end_epoch      exclusive end of the whole run (start_epoch + num_epochs)
    epoch_finished True if this save is at the end of `epoch`
    """
    weights_path, state_path = last_paths(save_root, name)
    _atomic_save(trainer.model.state_dict(), weights_path)
    _atomic_save({
        'optimizer': trainer.optimizer.state_dict(),
        'epoch': epoch,
        'epoch_finished': bool(epoch_finished),
        'next_epoch': epoch + 1 if epoch_finished else epoch,
        'end_epoch': end_epoch,
    }, state_path)


def load_resume(trainer, save_root, name):
    """Restore weights and optimizer from last_<name>.*; return (next_epoch, end_epoch).

    Raises FileNotFoundError if there is nothing to resume from.
    """
    weights_path, state_path = last_paths(save_root, name)
    if not (os.path.isfile(weights_path) and os.path.isfile(state_path)):
        raise FileNotFoundError(
            f'--resume: {weights_path} and/or {state_path} not found. '
            'Run once without --resume (they are written at the first checkpoint).')
    device = trainer.device
    trainer.model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    state = torch.load(state_path, weights_only=True, map_location=device)
    trainer.optimizer.load_state_dict(state['optimizer'])
    print(f"Resumed from {weights_path}: next epoch {state['next_epoch']}, "
          f"run ends at epoch {state['end_epoch']}")
    return state['next_epoch'], state['end_epoch']
