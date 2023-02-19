from functools import wraps
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import os

import cs

from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


# __all__ = [
#     'StopTraining',
#     'only_main_process',
#     'log_config',
# ]


class StopTraining(Exception):
    ...


def only_main_process(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        if cs.rank == 0:
            func(*args, **kwargs)

    return decorated


@only_main_process
def log_config(logger, config_vars):
    if cs.device_mode == 'cpu':
        logger.info(f'Use CPU.')
    elif cs.device_mode == 'single':
        logger.info(f'Use single gpu. Device number is {cs.device_no}')
        logger.info(f'GPU: {cs.device_name}')
    elif cs.device_mode == 'distributed':
        logger.info(f'Use {cs.num_devices} gpus.')
        logger.info(f'Devices: \n' + "\n".join([repr(w) for w in cs.workers]))

    begin = False
    for name, val in config_vars.items():
        if name == 'task_name':
            begin = True
        if begin:
            logger.info(f'{name}: {val}')


def all_exposed_methods(obj):
    temp = []
    excludes = {
        '__class__',
        '__delattr__',
        '__dir__',
        '__format__',
        '__getattribute__',
        '__init__',
        '__init_subclass__',
        '__new__',
        '__reduce__',
        '__reduce_ex__',
        '__setattr__',
        '__sizeof__',
        '__subclasshook__',
    }
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func) and name not in excludes:
            temp.append(name)
    return temp


def to_device(model, *, device=None, use_dist=None):
    if device is not None:
        return model.to(device)

    if use_dist is None:
        use_dist = cs.use_dist
    if use_dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cs.device)
        model = DistributedDataParallel(model, device_ids=[cs.local_rank], find_unused_parameters=True)
    else:
        model = model.to(cs.device)

    return model


def save_model(model, task_dir, file_name):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    os.makedirs(os.path.join(cs.SAVED_MODEL_DIR, task_dir), exist_ok=True)
    path = os.path.join(cs.SAVED_MODEL_DIR, task_dir, file_name)
    torch.save(model.state_dict(), path)


def get_training_set_loader(training_set, batch_size, *, use_dist=None, num_workers=0):
    if use_dist is None:
        use_dist = cs.use_dist
    dataset_iterable = isinstance(training_set, IterableDataset)
    sampler = DistributedSampler(training_set) if use_dist and not dataset_iterable else None
    loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=(not use_dist and not dataset_iterable),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=training_set.pack_mini_batch,
        pin_memory=cs.use_cuda,
    )
    return loader, sampler


def get_test_set_loader(test_set, batch_size, *, num_workers=0):
    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=test_set.pack_mini_batch,
        pin_memory=cs.use_cuda,
    )
    return loader
