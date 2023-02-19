# The package name "cs" means "common settings".


import os
import socket

import torch


# ==================================================================================================
# Resource directory

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

VOCAB_FILE = os.path.join(DATASET_DIR, 'vocab.txt')

SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_model')
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ==================================================================================================
# Setting of distributed training

# There are three choices, you can use any mode depend on your environment
# 'cpu': use CPU to train
# 'single': use single GPU
# 'distributed': use distributed GPUs
device_mode = 'single'
rank = 0

if device_mode == 'cpu':
    use_cuda = False
    use_dist = False
    num_devices = 1
    device = torch.device('cpu')
elif device_mode == 'single':
    use_cuda = True
    use_dist = False
    num_devices = 1
    device_no = 2  
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_no}'
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(device)
elif device_mode == 'distributed':
    use_cuda = True
    use_dist = True

    tcp_url = 'tcp://10.10.10.12:23457'  # It must be the first GPU's IP address, arbitrary port value
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'  

    workers = [
        {'hostname': 'main-server-12', 'device_id': 0},
        {'hostname': 'main-server-12', 'device_id': 1},
        {'hostname': 'main-server-12', 'device_id': 2},
        {'hostname': 'main-server-12', 'device_id': 3},
    ]
    num_devices = len(workers)
    assert num_devices > 1
    rank_list = []
    did = []
    hostname = socket.gethostname()
    for i, worker in enumerate(workers):
        if worker['hostname'] == hostname:
            rank_list.append(i)
            did.append(str(worker['device_id']))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(did)
    num_local_devices = len(rank_list)

    # Will be reset in the main entry.
    device = None
    local_rank = None
else:
    raise Exception('Wrong device mode.')


# ==================================================================================================
# pkl path

save_pkl_root = ''
