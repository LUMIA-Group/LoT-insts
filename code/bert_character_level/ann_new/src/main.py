import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cs
from tasks.masked_lm.train import train
# from tasks.bert_classifier.train import train
# from tasks.bert_metric.train import train

def main_with_dist(idx, task):
    cs.rank = cs.rank_list[idx]
    cs.local_rank = idx
    dist.init_process_group(
        backend='nccl',
        rank=cs.rank,
        world_size=cs.num_devices,
        init_method=cs.tcp_url,
    )
    torch.cuda.set_device(cs.local_rank)
    cs.device = torch.device("cuda", cs.local_rank)
    task()


if __name__ == '__main__':
    print("Hello world.")
    if not cs.use_dist:
        train()
    else:
        mp.spawn(main_with_dist, args=(train,), nprocs=cs.num_local_devices, join=True)
