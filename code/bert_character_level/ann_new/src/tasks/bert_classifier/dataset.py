from multiprocessing.managers import BaseManager

import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset
from .manager import Mapping

from tools.tokenizer import AnnTokenizer


class RemoteManager(BaseManager):
    def get_dataset(self) -> list:
        pass

    def get_mapping(self) -> Mapping:
        pass


RemoteManager.register("get_dataset")
RemoteManager.register("get_mapping")


def _input_helper(name, tokenizer):
    char_ids, char_position_ids = tokenizer.convert_string_to_ids(name)
    word_position_ids = tokenizer.get_word_position_ids(char_ids)
    char_ids = torch.from_numpy(char_ids)
    char_position_ids = torch.from_numpy(char_position_ids)
    word_position_ids = torch.from_numpy(word_position_ids)
    return char_ids, char_position_ids, word_position_ids


class BertClassifierDataset(IterableDataset):
    """
    Based on remote manager. Only for training.
    Test and validation set use `BertClassifierTestSet` based on local file which is defined below.
    """
    def __init__(
            self,
            tokenizer,
            *,
            manager_host,
            manager_port,
            manager_password,
            resample_exp,
    ):
        super().__init__()

        self.tokenizer: AnnTokenizer = tokenizer

        self.manager = None
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.manager_password = manager_password
        self.resample_exp = resample_exp

        self.dataset = None
        self.mapping = None
        self.classes = None
        self.cls_sizes = None
        self.prob = None

    def connect(self):
        if self.manager is None:
            self.manager = RemoteManager(address=(self.manager_host, self.manager_port), authkey=self.manager_password)
            self.manager.connect()
            self.dataset = self.manager.get_dataset()
            self.mapping = self.manager.get_mapping()

            self.classes = sorted(self.mapping.keys())
            self.cls_sizes = np.array([self.mapping.sub_len(k) for k in self.classes])
            self.prob = self.cls_sizes ** self.resample_exp
            self.prob = self.prob / np.sum(self.prob)

    def __getitem__(self, index):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.connect()

        chosen_cls = np.random.choice(self.classes, p=self.prob)
        chosen_idx = self.mapping.sub_choice(chosen_cls)
        name, label = self.dataset[chosen_idx]

        return *_input_helper(name, self.tokenizer), label

    @staticmethod
    def pack_mini_batch(data):
        bs = len(data)
        max_len = max(item[0].size(0) for item in data)

        char_ids = torch.zeros(bs, max_len, dtype=torch.long)
        mask = torch.zeros(bs, max_len, dtype=torch.long)
        char_position_ids = torch.zeros(bs, max_len, dtype=torch.long)
        word_position_ids = torch.zeros(bs, max_len, dtype=torch.long)
        labels = torch.zeros(bs, dtype=torch.long)

        for i, (cid, cpid, wpid, label) in enumerate(data):
            seq_len = cid.size(0)
            char_ids[i, :seq_len] = cid
            mask[i, :seq_len] = 1
            char_position_ids[i, :seq_len] = cpid
            word_position_ids[i, :seq_len] = wpid
            labels[i] = label

        return char_ids, mask, char_position_ids, word_position_ids, labels


class BertClassifierTestSet(Dataset):

    def __init__(
            self,
            tokenizer,
            dataset,
    ):
        super().__init__()

        self.tokenizer: AnnTokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        name, label = self.dataset[index]
        return *_input_helper(name, self.tokenizer), label

    @staticmethod
    def pack_mini_batch(data):
        return BertClassifierDataset.pack_mini_batch(data)
