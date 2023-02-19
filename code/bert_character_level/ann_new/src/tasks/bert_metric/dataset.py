from multiprocessing.managers import BaseManager

import torch
from torch.utils.data import Dataset, IterableDataset

from tools.tokenizer import AnnTokenizer
from .manager import Mapping
import numpy as np


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


class BertMetricDataset(IterableDataset):

    def __init__(
            self,
            tokenizer,
            *,
            manager_host,
            manager_port,
            manager_password,
            batch_size,
            samples_per_class,
            resample_exp,
    ):
        super().__init__()

        self.tokenizer: AnnTokenizer = tokenizer

        self.manager_host = manager_host
        self.manager_port = manager_port
        self.manager_password = manager_password
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.resample_exp = resample_exp

        self.manager = None
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

    def __next__(self):
        self.connect()

        cls_seq = np.random.choice(self.classes, size=self.batch_size, p=self.prob, replace=False)
        names = []
        labels = []
        for aff_id in cls_seq:
            n = min(self.samples_per_class, self.mapping.sub_len(aff_id), self.batch_size - len(names))
            samples = self.mapping.sub_sample(aff_id, n)
            samples_name, samples_label = zip(*(self.dataset[idx] for idx in samples))
            names.extend(samples_name)
            labels.extend(samples_label)
            if len(names) == self.batch_size:
                break
        labels = torch.LongTensor(labels)
        assert len(names) == len(labels) == self.batch_size

        max_len = max(len(name) for name in names) + 2  # will add [CLS], [EOS]
        char_ids = torch.zeros(self.batch_size, max_len, dtype=torch.long)
        mask = torch.zeros(self.batch_size, max_len, dtype=torch.long)
        char_position_ids = torch.zeros(self.batch_size, max_len, dtype=torch.long)
        word_position_ids = torch.zeros(self.batch_size, max_len, dtype=torch.long)

        for i, name in enumerate(names):
            cid, cpid, wpid = _input_helper(name, self.tokenizer)
            seq_len = cid.shape[0]
            char_ids[i, :seq_len] = cid
            mask[i, :seq_len] = 1
            char_position_ids[i, :seq_len] = cpid
            word_position_ids[i, :seq_len] = wpid

        return char_ids, mask, char_position_ids, word_position_ids, labels

    def __iter__(self):
        return self

    def __getitem__(self, index):
        pass

    @staticmethod
    def pack_mini_batch(items):
        return items


class BertMetricTestSet(Dataset):

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

    def __getitem__(self, item):
        name, label = self.dataset[item]
        return *_input_helper(name, self.tokenizer), label

    @staticmethod
    def pack_mini_batch(item):
        bs = len(item)
        max_len = max(item[0].size(0) for item in item)

        char_ids = torch.zeros(bs, max_len, dtype=torch.long)
        mask = torch.zeros(bs, max_len, dtype=torch.long)
        char_position_ids = torch.zeros(bs, max_len, dtype=torch.long)
        word_position_ids = torch.zeros(bs, max_len, dtype=torch.long)
        labels = torch.zeros(bs, dtype=torch.long)

        for i, (cid, cpid, wpid, label) in enumerate(item):
            seq_len = cid.size(0)
            char_ids[i, :seq_len] = cid
            mask[i, :seq_len] = 1
            char_position_ids[i, :seq_len] = cpid
            word_position_ids[i, :seq_len] = wpid
            labels[i] = label

        return char_ids, mask, char_position_ids, word_position_ids, labels
