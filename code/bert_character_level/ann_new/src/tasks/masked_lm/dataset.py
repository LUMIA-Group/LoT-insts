from multiprocessing.managers import BaseManager

import torch
from torch.utils.data import Dataset

from tools.tokenizer import AnnTokenizer


class RemoteManager(BaseManager):
    def get_corpus(self) -> list:
        pass


RemoteManager.register('get_corpus')


class MaskedLMDataset(Dataset):

    def __init__(
            self,
            tokenizer,
            *,
            manager_host,
            manager_port,
            manager_password,
            mask_prob=0.15,
            mask_mark_rate=0.8,
            random_mark_rate=0.1,
    ):
        super().__init__()

        self.tokenizer: AnnTokenizer = tokenizer

        assert mask_mark_rate + random_mark_rate <= 1
        self.mask_prob = mask_prob
        self.mask_mark_prob = mask_prob * mask_mark_rate
        self.random_mark_prob = mask_prob * random_mark_rate

        self.manager_host = manager_host
        self.manager_port = manager_port
        self.manager_password = manager_password
        self.manager = None
        self.corpus = None

    def connect(self):
        if self.manager is None:
            self.manager = RemoteManager(address=(self.manager_host, self.manager_port), authkey=self.manager_password)
            self.manager.connect()
            self.corpus = self.manager.get_corpus()

    def __len__(self):
        self.connect()
        return len(self.corpus)

    def __getitem__(self, index):
        self.connect()
        string = self.corpus[index]
        char_ids, char_position_ids = self.tokenizer.convert_string_to_ids(string)
        char_ids = torch.LongTensor(char_ids)
        char_position_ids = torch.LongTensor(char_position_ids)

        n = char_ids.shape[0]
        chosen = torch.rand(n)
        chosen[0] = 1  # [CLS] will never be masked
        chosen[-1] = 1  # [EOS] will never be masked
        mask_mark = torch.lt(chosen, self.mask_mark_prob)
        random_mark = torch.logical_and(
            torch.ge(chosen, self.mask_mark_prob),
            torch.lt(chosen, self.random_mark_prob)
        )
        chosen = chosen < self.mask_prob

        labels = torch.full((n, ), -100, dtype=torch.long)
        labels[chosen] = char_ids[chosen]

        char_ids[mask_mark] = self.tokenizer.mask_token_id
        char_ids[random_mark] = torch.randint(13, self.tokenizer.vocab_size, (random_mark.sum(), ))

        word_position_ids = self.tokenizer.get_word_position_ids(char_ids.numpy())
        word_position_ids = torch.LongTensor(word_position_ids)

        return char_ids, char_position_ids, word_position_ids, labels

    @staticmethod
    def pack_mini_batch(data):
        bs = len(data)
        max_len = max(item[0].size(0) for item in data)

        char_ids = torch.zeros(bs, max_len, dtype=torch.long)
        mask = torch.zeros(bs, max_len, dtype=torch.long)
        char_position_ids = torch.zeros(bs, max_len, dtype=torch.long)
        word_position_ids = torch.zeros(bs, max_len, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)

        for i, (cid, cpid, wpid, label) in enumerate(data):
            seq_len = cid.size(0)
            char_ids[i, :seq_len] = cid
            mask[i, :seq_len] = 1
            char_position_ids[i, :seq_len] = cpid
            word_position_ids[i, :seq_len] = wpid
            labels[i, :seq_len] = label

        return char_ids, mask, char_position_ids, word_position_ids, labels
