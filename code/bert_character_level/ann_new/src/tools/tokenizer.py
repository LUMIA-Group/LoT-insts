import re
from typing import Tuple, Optional

import numpy as np
from fold_to_ascii import fold


class AnnTokenizer:

    max_len = 256
    max_word_num = 64

    def __init__(
            self,
            vocab_file,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            eos_token="[EOS]",
    ):
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.eos_token = eos_token

        self.vocab = {}
        self.token_list = []
        self.vocab_size = 0
        with open(vocab_file, 'r') as f:
            for line in f:
                token = line.rstrip('\n')
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1
                self.token_list.append(token)

        self.unk_token_id = self.vocab[self.unk_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.eos_token_id = self.vocab[self.eos_token]

    @staticmethod
    def preprocess(s: str) -> Optional[str]:
        if not s:
            return None
        s_folder = fold(s)
        if len(s_folder) / len(s) < 0.8:
            return None
        s = s_folder.lower()
        s = re.sub(r'(?:#(?:n|tab|r)#)|(?:</?.+?>)|[\x00-\x1f\x7f%^<!$`=?|>\[\]+~/]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'^\s|\s$', '', s)
        letter_cnt = len(re.findall(r'[a-z]', s))
        if letter_cnt < 2:
            return None

        if len(s) + 2 > AnnTokenizer.max_len or len(s.split()) + 1 > AnnTokenizer.max_word_num:
            return None

        return s

    def convert_string_to_ids(self, s: str) -> Tuple[np.ndarray, np.ndarray]:
        char_ids = np.array([self.cls_token_id, *[self.vocab.get(c, self.unk_token_id) for c in s], self.eos_token_id])
        char_position_ids = np.arange(len(s) + 2)

        return char_ids, char_position_ids

    def get_word_position_ids(self, char_ids: np.ndarray):
        idx = np.where(char_ids == self.vocab[' '])[0]
        word_position_ids = np.zeros_like(char_ids)
        prev_idx = 0
        i = 0
        for i, p in enumerate(idx, 1):
            word_position_ids[prev_idx+1:p+1] = i
            prev_idx = p
        word_position_ids[prev_idx+1:] = i + 1
        return word_position_ids
