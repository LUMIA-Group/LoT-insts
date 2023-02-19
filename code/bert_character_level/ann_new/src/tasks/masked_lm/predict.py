import numpy as np
import torch

import cs
from . import config
from .utils import get_training_set


def masked_lm_predict(
        model,
        char_ids,
        mask,
        char_position_ids,
        word_position_ids
):
    model.eval()

    char_ids = char_ids.to(cs.device)
    mask = mask.to(cs.device)
    char_position_ids = char_position_ids.to(cs.device)
    word_position_ids = word_position_ids.to(cs.device)

    with torch.no_grad():
        score = model(
            char_ids,
            attention_mask=mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
        )

    return torch.argmax(score, dim=-1)


def masked_lm_validation(model):
    model.eval()

    val_set = get_training_set()
    size = len(val_set)

    samples = np.random.randint(0, size, (config.n_batches_in_validating, config.batch_size))
    tp = 0
    total = 0
    for i in range(config.n_batches_in_validating):
        inputs = [val_set[int(idx)] for idx in samples[i]]
        char_ids, mask, char_position_ids, word_position_ids, labels = val_set.pack_mini_batch(inputs)
        pred = masked_lm_predict(model, char_ids, mask, char_position_ids, word_position_ids)
        labels = labels.to(cs.device)
        k = (labels != -100)
        tp += torch.eq(labels[k], pred[k]).sum()
        total += k.sum()

    return tp / total
