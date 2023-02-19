import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import cs
from .utils import get_test_set
from tools.utils import get_test_set_loader
from tqdm import tqdm
import pickle
from . import config
import numpy as np


def bert_classifier_inference(
        model,
        char_ids,
        mask,
        char_position_ids,
        word_position_ids,
):
    model.eval()

    char_ids = char_ids.to(cs.device)
    mask = mask.to(cs.device)
    char_position_ids = char_position_ids.to(cs.device)
    word_position_ids = word_position_ids.to(cs.device)

    with torch.no_grad():
        output = model(
            char_ids,
            attention_mask=mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
        )
        logits = output[0]
    return logits


def bert_classifier_predict(
        model,
        char_ids,
        mask,
        char_position_ids,
        word_position_ids,
        k=1,
):
    logits = bert_classifier_inference(model, char_ids, mask, char_position_ids, word_position_ids)
    if k == 1:
        return torch.argmax(logits, dim=-1)
    else:
        _, idx = torch.sort(logits, dim=-1, descending=True)
        return idx[:, :k]


def bert_classifier_validation(model, test_set=None):
    model.eval()

    if test_set is None:
        test_set = get_test_set(part='validation')
    else:
        test_set = get_test_set()
    dataloader = get_test_set_loader(test_set, batch_size=128, num_workers=2)

    true = []
    pred = []
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(dataloader):
        true.append(labels)
        results = bert_classifier_predict(model, char_ids, mask, char_position_ids, word_position_ids)
        pred.append(results)
    true = torch.cat(true).cpu().numpy()
    pred = torch.cat(pred).cpu().numpy()

    with open(config.nor2len_file, 'rb') as f:
        nor2len = pickle.load(f)
    with open(config.aff_id_to_nor_file, 'rb') as f:
        id2nor = pickle.load(f)
    with open(config.id_to_cls_file, 'rb') as f:
        id_to_cls = pickle.load(f)
    cls_size = {}
    for aff_id, label in id_to_cls.items():
        cls_size[label] = nor2len[id2nor[aff_id]]

    r = _calc_split(5, 20, true, pred, cls_size)
    report_text = f'Overall A({r[0]:.1f}%) Overall P({r[1]:.1f}%) Overall R({r[2]:.1f}%) Overall F({r[3]:.1f}%) ' \
                  f'Many A({r[4]:.1f}%) Many P({r[5]:.1f}%) Many R({r[6]:.1f}%) Many F({r[7]:.1f}%) ' \
                  f'Medium A({r[8]:.1f}%) Medium P({r[9]:.1f}%) Medium R({r[10]:.1f}%) Medium F({r[11]:.1f}%) ' \
                  f'Few A({r[12]:.1f}%) Few P({r[13]:.1f}%) Few R({r[14]:.1f}%) Few F({r[15]:.1f}%)'
    return (report_text, r)


def _report(true, pred):
    all_labels = list(set(true))
    a = accuracy_score(true, pred) * 100
    p = precision_score(true, pred, average="macro", labels=all_labels, zero_division=0) * 100
    r = recall_score(true, pred, average="macro", labels=all_labels) * 100
    f = f1_score(true, pred, average="macro", labels=all_labels, zero_division=0) * 100
    return a, p, r, f


def _calc_split(low_margin, high_margin, true, pred, cls_size):
    test_set_size = np.array([cls_size[label] for label in true])
    low = test_set_size < low_margin
    high = test_set_size > high_margin
    mid = np.logical_and(test_set_size >= low_margin, test_set_size <= high_margin)

    r0 = _report(true, pred)
    r1 = _report(true[high], pred[high])
    r2 = _report(true[mid], pred[mid])
    r3 = _report(true[low], pred[low])
    return *r0, *r1, *r2, *r3
