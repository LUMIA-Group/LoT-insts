import torch
import pickle
import numpy as np

import cs
from . import config
from .utils import get_test_set
from tools.utils import get_test_set_loader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def bert_metric_inference(
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
        output = model(
            char_ids,
            attention_mask=mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
        )
        pooled = output[1]
    return pooled


def bert_metric_validation(model):
    model.eval()

    anchor_set = get_test_set(part='anchor')
    anchor_dl = get_test_set_loader(anchor_set, batch_size=16, num_workers=2)
    anchor_labels = []
    anchor_results = []
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(anchor_dl):
        anchor_results.append(bert_metric_inference(model, char_ids, mask, char_position_ids, word_position_ids))
        anchor_labels.append(labels)
    anchor_results = torch.cat(anchor_results, 0)
    anchor_labels = torch.cat(anchor_labels, 0)

    test_set = get_test_set(part='validation')
    test_dl = get_test_set_loader(test_set, batch_size=16, num_workers=2)
    test_labels = []
    indices = []
    true = []
    pred = []
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(test_dl):
        test_result = bert_metric_inference(model, char_ids, mask, char_position_ids, word_position_ids)
        test_labels.append(labels)
        true.append(labels)
        dist = torch.cdist(test_result, anchor_results)
        idx = torch.argmin(dist, dim=-1)
        indices.append(idx)
        pred.append(anchor_labels[idx])
    indices = torch.cat(indices, 0)
    test_labels = torch.cat(test_labels, 0)
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


def bert_metric_test(model):
    model.eval()

    anchor_set = get_test_set(part='anchor')
    anchor_dl = get_test_set_loader(anchor_set, batch_size=16, num_workers=2)
    anchor_labels = []
    anchor_results = []
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(anchor_dl):
        anchor_results.append(bert_metric_inference(model, char_ids, mask, char_position_ids, word_position_ids))
        anchor_labels.append(labels)
    anchor_results = torch.cat(anchor_results, 0)
    anchor_labels = torch.cat(anchor_labels, 0)

    test_set = get_test_set(part='test')
    test_dl = get_test_set_loader(test_set, batch_size=16, num_workers=2)
    test_labels = []
    indices = []
    true = []
    pred = []
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(test_dl):
        test_result = bert_metric_inference(model, char_ids, mask, char_position_ids, word_position_ids)
        test_labels.append(labels)
        true.append(labels)
        dist = torch.cdist(test_result, anchor_results)
        idx = torch.argmin(dist, dim=-1)
        indices.append(idx)
        pred.append(anchor_labels[idx])
    indices = torch.cat(indices, 0)
    test_labels = torch.cat(test_labels, 0)
    true = torch.cat(true).cpu().numpy()
    pred = torch.cat(pred).cpu().numpy()

#     tp = torch.sum(torch.eq(anchor_labels[indices], test_labels)).item()
    
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

# def ann_end_to_end(model, name_list, bs=128, preprocess=False):
#     tokenizer = AnnTokenizer(os.path.join(cs.DATASET_DIR, 'MAG-20-09-11', 'vocab.txt'))
#     if preprocess:
#         tmp = []
#         for name in name_list:
#             t = tokenizer.preprocess(name)
#             if t is not None:
#                 tmp.append(t)
#         name_list = tmp
#
#     vectors = []
#     ds = AnnTestSet(tokenizer, name_list)
#     dl = get_test_set_loader(ds, batch_size=bs, num_workers=2)
#     for char_ids, mask, char_position_ids, word_position_ids in tqdm(dl):
#         result = ann_predict(model, char_ids, mask, char_position_ids, word_position_ids)
#         vectors.append(result)
#     vectors = torch.cat(vectors, 0).cpu().numpy()
#     return vectors

def bert_metric_inference_cuda(
        model,
        char_ids,
        mask,
        char_position_ids,
        word_position_ids
):
    model.eval()

    char_ids = char_ids.to(torch.device("cuda"))
    mask = mask.to(torch.device("cuda"))
    char_position_ids = char_position_ids.to(torch.device("cuda"))
    word_position_ids = word_position_ids.to(torch.device("cuda"))

    with torch.no_grad():
        output = model(
            char_ids,
            attention_mask=mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
        )
        pooled = output[1]
    return pooled


def bert_metric_embeddings(model, val_dataset):
    model.eval()

    val_dl = get_test_set_loader(val_dataset, batch_size=128, num_workers=2)
    
    results = []
    same_labels = []
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(val_dl):
        results.append(bert_metric_inference(model, char_ids, mask, char_position_ids, word_position_ids))
        same_labels.append(labels)
    results = torch.cat(results, 0)
    same_labels = torch.cat(same_labels, 0)

    return results, same_labels
