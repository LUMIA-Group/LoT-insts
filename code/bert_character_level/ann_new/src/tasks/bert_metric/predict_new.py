import torch
import pickle
import numpy as np

import cs
from .utils import get_test_set
from tools.utils import get_test_set_loader
from tqdm import tqdm
from tools.utils import to_device
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .config import id_to_cls_file


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
    dataloader = get_test_set_loader(test_set, batch_size=128, num_workers=2)

    tp = 0
    total = 0
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(dataloader):
        labels = labels.to(cs.device)
        results = bert_classifier_predict(model, char_ids, mask, char_position_ids, word_position_ids)
        tp += torch.eq(results, labels).sum()
        total += labels.size(0)

    return tp / total


def report(true, pred):
    all_labels = list(set(true))
    a = accuracy_score(true, pred) * 100
    p = precision_score(true, pred, average="macro", labels=all_labels, zero_division=0) * 100
    r = recall_score(true, pred, average="macro", labels=all_labels) * 100
    f = f1_score(true, pred, average="macro", labels=all_labels, zero_division=0) * 100
    return a, p, r, f

def calc_split(low_margin, high_margin, true, pred, test_set_size):
    low = test_set_size < low_margin
    high = test_set_size > high_margin
    mid = np.logical_and(test_set_size >= low_margin, test_set_size <= high_margin)

    r1 = report(true[high], pred[high])
    r2 = report(true[mid], pred[mid])
    r3 = report(true[low], pred[low])
    
    return r1, r2, r3


def bert_classifier_test_detail(model, test_set=None):
    model = to_device(model)
    model.eval()

    if test_set is None:
        test_set = get_test_set(part='validation')
    elif test_set == 'test':
        test_set = get_test_set(part='test')
    dataloader = get_test_set_loader(test_set, batch_size=256, num_workers=2)
    
    id_to_cls = pickle.load(open(id_to_cls_file, 'rb'))
    cls_to_id = { v:k for k,v in id_to_cls.items() }
    afid2nor = pickle.load(open(cs.save_pkl_root+"afid2nor.pkl", "rb"))
    nor2len_dict = pickle.load(open(cs.save_pkl_root+'210422_nor2len_dict.pkl', 'rb'))

    true = []
    pred = []
    test_set_size = []
    
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(dataloader):
        labels = labels.to(cs.device)
        results = bert_classifier_predict(model, char_ids, mask, char_position_ids, word_position_ids)
        
        tmp_test_size = [nor2len_dict[afid2nor[cls_to_id[label_id.item()]]]  for label_id in labels]
        test_set_size = test_set_size + tmp_test_size
        
        true.append(labels.to(torch.device('cpu')))
        pred.append(results[:len(labels)].to(torch.device('cpu')))

    pred = torch.cat(pred).numpy()
    true = torch.cat(true).numpy()  
        
    test_set_size = np.array(test_set_size)
    acc, precision, recall, f1= report(true, pred)
    overall = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    high, middle, few = calc_split(5, 20, true, pred, test_set_size)
    part = {
        'high':{
            'accuracy': high[0],
            'precision': high[1],
            'recall': high[2],
            'f1': high[3],
        },
        'middle':{
            'accuracy': middle[0],
            'precision': middle[1],
            'recall': middle[2],
            'f1': middle[3],
        }, 
        'few':{
            'accuracy': few[0],
            'precision': few[1],
            'recall': few[2],
            'f1': few[3],
        },     
    }
    return (overall, part)


def bert_classifier_test_detail_result(model, test_set=None):
    model = to_device(model)
    model.eval()

    if test_set is None:
        test_set = get_test_set(part='validation')
    elif test_set == 'test':
        test_set = get_test_set(part='test')
    dataloader = get_test_set_loader(test_set, batch_size=256, num_workers=2)
    
    id_to_cls = pickle.load(open(id_to_cls_file, 'rb'))
    cls_to_id = { v:k for k,v in id_to_cls.items() }
    afid2nor = pickle.load(open(cs.save_pkl_root+"afid2nor.pkl", "rb"))
    nor2len_dict = pickle.load(open(cs.save_pkl_root+'210422_nor2len_dict.pkl', 'rb'))

    tp = 0
    total = 0
    true = []
    pred = []
    test_set_size = []
    
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(dataloader):
        labels = labels.to(cs.device)
        results = bert_classifier_predict(model, char_ids, mask, char_position_ids, word_position_ids)
        tp += torch.eq(results, labels).sum()
        total += labels.size(0)
        
        tmp_test_size = [nor2len_dict[afid2nor[cls_to_id[label_id.item()]]]  for label_id in labels]
        test_set_size = test_set_size + tmp_test_size
        
        true.append(labels.to(torch.device('cpu')))
        pred.append(results[:len(labels)].to(torch.device('cpu')))

    pred = torch.cat(pred).numpy()
    true = torch.cat(true).numpy()          
        
    test_set_size = np.array(test_set_size)
 
    
    acc, precision, recall, f1= report(true, pred)
    overall = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    high, middle, few = calc_split(5, 20, true, pred, test_set_size)
    part = {
        'high':{
            'accuracy': high[0],
            'precision': high[1],
            'recall': high[2],
            'f1': high[3],
        },
        'middle':{
            'accuracy': middle[0],
            'precision': middle[1],
            'recall': middle[2],
            'f1': middle[3],
        }, 
        'few':{
            'accuracy': few[0],
            'precision': few[1],
            'recall': few[2],
            'f1': few[3],
        },     
    }
    return [(overall, part), (true, pred)]


def bert_classifier_test_overall(model, test_set=None):
    model = to_device(model)
    model.eval()

    if test_set is None:
        test_set = get_test_set(part='validation')
    elif test_set == 'test':
        test_set = get_test_set(part='test')
    dataloader = get_test_set_loader(test_set, batch_size=256, num_workers=2)

    true = []
    pred = []
    
    for char_ids, mask, char_position_ids, word_position_ids, labels in tqdm(dataloader):
        labels = labels.to(cs.device)
        results = bert_classifier_predict(model, char_ids, mask, char_position_ids, word_position_ids)
        
        pred.append(results.cpu())
        true.append(labels)
        
    pred = torch.cat(pred).numpy()
    true = torch.cat(true).numpy()
    
    print(pred.shape, true.shape)
        
    acc, precision, recall, f1= report(true, pred)
    overall = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return overall