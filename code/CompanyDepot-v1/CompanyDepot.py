# -*- coding : utf-8 -*-

import features as fea
from tqdm import tqdm
from config import *
import Levenshtein
import heapq
from multiprocessing.pool import Pool
import pymysql
import elasticsearch
import time
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import re


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="preprocess", help="choose mode in preprocess, after_ranklib and get_result")
    return parser.parse_args()


# find in ES, return top K 
# return 3 lists: norm name, origin name, score
def get_Top_K_doc(es, index, words, K=10):
    # search_words = words.split()
    body = {
        "size": K,     
        "sort": [    
            {
              "_score": {
                "order": "desc"
              }
            }
        ],
        "query": {
            "match": {
                "original_name": words
            }
        },

        "_source": {
            "excludes": []
        },
        "stored_fields": [
            "*"
        ],
        "script_fields": {},
        "docvalue_fields": []
    }
    ret = es.search(index=index, body=body)

    if len(ret['hits']['hits']) == 0:
        return []
    else:
        search_result_list_score = []
        search_result_list_norm = []
        search_result_list_ori = []
        for item in ret['hits']['hits']:
            search_result_list_norm.append(item['_source']['normalized_name'])
            search_result_list_ori.append(item['_source']['original_name'])
            search_result_list_score.append(item['_score'])
    return search_result_list_norm[:K], search_result_list_ori[:K], search_result_list_score[:K]



# return: [query save, norm form, label]
def get_train_data(train_data_file):
    train_data = []
    file = train_data_file
    with open(file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            words = line.strip().split('\t\t')
            if len(words) == 2:
                train_data.append([words[0], words[1]])
    return train_data


# input: [train_data[k], k], train_data[k] = [query_name, norm_name, label]
# output: 14 * [norm, sore, ori], include features
def get_result(input_data):
    train_data = input_data[0]
    query = train_data[0]
    nor_result = train_data[1]

    index_es = 'cd_doc_0422'
    features_list = []

    try:
        infor = get_Top_K_doc(es, index_es, query, 100)
    except:
        return []
    if infor == []:
        return []

    nor = infor[0]
    ori = infor[1]
    score = infor[2]
    candicate = []

    if len(ori) < 10:
        for index in range(len(ori)):
            candicate.append([nor[index], score[index], ori[index]])
    else:
        for index in range(10):
            candicate.append([nor[index], score[index], ori[index]])
    distances = [Levenshtein.distance(query, index) for index in ori]

    if len(distances) == 1:
        candicate.append([nor[0], score[0], ori[0]])
    else:
        max_num = list(map(distances.index, heapq.nlargest(2, distances)))
        candicate.append([nor[max_num[0]], score[max_num[0]], ori[max_num[0]]])
        candicate.append([nor[max_num[1]], score[max_num[1]], ori[max_num[1]]])

    distances = [Levenshtein.distance(fea.calibration(query).replace(' ', ''), fea.calibration(index).replace(' ', ''))
                 for index in ori]
    if len(distances) == 1:
        candicate.append([nor[0], score[0], ori[0]])
    else:
        max_num = list(map(distances.index, heapq.nlargest(2, distances)))
        candicate.append([nor[max_num[0]], score[max_num[0]], ori[max_num[0]]])
        candicate.append([nor[max_num[1]], score[max_num[1]], ori[max_num[1]]])

    for index in range(len(candicate)):
        temp = {}
        if candicate[index][0] == nor_result:
            temp['label'] = 5
        else:
            temp['label'] = 0
        temp['qid'] = input_data[1]
        feature = []
        feature.extend(fea.get_length(query))  
        feature.extend(fea.extract_country(query))    
        feature.extend([candicate[index][1]])       
        feature.extend(fea.query_nor_equal(query, candicate[index][0]))      
        feature.extend(fea.query_ori_equal(query, candicate[index][2]))      
        feature.extend(fea.prefix_suffix(query, candicate[index][0]))        
        feature.extend(fea.common_words(query, candicate[index][0]))         
        feature.extend(fea.get_distance(query, candicate[index][0], candicate[index][2]))    
        feature.extend(fea.get_Jaccard(query, candicate[index][0], candicate[index][2]))     
        feature.extend(fea.country_same(query, candicate[index][0]))   
        if candicate[index][0] in nor2popular.keys():                  
            feature.extend([nor2popular[candicate[index][0]]])
        else:
            feature.extend([0])
        feature.extend([len(candicate[index][0])])                 
        feature.extend(fea.extract_country(candicate[index][0]))   
        temp['features'] = feature
        features_list.append(temp)
    return features_list, candicate


# save feature_list: label, pid , feature(37 in all)
def write_data(data, output_file):
    row_num = 0
    with open(output_file, 'w') as f:
        for item in data:
            for d in item:
                f.write(str(d['label']))
                f.write(' ' + 'qid:')
                f.write(str(d['qid']))
                f.write(' ')
                for i in range(len(d['features'])):
                    f.write(str(i + 1))
                    f.write(':')
                    f.write(str(d['features'][i]))
                    f.write(' ')
                f.write('#docid=' + str(d['qid']) + '-' + str(row_num % 14))
                row_num = row_num + 1
                f.write('\n')


def load_norm_labels(test_file):
    origin_names = []
    real_norm_labels = []
    real_norm_ids = []
    with open(test_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split('\t\t')
            if len(words) == 2:
                real_norm_labels.append(words[1])
                origin_names.append(words[0])
                real_norm_ids.append(nor2id[str(words[1])])
    return origin_names, real_norm_ids


def get_top_1_label_from_indri(indri_file, test_features_file, candicates):
    ids = []
    i = 0
    if_first = 1
    pre_label_lines = []
    pre_label = []

    with open(indri_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            if tokens[0] == str(i) and if_first:
                ids.append(tokens[2])
                if_first = 0
            elif tokens[0] != str(i):
                i += 1
                ids.append(tokens[2])
                if re.findall(find_if_exist, ids[-2])[0] == re.findall(find_if_exist, ids[-1])[0]:
                    ids[-1] = ids[-2]
                    ids[-2] = -1
                    
    with open(test_features_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        print(len(lines))
        k = 0
        for l, line in enumerate(lines):
            tokens = line.split()
            if k < len(ids) and ids[k] == -1:
                pre_label_lines.append(-1)
                k = k + 1
            elif k < len(ids) and tokens[-1] == "#" + ids[k]:
                pre_label_lines.append(l)
                k = k + 1

    pre_name = []
    for r in pre_label_lines:
        if r == -1:
            pre_name.append(-1)
            pre_label.append(-1)
        else:
            pre_name.append(str(candicates[r][0]))
            pre_label.append(nor2id[str(candicates[r][0])])
    return pre_label


def get_data():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) 
    train_data = get_train_data(train_data_file)

    features_list = []
    candicates = []
    with Pool() as pool:
        features = tqdm(pool.imap(get_result, [[train_data[i], i] for i in range(len(train_data))]))
        for item in features:
            if len(item) == 2:
                features_list.append(list(item[0]))
                candicates.extend(list(item[1]))
        write_data(features_list, output_file)
        print("Save Features Done!")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return candicates


def load_data(candicates):
    pred = get_top_1_label_from_indri(indri_file, test_features_file, candicates)
    with open(pre_id_result_pkl_path, 'wb') as f:
        pickle.dump(pred, f)

    origin_names, true = load_norm_labels(test_file)
    with open(true_id_result_pkl_path, 'wb') as f:
        pickle.dump(true, f)
    with open(origin_name_pkl_path, 'wb') as f:
        pickle.dump(origin_names, f)
    print("Load Indri File Done!")



def report(true, pred):
    all_labels = list(set(true))
    print(f'Accuracy: {accuracy_score(true, pred) * 100:.3f}%')
    print(f'Macro Avg Precision: {precision_score(true, pred, average="macro", labels=all_labels, zero_division=0) * 100:.3f}%')
    print(f'Macro Avg Recall: {recall_score(true, pred, average="macro", labels=all_labels) * 100:.3f}%')
    print(f'Macro Avg F1_score: {f1_score(true, pred, average="macro", labels=all_labels, zero_division=0) * 100:.3f}%')


def get_diff(h2m, m2l, real_result, predict_result, nor2len):
    real_h = []
    real_m = []
    real_l = []
    predict_h = []
    predict_m = []
    predict_l = []
    for i in range(len(real_result)):
        norm_name = id2nor.get(real_result[i])
        if nor2len.get(norm_name, -1) < m2l:
            real_l.append(real_result[i])
            predict_l.append(predict_result[i])
        elif nor2len.get(norm_name, -1) <= h2m:
            real_m.append(real_result[i])
            predict_m.append(predict_result[i])
        else:
            real_h.append(real_result[i])
            predict_h.append(predict_result[i])
    print("len_l: ", len(real_l))
    print("len_m: ", len(real_m))
    print("len_h: ", len(real_h))
    return real_h, real_m, real_l, predict_h, predict_m, predict_l


def get_performance(true, pred):
    print("all", '-' * 50)
    report(true, pred)

    real_h, real_m, real_l, predict_h, predict_m, predict_l = get_diff(100, 10, true, pred, nor2len)

    print("h", '-'*50)
    report(real_h, predict_h)

    print("m", '-' * 50)
    report(real_m, predict_m)

    print("l", '-' * 50)
    report(real_l, predict_l)



if __name__ == '__main__':
    arg = ArgumentParser()
    es = elasticsearch.Elasticsearch(es_host, timeout=3000)
    nor2popular = pickle.load(open("norm2popular_path", "rb"))
    nor2id = pickle.load(open(norm2id_path, 'rb'))
    id2nor = pickle.load(open(id2norm_path, 'rb'))
    nor2len = pickle.load(open(nor2len_path, 'rb'))
    
    find_if_exist = re.compile(r'docid=(.*?)-')
    
    if arg.mode == "preprocess":
        candicates = get_data()
        with open(candicates_save_path, 'wb') as f:
            pickle.dump(candicates, f)
        print("Save Candicates Done!")
    
    elif arg.mode == "after_ranklib":
        with open(candicates_save_path, 'rb') as f:
            candicates = pickle.load(f)
        print('Candicate Loaded')
        load_data(candicates)
        
    else:
        pred = pickle.load(open(pre_id_result_pkl_path, 'rb'))
        true = pickle.load(open(true_id_result_pkl_path, 'rb'))
        get_performance(true, pred)






















