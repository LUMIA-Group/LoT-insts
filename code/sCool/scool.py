from search import *
from tqdm import tqdm
from multiprocessing.pool import Pool


def scool_normalize(ori_l):
    nor_l = []
    with Pool(MP_NUM) as pool:
        for i in tqdm(pool.imap(predict, [calibrated(j) for j in ori_l]), total=len(ori_l)):
            nor_l.append(i)
    return nor_l


def scool_normalize_with_score(ori_l):
    nor_l = []
    with Pool(MP_NUM) as pool:
        for i, s in tqdm(pool.imap(predict_with_score, [calibrated(j) for j in ori_l]), total=len(ori_l)):
            nor_l.append((i, s))
    return nor_l

