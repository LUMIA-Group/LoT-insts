import re
from collections import defaultdict
from tqdm import tqdm
from collections import Counter
import math
import time
import pickle
from multiprocessing.pool import Pool
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from scipy.stats import entropy
from config import *

# word process
stopwords_eng = stopwords.words('english')
stopwords_eng.extend(['!', ',', '.', '?', '-s', '-ly', '</s>', 's', ''])
stopwords_eng = set(stopwords_eng)

# WordPunctTokenizer
from nltk.tokenize import WordPunctTokenizer

# TreebankWordTokenizer
from nltk.tokenize import TreebankWordTokenizer

# WhitespaceTokenizer
from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

pattern = r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |\.\.\.                # ellipsis
              |(?:[.,;"'?():-_`])    # special characters with meanings
            """

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def build_tokenizer(doc):
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(doc)


def extract_useful_tokens(sentence):
    # sklearn
    tokens = build_tokenizer(sentence)  # 864   861

    tokens = [i.lower() for i in tokens]
    tokens = [i for i in tokens if i not in stopwords_eng]
    return tokens


def replace_words(sentence):
    rep = {'+': ' ', '-': ' ', '!': ' ', '(': ' ', ')': ' ', '{': ' ', '}': ' ', '[': ' ', ']': ' ', '^': ' ', '~': ' ',
           '\\': ' ', ':': ' ', '/': ' ', '\"': ' ', '\"': ' ', '#': ' '}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    # print(rep)
    # print(rep.keys())
    pattern = re.compile("|".join(rep.keys()))

    # print(pattern)
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], sentence)


with open(cache_path, "rb") as f:
    aff_name2sentences, train_size, label_size, aff_names, vocabulary, aff_name2token2count = pickle.load(f)


all_token2count = defaultdict(int)
all_token = []
for aff_name, token2count in aff_name2token2count.items():
    all_token += (list(token2count.keys()))
    for key in token2count.keys():
        all_token2count[key] += token2count[key]


token_num = len(set(all_token))
all_token_num = sum(all_token2count.values())


token2log_p_w = {}
for key, value in all_token2count.items():
    token2log_p_w[key] = math.log((value + 1) / (all_token_num + token_num))


aff_name2sentence_count = {aff_name: len(sentences) for aff_name, sentences in aff_name2sentences.items()}
aff_name2all_token_count = {aff_name: sum(token2count.values()) for aff_name, token2count in
                            aff_name2token2count.items()}


aff_name2token2log_p_w_c = {}
print("Calculating Probs...")
for aff_name, token2count in tqdm(aff_name2token2count.items()):
    denominator = aff_name2all_token_count[aff_name]
    default_p_w_c = math.log(1 / (max(aff_name2all_token_count.values()) + len(vocabulary)))
    aff_name2token2log_p_w_c[aff_name] = defaultdict(lambda: default_p_w_c)
    token2log_p_w_c = aff_name2token2log_p_w_c[aff_name]
    for token, count in token2count.items():
        token2log_p_w_c[token] = math.log((count) / denominator)


# denominator = sum(aff_name2sentence_count.values()) + len(aff_name2sentence_count)
# aff_name2log_p_c = {aff_name: math.log(sentence_count + 1 / denominator) for aff_name, sentence_count in
#                     aff_name2sentence_count.items()}
del aff_name2sentences
del aff_name2token2count
del all_token2count
del token2log_p_w
del aff_name2sentence_count
del aff_name2all_token_count

latin_with_extend_dict = {
    ord("¡"): "!", ord("¢"): "c", ord("£"): "L", ord("¤"): "o", ord("¥"): "Y",
    ord("¦"): "|", ord("§"): "S", ord("¨"): "`", ord("©"): "c", ord("ª"): "a",
    ord("«"): "<<", ord("¬"): "-", 173: "-", ord("®"): "R", ord("¯"): "-",
    ord("°"): "o", ord("±"): "+-", ord("²"): "2", ord("³"): "3", ord("´"): "'",
    ord("µ"): "u", ord("¶"): "P", ord("·"): ".", ord("¸"): ",", ord("¹"): "1",
    ord("º"): "o", ord("»"): ">>", ord("¼"): "1/4", ord("½"): "1/2", ord("¾"): "3/4",
    ord("¿"): "?", ord("À"): "A", ord("Á"): "A", ord("Â"): "A", ord("Ã"): "A",
    ord("Ä"): "A", ord("Å"): "A", ord("Æ"): "Ae", ord("Ç"): "C", ord("È"): "E",
    ord("É"): "E", ord("Ê"): "E", ord("Ë"): "E", ord("Ì"): "I", ord("Í"): "I",
    ord("Î"): "I", ord("Ï"): "I", ord("Ð"): "D", ord("Ñ"): "N", ord("Ò"): "O",
    ord("Ó"): "O", ord("Ô"): "O", ord("Õ"): "O", ord("Ö"): "O", ord("×"): "*",
    ord("Ø"): "O", ord("Ù"): "U", ord("Ú"): "U", ord("Û"): "U", ord("Ü"): "U",
    ord("Ý"): "Y", ord("Þ"): "p", ord("ß"): "b", ord("à"): "a", ord("á"): "a",
    ord("â"): "a", ord("ã"): "a", ord("ä"): "a", ord("å"): "a", ord("æ"): "ae",
    ord("ç"): "c", ord("è"): "e", ord("é"): "e", ord("ê"): "e", ord("ë"): "e",
    ord("ì"): "i", ord("í"): "i", ord("î"): "i", ord("ï"): "i", ord("ð"): "d",
    ord("ñ"): "n", ord("ò"): "o", ord("ó"): "o", ord("ô"): "o", ord("õ"): "o",
    ord("ö"): "o", ord("÷"): "/", ord("ø"): "o", ord("ù"): "u", ord("ú"): "u",
    ord("û"): "u", ord("ü"): "u", ord("ý"): "y", ord("þ"): "p", ord("ÿ"): "y",
    ord("’"): "'",
    0x0100: "A", 0x0101: "a", 0x0102: "A", 0x0103: "a", 0x0104: "A", 0x0105: "a",
    0x0106: "C", 0x0107: "c", 0x0108: "C", 0x0109: "c", 0x010a: "C", 0x010b: "c", 0x010c: "C", 0x010d: "c",
    0x010e: "D", 0x010f: "d", 0x0110: "D", 0x0111: "d",
    0x0112: "E", 0x0113: "e", 0x0114: "E", 0x0115: "e", 0x0116: "E", 0x0117: "e", 0x0118: "E", 0x0119: "e", 0x011a: "E",
    0x011b: "e",
    0x011c: "G", 0x011d: "g", 0x011e: "G", 0x011f: "g", 0x0120: "G", 0x0121: "g", 0x0122: "G", 0x0123: "g",
    0x0124: "H", 0x0125: "h", 0x0126: "H", 0x0127: "h",
    0x0128: "I", 0x0129: "i", 0x012a: "I", 0x012b: "i", 0x012c: "I", 0x012d: "i", 0x012e: "I", 0x012f: "i", 0x0130: "I",
    0x0131: "i",
    0x0132: "IJ", 0x0133: "ij",
    0x0134: "J", 0x0135: "j",
    0x0136: "K", 0x0137: "k", 0x0138: "k",
    0x0139: "L", 0x013a: "l", 0x013b: "L", 0x013c: "l", 0x013d: "L", 0x013e: "l", 0x013f: "L", 0x0140: "l", 0x0141: "L",
    0x0142: "l",
    0x0143: "N", 0x0144: "n", 0x0145: "N", 0x0146: "n", 0x0147: "N", 0x0148: "n", 0x0149: "n", 0x0150: "N", 0x0151: "n",
    0x0152: "OE", 0x0153: "oe",
    0x0154: "R", 0x0155: "r", 0x0156: "R", 0x0157: "r", 0x0158: "R", 0x0159: "r",
    0x015a: "S", 0x015b: "s", 0x015c: "S", 0x015d: "s", 0x015e: "S", 0x015f: "s", 0x0160: "S", 0x0161: "s",
    0x0162: "T", 0x0163: "t", 0x0164: "T", 0x0165: "t", 0x0166: "T", 0x0167: "t",
    0x0168: "U", 0x0169: "u", 0x016a: "U", 0x016b: "u", 0x016c: "U", 0x016d: "u", 0x016e: "U", 0x016f: "u", 0x0170: "U",
    0x0171: "u", 0x0172: "U", 0x0173: "u",
    0x0174: "W", 0x0175: "w",
    0x0176: "Y", 0x0177: "y", 0x0178: "Y",
    0x0179: "Z", 0x017a: "z", 0x017b: "Z", 0x017c: "z", 0x017d: "Z", 0x017e: "z",
    0x017f: "s",
    0x2013: "-",
    0x2014: "-",
    0x2018: "'",
}


def latin2ascii(s):
    return s.translate(latin_with_extend_dict)


def alphabet_only(string):
    string_alphabet_only = re.sub('[^a-zA-Z0-9]+', ' ', string).strip().lower()  
    # use [^]+ to remove continuous non-alphabet characters, use strip to remove spaces at the two ends.
    return string_alphabet_only


def general_nb(sentence, top_k_candidates):
    tokens = extract_useful_tokens(sentence.lower())
    tokens = [i for i in tokens if i in vocabulary]
    aff_name2log_p = {}
    # top_k_candidates = list(top_k_candidates)
    # token2log_p_w = weight_cal(top_k_candidates)
    #     print(len(top_k_candidates))
    for aff_name in top_k_candidates:
        log_p = 0  # aff_name2log_p_c[aff_name]
        aff_name_tekens = extract_useful_tokens(aff_name)
        aff_name_tekens = set([i for i in aff_name_tekens if i in vocabulary])
        try:
            token2log_p_w_c = aff_name2token2log_p_w_c[aff_name]
        except:
            continue
        for token in tokens:
            log_p += token2log_p_w_c[token]
        score = log_p + 0.0001
        aff_name2log_p[aff_name] = score
    return aff_name2log_p


def nb_predict(sentence):
    top_k_candidates = list(aff_names)
    aff_name2log_p = general_nb(sentence, top_k_candidates)
    final_result = max(aff_name2log_p.keys(), key=lambda x: aff_name2log_p[x])
    del aff_name2log_p, top_k_candidates
    return final_result


def nb_predict_with_score(sentence):
    top_k_candidates = list(aff_names)
    aff_name2log_p = general_nb(sentence, top_k_candidates)
    final_result = max(aff_name2log_p.items(), key=lambda x: x[1])
    del aff_name2log_p, top_k_candidates
    return final_result


def js_divergence(p, q):
    m = (p+q) / 2
    js = entropy(p, m, axis=-1) / 2 + entropy(q, m, axis=-1) / 2
    return js


def nb_discriminate_jsd(p):
    s_a, s_b = p
    top_k_candidates = list(aff_names)
    aff2p_a = general_nb(s_a, top_k_candidates)
    aff2p_b = general_nb(s_b, top_k_candidates)
    dist_a = np.array([aff2p_a[i] for i in top_k_candidates])
    dist_b = np.array([aff2p_b[i] for i in top_k_candidates])
    final_result = js_divergence(dist_a, dist_b)
    return final_result

# basic normalization for list of original names
def bayes_normalize(ori_l):
    nor_l = []
    ori_l = [replace_words(latin2ascii(j)) for j in ori_l]
    with Pool(MP_NUM) as pool:
        for i in tqdm(pool.imap(nb_predict, ori_l), total=len(ori_l)):
            nor_l.append(i)
    return nor_l

# normalization for orginal names, and return list of tuples with (label, confidence)
def bayes_normalize_with_score(ori_l):
    nor_l = []
    ori_l = [replace_words(latin2ascii(j)) for j in ori_l]
    with Pool(MP_NUM) as pool:
        for i, s in tqdm(pool.imap(nb_predict_with_score, ori_l), total=len(ori_l)):
            nor_l.append((i, s))
    return nor_l

# discriminate the list of original name pairs, return list of JS-Distance
def bayes_discriminate(ori_a_l, ori_b_l):
    label_l = []
    ori_a_l = [replace_words(latin2ascii(j)) for j in ori_a_l]
    ori_b_l = [replace_words(latin2ascii(j)) for j in ori_b_l]
    pair_l = list(zip(ori_a_l, ori_b_l))
    with Pool(MP_NUM) as pool:
        for s in tqdm(pool.imap(nb_discriminate_jsd, pair_l), total=len(pair_l)):
            label_l.append(s)
    return label_l
