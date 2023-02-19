import re
from collections import defaultdict
from tqdm import tqdm
from fuzzywuzzy import fuzz, process
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
import math
import time
import pickle
from multiprocessing.pool import Pool
import sys
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy
from spacy.tokens import Doc
import stanza
import elasticsearch
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

print("Loading Train Data For NaiveBayes...")
surface_list, label_list = [], []
with open(train_txt_path, "r") as fin:
    for line in fin.readlines():
        line = line.strip()
        if not line:
            continue
        surface, label = line.split("\t\t")
        surface_list.append(surface)
        label_list.append(label)
aff_name2sentences = {}
train_size = len(label_list)
train_labels = set(label_list)
label_size = len(train_labels)
for i in tqdm(range(train_size)):
    surface, label = surface_list[i], label_list[i]
    if label not in aff_name2sentences:
        aff_name2sentences[label] = []
    aff_name2sentences[label].append(surface)
aff_names = set(aff_name2sentences.keys())
del train_labels

def count_tokens(args):
    aff_name, sentences = args
    token2count = defaultdict(int)
    for sentence in sentences:
        tokens = extract_useful_tokens(sentence)
        for token in tokens:
            token2count[token] += 1
    return aff_name, token2count


print("Counting Tokens...")
vocabulary = set()
aff_name2token2count = {}
with Pool() as pool:
    for aff_name, token2count in tqdm(pool.imap_unordered(count_tokens, aff_name2sentences.items()), total=label_size):
        vocabulary.update(token2count.keys())
        aff_name2token2count[aff_name] = token2count


cache_list = [aff_name2sentences, train_size, label_size, aff_names, vocabulary, aff_name2token2count]

with open(cache_path, "wb") as f:
    pickle.dump(cache_list, f)

