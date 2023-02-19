import re
from nltk.corpus import stopwords

stopwords_eng = stopwords.words('english')
stopwords_eng.extend(['!', ',', '.', '?', '-s', '-ly', '</s>', 's', ''])
stopwords_eng = set(stopwords_eng)


def build_tokenizer(doc: str):
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(doc)


def extract_useful_tokens(sentence: str) -> list:
    tokens = build_tokenizer(sentence)
    tokens = [i.lower() for i in tokens]
    tokens = [i for i in tokens if i not in stopwords_eng]
    return tokens


def calibrated(sentence: str) -> str:
    return " ".join(extract_useful_tokens(sentence))