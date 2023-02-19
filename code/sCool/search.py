import elasticsearch
from config import *
from fuzzywuzzy import fuzz
from data_utils import *


es = elasticsearch.Elasticsearch(db_host, timeout=30, maxsize=25)
print("Es Connection:", es.ping())


def query_body(sentence: str, query_type: str) -> dict:
    tokens = extract_useful_tokens(sentence)
    if query_type == "fuzzy":
        return {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": sentence,
                            "analyze_wildcard": True,
                            "time_zone": "Asia/Shanghai",
                            "fuzziness": "AUTO"
                        }
                    }
                ],
                "filter": [],
                "should": [],
                "must_not": []
            }
        }
    elif query_type == "parse":
        return {
            "bool": {
                "must": [],
                "filter": [],
                "should": [
                    {
                        "query_string": {
                            "query": i,
                            "analyze_wildcard": True,
                            "time_zone": "Asia/Shanghai"
                        }
                    } for i in tokens
                ],
                "must_not": []
            }
        }
    elif query_type == "exact":
        return {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": i,
                            "analyze_wildcard": True,
                            "time_zone": "Asia/Shanghai"
                        }
                    } for i in tokens
                ],
                "filter": [],
                "should": [],
                "must_not": []
            }
        }
    else:
        raise ValueError("No Such Type.")


def get_top_k_sim(words: str, query_type: str, K: int = 80):
    ret = es.search(index=db_name, body={
        "size": K,
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            }
        ],
        "_source": {
            "excludes": []
        },
        "script_fields": {},
        "docvalue_fields": [],
        "stored_fields": [
            "*"
        ],
        "query": query_body(words, query_type)
    })
    if len(ret['hits']['hits']) <= 0:
        return [[], [], []]
    else:
        search_result_list_score = []
        search_result_list_norm = []
        search_result_list_ori = []
        for i, k in enumerate(ret['hits']['hits']):
            search_result_list_norm.append(k['_source']['normalized_name'])
            search_result_list_ori.append(k['_source']['original_name'])
            search_result_list_score.append(k['_score'])
        return search_result_list_norm[:K], search_result_list_ori[:K], search_result_list_score[:K]


def weighted_score(s: float, w: float) -> float:
    return min(s * w, 100)


def retrieve_boost(sentence: str) -> list:
    F = get_top_k_sim(sentence, "exact")
    S = get_top_k_sim(sentence, "fuzzy")
    T = get_top_k_sim(sentence, "parse")
    ret = []

    def check_and_keep(X: list, w: float):
        label_field, surface_field, scores = X
        for i in range(len(scores)):
            score = weighted_score(scores[i], w)
            if score > weight_threshold:
                ret.append((label_field[i], surface_field[i], scores[i]))

    check_and_keep(F, weight_exact)
    check_and_keep(S, weight_fuzzy)
    check_and_keep(T, weight_parse)

    ret.sort(key=lambda x: x[2], reverse=True)

    return ret[:ret_num]


def predict(surface: str) -> str:
    try:
        candidates = retrieve_boost(surface)
    except:
        candidates = []
    if not candidates:
        pred = ""
    else:
        pred = max(candidates, key=lambda x: fuzz.ratio(x[1], surface))[0]
    return pred


def predict_with_score(surface: str) -> str:
    try:
        candidates = retrieve_boost(surface)
    except:
        candidates = []
    if not candidates:
        pred, score = "", 0
    else:
        pred, _, score = max(candidates, key=lambda x: fuzz.ratio(x[1], surface))
    return pred, score