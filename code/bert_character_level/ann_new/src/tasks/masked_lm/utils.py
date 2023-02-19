import os
import pickle

import torch

import cs
from . import config
from tools.tokenizer import AnnTokenizer
from .dataset import MaskedLMDataset
from .model import AnnBertForMaskedLM


def get_model(
        *,
        last_training_time=None,
        last_step=None,
):
    if last_training_time is None:
        last_training_time = config.last_training_time
    if last_step is None:
        last_step = config.last_step

    if last_training_time == 0:
        conf = config.model_conf
    else:
        conf_file = os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{last_training_time}', 'model_conf.pkl')
        with open(conf_file, 'rb') as f:
            conf = pickle.load(f)

    model = AnnBertForMaskedLM(conf)

    if last_training_time != 0:
        file = os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{last_training_time}', f'model_{last_step}.pt')
        model.load_state_dict(torch.load(file, map_location=torch.device("cpu")), strict=False)

    return model


def get_training_set():
    tokenizer = AnnTokenizer(cs.VOCAB_FILE)
    dataset = MaskedLMDataset(
        tokenizer,
        manager_host=config.manager_host,
        manager_port=config.manager_port,
        manager_password=config.manager_password,
    )
    return dataset

