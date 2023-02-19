import pickle
import random
from multiprocessing.managers import BaseManager
from tools.utils import all_exposed_methods

import tasks.masked_lm.config as config


def start_manager(port=config.manager_port, password=config.manager_password):
    with open(config.corpus_file, 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)
    print(f'Finish loading data. Dataset size: {len(data)}.')
    BaseManager.register('get_corpus', lambda: data, exposed=all_exposed_methods(data))
    m = BaseManager(address=('', port), authkey=password)
    s = m.get_server()
    s.serve_forever()
