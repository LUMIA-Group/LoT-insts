import pickle
import random
from multiprocessing.managers import BaseManager
from tqdm import tqdm
from . import config
from tools.utils import all_exposed_methods


# Dict[int, List[int]]
class Mapping(dict):

    def sub_len(self, key):
        return len(self[key])

    def sub_value(self, key, idx):
        return self[key][idx]

    def sub_choice(self, key):
        return random.choice(self[key])

    def sub_sample(self, key, k=1):
        return random.sample(self[key], k=k)


def start_manager(port=config.manager_port, password=config.manager_password):
    file_path = config.training_set_file
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Finish loading data. Dataset size: {len(dataset)}.')

    cls_to_idx = {}
    for i, (name, aff_id) in tqdm(enumerate(dataset)):
        aff_id = int(aff_id)
        dataset[i] = (name, aff_id)
        cls_to_idx.setdefault(aff_id, []).append(i)
    print(f'Finish create cls_id to index mapping. Size: {len(cls_to_idx)}.')
    mapping = Mapping(cls_to_idx)

    BaseManager.register('get_dataset', lambda: dataset, exposed=all_exposed_methods(dataset))
    BaseManager.register('get_mapping', lambda: mapping, exposed=all_exposed_methods(mapping))
    m = BaseManager(address=('', port), authkey=password)
    s = m.get_server()
    s.serve_forever()
