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


def start_manager(port=config.manager_port, password=config.manager_password, part='train'):
    with open(config.id_to_cls_file, 'rb') as f:
        id_to_cls = pickle.load(f)

    assert part in ['train', 'dev', 'test']
    if part == 'train':
        file_path = config.training_set_file
    elif part == 'dev':
        file_path = config.val_set_file
    else:
        file_path = config.test_set_file
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Finish loading data. Dataset size: {len(dataset)}.')

    cls_to_idx = {}
    for i in tqdm(range(len(dataset))):
        name, aff_id = dataset[i]
        dataset[i] = (name, id_to_cls[aff_id])
        cls_to_idx.setdefault(id_to_cls[aff_id], []).append(i)
    print(f'Finish create cls_id to index mapping. Size: {len(cls_to_idx)}.')
    mapping = Mapping(cls_to_idx)

    BaseManager.register('get_dataset', lambda: dataset, exposed=all_exposed_methods(dataset))
    BaseManager.register('get_mapping', lambda: mapping, exposed=all_exposed_methods(mapping))
    m = BaseManager(address=('', port), authkey=password)
    s = m.get_server()
    s.serve_forever()
