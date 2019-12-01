import pickle
# import yaml
# from utils.util import OrderedYaml
# Loader, Dumper = OrderedYaml()
#
# path = '../options/train/train_EDVR_M.yml'
# with open(path, mode='r') as f:
#     opt = yaml.load(f, Loader=Loader)
# print(opt['network_G'])
# for phase, dataset in opt['datasets'].items():
#     print(phase)
#     print(dataset['name'])

# import pickle
# path = '../dataset/train_x4_wval.lmdb/meta_info.pkl'
# a = sorted(pickle.load(open(path,'rb'))['keys'])
# print(a)

import torch
world_size = torch.distributed.get_world_size()
print(world_size)

