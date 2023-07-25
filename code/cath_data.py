# -*- coding: utf-8 -*
import torch
import pickle
import time
import numpy as np
import random
import copy
from CGraph import to_direct, soe_from_triedges

ROOT_DATA = '../data'


def cal_cb(n, ca, c):
    b = ca - n
    c = c - ca
    a = np.cross(b, c)
    cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + ca
    return cb

def append_cb(coo):
    return np.concatenate([coo, cal_cb(coo[:,0], coo[:,1], coo[:,2])[:, None]], 1)

def get_consistancy(mask):
    paded_mask = np.pad(mask, (1, 1), 'constant', constant_values=False)
    consistancy = paded_mask[1:] & paded_mask[:-1]
    consistancy = np.stack([consistancy[:-1], consistancy[1:]])[:, mask]
    return consistancy.T  # [2,L]->[L,2]

def get_rel_pos(edges, mask):
    abs_pos = np.arange(len(mask))[mask]
    rel_pos = np.array(abs_pos[edges][1] - abs_pos[edges[0]])
    return rel_pos

# fixed-length data grouper
def group_size(sizes, target_size):
    '''randomly group sizes'''
    random.shuffle(sizes)
    groups = [[]]
    gaps = np.array([target_size], dtype=int)
    for size in sizes:
        not_full = gaps>=size
        if not_full.any():
            idx = np.random.choice(not_full.nonzero()[0])
            groups[idx].append(size)
            gaps[idx] -= size
        else:
            groups.append([size])
            gaps = np.append(gaps, target_size-size)
    return groups

def fill_size_groups(size_groups, entrys_in_bins):
    '''
    randomly fill size_groups with entrys
    '''
    entrys_groups = []
    entrys_in_bins_copy = copy.deepcopy(entrys_in_bins)
    random.shuffle(size_groups)
    for size_group in size_groups:
        entrys_groups.append([entrys_in_bins_copy[size].pop(0) for size in size_group])
    return entrys_groups

def group_entrys(entrys_in_bins, sizes, target_size):
    '''
    entrys_in_bins: dict {size:[entry1, entry2...]}
    sizes: [N]
    '''
    size_groups = group_size(sizes, target_size)
    grouped_files = fill_size_groups(size_groups, entrys_in_bins)
    return grouped_files

class CATH_CGraph(object):
    def __init__(self, dataset='cath4.2', subset='test', batch_size=None, cutoff=10) -> None:
        time_start_init = time.time()
        with open(f'{ROOT_DATA}/{dataset}/{subset}-CGraph{cutoff}.pkl', 'rb') as f:
            self.entrys = pickle.load(f)
        
        self.length_list = []
        self.length_bins = [[] for _ in range(2000)]
        for name in self.entrys.keys():
            self.entrys[name]['name'] = name
            self.entrys[name] = preprocess(self.entrys[name])
            length = len(self.entrys[name]['seq'])
            self.length_list.append(length)
            self.length_bins[length].append(name)

        if batch_size != None:
            self.pack_entrys(batch_size)
        else:
            self.packed_entrys = list(self.entrys.values())
        
        time_finish_init = time.time()
        print('Dataset_init_time:', time_finish_init-time_start_init)

    def __len__(self):
        return len(self.packed_entrys)
    
    def pack_entrys(self, aa_batchsize):
        self.groups = group_entrys(self.length_bins, self.length_list, aa_batchsize)
        
        self.packed_entrys = []
        for group in self.groups:
            grouped_entrys = {'name' : group}
            for item in ['coo', 'seq', 'rel_pos', 'consistancy']:
                grouped_entrys[item] = torch.cat([
                    self.entrys[name][item] for name in group
                ], dim=0)
            
            edges = []
            graph_id = []
            soe = []

            cur_n = 0
            cur_g = 0
            cur_e = 0
            for name in group:
                entry = self.entrys[name]
                edges.append(entry['edges'] + cur_n)
                cur_n += len(entry['seq'])
                soe.append(entry['soe'] + cur_e)
                cur_e += len(entry['edges'][0])
                graph_id.append(entry['graph_id'] + cur_g)
                cur_g += 1
                
            grouped_entrys['edges'] = torch.cat(edges, dim=-1)
            grouped_entrys['graph_id'] = torch.cat(graph_id)
            grouped_entrys['soe'] = torch.cat(soe)

            self.packed_entrys.append(grouped_entrys)

    def __getitem__(self, idx):
        return self.packed_entrys[idx]

def preprocess(entry):
    entry['coo'] = torch.from_numpy(append_cb(entry['coo'])).float()
    entry['seq'] = torch.from_numpy(entry['seq']).long()
    entry['consistancy'] = torch.from_numpy(get_consistancy(entry['mask']))
    entry['graph_id'] = torch.zeros_like(entry['seq'], dtype=torch.long)
    entry['edges'] = torch.from_numpy(to_direct(entry['edges'])).long()
    entry['rel_pos'] = torch.from_numpy(get_rel_pos(entry['edges'], entry['mask'])).long()
    entry['soe'] = torch.from_numpy(soe_from_triedges(np.int64(entry['triedges']))).flatten(0,1)
    return entry
