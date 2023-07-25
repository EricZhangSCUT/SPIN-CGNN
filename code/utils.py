import pathlib
import os
import logging
import time
import numpy as np
        
        
def setup_savedir(exp_name, code_backup=True):
    save_dir = '../experiment/%s' % exp_name
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    if code_backup:
        pathlib.Path(f'{save_dir}/code').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir():
            if filename[-3:] == '.py':
                os.system(f"cp -rf ./{filename} {save_dir}/code/{filename}")
    return save_dir

def setup_logger(exp_name):
    logging.basicConfig(
        level=logging.INFO, filename='../log/%s.txt' % exp_name,
        filemode='w', format='%(message)s')
    logger = logging.getLogger(__name__)
    return logger

def setup_exp(args):
    exp_name = '_'.join([
        args.data,
        f'{args.arch}-L{args.layers}-d{args.dim}-h{args.heads}-c{args.cutoff}-v{args.virtual_atom}',
        f'b{args.batchsize}-lr{args.lr}-e{args.epochs}'
    ])
    
    if args.note != '':
        exp_name = '_'.join([exp_name, args.note])

    save_dir = setup_savedir(exp_name)
    logger = setup_logger(exp_name)
    return exp_name, save_dir, logger

def get_lcr(path, note=''):
        os.system(f'segmasker -in {path}/pred_seq{note}.fa -out {path}/seq{note}_lcr.interval')
        time.sleep(5)
        
        with open(f'{path}/pred_seq{note}.fa') as f:
            lines = f.read().split('\n')
            seq_len = {name[1:]:len(seq) for name, seq in zip(lines[::2], lines[1::2])}
            
        with open(f'{path}/seq{note}_lcr.interval') as f:
            lines = f.read()
            lcr_line = [l.split('\n')[:-1] for l in lines.split('>')[1:]]
            
        lcr_array = {}
        for line in lcr_line:
            name = line[0]
            lcr_array[name] = np.zeros(seq_len[name])
            
            for lcr in line[1:]:
                start, end = lcr.split(' - ')
                lcr_array[name][int(start):int(end)] = 1
        lcr = [la.mean() for la in lcr_array.values()]
        return np.mean(lcr)