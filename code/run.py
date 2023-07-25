### Run SPIN-CGraph to design sequence for given PDB ###

from data_preprocess import from_pdb, append_cgraph, aa_decode
from model import SPIN_CGNN
from data import preprocess
import torch
import argparse
import pathlib
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pdb', type=str,
                    default='../example/5flm.E.pdb')
parser.add_argument('--dir', type=str,
                    default=None)
parser.add_argument('--exp_name', type=str, 
                    default='cath_CGNN-L10-d128-h4-c12-v3_b4096-lr0.001-e100_final')
parser.add_argument('--output_dir', type=str,
                    default='../output')


default_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
def load_data(pdb_path):
    try:
        entrys = from_pdb(pdb_path)
        print(f'Load {len(entrys)} chains from {pdb_path}')
        print(list(entrys.keys()))
    except:
        entrys = {}
        print(f'Failed to load chains from {pdb_path}')
    return entrys
    

def load_model(exp_name):
    model_args = exp_name.split('_')[1]
    model_args = model_args.split('-')
    n_layers, n_dim, n_heads, dist_cutoff, n_virtual_atoms = [
        int(model_args[i][1:]) for i in range(1, 6)
    ]
    model = SPIN_CGNN(
        d_model=n_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        n_virtual_atoms=n_virtual_atoms,
    ).to(default_device)
    
    state_dict = torch.load(f'../experiment/{exp_name}/best_model.pth')
    model.load_state_dict(state_dict)
    print(f'Load Model from: {exp_name}')
    print(f'Locate Model on: {default_device}')
    print('Model Parameters: %.2f M' % (sum(p.numel() for p in model.parameters())/1e6))
    model.eval()
    return model, dist_cutoff

def sample_from_logits(logits, n_sample=1, temperature=1e-6):
    probs = torch.softmax(logits/temperature, dim=-1)
    seqs = torch.distributions.Categorical(probs).sample([n_sample])
    seqs = [aa_decode(seq) for seq in seqs.cpu().numpy()]
    return seqs

def main():
    args = parser.parse_args()
    model, dist_cutoff = load_model(args.exp_name)
    
    if args.dir == None:
        entrys = load_data(args.pdb)
    else:
        entrys = {}
        for pdb in os.listdir(args.dir):
            entrys.update(load_data(os.path.join(args.dir, pdb)))    
    
    for name in entrys.keys():
        entry = entrys[name]
        output_dir = f'{args.output_dir}/{name}'
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f'{output_dir}/parsed_entry.pkl', 'wb') as writer:
            pickle.dump(entry, writer)
        entry = append_cgraph(entry, cutoff=dist_cutoff)
        entry = preprocess(entry)
        nn_input = {item: entry[item].to(default_device) for item in [
            'coo', 'rel_pos', 'consistancy', 'edges', 'graph_id', 'soe'
        ]}
        
        with torch.no_grad():
            logits = model(**nn_input)['logits']
            np.save(f'{output_dir}/logits.npy', logits.cpu().numpy())
            seqs = sample_from_logits(logits)
            with open(f'{output_dir}/seqs.fa', 'w') as writer:
                for i, seq in enumerate(seqs):
                    writer.write(f'>design_{i}\n{seq}\n')
                    
if __name__ == '__main__':
    main()