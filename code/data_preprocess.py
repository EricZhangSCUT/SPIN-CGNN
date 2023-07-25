import numpy as np
from parsepdb import parse_pdb
from CGraph import ContactGraph
import os

# -----Global Setting-----
ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '.', 'X']
AA1_to_AA3 = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
              'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
              'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
              'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
AA3_to_AA1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
              'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
              'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
              'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
AA_INDEX = {}
for aa in ALPHABET:
    AA_INDEX[aa] = ALPHABET.index(aa)
    AA_INDEX[ALPHABET.index(aa)] = aa
def aa_encode(seq):
    return np.array([AA_INDEX[aa] for aa in seq])
def aa_decode(seq):
    return ''.join(np.array(ALPHABET)[seq])

atoms = ['N', 'CA', 'C', 'O']
n_atoms = len(atoms)
atom_idx = {atom:atoms.index(atom) for atom in atoms}
# -----Global Setting-----

def cal_cb(n, ca, c):
    b = ca - n
    c = c - ca
    a = np.cross(b, c)
    cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + ca
    return cb

def from_pdb(pdb_path, structure_name=None, chains=None):
    '''
    Get structure-sequence pairs from PDB
    '''
    
    structure = parse_pdb(pdb_path)
    if structure_name == None:
        structure_name = os.path.split(pdb_path)[-1].split('.')[0]
        
    entrys = {}
    for chain in structure:
        if chains != None:
            if chain.name not in  chains:
                continue
        entry = {'name': f'{structure_name}_{chain.name}'}
        seq = []
        coo = []
        for res in chain:
            seq.append(AA_INDEX[AA3_to_AA1[res.name]])
            coo.append(np.empty([n_atoms, 3]))
            coo[-1][:] = np.nan
            for atom in res:
                if atom.name in atoms:
                    coo[-1][atom_idx[atom.name]] = atom
        
        entry['mask'] = ~np.isnan(coo).any(axis=(1,2))
        entry['seq'] = np.array(seq)[entry['mask']]
        entry['coo'] = np.array(coo)[entry['mask']]
        entrys[entry['name']] = entry
    return entrys

def append_cgraph(entry, cutoff=12, center='cb'):
    coo = entry['coo']
    if center == 'cb':
        coo = cal_cb(coo[:, 0], coo[:, 1], coo[:, 2])
    else:
        coo = coo[:, 1]
    
    cg = ContactGraph(coo=coo, cutoff=cutoff)
    entry['edges'] = cg.edges
    entry['triedges'] = cg.get_triangle_edges()
    return entry