#!/bin/python
# -*- coding:utf-8 -*- 

AAindex=['GLY','ALA','VAL','LEU','ILE','PHE','TRP','TYR','ASP','ASN','GLU','LYS','GLN','MET','SER','THR','CYS','PRO','HIS','ARG']
aaindex='GAVLIFWYDNEKQMSTCPHR'
RNAindex=['A','U','C','G']
DNAindex = ['DA','DT','DC','DG']

class Atom(list):
    def __init__(self,name,num,x,y,z):
        self.name = name
        self.num = num
        self.extend([x,y,z])

class Residue(list):
    def __init__(self,name,num,atoms,insertAA):
        self.label = 0
        self.interface_label = 0
        self.insert = insertAA
        self.name = name
        self.num = num
        self.extend(atoms)

class Chain(list):
    def __init__(self,name,residues):
        self.name = name
        self.extend(residues)

class Model(list):
    def __init__(self,name,chains):
        self.name = name
        self.extend(chains)

class ParseLine:
    def __init__(self,line):
        self.atom_name = line[12:16].strip()
        self.res_name = line[17:20].strip()
        self.chain_name = line[20:22].strip()
        self.atom_num = int(line[6:11].strip())
        self.res_num = int(line[22:26].strip())
        self.insert = line[26]
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())

def parse_pdb(file_name):
    atoms = []
    residues = []
    chains = []
    old_line = ''

    i = 0
    for line in open(file_name):
        if line[:6] == 'ENDMDL':
            break
        if (line[0:4] == 'ATOM'):
            line = ParseLine(line)
            if i > 0:
                if  line.res_num != old_line.res_num or line.res_name != old_line.res_name:
                    residues.append(Residue(old_line.res_name,old_line.res_num,atoms,old_line.insert))
                    atoms = []
                if line.chain_name != old_line.chain_name:
                    chains.append(Chain(old_line.chain_name, residues))
                    residues = []

            if line.atom_name[0] == 'H':
                continue

            atoms.append(Atom(line.atom_name, line.atom_num, line.x, line.y, line.z))
            old_line = line
            i += 1

    residues.append(Residue(old_line.res_name, old_line.res_num, atoms,old_line.insert))
    chains.append(Chain(old_line.chain_name, residues))
    return Model(file_name, chains)