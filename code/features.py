import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cgnn import LinearUpdate

# Thanks for StructTrans and PiFold
# https://github.com/jingraham/neurips19-graph-protein-design
# https://github.com/A4Bio/PiFold

default_device = torch.device('cuda:0')

def nan_to_num(tensor, nan=0.0):
    tensor[torch.isnan(tensor)] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

class RBF(object):
    def __init__(self, D_min=0, D_max=20, n_bins=16, device=default_device) -> None:
        D_mu = torch.linspace(D_min, D_max, n_bins).to(device)
        self.n_bins = n_bins
        self.D_mu = D_mu.view([1,1,-1])
        self.D_sigma = (D_max - D_min) / n_bins
    
    def __call__(self, D):
        rbf = torch.exp(-((D.unsqueeze(-1) - self.D_mu) / self.D_sigma)**2)
        return rbf

class PositionalEncoding(nn.Module):
    def __init__(self, dim, device=default_device) -> None:
        super().__init__()
        self.dim = dim
        self.frequency = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            * -(np.log(10000.0) / dim)
        )
    
    def forward(self, rel_pos):
        '''
        rel_pos [*]
        pos_encoding [*, D]
        '''
        angles = rel_pos.unsqueeze(-1) * self.frequency
        pos_encoding = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return pos_encoding

class GraphProteinFeaturizer(nn.Module):
    def __init__(self,
        d_model,
        n_positional_embeddings=16,
        n_virtual_atoms=3,
        n_rbf=16,
        ) -> None:
        super().__init__()
        n_real_atoms = 5
        self.d_model = d_model
        self.n_virtual_atoms = n_virtual_atoms
        
        self.positional_embeddings = PositionalEncoding(dim=n_positional_embeddings)
        self.dist_rbf = RBF(n_bins=n_rbf)
        
        d_node = 96
        # d_dihedrals 14 + d_adjacent_directions 6 + d_inner_dist 64 + d_inner_orientation 12
        d_edge = (n_real_atoms**2) * (n_rbf + 3) + n_positional_embeddings + 4
        # (5**2)*(16+3) + 16 + 4 = 495

        if self.n_virtual_atoms != 0:
            d_virtual_edge = ((n_virtual_atoms+n_real_atoms)**2-(n_real_atoms**2)) * n_rbf  #* (n_rbf + 3)
            self.virtual_atoms_direction = nn.Parameter(_normalize(torch.rand([n_virtual_atoms, 3])))
            d_edge += d_virtual_edge
            
        self.node_norm = nn.LayerNorm(d_node)
        self.node_emb = LinearUpdate(d_node, d_model)
        self.edge_norm =  nn.LayerNorm(d_edge)
        self.edge_emb = LinearUpdate(d_edge, d_model)
        
    @staticmethod
    def _local_coordinate_system(coo):
        '''
        coo [L, 5, 3]
        '''
        
        n, ca, c = coo[:, 0], coo[:, 1], coo[:, 2]
        u = _normalize(ca - n)
        v = _normalize(c - ca)
        b = _normalize(u - v)
        n = _normalize(torch.cross(u, v), dim=-1)
        Q = torch.stack([b, n, torch.cross(b, n)], 1)
        return Q  # [L, 3, 3]
    
    @staticmethod
    def _dihedrals(coo, consistancy, eps=1e-7):
        '''
        coo [L, 5, 3]
        consistancy [L, 2]
        D_features: sin/cos of 3 dihedral and 3 angle with consistancy [L,6]+[L,6]+[L,2]
        '''

        backbone = coo[:, :3]
        backbone = backbone.flatten(0,1) # [L*3, 3]
        
        dX = backbone[1:,:] - backbone[:-1,:] # [L*3-1, 3]
        U = _normalize(dX, dim=-1)  # [L*3-1, 3]
        u_0 = U[:-2,:] # [L*3-3, 3]
        u_1 = U[1:-1,:] # [L*3-3, 3]
        u_2 = U[2:,:] # [L*3-3, 3]

        n_0 = _normalize(torch.cross(u_0, u_1), dim=-1) # [L*3-3, 3]
        n_1 = _normalize(torch.cross(u_1, u_2), dim=-1) # [L*3-3, 3]
        
        cosD = (n_0 * n_1).sum(-1)  # [L*3-3]
        cosD = torch.clamp(cosD, -1+eps, 1-eps)  # [L*3-3]
        v = _normalize(torch.cross(n_0, n_1), dim=-1)  # [L*3-3, 3]
        D = torch.sign((-v* u_1).sum(-1)) * torch.acos(cosD)  # [L*3-3]
        D = F.pad(D, (1,2), 'constant', 0)  # [L*3]
        D = D.view(-1, 3)  # [L, 3]
        Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), -1)  # [L, 6]

        # alpha, beta, gamma
        cosD = (u_0*u_1).sum(-1)  # [L*3-3]
        cosD = torch.clamp(cosD, -1+eps, 1-eps)  # [L*3-3]
        D = torch.acos(cosD)  # [L*3-3]
        D = F.pad(D, (1,2), 'constant', 0)  # [L*3]
        D = D.view(-1, 3)  # [L, 3]
        Angle_features = torch.cat((torch.cos(D), torch.sin(D)), -1)  # [L, 6]
        
        consistancy = torch.cat([
            consistancy, consistancy[:, 1].unsqueeze(1),  # [L, 3] cos
            consistancy, consistancy[:, 1].unsqueeze(1),  # [L, 3] sin
            ], dim=-1) # [L, 6]
        D_features = torch.cat([
            Dihedral_Angle_features * consistancy,  # [L, 6]
            Angle_features * consistancy,  # [L, 6]
            consistancy[:, :2]  # [L, 2]
            ], dim=-1)
        return D_features  # [L, 14]
    
    @staticmethod
    def _inner_direction(coo, Q):
        '''
        coo [L, A, 3]
        Q [L, 3, 3]
        '''

        inner_directions = _normalize(coo[:, [0,2,3,4]] - coo[:, 1].unsqueeze(1))  # [L, 4, 3]
        inner_directions = inner_directions.unsqueeze(-1) # [L, 4, 3, 1]
        Q = Q.unsqueeze(-3)  # [L, 1, 3, 3]
        local_inner_directions = _normalize(torch.matmul(Q, inner_directions).squeeze(-1))
        return local_inner_directions  # [L, 4, 3]
    
    def _inner_distance(self, coo):
        '''
        coo [L, A, 3]
        '''
        inner_dist = (coo[:, [0,2,3,4]] - coo[:, 1].unsqueeze(1)).norm(dim=-1) # [L, 4]
        inner_dist = self.dist_rbf(inner_dist).flatten(-2) # [L, 4]->[L, 4, 16]->[L, 64]
        return inner_dist  # [L, 64]
    
    @staticmethod
    def _adjacent_direction(coo, Q, consistancy, return_consistancy=False):
        '''
        coo [L, A, 3]
        Q [L, 3, 3]
        consistancy [L, 2]
        '''

        ca = coo[:, 1]  # [L, 3]
        forward_directions = _normalize(ca[1:] - ca[:-1])  # [L-1, 3]
        reward_directions = -forward_directions  # [L-1, 3]
        adjacent_directions = torch.stack(
            [F.pad(reward_directions, (0,0,1,0), 'constant', 0),
             F.pad(forward_directions, (0,0,0,1), 'constant', 0)], dim=1
            )  # [L, 2, 3]
        adjacent_directions = adjacent_directions.unsqueeze(-1) # [L, 2, 3, 1]
        Q = Q.unsqueeze(-3)  # [L, 1, 3, 3]
        local_adjacent_directions = _normalize(torch.matmul(Q, adjacent_directions).squeeze(-1))  # [L, 2, 3]
        
        consistancy = consistancy.float().unsqueeze(-1).repeat(1,1,3) # [L, 2, 3]
        local_adjacent_directions = local_adjacent_directions * consistancy
        if return_consistancy:
            local_adjacent_directions = torch.cat(
                [local_adjacent_directions, consistancy], dim=1
            )
        return local_adjacent_directions  # [L, 2, 3] or [L, 4, 3] with consistancy
    
    @staticmethod
    def _quaternion(Q, edges):
        R = torch.matmul(Q[edges[0]].transpose(-1,-2), Q[edges[1]])
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
                Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        return _normalize(Q, dim=-1)  # [E, 4]
    
    def get_virtual_atoms(self, coo, Q):
        '''
        coo [L, 5, 3]
        Q [L, 3, 3]
        '''

        local_directions = _normalize(self.virtual_atoms_direction)  # [V, 3]
        global_dircetions = torch.matmul(local_directions, Q)  # [L, V, 3]
        cb = coo[:,-1].unsqueeze(1)  # [L, 1, 3]
        virtual_coords = global_dircetions + cb
        return virtual_coords  # [L, V, 3]
    
    def _neighbor_dist(self, coo, edges, coo_2=None):
        '''
        coo [L, A1, 3]
        edges [2, E]
        coo_2 [L, A2, 3]
        '''

        if coo_2 == None:
            coo_2 = coo  # else A2=A1
        coo_src = coo[edges[0]].unsqueeze(-3)  # [E, 1, A1, 3]
        coo_tgt = coo_2[edges[1]].unsqueeze(-2)  # [E, A2, 1, 3]
        d_coo = coo_tgt -  coo_src  # [E, A2, A1, 3]
        
        # distance
        dist = d_coo.norm(dim=-1)  # [E, A2, A1]
        dist = dist.flatten(-2) # [E, A2*A1]
        dist = self.dist_rbf(dist).flatten(-2)  # [E, A2*A1, 16] -> [E, A2*A1*16]
        return dist  # [E, D]
    
    def _neighbor_direction(self, coo, edges, Q, coo_2=None):
        '''
        coo [L, A1, 3]
        edges [2, E]
        coo_2 [L, A2, 3]
        '''

        if coo_2 == None:
            coo_2 = coo  # else A2=A1
        coo_src = coo[edges[0]].unsqueeze(-3)  # [E, 1, A1, 3]
        coo_tgt = coo_2[edges[1]].unsqueeze(-2)  # [E, A2, 1, 3]
        d_coo = coo_tgt -  coo_src  # [E, A2, A1, 3]

        # direction
        directions = _normalize(d_coo.flatten(1,2))  # [E, A2*A1, 3]
        directions = directions.unsqueeze(-1) # [E, A2*A1, 3, 1]
        Q = Q[edges[0]].unsqueeze(-3)  # [E, 1, 3, 3]
        local_neighbor_directions = _normalize(torch.matmul(Q, directions).squeeze(-1))  # [L*30, A2*A1, 3]
        return local_neighbor_directions  # [E, A2*A1, 3]

    def get_node_features(self, coo, consistancy, Q):
        '''
        f_node:
            1.dihedrals [L, 14]
            2.adjacent directions in local coordinate system  [L, 2*3]
            3.rbf encoded inner distance [L, 4*n_bins]
            4.inner orientations [L, 4*3]
        '''

        dihedrals = self._dihedrals(coo, consistancy)  # [L, 14]
        local_adjacent_directions = self._adjacent_direction(coo, Q, consistancy)  # [L, 2, 3]
        inner_dist = self._inner_distance(coo)  # [L, 64]
        inner_directions = self._inner_direction(coo, Q)  # [L, 4, 3]
        
        f_node = torch.cat([
            dihedrals,
            local_adjacent_directions.flatten(1),
            inner_dist,
            inner_directions.flatten(1)
            ], -1)  # [L, 96]
        return f_node  # [L, 96]

    def get_edge_features(self, coo, rel_pos, edges, Q):
        '''
        f_edge:
            1.rbf encoded fullatom distance [E, (5+V)*(5+V)*n_bins]
            2.fullatom orientations [E, (5+V)*(5+V)*3]
            3.relative positional embedding [E, num_positional_embeddings]
            4.quaternion of transformation from neighbor Q to center Q [E, 4]
        '''

        pos_embeddings = self.positional_embeddings(rel_pos)  # [E, 16]
        f_dist = self._neighbor_dist(coo, edges)  # [E, 5*5*16=400]
        f_neighbor_directions = self._neighbor_direction(coo, edges, Q)  # [E, 5*5*3=75]
        f_quat = self._quaternion(Q, edges)  # [E, 4]
        
        f_edge = torch.cat([
            f_dist,
            f_neighbor_directions.flatten(1),
            pos_embeddings,
            f_quat
            ], -1)
        return f_edge  # [E, 495]

    def forward(self, coo, rel_pos, edges, consistancy, noise=0.):
        if noise!=0:
            coo += torch.normal(0, noise, size=coo.shape, device=default_device)
        
        with torch.no_grad():
            Q = self._local_coordinate_system(coo)
            f_node = self.get_node_features(coo, consistancy, Q)
            f_edge = self.get_edge_features(coo, rel_pos, edges, Q)
        
        if self.n_virtual_atoms != 0:
            virtual_atoms = self.get_virtual_atoms(coo, Q)
            f_edge = torch.cat([
                f_edge,
                self._neighbor_dist(coo, edges, virtual_atoms),
                self._neighbor_dist(virtual_atoms, edges, coo),
                self._neighbor_dist(virtual_atoms, edges, virtual_atoms)
            ], dim=-1)
            
        f_node = self.node_emb(self.node_norm(f_node))
        f_edge = self.edge_emb(self.edge_norm(f_edge))
        return f_node, f_edge