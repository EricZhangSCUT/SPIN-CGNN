import torch
import torch.nn as nn
import einops as ein
from torch_scatter import scatter_mean, scatter_sum


default_activation = nn.GELU()
default_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
neginf = torch.tensor(-float('inf')).to(default_device)

from torch_scatter import scatter_max
from torch_scatter.utils import broadcast
def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
    '''modified official source code: add default eps'''
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max(src, index, dim=dim)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_sum(recentered_scores_exp, index, dim)
    normalizing_constants = sum_per_index.gather(dim, index)
    return recentered_scores_exp.div(normalizing_constants.add(eps)) # modification


class CGNNBlock(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.edge_update = EdgeUpdate(d_model)
        self.node_update = NodeUpdate(d_model, n_heads)

    def forward(self, f_node, f_edge, edge_index, graph_id, soe):
        f_edge = self.edge_update(f_node, f_edge, edge_index, soe)
        f_node = self.node_update(f_node, f_edge, edge_index, graph_id)
        return f_node, f_edge


class NodeUpdate(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.incoming_attn = GraphAttentionUpdate(
            d_model, kdim=d_model, vdim=3*d_model, n_heads=n_heads
        )

        self.graph_pooling = AttnGraphPooling(d_model)
        self.graph_update = LinearUpdate(2*d_model, d_model)
        
        self.sk = KernelSelection(d_model, n_fea=2)
        self.ff = PositionwiseFeedForward(d_model)

    def forward(self, f_node, f_edge, edge_index, graph_id):
        '''
        f_node [L, D]
        f_edge [E, D]
        edge_index [2, E]
        '''
        tgt_idx, src_idx = edge_index
        f_src_e = torch.cat([f_node[src_idx], f_edge, f_node[tgt_idx]], dim=-1)
        
        f_node_in = self.incoming_attn(
            f=f_node, k=f_edge, v=f_src_e, agg_idx=tgt_idx
        )
        
        f_graph = self.graph_pooling(f_node, graph_id)
        f_node_global = torch.cat([f_node, f_graph[graph_id]], dim=-1)  # [L, 2*D]
        f_node_global = self.graph_update(f_node_global)  # [L, D]
        f_node_update = [f_node_in, f_node_global]
        return self.ff(self.sk(f_node, f_node_update))


class EdgeUpdate(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        self.connection = nn.Linear(3*d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.soe_update = TriangleUpdate(d_model)
        self.symmetric_update = SymmetricUpdate(d_model)
        self.connection_update = LinearUpdate(d_model, residual=False)
        
        self.sk = KernelSelection(d_model, n_fea=3)  
        self.ff = PositionwiseFeedForward(d_model)

    def forward(self, f_node, f_edge, edges, soe):
        '''
        f_node [L, D]
        f_edge [E, D]
        edges [2, E]
        soe [T, 3]  [ik,kj -> ij]
        '''

        f_edge_connect = torch.cat([f_node[edges[0]],  # [E, D]
                                    f_edge,  # [E, D]
                                    f_node[edges[1]]], -1)  # [E, D]
        f_edge_connect = self.norm(self.connection(f_edge_connect))
        
        f_edge_update = [self.connection_update(f_edge_connect)]
        f_edge_update.append(self.symmetric_update(f_edge_connect))
        f_edge_update.append(self.soe_update(f_edge_connect, soe))
        return self.ff(self.sk(f_edge, f_edge_update))

class TriangleUpdate(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 act=default_activation) -> None:
        super().__init__()
        self.activation = act
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.triangle = nn.Linear(3*d_model, d_model)
    
    def forward(self, f_edge, triangle):
        '''
        f_edge [E, D]
        triangle [T, 3]  [edge0, edge1 -> edge2]
        '''  
        f_triangle = torch.flatten(f_edge[triangle], 1)
        f_triangle = self.activation(self.triangle(f_triangle))
        f_triangle = self.norm(self.dropout(scatter_mean(
            f_triangle, triangle[:, 2], dim=0, dim_size=len(f_edge))))  # [E, D]
        return f_triangle
    
    
class SymmetricUpdate(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.symmetric = LinearUpdate(2*d_model, d_model, residual=False)
    
    def forward(self, f_edge):
        '''
        f_edge [E, D]
        '''
        n_edge = len(f_edge) // 2
        f_symmetric = torch.cat([f_edge[n_edge:], f_edge[:n_edge]], dim=0)
        f_symmetric = torch.cat([f_edge, f_symmetric], dim=-1)
        f_symmetric = self.symmetric(f_symmetric)
        return f_symmetric

    
class KernelSelection(nn.Module):
    def __init__(self, d_model, n_fea, sk_ratio=4,
                 act=default_activation) -> None:
        super().__init__()
        self.n_fea = n_fea
        self.activation = act
        self.norm_fea = nn.LayerNorm(d_model)
        self.squeeze = nn.Linear(d_model, d_model//sk_ratio)
        self.excitation = nn.Linear(d_model//sk_ratio, n_fea*d_model)
        self.norm_update = nn.LayerNorm(d_model)
        
    def forward(self, f, f_update):
        '''
        f [(L or E), D]
        f_update [F * [(L or E), D]]  F groups of update feature in a list
        '''
        f_update = torch.stack(f_update, dim=1)  # [E, F, D]
        update_attn = self.squeeze(self.norm_fea(f_update.sum(1)))  # [E, D]
        update_attn = self.excitation(self.activation(update_attn))
        update_attn = ein.rearrange(update_attn, 'e (h d) -> e h d', h=self.n_fea)
        update_attn = torch.softmax(update_attn, dim=1)  # [E, F, D]
        f_update = (f_update * update_attn).sum(1)  # [E, D]
        return self.norm_update(f + f_update)


class GraphAttentionUpdate(nn.Module):
    def __init__(self, d_model, kdim=None, vdim=None, n_heads=4,
                 dropout=0.1, act=default_activation) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_dim = d_model//n_heads
        self.scale = self.attn_dim ** -0.5
        self.activation = act
        self.p_mask = dropout
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        kdim = d_model if kdim == None else kdim
        vdim = d_model if vdim == None else vdim
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(kdim, d_model)
        self.value = nn.Linear(vdim, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, f, k, v, agg_idx):
        '''
        f [L, D], k [L_k, D_k], v [L_v, D_v]
        agg_idx [L_k] scatter v to f
        '''

        q = self.query(f)  # [L, D]
        k = self.key(k)  # [L_k, D]
        v = self.value(v)  # [L_v, D]

        q = ein.rearrange(q, 'l (h d) -> l h d', h=self.n_heads)
        k = ein.rearrange(k, 'e (h d) -> e h d', h=self.n_heads)
        v = ein.rearrange(v, 'e (h d) -> e h d', h=self.n_heads)

        attn = torch.einsum(
            'ehd,ehd->eh', q[agg_idx], k) * self.scale  # [L_k, H]
        ### dropout on attention
        if self.dropout.training:
            attn = attn.masked_fill(torch.rand_like(attn)<self.p_mask, neginf)
        attn = scatter_softmax(attn, agg_idx, dim=0)  # [L_k, H]

        v = v * attn.unsqueeze(-1)
        v = ein.rearrange(v, 'e h d -> e (h d)')
        v = scatter_sum(v, agg_idx, dim=0, dim_size=len(f))  # [L, D]
        return self.norm(self.dropout(self.out(v)))


class AttnGraphPooling(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 act=default_activation) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = act
        self.p_mask = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, f_node, graph_id):
        '''
        f_node [L, D]
        graph_id [L]
        '''
        attn = self.key(f_node)
        if self.dropout.training:
            attn = attn.masked_fill(torch.rand_like(attn)<self.p_mask, neginf)
        attn = scatter_softmax(attn, graph_id, dim=0)
        f_graph = scatter_sum(self.value(f_node)*attn, graph_id, dim=0)  # [G,D]
        f_graph = self.norm(self.dropout(f_graph))
        return f_graph


class LinearUpdate(nn.Module):
    def __init__(self, d_model, d_out=None, residual=None,
                 dropout=0.1, act=default_activation) -> None:
        '''linear->act->drop(->add)->norm'''
        super().__init__()
        d_out = d_model if d_out == None else d_out
        self.residual = (d_model == d_out) if residual == None else residual
        self.linear = nn.Linear(d_model, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = act

    def forward(self, f_in):
        if self.residual:
            return self.norm(f_in + self.dropout(self.activation(self.linear(f_in))))
        else:
            return self.norm(self.dropout(self.activation(self.linear(f_in))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1) -> None:
        super().__init__()
        d_ff = 2*d_model if d_ff == None else d_ff
        self.ff1 = GatedLinearUnit(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, f):
        f_update = self.ff2(self.dropout(self.ff1(f)))
        return self.norm(f + self.dropout(f_update))


class GatedLinearUnit(nn.Module):
    def __init__(self, d_model, d_out=None, act=default_activation) -> None:
        super().__init__()
        d_out = d_model if d_out == None else d_out
        self.linear = nn.Linear(d_model, 2*d_out, bias=False)
        self.activation = act
    
    def forward(self, f):
        value, gate = torch.chunk(self.linear(f), 2, dim=-1)
        return self.activation(gate) * value
