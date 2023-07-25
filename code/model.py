import torch.nn as nn
from features import GraphProteinFeaturizer
from cgnn import CGNNBlock


class SPIN_CGNN(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_virtual_atoms) -> None:
        super().__init__()

        self.featurizer = GraphProteinFeaturizer(
            d_model=d_model, n_virtual_atoms=n_virtual_atoms)
        self.layers = nn.ModuleList(
            CGNNBlock(d_model, n_heads) for _ in range(n_layers))

        self.W_out = nn.Linear(d_model, 20)

        for name, p in self.named_parameters():
            if name in [
                'featurizer.virtual_atoms_direction',
            ]:
                print(f'skip init {name}')
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, coo, rel_pos, consistancy, edges, graph_id, soe, noise=0.):
        f_node, f_edge = self.featurizer(coo, rel_pos, edges, consistancy, noise)
        for layer in self.layers:
            f_node, f_edge = layer(f_node, f_edge, edges, graph_id, soe)
        logits = self.W_out(f_node)  # [L, 20]
        return {'logits': logits}