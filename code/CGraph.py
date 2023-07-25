import numpy as np


class EdgeIndex(object):
    '''
    find the index of the edge from node i to node j in edges
    '''
    def __init__(self, edge) -> None:
        self.num_nodes = edge.max() + 1
        self.index = {code:idx for idx, code in enumerate(self.encode(edge[0], edge[1]))}
    
    def encode(self, i, j):
        '''
        encode the edge from node i to node j as i*l+j, where l is the number of nodes
        '''
        return i * self.num_nodes + j
    
    def find(self, i, j):
        '''
        find the index of the edge from node i to node j in edges
        '''
        return self.index.get(self.encode(i, j))
    

class ContactGraph(object):
    def __init__(self, edges=None, triangles=None, coo=None, cutoff=None) -> None:
        '''
        edges [2, E] undirected edges, [i, j] idx of nodes
        triangles [T, 3] node sets [i, j, k], each node is neighbor node of others
        coo [L, 3]
        cutoff int
        '''
        if type(edges)==np.ndarray:
            self.edges = edges
        elif type(coo)==np.ndarray and cutoff!=None:
            self.from_coo(coo, cutoff)
        else:
            raise SystemExit('No edges nor coo and cutoff for CGraph construction!!!')
        
        if type(triangles)==np.ndarray:
            self.triangles = triangles
        else:
            self.get_triangles()
    
    def from_coo(self, coo, cutoff):
        '''
        construct contact graph from coo with cutoff
        coo [L, 3]
        cutoff int
        edges [2, E] undirected edges
        '''
        dist_map = np.linalg.norm(coo[:, None] - coo[None, :], axis=-1)
        edges = np.array((dist_map < cutoff).nonzero())
        self.edges = to_undirect(edges)
    
    def get_triangles(self):
        '''
        edges [2, E] undirected edges
        triangles [T, 3]
        '''
        num_nodes = self.edges.max()+1
        triangles = []
        for i in range(num_nodes-2):
            i_neighbor = self.edges[1][self.edges[0]==i]
            for j in i_neighbor:
                j_neighbor = self.edges[1][self.edges[0]==j]
                shared_neighbor = set(i_neighbor) & set(j_neighbor)
                for k in shared_neighbor:
                    triangles.append([i,j,k])
        self.triangles = np.array(triangles)

    def get_triangle_edges(self):
        '''
        get triangle_edges index from triangles and edges
        edges [2, E] undirected edges
        triangles [T, 3]
        triangle_edges [T, 3] '''
        finder = EdgeIndex(self.edges)
        trianlge_edges = []
        for i, j, k in self.triangles:
            trianlge_edges.append([finder.find(i,j),
                                   finder.find(i,k),
                                   finder.find(j,k)]) # [ij, ik, jk]
        trianlge_edges = np.array(trianlge_edges) # [T, 3]
        return trianlge_edges  # [T, 3]


# apply triangle_to_soe on triangle_edges to get all soe
# triangle_edges [ij, ik, jk, ji, ki, kj]
Tri_to_SOE = np.array([
    [4, 0, 5], [3, 1, 2],  # [ki, ij -> kj], [ji, ik -> jk]
    [0, 2, 1], [5, 3, 4],  # [ij, jk -> ik], [kj, ji -> ki]
    [1, 5, 0], [2, 4, 3],  # [ik, kj -> ij], [jk, ki -> ji]
    ])

def soe_from_triedges(trianlge_edges):
    '''
    get second order edge index from triangle_edges
    triangle_edges [T, 3] [ij, ik, jk]
    soe [T, 6, 3] index on edges, [e1, e2, e3] representing e1,e2->e3
    for example: edges[e1] is [i, k], edges[e2] is [k, j], edges[e3] is [i, j],
                 and ik,kj is the second order edge from i to j though k, for ij update
    '''
    num_edges = trianlge_edges.max()+1
    trianlge_edges = np.concatenate(
            [trianlge_edges, trianlge_edges + num_edges],
            axis=-1) # [T, 6]
    soe = trianlge_edges[:, Tri_to_SOE]  # [T, 6, 3]
    return soe

def to_undirect(edges):
    return edges[:, edges[0] < edges[1]] # unique undirect edges

def to_direct(edges):
    '''
    concate the reverse edges of input undirect edges, return direct edges
    so that the reverse edge ji of edge ij can be index by +num_edges
    '''
    return np.concatenate([edges, edges[[1,0]]], axis=-1) # direct edges