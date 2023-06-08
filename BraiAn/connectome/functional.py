import igraph as ig
import numpy as np

from .connectome import Connectome
from .connectome_adjacency import ConnectomeAdjacency
from .cross_correlation import CrossCorrelation

class FunctionalConnectome(Connectome):
    def __init__(self, cc: CrossCorrelation,
                 p_cutoff: float, r_cutoff: float,
                 negatives=False, weighted=False, n=None,
                 graph=None, 
                 **kwargs) -> None:
        self.n = cc.n if cc is not None else n
        if graph is not None:
            self.r_cutoff = r_cutoff
            return super().__init__(None, None, None, None, graph=graph, **kwargs)

        if negatives:
            above_threshold = (cc.p.data <= p_cutoff) & (cc.r.data.abs() >= r_cutoff)
        else:
            above_threshold = (cc.p.data <= p_cutoff) & (cc.r.data >= r_cutoff)

        self.r_cutoff = r_cutoff
        if weighted:
            A = cc.r.data.copy(deep=True)
            # zero in weighted matrix[i,j] corresponds to NO edge between i and j
            A[~above_threshold] = 0
        else:
            A = above_threshold
        super().__init__(A, weighted=weighted, directed=False, name=cc.name, weight_str="Pearson r", **kwargs)
        self.__add_vertices_attributes(cc.r)
        self.__add_edges_attributes(cc)

    def __add_vertices_attributes(self, M: ConnectomeAdjacency):
        for v in self.G.vs:
            v_name = v["name"]
            v["is_undefined"] = M.data[v_name].isna().all()
            v["upper_region"] = M.upper_regions[v_name]
    
    def __add_edges_attributes(self, cc: CrossCorrelation):
        for e in self.G.es:
            e["p-value"] = cc.p.data.loc[e.source_vertex["name"], e.target_vertex["name"]]
            if not self.G.is_weighted():
                e["r-value"] = cc.r.data.loc[e.source_vertex["name"], e.target_vertex["name"]]