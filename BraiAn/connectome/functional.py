import igraph as ig
import numpy as np

from .cross_correlation import CrossCorrelation

class FunctionalConnectome:
    def __init__(self, cc: CrossCorrelation,
                 p_cutoff: float, r_cutoff: float,
                 negatives=False, isolated_vertices=True,
                 weighted=False) -> None:
        if negatives:
            above_threshold = (cc.p <= p_cutoff) & (cc.r.abs() >= r_cutoff)
        else:
            above_threshold = (cc.p <= p_cutoff) & (cc.r >= r_cutoff)
        self.r_cutoff = r_cutoff
        if weighted:
            A = cc.r.copy(deep=True)
            # zero in weighted matrix[i,j] corresponds to NO edge between i and j
            A[~above_threshold] = 0
            new_graph = ig.Graph.Weighted_Adjacency
        else:
            A = above_threshold
            new_graph = ig.Graph.Adjacency
        if not isolated_vertices:
            # remove nodes/regions with no connections
            A = A.loc[above_threshold.any(axis=0), above_threshold.any(axis=0)]
        self.G: ig.Graph = new_graph(A, mode="lower", loops=False)
        self.__add_vertices_attributes(cc)
        self.__add_edges_attributes(cc)

    def __add_vertices_attributes(self, cc: CrossCorrelation):
        for v in self.G.vs:
            v_name = v["name"]
            v["is_undefined"] = cc.r[v_name].isna().all()
            v["upper_region"] = cc.upper_regions[v_name]
    
    def __add_edges_attributes(self, cc: CrossCorrelation):
        for e in self.G.es:
            e["p-value"] = cc.p.loc[e.source_vertex["name"], e.target_vertex["name"]]
            if not self.G.is_weighted():
                e["r-value"] = cc.r.loc[e.source_vertex["name"], e.target_vertex["name"]]