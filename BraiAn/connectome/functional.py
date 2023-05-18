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
        A = cc.r.copy(deep=True)
        # zero in weighted matrix[i,j] corresponds to NO edge between i and j
        A[~above_threshold] = 0
        if not isolated_vertices:
            # remove nodes/regions with no connections
            A = A.loc[above_threshold.any(axis=0), above_threshold.any(axis=0)]
        if weighted:
            self.G: ig.Graph = ig.Graph.Weighted_Adjacency(A.values, mode="lower", loops=False)
        else:
            self.G: ig.Graph = ig.Graph.Adjacency(above_threshold, mode="lower", loops=False)
        self.G.vs["label"] = list(A.index)
        self.G.vs["is_undefined"] = cc.r.isna().all().values
        for e in self.G.es:
            e["p-value"] = cc.p.loc[e.source_vertex["label"], e.target_vertex["label"]]
            if not weighted:
                e["r-value"] = cc.r.loc[e.source_vertex["label"], e.target_vertex["label"]]

        self.G.vs["upper_region"] = [cc.upper_regions[v["label"]] for v in self.G.vs]