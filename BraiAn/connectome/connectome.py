import igraph as ig
import numpy as np
import pandas as pd

from .utils_bu import participation_coefficient

class Connectome:
    def __init__(self, A: pd.DataFrame,
                 isolated_vertices=True,
                 weighted=False, directed=False,
                 name="", weight_str="",
                 graph: ig.Graph=None) -> None:
        self.name = name
        self.weight_str = weight_str
        if graph is not None:
            self.G = graph
            self.is_weighted = graph.is_weighted()
            self.is_directed = graph.is_directed()
            self.vc = None
            return

        self.is_weighted = weighted
        self.is_directed = directed
        mode = "directed" if directed else "lower"
        if not isolated_vertices:
            # remove nodes/regions with no 
            A_binary = A != 0 if weighted else A
            is_not_isolated = np.maximum(A_binary, A_binary.T).any(axis=0)
            A = A.loc[is_not_isolated, is_not_isolated]
        if weighted:
            # zero in weighted matrix[i,j] corresponds to NO edge between i and j
            new_graph = ig.Graph.Weighted_Adjacency
        else:
            new_graph = ig.Graph.Adjacency
        
        self.G: ig.Graph = new_graph(A, mode=mode, loops=False)
        self.vc = None

    def cluster_regions(self, clustering_fun: callable, *args, **kwargs):
        self.vc = clustering_fun(self.G, *args, **kwargs)
        if not isinstance(self.vc, ig.VertexClustering):
            self.vc = self.vc.as_clustering()
        for i,cluster in enumerate(self.vc):
            for node in cluster:
                self.G.vs[node]["cluster"] = i
        return
    
    def participation_coefficient(self, weights=None):
        if self.is_directed or (self.is_weighted and weights):
            raise NotImplementedError("This centrality metric is not yet implemented for this type of connectomic.")
        self.G.vs["Participation coefficient"] = participation_coefficient(self.G, self.vc)
        return

    def region_subgraph(self, region: str, isolated_vertices=True):
        try:
            v = self.G.vs.select(name=region)[0]
        except Exception:
            # happens if connectome is created with isolated_vertices=False
            raise NameError(f"'{region}' has no correlation links with other regions!")
        G = self.G.subgraph_edges(v.incident(), delete_vertices=not isolated_vertices)
        return Connectome(None, None, None, None,
                          name=self.name, weight_str=self.weight_str, graph=G)