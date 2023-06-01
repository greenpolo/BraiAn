import igraph as ig
import numpy as np

from .cross_correlation import CrossCorrelation

class FunctionalConnectome:
    def __init__(self, cc: CrossCorrelation,
                 p_cutoff: float, r_cutoff: float,
                 negatives=False, isolated_vertices=True,
                 weighted=False, graph: ig.Graph=None, n=None, name="") -> None:
        self.name = cc.name if cc is not None else name
        self.n = cc.n if cc else n
        if graph is not None:
            self.G = graph
            self.r_cutoff = r_cutoff
            self.vc = None
            return

        if negatives:
            above_threshold = (cc.p.A <= p_cutoff) & (cc.r.A.abs() >= r_cutoff)
        else:
            above_threshold = (cc.p.A <= p_cutoff) & (cc.r.A >= r_cutoff)
        self.r_cutoff = r_cutoff
        if weighted:
            A = cc.r.A.copy(deep=True)
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
        self.vc = None
        self.__add_vertices_attributes(cc)
        self.__add_edges_attributes(cc)
    
    def cluster_regions(self, clustering_fun: callable, *args, **kwargs):
        self.vc = clustering_fun(self.G, *args, **kwargs)
        if not isinstance(self.vc, ig.VertexClustering):
            self.vc = self.vc.as_clustering()
        for i,cluster in enumerate(self.vc):
            for node in cluster:
                self.G.vs[node]["cluster"] = i
        return

    def region_subgraph(self, region: str, isolated_vertices=True):
        try:
            v = self.G.vs.select(name=region)[0]
        except Exception:
            # happens if connectome is created with isolated_vertices=False
            raise NameError(f"'{region}' has no correlation links with other regions!")
        G = self.G.subgraph_edges(v.incident(), delete_vertices=not isolated_vertices)
        return FunctionalConnectome(None, None, self.r_cutoff, graph=G, n=self.n, name=self.name)

    def __add_vertices_attributes(self, cc: CrossCorrelation):
        for v in self.G.vs:
            v_name = v["name"]
            v["is_undefined"] = cc.r.A[v_name].isna().all()
            v["upper_region"] = cc.r.upper_regions[v_name]
    
    def __add_edges_attributes(self, cc: CrossCorrelation):
        for e in self.G.es:
            e["p-value"] = cc.p.A.loc[e.source_vertex["name"], e.target_vertex["name"]]
            if not self.G.is_weighted():
                e["r-value"] = cc.r.A.loc[e.source_vertex["name"], e.target_vertex["name"]]