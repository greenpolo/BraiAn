import igraph as ig
import numpy as np
import pandas as pd
from typing import Self

from braian.connectome.utils_bu import participation_coefficient
from braian.ontology import AllenBrainOntology

class Connectome:
    def __init__(self, A: pd.DataFrame,
                 isolated_vertices=True,
                 weighted=False, directed=False,
                 name="", weight_str="",
                 graph: ig.Graph=None) -> None:
        """
        _summary_

        Parameters
        ----------
        A
            _description_
        isolated_vertices, optional
            _description_, by default True
        weighted, optional
            _description_, by default False
        directed, optional
            _description_, by default False
        name, optional
            _description_, by default ""
        weight_str, optional
            _description_, by default ""
        graph, optional
            _description_, by default None
        """
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

    def collapse_region(self, atlas: AllenBrainOntology, region_acronym: str) -> Self:
        # returns a Connectome with all subrregions of region_acronym collapsed in one single node
        # NOTE: if connectome is weighted, the weights won't be retained
        all_subregions = atlas.list_all_subregions(region_acronym) # region_acronym included
        v_collapsed = []
        v_mapping = np.full(self.G.vcount(), 0)
        for v in self.G.vs:
            if v["name"] in all_subregions:
                v_mapping[v.index] = 0
                v_collapsed.append(v["name"])
            else:
                v_mapping[v.index] = (v.index+1)-len(v_collapsed)
        if len(v_collapsed) == 1:
            return Connectome(None, None, None, None, graph=self.G, name=self.name, weight_str=self.weight_str)
        G = self.G.copy()
        def contract_names(names: list[str]):
            if len(names) == 1:
                return names[0]
            return region_acronym

        def contract_upper_regions(upper_regions: list[str]):
            if all(region == upper_regions[0] for region in upper_regions):
                return upper_regions[0]
            return "root"

        G.vs[0]["collapsed"] = "+".join(v_collapsed)
        G.contract_vertices(v_mapping, dict(name=contract_names, upper_region=contract_upper_regions, collapsed="last"))
        G = G.simplify(multiple=True, loops=True, combine_edges=None) # loses all attirbutes (weight, p-value, normalized connection density)
        collapsed = Connectome(None, None, None, None, graph=G, name=f"{self.name} & Collapsed", weight_str=self.weight_str)
        return collapsed