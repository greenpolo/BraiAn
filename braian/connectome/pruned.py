from braian.connectome.connectome import Connectome
from braian.connectome.connectome_adjacency import ConnectomeAdjacency
from braian.connectome.cross_correlation import CrossCorrelation
from braian.connectome.structural import StructuralConnectome
from braian.utils import get_indices_where

import numpy as np

class PrunedConnectomics(Connectome):
    def __init__(self, sc: StructuralConnectome, cc: CrossCorrelation,
                 p_cutoff: float, r_cutoff: float,
                 negatives=False, weighted=True,
                 isolated_vertices=False) -> None:
        self.p_cutoff = p_cutoff
        self.r_cutoff = r_cutoff
        if negatives:
            self.fc_mask = (cc.p.data <= self.p_cutoff) & (cc.r.data.abs() >= self.r_cutoff)
        else:
            self.fc_mask = (cc.p.data <= self.p_cutoff) & (cc.r.data >= self.r_cutoff)
        mask = sc.mask & self.fc_mask
        # If make_comparable() or remove_insufficient_regions() were called on cc,
        # the functional connectome wouldn't be compatible with the structural connectome.
        # Hence, we have to select the same regions
        try:
            mask = mask.loc[cc.r.data.columns, cc.r.data.index]
        except KeyError as e:
            missing_region = str(e).split("'")[1]
            raise ValueError(f"The region '{missing_region}' is present in the CrossCorrelation but absent in the StructuralConnectome!")
        if weighted:
            r_masked = cc.r.mask(mask)
            self.A = r_masked.data.fillna(0, inplace=False)
        else:
            self.A = mask
        super().__init__(self.A,
                         isolated_vertices=isolated_vertices,
                         weighted=weighted,
                         directed=True,
                         name=f"{cc.name} - Pruned",
                         weight_str="Pearson r")
        self.__add_vertices_attributes(cc.r)
        # self.__add_edges_attribute("r-value", cc.r)
        self.__add_edges_attribute("p-value", cc.p)
        self.__add_edges_attribute("normalized connection density", sc.A)
    
    def get_functional_neighbors_distances(self):
        # we don't make care about direction.
        # We take the minimum distance of the two directions
        # e.g. if A -- Z functionally, but:
        #   * A --> ... --> Z does not exist in pruned connectome
        #   * Z --> ... --> A exists in pruned connectome
        # ==> we take the length of path Z to A
        fc_mask = self.fc_mask.copy()
        fc_mask.values[*np.triu_indices_from(fc_mask)] = False
        es = get_indices_where(fc_mask)
        ds = self.G.distances()
        ds_pruned = []
        for source, target in es:
            try:
                source_id = self.G.vs.select(name=source)[0].index #self.G.vs.select(name=e.source_vertex["name"])[0].index
                target_id = self.G.vs.select(name=target)[0].index #self.G.vs.select(name=e.target_vertex["name"])[0].index
            except IndexError:
                ds_pruned.append(np.inf)
                continue
            d = min(ds[source_id][target_id], ds[target_id][source_id])
            ds_pruned.append(d)
        return np.asarray(ds_pruned)
    
    def get_fully_pruned_edges(self, sc: StructuralConnectome):
        # returns the list of edges (A -- B) that are no longer directly connected.
        # neither in (A -> B) nor in (A <- B)
        pruned_edges_mask = ~sc.mask & self.fc_mask & ~sc.mask.T
        pruned_edges_mask.values[*np.triu_indices_from(pruned_edges_mask)] = False
        es = get_indices_where(pruned_edges_mask)
        return es

    def get_pruned_edges(self, sc: StructuralConnectome):
        # returns the list of directed links (A -> B) that are no longer directly connected.
        pruned_links_mask = ~sc.mask & self.fc_mask
        es = get_indices_where(pruned_links_mask)
        return es

    def __add_vertices_attributes(self, M: ConnectomeAdjacency):
        for v in self.G.vs:
            v_name = v["name"]
            v["upper_region"] = M.upper_regions[v_name]
    
    def __add_edges_attribute(self, attr: str, M: ConnectomeAdjacency):
        for e in self.G.es:
            e[attr] = M.data.loc[e.source_vertex["name"], e.target_vertex["name"]]