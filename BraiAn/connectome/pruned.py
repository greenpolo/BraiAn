from .connectome import Connectome
from .connectome_adjacency import ConnectomeAdjacency
from .cross_correlation import CrossCorrelation
from .structural import StructuralConnectome

class PrunedConnectomics(Connectome):
    def __init__(self, sc: StructuralConnectome, cc: CrossCorrelation,
                 r_cutoff: float, p_cutoff: float,
                 isolated_vertices=False, weighted=True) -> None:
        fc_mask = (cc.p.data <= p_cutoff) & (cc.r.data >= r_cutoff)
        mask = sc.mask & fc_mask
        if weighted:
            r_masked = cc.r.mask(mask)
            A = r_masked.data.fillna(0, inplace=False)
        else:
            A = mask
        super().__init__(A,
                         isolated_vertices=isolated_vertices,
                         weighted=weighted,
                         directed=True,
                         name=f"{cc.name} - Pruned")
        self.__add_vertices_attributes(cc.r)
        # self.__add_edges_attribute("r-value", cc.r)
        self.__add_edges_attribute("p-value", cc.p)
        self.__add_edges_attribute("normalized connection density", sc.A)

    def __add_vertices_attributes(self, M: ConnectomeAdjacency):
        for v in self.G.vs:
            v_name = v["name"]
            v["upper_region"] = M.upper_regions[v_name]
    
    def __add_edges_attribute(self, attr: str, M: ConnectomeAdjacency):
        for e in self.G.es:
            e[attr] = M.data.loc[e.source_vertex["name"], e.target_vertex["name"]]