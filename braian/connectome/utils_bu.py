import igraph as ig
import numpy as np

# utils for Binary Undirected graphs

def participation_coefficient(G: ig.Graph, vc: ig.VertexClustering):
    n = G.vcount()
    neighborhood = G.neighborhood()
    neighborhood_size = G.neighborhood_size()
    clusters = [set(cluster) for cluster in vc]
    res = np.full(n, 0, dtype=float)
    for v in range(n):
        k_i = neighborhood_size[v]
        res_v = 0
        v_neighborhood = set(neighborhood[v])
        for cluster in clusters:
            # in-cluster neighborhood size
            k_is = k_i - len(v_neighborhood - cluster)
            res_v += (k_is/k_i)**2
        res[v] = 1 - res_v
    return res