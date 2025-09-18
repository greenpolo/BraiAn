import igraph as ig

from collections.abc import Iterable, Sequence

def remove_branch(g: ig.Graph,
                   branch_acronyms: Iterable[str],
                   attr: str="name"):
    to_remove = []
    for branch_to_remove in g.vs.select(**{attr+"_in": branch_acronyms}):
        whole_branch = g.dfs(branch_to_remove.index, mode="out")[0]
        to_remove.extend(whole_branch)
    g.delete_vertices(to_remove)

def blacklist_regions(g: ig.Graph,
                       blacklisted: Iterable[str],
                       unreferenced: Iterable[str]=None,
                       attr: str="name"):
    for region, depth, parent in g.dfsiter(0, mode="out", advanced=True):
        region["blacklisted"] = (parent is not None and parent["blacklisted"]) or region[attr] in blacklisted
        if unreferenced:
            region["referenced"] = (parent is None or parent["referenced"]) and region[attr] not in unreferenced

def select_regions(g: ig.Graph, regions: Iterable, attr: str="name"):
    for region in g.dfsiter(0, mode="out"):
        region["selected"] = region[attr] in regions

def _attr(v: ig.Vertex, mode: str):
    if mode == "index":
        return v.index
    else:
        return v[mode]

def minimum_treecover(vs: Sequence[ig.Vertex]):
    vs = set(vs)
    vs_ = {parent if all(child in vs for child in parent.neighbors(mode="out")) else v
           for v in vs for parent in v.neighbors(mode="in")}
    if vs_ == vs:
        return vs
    return minimum_treecover(vs_)

def _selection_cut(root: ig.Vertex,
                   attr: str,
                   advanced: bool,      # returns also the list of those leaves left out
                   c: list[list]=None,  # covered
                   u: list=None):       # uncovered
    if c is None:
        c = [[]]
    if u is None:
        u = []
    if root["selected"]:
        c[-1].append(_attr(root, attr)) # append to the last contiguous list of nodes
    elif root.outdegree() == 0:
        if advanced:
            u.append(root)
        if len(c[-1]) != 0: # root is a leaf and the current last contiguous list of nodes is not empty
            c.append([])    # create a new empty list of nodes
    else:
        for v in root.neighbors(mode="out"):
            _selection_cut(root=v,attr=attr,advanced=advanced,c=c,u=u)
    return (c,[_attr(v, attr) for v in minimum_treecover(u)]) if advanced else c

def brainwide_selection(g: ig.GraphBase, attr="index", advanced: bool=False) -> list[list[str]]:
    if "selected" not in g.vs.attributes():
        raise ValueError("The current ontology has no active selection.")
    return _selection_cut(g.vs[0], attr=attr, advanced=advanced)

def _is_selection_leaves_cover(root: ig.Vertex):
    if root["selected"]:
        return True
    if root.outdegree() == 0:
        return False
    return all(_is_selection_leaves_cover(v) for v in root.neighbors(mode="out"))

def is_brainwide_selection(g: ig.GraphBase):
    if "selected" not in g.vs[0].attributes():
        raise ValueError("The current ontology has no active selection.")
    return _is_selection_leaves_cover(g.vs[0])