import brainglobe_atlasapi as bga
import numpy as np
import warnings

from braian.utils import deprecated
from collections import OrderedDict
from copy import deepcopy
from treelib import Tree, Node
from typing import Sequence, Iterable, Callable, Container, Literal

__all__ = ["AtlasOntology"]

class RegionNode(Node):
    def __init__(self,
                 acronym: str,
                 identifier: int,
                 name: str,
                 rgb_triplet: Sequence[int],
                 blacklisted: bool,
                 selected: bool):
        super().__init__(
            tag=acronym,
            identifier=identifier,
            expanded=blacklisted
        )
        self.name = name
        self.rgb_triplet = rgb_triplet
        self.selected = selected

    @property
    def acronym(self):
        return self.tag

    @property
    def blacklisted(self):
        return self.expanded

    @blacklisted.setter
    def blacklisted(self, value):
        self.expanded = value

    @property
    def id(self):
        return self.identifier

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = [
            f"acronym={repr(self.tag)}",
            f"id={repr(self.identifier)}",
            f"name={repr(self.name)}",
            f"rgb_triplet={repr(self.rgb_triplet)}",
            f"blacklisted={repr(self.expanded)}",
            f"selected={repr(self.selected)}"
        ]
        return f"{name}({', '.join(kwargs)})"

def convert_to_region_nodes(atlas: bga.BrainGlobeAtlas) -> Tree:
    tree = Tree(node_class=RegionNode)
    old_nodes: dict[int,Node] = atlas.structures.tree.nodes
    for nid, old_node in old_nodes.items():
    # ids: list[int] = atlas.structures.tree.expand_tree(sorting=False)
    # for nid in ids:
    #     old_node = atlas.structures.tree[nid]
        new_node = RegionNode(
            acronym=atlas.structures[nid]["acronym"],
            identifier=nid,
            name=atlas.structures[nid]["name"],
            rgb_triplet=atlas.structures[nid]["rgb_triplet"],
            blacklisted=False,
            selected=False
            )
        parent_id = old_node.predecessor(atlas.structures.tree.identifier)
        tree.add_node(new_node, parent=parent_id)
    return tree

def first_subtrees(tree: Tree, func: Callable[[Node],bool]) -> Iterable[Node]:
    def first(n: Node) -> bool:
        return func(n) and (n.is_root(tree.identifier) or not func(tree.parent(n.identifier)))
    return tree.filter_nodes(first)

def minimum_treecover(tree: Tree, nids: Iterable) -> set:
    nids = set(nids)
    nids_ = {tree.parent(nid).identifier if all(child.identifier in nids for child in tree.children(tree.parent(nid).identifier)) else nid
            for nid in nids}
    if nids_ == nids:
        return nids
    return minimum_treecover(tree, nids_)

class AtlasOntology:
    def __init__(self,
                 atlas: str|bga.BrainGlobeAtlas,
                 blacklisted: Iterable[str]=[],
                 unreferenced: bool=False):
        if not isinstance(atlas, bga.BrainGlobeAtlas):
            atlas = bga.BrainGlobeAtlas(atlas)
        self._atlas: bga.BrainGlobeAtlas = atlas
        self._acronym2id: dict[str,int] = deepcopy(atlas.structures.acronym_to_id_map)
        self._tree_full: Tree = convert_to_region_nodes(self._atlas)
        self._tree: Tree = Tree(self._tree_full, deep=False)
        self._selected: bool = False
        if not unreferenced:
            unreferenced_regions = self._unreferenced(key="id") # in allen_mouse_50um, only 'RSPd4 (545)' is unreferenced
            self.blacklist(unreferenced_regions, unreferenced=True)
        if blacklisted:
            blacklisted_ids = self._to_ids(blacklisted, unreferenced=True, check_all=False)
            self.blacklist(blacklisted_ids, unreferenced=False)

    def _nodes_to_attr(self,
                       regions: Iterable[RegionNode],
                       *,
                       attr: Literal["acronym","id"]
                       ) -> list:
        if attr not in ("acronym", "tag", "id", "identifier"):
            raise ValueError(f"Unknown unique identifier of brain regions: '{attr}'")
        return [r.__getattribute__(attr) for r in regions]

    def _unreferenced(self,
                      *,
                      key: Literal["id","acronym"]="acronym") -> list:
        atlas_meshes_dir = self._atlas.root_dir/"meshes"
        if not atlas_meshes_dir.exists():
            raise ValueError(f"BrainGlobe atlas meshes not downloaded: '{self._atlas.atlas_name}'")
        regions_w_annotation = [int(p.stem) for p in atlas_meshes_dir.iterdir() if p.suffix == ".obj"]
        regions_wo_annotation = self._tree_full.filter_nodes(lambda n: n.identifier not in regions_w_annotation)
        return self._nodes_to_attr(regions_wo_annotation, attr=key)

    def _to_id(self,
               region: int|str,
               *,
               unreferenced: bool) -> int:
        """
        Checks the existance of a region and returns the corresponding IDs.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        unreferenced
            If `True`, it accepts a `region` that has no reference in the atlas annotations.

        Returns
        -------
        :
            A region ID.

        Raises
        ------
        KeyError
            If the given `region` is not found in the ontology.
        """
        id = self._acronym2id.get(region)
        if id is None:
            try:
                id = int(region)
            except ValueError:
                pass
        if id is None:
            raise KeyError(f"Region not found ({repr(region)})")
        if not unreferenced and id not in self._tree:
            raise KeyError(f"Region not found ({repr(region)}). The region may be unreferenced.")
        return id

    def _to_ids(self,
                regions: Sequence,
                *,
                unreferenced: bool, # include them or no?
                duplicated: bool, # check for duplicated regions or not?
                check_all: bool) -> list[int]:
        """
        Checks the existance of a list of regions and returns the corresponding IDs.

        Parameters
        ----------
        regions
            A sequence of regions, identified by their ID or their acronym.
        unreferenced
            If `True`, it accepts `regions` that have no reference in the atlas annotations.
        duplicated
            If `True`, it accepts the same region to appear twice (or more) in `regions`.\
            Otherwise, it raises `ValueError`.
        check_all
            If `True`, the returned list may contain some `None` values, corresponding to unknown regions.

        Returns
        -------
        :
            A list of region IDs.

        Raises
        ------
        ValueError
            If `duplicated=False` and there is at least one region that appears twice in `regions`.
        KeyError
            If `check_all=False` and at least one of the given `regions` is not found in the ontology.
        """
        ids = []
        for region in regions:
            try:
                id = self._to_id(region, unreferenced=unreferenced)
            except KeyError:
                if check_all:
                    id = None
                else:
                    raise
            ids.append(id)
        if not duplicated and len(ids) > 1:
            duplicates = np.rec.find_duplicate(np.array(ids)) # may contain None
            duplicates = [self._tree_full[id].tag for id in duplicates if id is not None]
            if len(duplicates) > 0:
                raise ValueError(f"Duplicates detected in the list of regions: '{', '.join(duplicates)}'")
        return ids

    def _to_nodes(self,
                  regions: Container[int],
                  unreferenced: bool,
                  duplicated: bool) -> list[RegionNode]:
        ids = self._to_ids(regions, unreferenced=unreferenced, duplicated=duplicated, check_all=False)
        return [self._tree_full[id] for id in ids]

    def _sort(self,
              regions: Container[RegionNode|int],
              mode: Literal["breadth", "depth"]|None="depth") -> list[RegionNode]:
        if len(regions) == 0:
            return []
        match mode:
            case "breadth":
                mode = Tree.WIDTH
            case "depth":
                mode = Tree.DEPTH
            case _:
                raise ValueError(f"Unsupported sorting mode: '{mode}'. Available modes are 'breadth' or 'depth'.")
        # NOTE: avoiding to sort to match the order used in the ontology
        sorted_ids = self._tree_full.expand_tree(mode=mode, sorting=False)
        regions = list(regions)
        if isinstance(regions[0], RegionNode):
            # NOTE: no check whether 'regions' are actually in self._tree_full
            ids = {r.id for r in regions}
            return [self._tree_full[id] for id in sorted_ids if id in ids]
        else: # isinstance(regions[0], int):
            # NOTE: no check whether the IDs are actually in self._tree_full
            return [id for id in sorted_ids if id in regions]

    @deprecated(since="1.1.0", message="Use 'blacklist' instead.")
    def blacklist_regions(self,
                          regions: Iterable,
                          *,
                          unreferenced: bool=False,
                          has_reference: bool=None):
        if has_reference is not None:
            warning_message = "'has_reference' is deprecated since 1.1.0 and may be removed in future versions. Use 'unreferenced' instead."
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            unreferenced = not has_reference
        self.blacklist(regions, unreferenced=unreferenced)

    def blacklist(self,
                  regions: Iterable,
                  *,
                  unreferenced: bool=False):
        """Blacklists from further analysis the given `regions` the ontology, as well as all their sub-regions."""
        ids = self._to_ids(regions, unreferenced=True, duplicated=False, check_all=False)
        for id in ids: #self._tree_full.filter_nodes
            if unreferenced and id in self._tree: # ids contains regions that may already be unreferenced
                subtree = self._tree.remove_subtree(id) # self._tree.remove_node(id) is not enough
                # we want to blacklist all regions below so that get_blacklisted_trees(unreferenced=True) works
                for subregion_id in subtree.expand_tree():
                    subregion: RegionNode = subtree[subregion_id]
                    subregion.blacklisted = True
            elif not unreferenced:
                rn: RegionNode = self._tree_full[id]
                if rn.blacklisted:
                    continue
                for child_id in self._tree_full.expand_tree(id):
                    child_rn: RegionNode = self._tree_full[child_id]
                    child_rn.blacklisted = True

    @deprecated(since="1.1.0", message="Use 'blacklisted' instead.")
    def get_blacklisted_trees(self,
                              unreferenced: bool=False,
                              key: Literal["id","acronym"]="acronym") -> list:
        return self.blacklisted(unreferenced=unreferenced, key=key)

    def blacklisted(self,
                    unreferenced: bool=False,
                    key: Literal["id","acronym"]="acronym") -> list:
        """Returns the biggest brain region of each branch in the ontology that was blacklisted."""
        tree = self._tree_full if unreferenced else self._tree
        blacklisted_trees = first_subtrees(tree, lambda n: n.blacklisted)
        return self._nodes_to_attr(blacklisted_trees, attr=key)

    def is_region(self,
                  value: int|str,
                  unreferenced: bool=False) -> bool:
        try:
            _ = self._to_id(value, unreferenced=unreferenced)
        except KeyError:
            return False
        return True

    def are_regions(self,
                    values: Iterable,
                    unreferenced: bool=False,
                    duplicated: bool=True):
        ids = self._to_ids(values, unreferenced=unreferenced, duplicated=duplicated, check_all=True)
        return np.array([id is not None for id in ids], dtype=bool)

    def minimum_treecover(self,
                          regions: Iterable,
                          *,
                          unreferenced: bool=False,
                          blacklisted: bool=True,
                          key: Literal["id","acronym"]="acronym") -> list[str]:
        """
        Returns the minimum set of regions that covers all the given regions, and not more.

        Parameters
        ----------
        acronyms
            The acronyms of the regions to cover
        unreferenced
            If False, it does not consider from the cover all those brain regions that have no references in the atlas annotations.
        blacklisted
            If False, it does not consider from the cover all those brain regions that are currently blacklisted from the ontology.\
            If `unreferenced` is True, it is ignored.
        key
            The region identifier of the returned sibling structures.

        Returns
        -------
        :
            A list of acronyms of regions

        Examples
        --------
        >>> atlas_ontology = braian.AtlasOntology("allen_mouse_10um")
        >>> sorted(atlas_ontology.minimum_treecover(['P', 'MB', 'TH', 'MY', 'CB', 'HY']))
        ['BS', 'CB']

        >>> sorted(atlas_ontology.minimum_treecover(["RE", "Xi", "PVT", "PT", "TH"]))
        ['MTN', 'TH']
        """
        tree = self._tree_full if unreferenced else self._tree
        if not unreferenced and not blacklisted:
            tree = Tree(tree, deep=False)
            for blacklisted_id in self.blacklisted(unreferenced=False, key="id"):
                if blacklisted_id in tree:
                    tree.remove_node(blacklisted_id)
        ids = self._to_ids(regions, unreferenced=unreferenced, duplicated=True, check_all=True)
        ids = [id for id in ids if id is not None]
        treecover_ids = minimum_treecover(tree, ids)
        # NOTE: we don't use self._to_nodes() because we don't need to check the IDs
        treecover = [self._tree_full[id] for id in treecover_ids]
        treecover = self._sort(treecover, mode="depth")
        return self._nodes_to_attr(treecover, attr=key)

    @deprecated(since="1.1.0", message="Use 'siblings' instead.")
    def get_sibiling_regions(self,
                             region: str|int,
                             key: Literal["id","acronym"]="acronym") -> list:
        """
        Get all brain regions that, combined, make the whole parent of the given `region`.\\
        It does not include the regions that have no reference in the atlas annotations.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        key
            The region identifier of the returned sibling structures.

        Returns
        -------
        :
            All `region`'s sibilings, including itself

        Raises
        ------
        KeyError
            If the given `region` is not found in the ontology.
        """
        id = self._to_id(region, unreferenced=False)
        pid = self._tree[id].predecessor(self._tree.identifier)
        if pid is None:
            siblings = [self._tree[id]]
        else:
            siblings = [self._tree[sid] for sid in self._tree[pid].successors(self._tree.identifier)]
        return self._nodes_to_attr(siblings, attr=key)

    def siblings(self,
                 region: str|int,
                 *,
                 key: Literal["id","acronym"]="acronym") -> list:
        """
        Retrieve the sibling structures of `region`, such that together with `region` they make the parent region.\\
        It does not include the regions that have no reference in the atlas annotations.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        key
            The region identifier of the returned sibling structures.

        Returns
        -------
        :
            All `region`'s sibilings, excluding itself

        Raises
        ------
        KeyError
            If the given `region` is not found in the ontology.
        """
        id = self._to_id(region, unreferenced=False)
        siblings = self._tree.siblings(id)
        return self._nodes_to_attr(siblings, attr=key)

    @deprecated(since="1.1.0", message="Use 'to_acronym' instead.")
    def ids_to_acronym(self, ids: Container[int], mode: Literal["breadth", "depth"]|None="depth") -> list[str]:
        return self.ids_to_acronym(ids, mode=mode)

    def to_acronym(self, regions: Container[int], mode: Literal["breadth", "depth"]|None="depth") -> list[str]:
        regions = self._to_nodes(regions, unreferenced=True, duplicated=False)
        if mode is not None:
            regions = self._sort(regions, mode=mode)
        return self._nodes_to_attr(regions, attr="acronym")

    @deprecated(since="1.1.0", message="Use 'to_id' instead.")
    def acronyms_to_id(self, acronyms: Container[str], mode: Literal["breadth", "depth"]|None="depth") -> list[int]:
        return self.to_id(acronyms, mode=mode)

    def to_id(self, regions: Container[str], mode: Literal["breadth", "depth"]|None="depth") -> list[int]:
        ids = self._to_ids(regions, unreferenced=True, duplicated=False, check_all=False)
        if mode is not None:
            return self._sort(ids, mode=mode)
        return ids

    # def get_parent_regions(self, regions: Iterable, key: str="acronym") -> dict:
    #     result = {}
    #     for region in regions:
    #         node = next((n for n in self._tree_full.all_nodes() if n.data and n.data.get(key) == region), None)
    #         if node is None:
    #             raise KeyError(f"Parent region not found ('{key}'={region})")
    #         parent = self._tree_full.parent(node.identifier)
    #         result[region] = parent.data[key] if parent and parent.data and key in parent.data else None
    #     return result

    # def list_all_subregions(self, acronym: str, mode: Literal["breadth", "depth"]="breadth", blacklisted: bool=True, unreferenced: bool=False) -> list:
    #     node = next((n for n in self._tree_full.all_nodes() if n.data and n.data.get("acronym") == acronym), None)
    #     if node is None:
    #         raise KeyError(f"Region not found ('acronym'='{acronym}')")
    #     subregions = []
    #     if mode == "breadth":
    #         queue = list(self._tree_full.children(node.identifier))
    #         while queue:
    #             n = queue.pop(0)
    #             if n.data and (blacklisted or not n.data.get("blacklisted", False)) and (unreferenced or n.data.get("has_reference", True)):
    #                 subregions.append(n.data["acronym"])
    #             queue.extend(self._tree_full.children(n.identifier))
    #     elif mode == "depth":
    #         stack = list(self._tree_full.children(node.identifier))
    #         while stack:
    #             n = stack.pop()
    #             if n.data and (blacklisted or not n.data.get("blacklisted", False)) and (unreferenced or n.data.get("has_reference", True)):
    #                 subregions.append(n.data["acronym"])
    #             stack.extend(self._tree_full.children(n.identifier))
    #     else:
    #         raise ValueError(f"Unsupported mode '{mode}'. Available modes are 'breadth' and 'depth'.")
    #     return subregions

    # def get_regions_above(self, acronym: str) -> list[str]:
    #     node = next((n for n in self._tree_full.all_nodes() if n.data and n.data.get("acronym") == acronym), None)
    #     if node is None:
    #         raise KeyError(f"Region not found ('acronym'='{acronym}')")
    #     path = []
    #     parent = self._tree_full.parent(node.identifier)
    #     while parent is not None:
    #         if parent.data and "acronym" in parent.data:
    #             path.append(parent.data["acronym"])
    #         parent = self._tree_full.parent(parent.identifier)
    #     return path

    # def get_corresponding_md(self, acronym: str, *acronyms: str) -> OrderedDict[str, str]:
    #     """Finds the corresponding major division for each of the acronyms."""
    #     raise NotImplementedError

    # def get_layer1(self) -> list[str]:
    #     """Returns the layer 1 in the Isocortex accordingly to CCFv3"""
    #     raise NotImplementedError

    # def _get_full_names(self) -> dict[str,str]:
    #     return {n.data["acronym"]: n.data["name"] for n in self._tree_full.all_nodes() if n.data and "acronym" in n.data and "name" in n.data}

    # def get_region_colors(self) -> dict[str,str]:
    #     return {n.data["acronym"]: "#"+n.data["color_hex_triplet"] for n in self._tree_full.all_nodes() if n.data and "acronym" in n.data and "color_hex_triplet" in n.data}

    # def to_igraph(self, unreferenced: bool=False, blacklisted: bool=True):
    #     """Translates the current brain ontology into an igraph directed Graph."""
    #     raise NotImplementedError

    def has_selection(self) -> bool:
        return self._selected

    def unselect_all(self):
        for n in self._tree_full.all_nodes():
            n.selected = False
        self._selected = False

    def _select(self, regions: Iterable[RegionNode]):
        for n in regions:
            n.selected = True
        self._selected = True

    @deprecated(since="1.1.0", message="Use 'selected' instead.")
    def get_selected_regions(self,
                            *,
                            key: Literal["id","acronym"]="acronym") -> list:
        return self.selected(key=key)

    def selected(self,
                 *,
                 key: Literal["id","acronym"]="acronym") -> list:
        selected_trees = first_subtrees(self._tree, lambda n: not n.blacklisted and n.selected)
        return self._nodes_to_attr(selected_trees, attr=key)

    def select_at_depth(self, depth: int):
        def select(n: RegionNode) -> bool:
            depth_ = self._tree.depth(n)
            return depth_ == depth or (depth_ < depth and n.is_leaf(self._tree.identifier))
        self._select(first_subtrees(self._tree, select))

    # def select_at_structural_level(self, level: int):
    #     """Select all non-overlapping brain regions at the same structural level in the ontology."""
    #     raise NotImplementedError

    def select_leaves(self):
        self._select(self._tree.leaves())

    # def select_summary_structures(self):
    #     """Select all summary structures in the ontology."""
    #     raise NotImplementedError

    # def select_regions(self, regions: Iterable, key: str="acronym"):
    #     """Select the given regions in the ontology"""
    #     raise NotImplementedError

    # def add_to_selection(self, regions: Iterable, key: str="acronym"):
    #     """Add the given brain regions to the current selection in the ontology"""
    #     raise NotImplementedError

    # def get_regions(self, selection_method: str) -> list[str]:
    #     """Returns a list of acronyms of non-overlapping regions based on the selection method."""
    #     raise NotImplementedError