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
    old_nodes: dict[int,Node] = atlas.structures.tree.nodes # expand_tree(mode=Tree.DEPTH):
    for nid, old_node in old_nodes.items():
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

def first_subtrees(tree: Tree, func: Callable[[Node],bool]) -> list[Node]:
    return tree.filter_nodes(lambda n: func(n) and (not func(tree.parent(n.identifier) or tree.root == n.id)))

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
                 blacklisted_acronyms: Iterable[str]=[],
                 unreferenced: bool=False):
        if not isinstance(atlas, bga.BrainGlobeAtlas):
            atlas = bga.BrainGlobeAtlas(atlas)
        self._atlas: bga.BrainGlobeAtlas = atlas
        self._acronym2id: dict[str,int] = deepcopy(atlas.structures.acronym_to_id_map)
        self._tree_full: Tree = convert_to_region_nodes(self._atlas)
        self._tree: Tree = Tree(self._tree_full, deep=False)
        if not unreferenced:
            unreferenced_regions = self._unreferenced(key="id") # in allen_mouse_50um, only 'RSPd4 (545)' is unreferenced
            self.blacklist_regions(unreferenced_regions, has_reference=False)
        if blacklisted_acronyms:
            self.blacklist_regions(blacklisted_acronyms)

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
                check_all: bool) -> list[int]:
        """
        Checks the existance of a list of regions and returns the corresponding IDs.

        Parameters
        ----------
        regions
            A sequence of regions, identified by their ID or their acronym.
        unreferenced
            If `True`, it accepts `regions` that have no reference in the atlas annotations.
        check_all
            If `True`, the returned list may contain some `None` values, corresponding to unknown regions.

        Returns
        -------
        :
            A list of region IDs.

        Raises
        ------
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
        return ids

    def blacklist_regions(self,
                          regions: Iterable,
                          *,
                          unreferenced: bool=False,
                          has_reference: bool=None):
        """Blacklists from further analysis the given `regions` the ontology, as well as all their sub-regions."""
        if has_reference is not None:
            warning_message = "'has_reference' is deprecated since 1.1.0 and may be removed in future versions. Use 'unreferenced' instead."
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            unreferenced = not has_reference
        ids = self._to_ids(regions, unreferenced=True, check_all=False)
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
                    unreferenced: bool=False):
        ids = self._to_ids(values, unreferenced=unreferenced, check_all=True)
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
        ids = self._to_ids(regions, unreferenced=unreferenced, check_all=True)
        ids = [id for id in ids if id is not None]
        treecover_ids = minimum_treecover(tree, ids)
        # sort depth-first
        sorted_ids = tree.expand_tree()
        return self._nodes_to_attr([tree[id] for id in sorted_ids if id in treecover_ids], attr=key)

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

    # def acronyms_to_id(self, acronyms: Container[str], mode: Literal["breadth", "depth"]="depth") -> list[int]:
    #     """Converts the given brain regions acronyms into their corresponding IDs"""
    #     raise NotImplementedError

    # def ids_to_acronym(self, ids: Container[int], mode: Literal["breadth", "depth"]="depth") -> list[str]:
    #     """Converts the given brain regions IDs into their corresponding acronyms."""
    #     raise NotImplementedError

    # def select_at_depth(self, depth: int):
    #     """Select all non-overlapping brain regions at the same depth in the ontology."""
    #     raise NotImplementedError

    # def select_at_structural_level(self, level: int):
    #     """Select all non-overlapping brain regions at the same structural level in the ontology."""
    #     raise NotImplementedError

    # def select_leaves(self):
    #     """Select all the non-overlapping smallest brain regions in the ontology."""
    #     raise NotImplementedError

    # def select_summary_structures(self):
    #     """Select all summary structures in the ontology."""
    #     raise NotImplementedError

    # def select_regions(self, regions: Iterable, key: str="acronym"):
    #     """Select the given regions in the ontology"""
    #     raise NotImplementedError

    # def add_to_selection(self, regions: Iterable, key: str="acronym"):
    #     """Add the given brain regions to the current selection in the ontology"""
    #     raise NotImplementedError

    # def has_selection(self) -> bool:
    #     """Check whether the current ontology is currently selecting any brain region."""
    #     raise NotImplementedError

    # def get_selected_regions(self, key: str="acronym") -> list:
    #     """Returns a non-overlapping list of selected non-blacklisted brain regions"""
    #     raise NotImplementedError

    # def unselect_all(self):
    #     """Resets the selection in the ontology"""
    #     raise NotImplementedError

    # def get_regions(self, selection_method: str) -> list[str]:
    #     """Returns a list of acronyms of non-overlapping regions based on the selection method."""
    #     raise NotImplementedError