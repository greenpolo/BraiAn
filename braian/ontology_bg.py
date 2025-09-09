import brainglobe_atlasapi as bga
import warnings

from treelib import Tree, Node
from collections import OrderedDict
from copy import deepcopy
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
            unreferenced_regions = self._get_unreferenced_regions() # in allen_mouse_50um, only 'RSPd4 (545)' is unreferenced
            self.blacklist_regions(unreferenced_regions, has_reference=False)
        if blacklisted_acronyms:
            self.blacklist_regions(blacklisted_acronyms)

    def _get_unreferenced_regions(self) -> list[int]:
        atlas_meshes_dir = self._atlas.root_dir/"meshes"
        if not atlas_meshes_dir.exists():
            raise ValueError(f"BrainGlobe atlas meshes not downloaded: '{self._atlas.atlas_name}'")
        regions_w_annotation = [int(p.stem) for p in atlas_meshes_dir.iterdir() if p.suffix == ".obj"]
        regions_wo_annotation = self._tree_full.filter_nodes(lambda n: n.identifier not in regions_w_annotation)
        return [region.identifier for region in regions_wo_annotation]

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
                ids.append(int(region))
                continue
            except ValueError:
                pass
            if region not in self._acronym2id and not check_all:
                raise KeyError(f"Region not found ('acronym'='{region}')")
            ids.append(self._acronym2id.get(region, None))
        if not unreferenced:
            for i in range(len(ids)):
                if ids[i] not in self._tree: # region is unreferenced
                    if check_all:
                        ids[i] = None
                    else:
                        raise KeyError(f"Region not found ('acronym'='{self._tree_full[ids[i]].tag}')")
        return ids

    def _nodes_to_attr(self,
                       regions: Iterable[RegionNode],
                       *,
                       attr: Literal["acronym","id"]
                       ) -> list:
        if attr not in ("acronym", "tag", "id", "identifier"):
            raise ValueError(f"Unknown unique identifier of brain regions: '{attr}'")
        return [r.__getattribute__(attr) for r in regions]

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
                # self._tree.remove_node(id) # might want to use 'remove_subtree' if we want to modify the nodes below
                subtree = self._tree.remove_subtree(id)
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

    def get_blacklisted_trees(self,
                              unreferenced: bool=False,
                              key: Literal["id","acronym"]="acronym") -> list:
        """Returns the biggest brain region of each branch in the ontology that was blacklisted."""
        tree = self._tree_full if unreferenced else self._tree
        blacklisted_trees = first_subtrees(tree, lambda n: n.blacklisted)
        return self._nodes_to_attr(blacklisted_trees, attr=key)

    # def is_region(self, r: int|str, key: str="acronym", unreferenced: bool=False) -> bool:
    #     for node in self._tree_full.all_nodes():
    #         if node.data and key in node.data and node.data[key] == r:
    #             return True
    #     return False

    # def are_regions(self, a: Iterable, key: str="acronym", unreferenced: bool=False):
    #     a = list(a)
    #     found = [self.is_region(x, key=key, unreferenced=unreferenced) for x in a]
    #     return found

    # def get_sibiling_regions(self, region: str|int, key: str="acronym") -> list:
    #     node = next((n for n in self._tree_full.all_nodes() if n.data and n.data.get(key) == region), None)
    #     if node is None:
    #         raise KeyError(f"Region not found ('{key}'={region})")
    #     parent = self._tree_full.parent(node.identifier)
    #     if parent is None:
    #         return [region]
    #     return [n.data[key] for n in self._tree_full.children(parent.identifier) if n.data and key in n.data]

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

    # def minimum_treecover(self, acronyms: Iterable[str], unreferenced: bool=False, blacklisted: bool=True) -> set[str]:
    #     """Returns the minimum set of regions that covers all the given regions, and not more."""
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