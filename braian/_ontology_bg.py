import brainglobe_atlasapi as bga
import numpy as np
import warnings

from collections import OrderedDict
from copy import deepcopy
from treelib import Node, Tree
from typing import Callable, Container, Generator, Iterable, Literal, Sequence

from braian.utils import deprecated

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

    @property
    def hex_color(self):
        return f"#{self.rgb_triplet[0]:02X}{self.rgb_triplet[1]:02X}{self.rgb_triplet[2]:02X}"

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
    root = tree[tree.root]
    if func(root):
        yield root
        return
    queue = [tree[i] for i in root.successors(tree.identifier)]
    while queue:
        n = queue[0]
        queue = queue[1:]
        if func(n):
            yield n
        else:
            expansion = [tree[i] for i in n.successors(tree.identifier)]
            queue = expansion + queue  # depth-first

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
        """The name of the atlas accordingly to ABBA/BrainGlobe"""
        self.full_name: dict[str,str] = self._map_to_name(key="acronym")
        """A dictionary mapping a regions' acronym to its full name. It also contains the names for the blacklisted and unreferenced regions."""
        self.parent_region: dict[str,str] = self._map_to_parent(key="acronym") #: A dictionary mapping region's acronyms to the parent region. It does not have 'root'.
        self.direct_subregions: dict[str,list[str]] = self._map_to_subregions(key="acronym")
        """A dictionary mappin region's acronyms to a list of direct subregions.

        Examples
        --------
        >>> atlas_ontology = braian.AtlasOntology("allen_mouse_50um")
        >>> atlas_ontology.direct_subregions["ACA"]
        ["ACAv", "ACAd"]                                    # dorsal and ventral part
        >>> atlas_ontology.direct_subregions["ACAv"]
        ["ACAv5", "ACAv2/3", "ACAv6a", "ACAv1", "ACAv6b"]   # all layers in ventral part
        """

    def _to_ids(self,
                regions: Iterable,
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
                  regions: Iterable,
                  unreferenced: bool,
                  duplicated: bool) -> list[RegionNode]:
        ids = self._to_ids(regions, unreferenced=unreferenced, duplicated=duplicated, check_all=False)
        return [self._tree_full[id] for id in ids]

    def _check_node_attr(self, attr: str):
        if attr not in ("acronym", "tag", "id", "identifier"):
            raise ValueError(f"Unknown unique identifier of brain regions: '{attr}'")

    def _node_to_attr(self,
                       region: RegionNode,
                       *,
                       attr: Literal["acronym","id"]
                       ) -> str|int:
        return region.__getattribute__(attr)

    def _nodes_to_attr(self,
                       regions: Iterable[RegionNode],
                       *,
                       attr: Literal["acronym","id"]
                       ) -> list:
        self._check_node_attr(attr)
        return [self._node_to_attr(r, attr=attr) for r in regions]

    def _unreferenced(self,
                      *,
                      key: Literal["id","acronym"]="acronym") -> Generator:
        atlas_meshes_dir = self._atlas.root_dir/"meshes"
        if not atlas_meshes_dir.exists():
            raise ValueError(f"BrainGlobe atlas meshes not downloaded: '{self._atlas.atlas_name}'")
        regions_w_annotation = [int(p.stem) for p in atlas_meshes_dir.iterdir() if p.suffix == ".obj"]
        regions_wo_annotation = self._tree_full.filter_nodes(lambda n: n.identifier not in regions_w_annotation)
        return self._nodes_to_attr(regions_wo_annotation, attr=key)

    def _map_to_name(self, key: Literal["id","acronym"]="acronym") -> dict[str,str]:
        if key not in ("acronym", "tag", "id", "identifier"):
            raise ValueError(f"Unknown unique identifier of brain regions: '{key}'")
        return {n.acronym: n.name for n in self._tree_full.all_nodes()}

    def _map_to_parent(self, key: Literal["id","acronym"]="acronym") -> dict:
        """
        Finds, for each brain region in the ontology, the corresponding parent region.
        The "root" region has no entry in the returned dictionary

        The resulting dictionary does not include unreferenced brain regions.

        Parameters
        ----------
        key
            The region identifier used in the returned dictionary.

        Returns
        -------
        :
            A dictionary mapping region→parent
        """
        if key not in ("acronym", "tag", "id", "identifier"):
            raise ValueError(f"Unknown unique identifier of brain regions: '{key}'")
        return {node.__getattribute__(key): self._tree.parent(id).__getattribute__(key)
                for id,node in self._tree.nodes.items() if id != self._tree.root}

    def _map_to_subregions(self, key: Literal["id","acronym"]="acronym") -> dict:
        """
        returns a dictionary where all parent regions are the keys,
        and the subregions that belong to the parent are stored
        in a list as the value corresponding to the key.
        Regions with no subregions have no entries in the dictionary.

        The resulting dictionary does not include unreferenced brain regions.

        Parameters
        ----------
        key
            The region identifier used in the returned dictionary.

        Returns
        -------
        :
            a dictionary that maps region→[subregions...]
        """
        subregions = {self._node_to_attr(node, attr=key): self._nodes_to_attr(self._tree.children(id), attr=key)
                for id,node in self._tree.nodes.items()}
        return {parent: children for parent,children in subregions.items() if children}

    def get_region_colors(self) -> dict[str,str]:
        return {n.acronym: n.hex_color for n in self._tree_full.all_nodes()}

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

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.blacklist"])
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
        self.direct_subregions = self._map_to_subregions(key="acronym")
        self.parent_region = self._map_to_parent(key="acronym")

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.blacklisted"])
    def get_blacklisted_trees(self,
                              unreferenced: bool=False,
                              key: Literal["id","acronym"]="acronym") -> list:
        """
        Returns the biggest brain region of each branch in the ontology that was blacklisted.

        Parameters
        ----------
        unreferenced
            If True, it includes also the brain regions that were blacklisted from the ontology because they don't have an annotation in the atlas.
            Otherwise, it returns only those blacklisted regions that has a reference in the annotation volume.
        key
            The key in Allen's structural graph used to indentify the returned regions

        Returns
        -------
        :
            A list of blacklisted regions, each of which is identifiable with `key`
        """
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
        regions
            The regions to cover
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

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.parents"])
    def get_parent_regions(self,
                           regions: Iterable,
                           *,
                           key: Literal["id","acronym"]="acronym") -> dict:
        return self.parents(regions, key=key)

    def parents(self,
                regions: Iterable,
                *,
                key: Literal["id","acronym"]="acronym") -> dict:
        self._check_node_attr(key)
        children = self._to_nodes(regions, unreferenced=True, duplicated=False)
        parents = (self._node_to_attr(self._tree_full.parent(child.id), attr=key)
                   if child.id != self._tree_full.root
                   else None
                   for child in children)
        children = self._nodes_to_attr(children, attr=key)
        return dict(zip(children, parents))

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.siblings"])
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

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.subregions"])
    def list_all_subregions(self,
                            region: str,
                            mode: Literal["breadth", "depth"]="breadth",
                            blacklisted: bool=True,
                            unreferenced: bool=False) -> list:
        return self.subregions(region, mode=mode, blacklisted=blacklisted, unreferenced=unreferenced)

    def subregions(self,
                   region: str,
                   *,
                   mode: Literal["breadth", "depth"]="breadth",
                   blacklisted: bool=True,
                   unreferenced: bool=False) -> list:
        id = self._to_id(region, unreferenced=unreferenced)
        match mode:
            case "depth":
                mode = Tree.DEPTH
            case "breadth":
                mode = Tree.WIDTH
            case _:
                raise ValueError(f"Unsupported mode '{mode}'. Available modes are 'breadth' and 'depth'.")
        tree = self._tree_full if unreferenced else self._tree
        visit_tree = (lambda _: True) if blacklisted else (lambda n: not n.blacklisted)
        subregions = list(tree.expand_tree(id, filter=visit_tree, mode=mode, sorting=False))
        return self._nodes_to_attr((tree[r] for r in subregions), attr="acronym")

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.ancestors"])
    def get_regions_above(self, acronym: str) -> list[str]:
        return self.ancestors(acronym)

    def ancestors(self, acronym: str) -> list[str]:
        node = self._tree[self._to_id(acronym, unreferenced=False)]
        nodes = []
        while (node:=self._tree.parent(node.id)) is not None:
            nodes.append(node)
        return self._nodes_to_attr(nodes, attr="acronym")

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.to_acronym"])
    def ids_to_acronym(self, ids: Container[int], mode: Literal["breadth", "depth"]|None="depth") -> list[str]:
        return self.to_acronym(ids, mode=mode)

    def to_acronym(self, regions: Iterable[int], mode: Literal["breadth", "depth"]|None="depth") -> list[str]:
        regions = self._to_nodes(regions, unreferenced=True, duplicated=False)
        if mode is not None:
            regions = self._sort(regions, mode=mode)
        return self._nodes_to_attr(regions, attr="acronym")

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.to_id"])
    def acronyms_to_id(self, acronyms: Container[str], mode: Literal["breadth", "depth"]|None="depth") -> list[int]:
        return self.to_id(acronyms, mode=mode)

    def to_id(self, regions: Iterable[str], mode: Literal["breadth", "depth"]|None="depth") -> list[int]:
        ids = self._to_ids(regions, unreferenced=True, duplicated=False, check_all=False)
        if mode is not None:
            return self._sort(ids, mode=mode)
        return ids

    # def get_corresponding_md(self, acronym: str, *acronyms: str) -> OrderedDict[str, str]:
    #     """Finds the corresponding major division for each of the acronyms."""
    #     raise NotImplementedError

    # def get_layer1(self) -> list[str]:
    #     """Returns the layer 1 in the Isocortex accordingly to CCFv3"""
    #     raise NotImplementedError

    # def to_igraph(self, unreferenced: bool=False, blacklisted: bool=True):
    #     """Translates the current brain ontology into an igraph directed Graph."""
    #     raise NotImplementedError

    def has_selection(self) -> bool:
        return self._selected

    def unselect_all(self):
        for n in self._tree_full.all_nodes():
            n.selected = False
        self._selected = False

    def _select(self, regions: Iterable[RegionNode], *, add: bool):
        if not add:
            self.unselect_all()
        for n in regions:
            n.selected = True
        self._selected = True

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.select"])
    def select_regions(self, regions: Iterable, key: str="acronym"):
        return self.select(regions)

    def select(self, regions: Iterable):
        regions = self._to_nodes(regions, unreferenced=False, duplicated=False)
        self._select(regions, add=False)

    def add_to_selection(self, regions: Iterable, key: str=None):
        if key is not None:
            warning_message = "'key' is deprecated since 1.1.0 and may be removed in future versions."
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
        regions = self._to_nodes(regions, unreferenced=False, duplicated=True)
        self._select(regions, add=True)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.selected"])
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
        self._select(first_subtrees(self._tree, select), add=False)

    # def select_at_structural_level(self, level: int):
    #     """Select all non-overlapping brain regions at the same structural level in the ontology."""
    #     raise NotImplementedError

    def select_leaves(self):
        self._select(self._tree.leaves(), add=False)

    # def select_summary_structures(self):
    #     """Select all summary structures in the ontology."""
    #     raise NotImplementedError

    # def get_regions(self, selection_method: str) -> list[str]:
    #     """Returns a list of acronyms of non-overlapping regions based on the selection method."""
    #     raise NotImplementedError