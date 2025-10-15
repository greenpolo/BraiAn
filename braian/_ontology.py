import brainglobe_atlasapi as bga
import igraph as ig
import numpy as np
import pandas as pd

from braian import utils, _graph_utils
from collections import OrderedDict
from copy import deepcopy
from treelib import Node, Tree
from typing import Callable, Container, Generator, Iterable, Literal, Sequence

from braian.utils import deprecated

__all__ = [
    "AtlasOntology",
    "MissingWholeBrainPartitionError"
]

class MissingWholeBrainPartitionError(FileNotFoundError):
    def __init__(self, atlas: str, partition: str):
        super().__init__(f"Unknown '{partition}' partition for '{atlas}' atlas. If you think think BraiAn should support this atlas, please open an issue on https://codeberg.org/SilvaLab/BraiAn")

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
    def acronym(self) -> str:
        return self.tag

    @property
    def blacklisted(self) -> bool:
        return self.expanded

    @blacklisted.setter
    def blacklisted(self, value: bool):
        self.expanded = value

    @property
    def id(self) -> int:
        return self.identifier

    @property
    def hex_triplet(self) -> str:
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
        """
        Creates an ontology of the brain structures as defined by the corresponding
        [BrainGlobe](https://brainglobe.info/documentation/brainglobe-atlasapi) atlas.

        This is the basic module for any whole-brain analysis.
        The atlas should be the same one that was used to register the whole-brain data to.

        Parameters
        ----------
        atlas
            The atlas or the corresponding codename, as defined by
            [BrainGlobe's API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html#atlases-available)
        blacklisted
            The brain regions that should be excluded completely from the analysis.
            Any data manipulated, analysed and exported by the ontology, then, will not have
            any reference of the blacklisted regions and subregions.
        unreferenced
            If True, it considers as part of the ontology all those brain regions that have no references in the atlas annotations.
            Otherwise, it removes them from the ontology.
            These "unreferenced" regions are structures that the atlas creators might consider as
            existing, but thought that the scientific community did not yet reach a consesus on their
            shape and position.
            As an example, on Allen's website, unreferenced brain regions are identified in grey italic text: [](https://atlas.brain-map.org).

        Raises
        ------
        ValueError
            If `atlas` is not a valid atlas name, accordingly to BrainGlobe.
        KeyError
            If any of the `blacklisted` structures is not found in the ontology.
        ValueError
            If `blacklisted` contains the same brain structure more than once.
        """
        if not isinstance(atlas, bga.BrainGlobeAtlas):
            atlas = bga.BrainGlobeAtlas(atlas, check_latest=False)
        self._atlas: bga.BrainGlobeAtlas = atlas
        self._acronym2id: dict[str,int] = deepcopy(atlas.structures.acronym_to_id_map)
        self._tree_full: Tree = convert_to_region_nodes(self._atlas)
        self._tree: Tree = Tree(self._tree_full, deep=False)
        self._selected: bool = False
        if not unreferenced:
            unreferenced_ids = self._unreferenced(key="id")
            # In allen_mouse 25um & 50um, 'RSPd4 (545)' is the only unreferenced brain region that exists
            self.blacklist(unreferenced_ids, unreferenced=True)
        if blacklisted:
            blacklisted_ids = self._to_ids(blacklisted, unreferenced=True, duplicated=False, check_all=False)
            self.blacklist(blacklisted_ids, unreferenced=False)
        self.full_name: dict[str,str] = self._map_to_name(key="acronym")
        """A dictionary mapping a regions' acronym to its full name. It also contains the names for the blacklisted and unreferenced regions."""
        self.parent_region: dict[str,str] = self._map_to_parent(key="acronym")
        """A dictionary mapping region's acronyms to the parent region. It does not have 'root'."""
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

    @property
    def name(self) -> str:
        """The codename of the atlas according to BrainGlobe"""
        return self._atlas.atlas_name

    def is_compatible(self, atlas_name: str) -> bool:
        """
        Checks whether the ontology is compatible with another atlas, given its codename.

        Currently, it only checks whether the current atlas is compatible with
        [ABBA](https://go.epfl.ch/abba)'s atlases. For example,
        ABBA's "Adult Mouse Brain - Allen Brain Atlas V3p1" is fully compatible
        with [BrainGlobe](https://brainglobe.info/documentation/brainglobe-atlasapi)'s
        "allen_mouse_10um".

        Parameters
        ----------
        atlas_name
            The name of an atlas.

        Returns
        -------
        :
            True, if `atlas_name` referes to an atlas that is equivalent the ontology's atlas.
        """
        return self.name == atlas_name or \
        (self.name == "allen_mouse_10um" and atlas_name in {"allen_mouse_10um_java",
                                                            "Adult Mouse Brain - Allen Brain Atlas V3p1",
                                                            "Adult Mouse Brain - Allen Brain Atlas V3"}) or\
        (self.name == "whs_sd_rat_39um" and atlas_name in {"whs_sd_rat_39um",
                                                           "Rat - Waxholm Sprague Dawley V4p2",
                                                           "Rat - Waxholm Sprague Dawley V4",
                                                           "waxholm_sprague_dawley_rat_v4"})

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
            If False, it raises `ValueError` if `values` contains the same region more than once.
            Otherwise, it does nothing.
        check_all
            If `True`, the returned list may contain some `None` values, corresponding to unknown regions.

        Returns
        -------
        :
            A list of region IDs.

        Raises
        ------
        ValueError
            If `duplicated=False` and there is at least one brain structure that appears twice in `regions`.
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
        """
        Raises
        ------
        KeyError
            If at least one of the given `regions` is not found in the ontology.
        ValueError
            If `duplicated=False` and there is at least one brain structure that appears twice in `regions`.
        """
        ids = self._to_ids(regions, unreferenced=unreferenced, duplicated=duplicated, check_all=False)
        return [self._tree_full[id] for id in ids]

    def _check_node_attr(self, attr: str):
        """
        Raises
        ------
        ValueError
            If `attr` is not known as an unique identifier for brain structures
        """
        if attr not in ("acronym", "tag", "id", "identifier"):
            raise ValueError(f"Unknown unique identifier for brain structures: '{attr}'")

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
        """
        Raises
        ------
        ValueError
            If `attr` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(attr)
        return [self._node_to_attr(r, attr=attr) for r in regions]

    def _unreferenced(self,
                      *,
                      key: Literal["id","acronym"]="acronym") -> Generator:
        """
        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(key)
        return [s[key] for s in self._atlas.structures_list
                if not s["mesh_filename"].exists() and\
                    # BrainGlobe does not have a mesh for 'RSPd4 (545)' in allen_mouse_10um,
                    # but it is not an unreferenced region.
                    # see: https://github.com/brainglobe/brainglobe-atlasapi/issues/647
                    (self._atlas.atlas_name not in ("allen_mouse_10um", "silvalab_mouse_10um") or s["id"] != 545)]
        # atlas_meshes_dir = self._atlas.root_dir/"meshes"
        # if not atlas_meshes_dir.exists():
        #     raise ValueError(f"BrainGlobe atlas meshes not downloaded: '{self._atlas.atlas_name}'")
        # regions_w_annotation = [int(p.stem) for p in atlas_meshes_dir.iterdir() if p.suffix == ".obj"]
        # regions_wo_annotation = self._tree_full.filter_nodes(lambda n: n.identifier not in regions_w_annotation)
        # return self._nodes_to_attr(regions_wo_annotation, attr=key)

    def _map_to_name(self, key: Literal["id","acronym"]="acronym") -> dict[str,str]:
        """
        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(key)
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

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(key)
        return {self._node_to_attr(node, attr=key): self._node_to_attr(self._tree.parent(id), attr=key)
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

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(key)
        subregions = {self._node_to_attr(node, attr=key): self._nodes_to_attr(self._tree.children(id), attr=key)
                for id,node in self._tree.nodes.items()}
        return {parent: children for parent,children in subregions.items() if children}

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.colors"])
    def get_region_colors(self) -> dict[str,str]:
        """
        Gets the dictionary that maps brain structures to their hex color triplet,
        as defined by the creators' of the atlas.\
        If they didn't specify one, it defaults to white, as decided by BrainGlobe.

        Returns
        -------
        :
            A dictionary that maps region→color

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        return self.colors(key="acronym")

    def colors(self, *, key: Literal["id","acronym"]="acronym") -> dict[int|str,str]:
        """
        Gets the dictionary that maps brain structures to their hex color triplet,
        as defined by the creators' of the atlas.\
        If they didn't specify one, it defaults to white, as decided by BrainGlobe.

        Parameters
        ----------
        key
            The unique identifier used to identify a brain structure

        Returns
        -------
        :
            A dictionary that maps region→color

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(key)
        return {self._node_to_attr(n, attr=key): n.hex_triplet for n in self._tree_full.all_nodes()}

    def _to_id(self,
               region: int|str,
               *,
               unreferenced: bool) -> int:
        """
        Checks the existance of a region and returns the corresponding ID.

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
            If `region` is not found in the ontology.
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
              mode: Literal["breadth", "depth"]|None="depth") -> list[RegionNode|int]:
        """
        Parameters
        ----------
        regions
            A list of brain structures to be sorted according to the ontology's hierarchy.
            The structures can either be of `RegionNode`s or of IDs.
            No check on the lists type is made
        mode
            The mode in which to visit the hierarchy of the atlas ontology, which dictates
            how to sort a linearised list of regions.

        Raises
        ------
        ValueError
            When `mode` has an invalid value.
        """
        if len(regions) == 0:
            return []
        match mode:
            case "breadth":
                mode = Tree.WIDTH
            case "depth":
                mode = Tree.DEPTH
            case _:
                raise ValueError(f"Unsupported sorting mode: '{mode}'. Available modes are 'breadth' or 'depth'.")
        # NOTE: avoiding 'sorting' to match the order used in the ontology
        sorted_ids = self._tree_full.expand_tree(mode=mode, sorting=False)
        regions = list(regions)
        if isinstance(regions[0], RegionNode):
            # NOTE: no check whether 'regions' are actually in self._tree_full
            ids = {r.id for r in regions}
            return [self._tree_full[id] for id in sorted_ids if id in ids]
        else: # isinstance(regions[0], int):
            # NOTE: no check whether the IDs are actually in self._tree_full
            return [id for id in sorted_ids if id in regions]

    def sort(self, regions: Iterable|pd.DataFrame|pd.Series,
             *, mode: Literal["depth","width"]="depth",
             blacklisted: bool=True, unreferenced: bool=False,
             fill: bool=False, fill_value=np.nan,
             key: Literal["id", "acronym"]=None) -> list|pd.DataFrame|pd.Series:
        """
        Sorts some regionalised data in depth-first order (or width-first), based on the
        atlas hierarchical ontology.\\
        If `fill=True` and `regions` contains values for regions not present in the ontology,
        it removes them from the data.

        A new object is always produced.

        Parameters
        ----------
        regions
            The brain structures to be sorted according to the ontology's hierarchy,
            uniquely identified by their ID or their acronym.

            It also accepts regionalised data. In that case it should be a
            [`DataFrame`][pandas.DataFrame] or a [`Series`][pandas.Series] indexed by
            the brain region identifiers.
        mode
            The mode in which to visit the hierarchy of the atlas ontology, which dictates
            how to sort a linearised list of regions.
        blacklisted
            If `True`, it fills the data with `fill_value` also in correspondance to
            structures that are blacklisted in the ontology.
        unreferenced
            If `True`, it fills the data with `fill_value` also in correspondance to
            structures that have no reference in the atlas annotations.
        fill
            If `True` and `regions` is a [`DataFrame`][pandas.DataFrame] or a
            [`Series`][pandas.Series], it fills the data with `fill_value` corresponding
            to the regions present in the ontology but missing in `regions`.\\
            If `regions` contains values for regions not present in the ontology,
            it removes them from the data.

            If `False`, it raises a `KeyError` if `regions` contains values for
            regions not present in the ontology (or that are unreferenced when
            `unreferenced=False`)
        fill_value
            The value used to fill the data with, when `fill=True`
        key
            The region identifier of the returned brain structures.

            If not specified, it uses the first value in `regions` to decide
            whether to use IDs or acronyms.

        Returns
        -------
        :
            A copy of `regions`, but sorted according to the ontology hierarchy.

        Raises
        ------
        KeyError
            If at least one of `regions` is not found in the ontology.
        ValueError
            When `mode` has an invalid value.
        ValueError
            If `fill=True` but `regions` is just a list of regions with no data.

        See also
        --------
        [`sort`][braian.sort]
        [`BrainData.sort`][braian.BrainData.sort]
        [`AnimalBrain.sort`][braian.AnimalBrain.sort]
        """
        if isinstance(regions, (pd.Series, pd.DataFrame)):
            regions_ = regions.index
        elif fill:
            raise ValueError(f"Can't fill any missing data in '{type(regions)}' type")
        else:
            regions_ = regions
        key = "id" if isinstance(regions_[0], int) else "acronym"
        if fill:
            # before overwriting, check that the all `regions` exist in the ontology
            _ = self._to_ids(regions_, unreferenced=unreferenced, duplicated=True, check_all=False)
            regions_ = self.subregions(self._tree_full.root, mode=mode,
                                       blacklisted=blacklisted, key=key,
                                       unreferenced=unreferenced)
        else:
            # might want to first call self.are_regions(), so that the error is complete of ALL unkown regions
            regions_ = [self._node_to_attr(self._tree_full[id], attr=key)
                        for id in self._sort(self._to_ids(regions_, unreferenced=unreferenced,
                                                          duplicated=True, check_all=False),
                                            mode=mode)]
        if not isinstance(regions, (pd.Series, pd.DataFrame)):
            return regions_
        # else: 'regions' contains data
        # NOTE: if fill_value=np.nan -> converts dtype to float
        sorted = regions.reindex(regions_, copy=False, fill_value=fill_value)
        return sorted

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.blacklist"])
    def blacklist_regions(self,
                          regions: Iterable,
                          *,
                          has_reference: bool=None):
        """
        Blacklists from the ontology the brain regions that should be excluded completely from the analysis.
        Any data manipulated, analysed and exported by the ontology, then, will not have
        any reference of the blacklisted regions and subregions.

        Parameters
        ----------
        regions
            Regions to blacklist from the ontology
        has_reference
            Usually users should not use this parameter.
            If True, it treats the blacklisted structures as if they didn't even have a reference in the atlas annotations.
            This is a way to hide `regions` even deeper.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        ValueError
            If `regions` contains the same brain structure more than once.
        """
        self.blacklist(regions, unreferenced=not has_reference)

    def blacklist(self,
                  regions: Iterable,
                  *,
                  unreferenced: bool=False):
        """
        Blacklists from the ontology the brain regions that should be excluded completely from the analysis.
        Any data manipulated, analysed and exported by the ontology, then, will not have
        any reference of the blacklisted regions and subregions.

        Parameters
        ----------
        regions
            Regions to blacklist from the ontology
        unreferenced
            Usually users should not use this parameter.
            If False, it treats the blacklisted structures as if they didn't even have a reference in the atlas annotations.
            This is a way to hide `regions` even deeper.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        ValueError
            If `regions` contains the same brain structure more than once.
        """
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
            If True, it includes also the brain regions that were blacklisted from the ontology
            because they didn't have an annotation in the atlas.
            Otherwise, it returns only those blacklisted structures that has a reference
            in the annotation volume.
        key
            The unique identifier used to identify a brain structure

        Returns
        -------
        :
            A list of blacklisted regions, each of which is identifiable with `key`

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        return self.blacklisted(unreferenced=unreferenced, key=key)

    def blacklisted(self,
                    unreferenced: bool=False,
                    key: Literal["id","acronym"]="acronym") -> list:
        """
        Returns the biggest brain region of each branch in the ontology that was blacklisted.

        Parameters
        ----------
        unreferenced
            If True, it includes also the brain regions that were blacklisted from the ontology
            because they didn't have an annotation in the atlas.
            Otherwise, it returns only those blacklisted structures that has a reference
            in the annotation volume.
        key
            The unique identifier used to identify a brain structure

        Returns
        -------
        :
            A list of blacklisted regions, each of which is identifiable with `key`

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        tree = self._tree_full if unreferenced else self._tree
        blacklisted_trees = first_subtrees(tree, lambda n: n.blacklisted)
        return self._nodes_to_attr(blacklisted_trees, attr=key)

    def is_region(self,
                  value: int|str,
                  unreferenced: bool=False) -> bool:
        """
        Checks whether a region is in the ontology.

        Parameters
        ----------
        value
            A value that uniquely indentifies a brain region (e.g. its acronym or its ID).
        unreferenced
            If True, it considers as region also those structures that have no reference
            in the atlas annotation.

        Returns
        -------
        :
            True, if a region identifiable by `value` exists in the ontoloy. False otherwise.
        """
        try:
            _ = self._to_id(value, unreferenced=unreferenced)
        except KeyError:
            return False
        return True

    def are_regions(self,
                    values: Iterable,
                    unreferenced: bool=False,
                    duplicated: bool=True) -> np.ndarray:
        """
        Checks whether each of the elements in `values` are in the ontology.

        Parameters
        ----------
        values
            List of values that identify uniquely a brain region (e.g. their acronyms or their IDs).
        unreferenced
            If True, it considers as regions also those structures that have no reference
            in the atlas annotation.
        duplicated
            If False, it raises `ValueError` if `values` contains the same region more than once.
            Otherwise, it does nothing.

        Returns
        -------
        :
            A numpy array of bools being True where the corresponding value in `values` is a region and False otherwise.

        Raises
        ------
        ValueError
            If duplicates are detected in the list of acronyms.

        See also
        --------
        [`is_region`][braian.AtlasOntology.is_region]
        """
        ids = self._to_ids(values, unreferenced=unreferenced, duplicated=duplicated, check_all=True)
        return np.array([id is not None for id in ids], dtype=bool)

    def __contains__(self, key) -> bool:
        return self.is_region(key, unreferenced=False)

    def minimum_treecover(self,
                          regions: Iterable,
                          *,
                          unreferenced: bool=False,
                          blacklisted: bool=True,
                          key: Literal["id","acronym"]="acronym") -> list:
        """
        Gives the minimum set of structures that covers all the given regions, and _not more_.

        Parameters
        ----------
        regions
            The brain structures to cover, uniquely identified by their acronyms or their IDs.
        unreferenced
            If False, it does not consider from the cover all those brain regions that have
            no references in the atlas annotations.
        blacklisted
            If False, it does not consider from the cover all those brain regions that are
            currently blacklisted from the ontology.
            If `unreferenced` is True, it is ignored.
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            A list of acronyms of regions

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        ValueError
            If `key` is not known as an unique identifier for brain structures

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

    @deprecated(since="1.1.0")
    def contains_all_children(self, parent: str, regions: Container[str]) -> bool:
        """
        Checks whether a brain region contains all the given regions.

        Parameters
        ----------
        parent
            An acronym of a brain region
        regions
            The regions to check as subregions of `parent`

        Returns
        -------
        :
            True if all `regions` are direct subregions of `parent`

        Raises
        ------
        KeyError
            If it can't find `parent` in the ontology
        """
        return set(self.direct_subregions[parent]) == set(regions)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.parents"])
    def get_parent_regions(self,
                           regions: Iterable,
                           *,
                           key: Literal["id","acronym"]="acronym") -> dict:
        """
        Finds, for each of the given brain regions, their parent region in the ontology.
        It accepts blacklisted and unreferenced structures, too.

        Parameters
        ----------
        regions
            The brain structures to search the parent for.
        key
            The unique identifier used to identify a brain structure

        Returns
        -------
        :
            A dicrtionary mapping region→parent

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        ValueError
            If `key` is not known as an unique identifier for brain structures
        ValueError
            If there is at least one brain structure that appears twice in `regions`.
        """
        return self.parents(regions, key=key)

    def parents(self,
                regions: Iterable,
                *,
                key: Literal["id","acronym"]="acronym") -> dict:
        """
        Finds, for each of the given brain regions, their parent region in the ontology.
        For the root brain region, its parent will be `None`.\\
        It accepts blacklisted and unreferenced structures, too.

        Parameters
        ----------
        regions
            The brain structures to search the parent for.
        key
            The unique identifier used to identify the returned brain structure

        Returns
        -------
        :
            A dicrtionary mapping region→parent

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        ValueError
            If `key` is not known as an unique identifier for brain structures
        ValueError
            If there is at least one brain structure that appears twice in `regions`.
        """
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
        Get all brain structures that, combined, make the whole parent region of `region`.\\
        It does not include the regions that have no reference in the atlas annotations.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            All `region`'s sibilngs, including itself

        Raises
        ------
        KeyError
            If `region` is not found in the ontology.
        ValueError
            If `key` is not known as an unique identifier for brain structures
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
        Retrieve the sibling structures of `region` such that, together with `region`, they make the parent region.\\
        It does not include the regions that have no reference in the atlas annotations.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            All `region`'s sibilngs, excluding itself.

        Raises
        ------
        KeyError
            If `region` is not found in the ontology.
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        id = self._to_id(region, unreferenced=False)
        siblings = self._tree.siblings(id)
        return self._nodes_to_attr(siblings, attr=key)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.subregions"])
    def list_all_subregions(self,
                            acronym: str,
                            mode: Literal["breadth", "depth"]="breadth",
                            blacklisted: bool=True,
                            unreferenced: bool=False) -> list:
        """
        Gets the complete set of subregions of `acronym`, in hierarchical order.

        Parameters
        ----------
        acronym
            The acronym of the brain region.
        mode
            The hiearchical order in which the sequence of brain structures will be presented:
            -breadth-first or depth-first.
        blacklisted
            If True, it considers as subregion any possible blacklisted structure, too.
        unreferenced
            If True, it considers as subregion any possible structure having no reference
            in the atlas annotations.

        Raises
        ------
        KeyError
            If `acronym` is not found in the ontology.
        ValueError
            When `mode` has an invalid value.
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        return self.subregions(acronym, mode=mode,
                               blacklisted=blacklisted, unreferenced=unreferenced,
                               key="acronym")

    def subregions(self,
                   region: str|int,
                   *,
                   mode: Literal["breadth", "depth"]="breadth",
                   blacklisted: bool=True,
                   unreferenced: bool=False,
                   key: Literal["id", "acronym"]="acronym") -> list:
        """
        Gets the complete set of subregions of `region`, in hierarchical order.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        mode
            The hiearchical order in which the sequence of brain structures will be presented:
            -breadth-first or depth-first.
        blacklisted
            If True, it considers as subregion any possible blacklisted structure, too.
        unreferenced
            If True, it considers as subregion any possible structure having no reference
            in the atlas annotations.
        key
            The unique identifier used to identify the returned brain structure

        Raises
        ------
        KeyError
            If `region` is not found in the ontology.
        ValueError
            When `mode` has an invalid value.
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
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
        return self._nodes_to_attr((tree[r] for r in subregions), attr=key)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.ancestors"])
    def get_regions_above(self, acronym: str) -> list[str]:
        """
        Gets all the regions for which `region` is a subregion.

        It raises `KeyError` if `acronym` is an unreferenced brain structure.

        Parameters
        ----------
        acronym
            A region, identified by its acronym.

        Returns
        -------
        :
            The whole ancestry of structures containing `acronym`, excluding `acronym` itself

        Raises
        ------
        KeyError
            If `acronym` is not found in the ontology.
        """
        return self.ancestors(acronym, key="acronym")

    def ancestors(self,
                  region: str|int,
                  *,
                  unreferenced: bool=False,
                  key: Literal["id","acronym"]="acronym") -> list:
        """
        Gets all the regions for which `region` is a subregion.

        It raises `KeyError` if `region` is an unreferenced brain structure.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        unreferenced
            If True, `region` can also be a structure with no reference
            in the atlas annotations.
        key
            The region identifier of the returned brain structures.


        Returns
        -------
        :
            The whole ancestry of structures containing `region`, excluding `region` itself

        Raises
        ------
        KeyError
            If `region` is not found in the ontology, or if `unreferenced=False`
            and `region` is an unreferenced structure.
        ValueError
            If `key` is not known as an unique identifier for brain structures
        """
        self._check_node_attr(key)
        id = self._to_id(region, unreferenced=unreferenced)
        ancestors_ids = self._atlas.structures[id]["structure_id_path"][-2::-1]
        if key == "id":
            return ancestors_ids
        else:
            return self.to_acronym(ancestors_ids, mode=None)
        # node = self._tree[id]
        # nodes = []
        # while (node:=self._tree.parent(node.id)) is not None:
        #     nodes.append(node)
        # return self._nodes_to_attr(nodes, attr=key)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.to_acronym"])
    def ids_to_acronym(self, ids: Container[int],
                       mode: Literal["breadth", "depth"]|None="depth") -> list[str]|str:
        """
        Converts the given brain region IDs into their corresponding acronyms.

        Parameters
        ----------
        ids
            The brain structure(s) uniquely identified by their IDs.
        mode
            If None, it returns the acronyms in the same order as `ids`.
            Otherwise, it returns them in breadth-first or depth-first order.

        Returns
        -------
        :
            The region acronym(s).

        Raises
        ------
        KeyError
            If it can't find at least one of the `ids` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `ids`.
        ValueError
            When `mode` has an invalid value.
        """
        return self.to_acronym(ids, mode=mode)

    def to_acronym(self, regions: Iterable[int]|int,
                   *, mode: Literal["breadth", "depth"]|None="depth") -> list[str]|str:
        """
        Converts the given brain regions into their corresponding acronyms.

        Parameters
        ----------
        regions
            The brain structure(s) uniquely identified by their IDs (or their acronyms, even).
        mode
            If None, it returns the acronyms in the same order as `regions`.
            Otherwise, it returns them in breadth-first or depth-first order.

        Returns
        -------
        :
            The region acronym(s).

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `regions`.
        ValueError
            When `mode` has an invalid value.
        """
        if isinstance(regions, int):
            return self._node_to_attr(self._tree_full[regions], attr="acronym")
        regions = self._to_nodes(regions, unreferenced=True, duplicated=False)
        if mode is not None:
            regions = self._sort(regions, mode=mode)
        return self._nodes_to_attr(regions, attr="acronym")

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.to_id"])
    def acronyms_to_id(self, acronyms: Container[str]|str,
                       mode: Literal["breadth", "depth"]|None="depth") -> list[int]|int:
        """
        Converts the given brain region acronyms into their corresponding IDs.

        Parameters
        ----------
        acronyms
            The brain structure(s) uniquely identified by their acronyms.
        mode
            If None, it returns the IDs in the same order as `acronyms`.
            Otherwise, it returns them in breadth-first or depth-first order.

        Returns
        -------
        :
            The region ID(s).

        Raises
        ------
        KeyError
            If it can't find at least one of the `acronyms` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `acronyms`.
        ValueError
            When `mode` has an invalid value.
        """
        return self.to_id(acronyms, mode=mode)

    def to_id(self, regions: Iterable[str],
              *, mode: Literal["breadth", "depth"]|None="depth") -> list[int]|int:
        """
        Converts the given brain regions into their corresponding IDs.

        Parameters
        ----------
        regions
            The brain structure(s) uniquely identified by their acronyms (or their IDs, even).
        mode
            If None, it returns the IDs in the same order as `regions`.
            Otherwise, it returns them in breadth-first or depth-first order.

        Returns
        -------
        :
            The region ID(s).

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `regions`.
        ValueError
            When `mode` has an invalid value.
        """
        if isinstance(regions, str):
            return self._to_id(regions, unreferenced=True)
        ids = self._to_ids(regions, unreferenced=True, duplicated=False, check_all=False)
        if mode is not None:
            return self._sort(ids, mode=mode)
        return ids

    def has_selection(self) -> bool:
        """
        Checks if there are structures currently selected in the ontology.

        Returns
        -------
        :
            True, if the ontology has at least one brain structure selected. Otherwise, False.

        See also
        --------
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        return self._selected

    def unselect_all(self):
        """
        Resets the selection in the ontology.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        for n in self._tree_full.all_nodes():
            n.selected = False
        self._selected = False

    def _select(self, regions: Iterable[RegionNode], *, add: bool):
        # NOTE: it selects also unreferenced and blacklisted regions
        if not add:
            self.unselect_all()
        for n in regions:
            n.selected = True
        self._selected = True

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.select"])
    def select_regions(self, regions: Iterable, key: str="acronym"):
        """
        Select `regions` in the ontology.

        Parameters
        ----------
        regions
            The brain structures uniquely identified by their IDs or their acronyms.
        key
            The region identifier of the returned brain structures.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `regions`.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        return self.select(regions)

    def select(self, regions: Iterable):
        """
        Select `regions` in the ontology.

        Parameters
        ----------
        regions
            The brain structures uniquely identified by their IDs or their acronyms.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `regions`.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        regions = self._to_nodes(regions, unreferenced=True, duplicated=False)
        self._select(regions, add=False)

    @deprecated(since="1.1.0", params=["key"])
    def add_to_selection(self, regions: Iterable, key: str=None):
        """
        Add the given brain regions to the current selection in the ontology

        Parameters
        ----------
        regions
            The brain structures uniquely identified by their IDs or their acronyms.
        key
            The region identifier of the returned brain structures.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `regions`.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        regions = self._to_nodes(regions, unreferenced=False, duplicated=True)
        self._select(regions, add=True)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.selected"])
    def get_selected_regions(self,
                            *,
                            key: Literal["id","acronym"]="acronym") -> list:
        """
        Gets the list of the selected, non-overlapping, non-blacklisted brain structures.\\
        If a region $R$ and its subregions $R_1$, $R_2$,..., $R_n$ are selected,
        it will return only $R$.

        Parameters
        ----------
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            A list of brain structures uniquely identified by `key`.

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        return self.selected(blacklisted=False, unreferenced=False, key=key)

    def selected(self,
                 *,
                 blacklisted: bool=False,
                 unreferenced: bool=False,
                 key: Literal["id","acronym"]="acronym") -> list:
        """
        Gets the list of the selected, non-overlapping, non-blacklisted brain structures.\\
        If a region $R$ and its subregions $R_1$, $R_2$,..., $R_n$ are selected,
        it will return only $R$.

        Parameters
        ----------
        blacklisted
            If True, it includes also structures blacklisted in the ontology.
        unreferenced
            If True, it includes also structures with no reference in the atlas annotations.
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            A list of brain structures uniquely identified by `key`.

        Raises
        ------
        ValueError
            If `key` is not known as an unique identifier for brain structures.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        tree = self._tree_full if unreferenced else self._tree
        selected_trees = first_subtrees(tree, lambda n: (blacklisted or not n.blacklisted) and n.selected)
        return self._nodes_to_attr(selected_trees, attr=key)

    def select_at_depth(self, depth: int):
        """
        Select all non-overlapping brain regions at the same depth in the ontology.\\
        If a brain region is above the given depth but has no sub-regions, it is selected anyway.
        This grants that the list of [selected][braian.AtlasOntology.selected] brain structures
        will be non-overlapping and comprehensive of the whole-brain, excluding eventual blacklisted regions.

        Parameters
        ----------
        depth
            The desired depth in the ontology to select

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        def select(n: RegionNode) -> bool:
            depth_ = self._tree.depth(n)
            return depth_ == depth or (depth_ < depth and n.is_leaf(self._tree.identifier))
        self._select(first_subtrees(self._tree, select), add=False)

    def select_leaves(self):
        """
        Select all the non-overlapping smallest brain regions in the ontology.
        A region is also selected if it's not the smallest possible,
        but all of its sub-regions have no reference in the atlas annotations.
        This grants that the list of [selected][braian.AtlasOntology.selected] brain structures
        will be non-overlapping and comprehensive of the whole-brain, excluding eventual blacklisted regions.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        self._select(self._tree.leaves(), add=False)

    def _select_partition(self, partition: str):
        """
        Raises
        ------
        MissingWholeBrainPartitionError
            If `partition` is not known for this atlas's ontology.
        """
        file = utils.resource(f"partitions/{self.name}/{partition.replace(' ', '_')}.csv")
        if not file.exists():
            raise MissingWholeBrainPartitionError(self.name, partition)
        key = "id"
        regions = pd.read_csv(file, sep="\t", index_col=0)
        self.select(regions[key].values)

    def select_major_divisions(self):
        """
        Select all major divisions in the ontology.\\
        This is short list of larger macro brain structures.

        The term "Major Divisions" comes from
        [Wang et al., 2020](https://doi.org/10.1016/j.cell.2020.04.007)
        by the Allen Institute.

        Raises
        ------
        MissingWholeBrainPartitionError
            If there is no known list of "major divisions" for this atlas's ontology.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`partition`][braian.AtlasOntology.partition]
        """
        self._select_partition("major divisions")

    def select_summary_structures(self):
        """
        Select all summary structures in the ontology.\\
        This is a list of finer brain structures that usually reflects the
        scientific community's interest when investigating brain circuits.

        The term "Summary Structures" comes from the Allen Institute, when they defined
        a set of non-overlapping, finer divisions in
        [Wang et al., 2020](https://doi.org/10.1016/j.cell.2020.04.007)

        Raises
        ------
        MissingWholeBrainPartitionError
            If there is no known list of "summary structures" for this atlas's ontology.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`partition`][braian.AtlasOntology.partition]
        """
        self._select_partition("summary structures")

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.partition"])
    def get_regions(self, selection_method: str) -> list[str]:
        """
        Returns a list of non-overlapping, brain structures representing
        the whole non-blacklisted brain.

        Parameters
        ----------
        selection_method
            In order to retrieve the partition, it uses some of the available section methods.
            Must be "summary structure", "major divisions", "leaves" or "depth <d>"

        Returns
        -------
        :
            A list of brain structures uniquely identified by `key`.

        Raises
        ------
        MissingWholeBrainPartitionError
            If `selection_method` is `"summary structures" or "major divisions",
            but no corresponding list of regions is not known for this atlas's ontology.
        ValueError
            If `selection_method` is not recognised
        ValueError
            If `key` is not known as an unique identifier for brain structures.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        """
        return self.partition(selection_method)

    def _partition(self,
                   selection: Sequence|Callable[[],None],
                   *,
                   blacklisted: bool=False,
                   unreferenced: bool=False,
                   key: Literal["id","acronym"]="acronym") -> list:
        """
        Raises
        ------
        KeyError
            If at least one of the given `selection` regions is not found in the ontology.
        ValueError
            If `key` is not known as an unique identifier for brain structures.
        """
        old_selection = self.selected(blacklisted=True, unreferenced=True, key="id")
        if isinstance(selection, Callable):
            selection()
        else:
            selection = self._to_nodes(selection, unreferenced=True, duplicated=True)
            self._select(selection, add=False)
        selected = self.selected(blacklisted=blacklisted, unreferenced=unreferenced, key=key)
        self.select(old_selection)
        return selected

    def partition(self,
                  selection_method: Literal["summary structures","major divisions","smallest","leaves","depth <n>"],
                  *,
                  blacklisted: bool=False,
                  unreferenced: bool=False,
                  key: Literal["id","acronym"]="acronym") -> list:
        """
        Returns a list of non-overlapping, brain structures representing
        the whole non-blacklisted brain.

        Parameters
        ----------
        selection_method
            In order to retrieve the partition, it uses some of the available section methods.
            Must be "summary structure", "major divisions", "leaves" or "depth <d>"
        blacklisted
            If True, it includes also structures blacklisted in the ontology.
        unreferenced
            If True, it includes also structures with no reference in the atlas annotations.

        Returns
        -------
        :
            A list of brain structures uniquely identified by `key`.

        Raises
        ------
        MissingWholeBrainPartitionError
            If `selection_method` is `"summary structures"` or `"major divisions"`,
            but no corresponding list of regions is not known for this atlas's ontology.
        ValueError
            If `selection_method` is not recognised
        ValueError
            If `key` is not known as an unique identifier for brain structures.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        """
        match selection_method:
            case "summary structures":
                select = self.select_summary_structures
            case "major divisions":
                select = self.select_major_divisions
            case "smallest" | "leaves":
                # excluded_from_leaves = set(braian.MAJOR_DIVISIONS) - {"Isocortex", "OLF", "CTXsp", "HPF", "STR", "PAL"}
                # excluded_from_leaves = self.minimimum_treecover(excluded_from_leaves)
                # self.blacklist_regions(excluded_from_leaves)
                select = self.select_leaves
            case s if s.startswith("depth"):
                n = selection_method.split(" ")[-1]
                try:
                    depth = int(n)
                except Exception:
                    raise ValueError("Could not retrieve the <n> parameter of the 'depth' method for 'selection_method'")
                select = lambda: self.select_at_depth(depth)  # noqa: E731
            case _:
                raise ValueError(f"Invalid value '{selection_method}' for selection_method")
        return self._partition(select, blacklisted=blacklisted, unreferenced=unreferenced, key=key)

    @deprecated(since="1.1.0", alternatives=["braian.AtlasOntology.partitioned"])
    def get_corresponding_md(self, acronym: str, *acronyms: str) -> OrderedDict[str, str]:
        """
        Finds the corresponding major division for each one of the `acronyms`.

        While it accepts blacklisted regions, it raises `KeyError`
        if given a region with no reference in the atlas annotations.

        Parameters
        ----------
        acronym
            Acronym of the brain region of which you want to know the corresponding major division

        Returns
        -------
        :
            An OrderedDict mapping $\\text{region acronym}→\\text{major division}$

        Raises
        ------
        KeyError
            If it can't find some `acronym` in the ontology
        """
        acronyms = (acronym, *acronyms)
        return self.partitioned(acronyms, partition="major divisions", key="acronym")

    def partitioned(self,
                    regions: Sequence,
                    *,
                    partition: Sequence|Literal["selection","summary structures","major divisions","smallest","leaves","depth <n>"],
                    blacklisted: bool=True,
                    unreferenced: bool=False,
                    key: Literal["id","acronym"]="acronym") -> OrderedDict[str|int,str|int]:
        """
        Partitions the given regions based on the specified partitioning method.

        Parameters
        ----------
        regions
            The brain structures uniquely identified by their IDs or their acronyms.
        partition
            The partitioning method to use.

            This can be one of the [predefined][braian.AtlasOntology.partition]
            partitioning strategies, as well as a custom partition defined by the current
            selection (`partition="selection"`) or by a sequence of regions identified
            by their IDs or their acronyms.
        blacklisted
            If True, `region` can also be a blacklisted structure.
        unreferenced
            If True, `region` can also be a structure with no reference
            in the atlas annotations.
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            An OrderedDict mapping $\\text{region}→\\text{partition}$

        Raises
        ------
        KeyError
            If at least one of the structures in `regions` or `partition`
            is not found in the ontology.
        KeyError
            If `unreferenced=False` one of the structures in `region` is an
            unreferenced structure.
        ValueError
            If there is at least one brain structure that appears twice in
            `regions` or `partition`.
        ValueError
            If `key` is not known as an unique identifier for brain structures.
        ValueError
            If `partition` is not recognised as a valid partitioning method.
            Please refer to [the documentation][braian.AtlasOntology.partition]
            for the list of valid partitioning methods.

        See also
        --------
        [`ancestors`][braian.AtlasOntology.ancestors]
        [`partition`][braian.AtlasOntology.partition]
        [`selected`][braian.AtlasOntology.selected]
        """
        self._check_node_attr(key)
        if key == "id":
            _regions = self._to_ids(regions, unreferenced=unreferenced, duplicated=False, check_all=False)
        else:
            _regions = self.to_acronym(regions, mode=None)
        if partition == "selection":
            _partition = self.selected(blacklisted=blacklisted, unreferenced=unreferenced, key=key)
        elif isinstance(partition, str):
            _partition = self.partition(partition, blacklisted=blacklisted, unreferenced=unreferenced, key=key)
        else:
            _partition = self._partition(partition, blacklisted=blacklisted, unreferenced=unreferenced, key=key)
        _partition = set(_partition)
        partition = OrderedDict()
        for region in _regions:
            if region in _partition:
                partition[region] = region
                continue
            for ancestor in self.ancestors(region, unreferenced=unreferenced, key=key):
                if ancestor in _partition:
                    partition[region] = ancestor
                    break
            if region not in partition:
                partition[region] = None
        return partition

    @deprecated(since="1.1.0")
    def get_layer1(self, *, key: Literal["id","acronym"]) -> list:
        """
        Returns the smallest regions in the cortical layer 1.

        Parameters
        ----------
        key
            The region identifier of the returned brain structures.

        Returns
        -------
        :
            A list of [leaf][braian.AtlasOntology.select_leaves] brains structures

        Raises
        ------
        ValueError
            If the cortical layer1 is not known for this ontology.
        """
        self._check_node_attr(key)
        import itertools
        import re
        if not re.match(r"allen_mouse_(10|25|50|100)um", self.name):
            raise ValueError(f"Could not extract layer 1 of the cortex. Incompatible atlas: '{self.name}'.")
        layer1 = [s for s in
                    itertools.chain(self.subregions("Isocortex", key="acronym"), self.subregions("OLF", key="acronym"))
                    if s.endswith("1")]
        if key == "acronym":
            return layer1
        return self._to_ids(layer1, unreferenced=False, duplicated=True, check_all=False)

    def to_igraph(self, unreferenced: bool=False, blacklisted: bool=True):
        """
        Translates the ontology into an `igraph` directed acyclic `Graph`,
        where nodes are brain structures and edges are region→subregion relationships.

        Parameters
        ----------
        unreferenced
            If False, it removes from the graph all those brain regions that have no references in the atlas annotations.
        blacklisted
            If False, it removes from the graph all those brain regions that are currently blacklisted from the ontology.\
            If `unreferenced` is True, it is ignored.

        Returns
        -------
        :
            A graph
        """
        tree = self._tree_full if unreferenced else self._tree
        id2n = dict()
        edges = []
        for n,id in enumerate(tree.expand_tree(mode=Tree.DEPTH, sorting=False)):
            id2n[id] = n
            parent = tree.parent(id)
            if parent is None:
                continue
            pid = parent.identifier
            edges.append((id2n[pid], n))
        g = ig.Graph(edges=edges, directed=True)
        for region in tree.all_nodes():
            graph_order = id2n[region.id]
            v = g.vs[graph_order]
            v["id"] = region.id
            v["name"] = region.acronym
            # v["full_name"] = region.name
            # v["depth"] = depth
            # v["color"] = region.hex_triplet
            # v["structural_level"] = region["st_level"]
        blacklisted_trees = set(self.blacklisted(unreferenced=False, key="acronym"))
        if blacklisted:
            _graph_utils.blacklist_regions(g, blacklisted_trees)
        else:
            _graph_utils.remove_branch(g, blacklisted_trees)
        if self.has_selection():
            _graph_utils.select_regions(g, self.selected())
        return g