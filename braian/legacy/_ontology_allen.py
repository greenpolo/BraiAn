import braian.utils
import igraph as ig
import json
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests

from brainglobe_atlasapi import list_atlases
from bs4 import BeautifulSoup
from collections import OrderedDict
from collections.abc import Container, Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Literal

from braian import _graph_utils
from braian.legacy import _visit_dict
from braian.utils import deprecated

__all__ = ["AllenBrainOntology", "MAJOR_DIVISIONS"]

MAJOR_DIVISIONS = [
    "Isocortex",
    "OLF",
    "HPF",
    "CTXsp",
    "STR",
    "PAL",
    "TH",
    "HY",
    "MB",
    "P",
    "MY",
    "CB"
]

UPPER_REGIONS = ["root", *MAJOR_DIVISIONS]

def set_blacklisted(node, is_blacklisted):
    node["blacklisted"] = is_blacklisted

def is_blacklisted(node) -> bool:
    return "blacklisted" in node and node["blacklisted"]

def set_reference(node, has_reference):
    node["has_reference"] = has_reference

def has_reference(node) -> bool:
    return "has_reference" in node and node["has_reference"]

def _version_from_abba_name(name: str) -> str:
    match name:
        case "Adult Mouse Brain - Allen Brain Atlas V3" | "Adult Mouse Brain - Allen Brain Atlas V3p1" | "allen_mouse_10um_java":
            return "CCFv3"
        case _:
            raise ValueError(f"Name not recognised as an Allen Mouse Brain-compatible atlas: '{name}'. Please, provide the version of the Common Coordinate Framework (CCF) manually.")

def _is_brainglobe_atlas(name: str) -> bool:
    last_atlases_versions: dict[str,str] = list_atlases.get_all_atlases_lastversions()
    return name in last_atlases_versions

def _get_brainglobe_name(name: str) -> bool:
    last_atlases_versions: dict[str,str] = list_atlases.get_all_atlases_lastversions()
    if name not in last_atlases_versions:
        raise ValueError(f"BrainGlobe atlas not found: '{name}'")
    last_version = last_atlases_versions[name]
    bg_atlas_name = (name+"_v"+last_version)
    return bg_atlas_name

@deprecated(since="1.1.0", alternatives=["braian.AtlasOntology"])
class AllenBrainOntology:
    # https://community.brain-map.org/t/allen-mouse-ccf-accessing-and-using-related-data-and-tools/359
    def __init__(self,
                 allen_json: str|Path|dict,
                 blacklisted_acronyms: Iterable=[],
                 name: str="allen_mouse_10um_java",
                 version: str|None=None,
                 unreferenced: bool=False):
        """
        Crates an ontology of brain regions based on Allen Institute's structure graphs.
        To know more where to get the structure graphs, read the
        [official guide](https://community.brain-map.org/t/downloading-an-ontologys-structure-graph/2880)
        from Allen Institute.

        This implementation, compared to [`braian.AllenBrainOntology`][braian.AllenBrainOntology], relies
        on public API from Allen Institute. Its reimplementation, instead, relies on
        [`braian.AtlasOntology`][braian.AtlasOntology] which, internally, uses
        [BrainGlobe](https://brainglobe.info/)'s API.

        Parameters
        ----------
        allen_json
            The path to an Allen structural graph json
        blacklisted_acronyms
            Acronyms of branches from the onthology to exclude completely from the analysis
        name
            The name of a Allen-compatible mouse brain atlas from ABBA, BrainGlobe or others.
            (e.g., "allen_mouse_10um" or "Adult Mouse Brain - Allen Brain Atlas V3p1").
        version
            The version of the Common Coordinates Framework from Allen's mouse brain atlas.
            Must be `"CCFv1"`, `"CCFv2"`, `"CCFv3"`, `"CCFv4"` or `None`.
            If `version` is None, it defaults tries to deduce it from the `name` of the atlas.
        unreferenced
            If True, it considers as part of the ontology all those brain regions that have no references in the atlas annotations.
            Otherwise, it removes them from the ontology.
            On Allen's website, unreferenced brain regions are identified in grey italic text: [](https://atlas.brain-map.org).

        Raises
        ------
        ValueError
            If no `version` is provided and the given `name` is not compatible with ABBA nor with BrainGlobe,
            thus making it impossible to deduce the version of the Common Coordinate Framework.
        ValueError
            If the provided `name` is from a BrainGlobe atlas but its not available locally on the computer.
        ValueError
            If the provided `version` is not recognised.
        ValueError
            If any of the `blacklisted_acronyms` is not found in the ontology.
        """
        # TODO: we should probably specify the size (10nm, 25nm, 50nm) to which the data was registered
        if isinstance(allen_json, (str, Path)):
            with open(allen_json, "r") as file:
                allen_data = json.load(file)
            if "msg" in allen_data:
                self.dict = allen_data["msg"][0]
            else:
                self.dict = allen_data
        else:
            assert isinstance(allen_json, dict)
            self.dict = deepcopy(allen_json)
        # First label every region as "not blacklisted"
        _visit_dict.visit_bfs(self.dict, "children", lambda n,d: set_blacklisted(n, False))
        _visit_dict.visit_bfs(self.dict, "children", lambda n,d: set_reference(n, True))
        if blacklisted_acronyms:
            self.blacklist(blacklisted_acronyms, key="acronym", unreferenced=False)
            # we don't prune, otherwise we won't be able to work with region_to_exclude (QuPath output)
            # prune_where(self.dict, "children", lambda x: x["acronym"] in blacklisted_acronyms)
        self.name = name
        if version is None:
            if _is_brainglobe_atlas(self.name):
                version = "CCFv3"
            else:
                version = _version_from_abba_name(self.name)
        self.annotation_version = self._get_allen_version(version) # used to remove unreferenced regions and to select isocortex layer 1
        if not unreferenced:
            if _is_brainglobe_atlas(self.name):
                unannoted_regions = self._get_unannoted_bg_regions() # throws ValueError if the atlas is not downloaded
            else:
                unannoted_regions = self._get_unannoted_regions()
            self.blacklist(unannoted_regions, key="id", unreferenced=True)

        self._add_depth_to_regions()
        self._mark_major_divisions()
        """The name of the atlas accordingly to ABBA/BrainGlobe"""
        self.full_name: dict[str,str] = self._get_full_names()
        """A dictionary mapping a regions' acronym to its full name. It also contains the names for the blacklisted and unreferenced regions."""
        self.parent_region: dict[str,str] = self._get_all_parent_areas() #: A dictionary mapping region's acronyms to the parent region. It does not have 'root'.
        self.direct_subregions: dict[str,list[str]] = self._get_all_subregions()
        """A dictionary mappin region's acronyms to a list of direct subregions.

        Examples
        --------
        >>> braian.utils.cache("ontology.json", "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        >>> brain_ontology = braian.legacy.AllenBrainOntology("ontology.json", [])
        >>> brain_ontology.direct_subregions["ACA"]
        ["ACAv", "ACAd"]                                    # dorsal and ventral part
        >>> brain_ontology.direct_subregions["ACAv"]
        ["ACAv5", "ACAv2/3", "ACAv6a", "ACAv1", "ACAv6b"]   # all layers in ventral part
        """

    def _get_allen_version(self, version):
        match version:
            case "2015" | "CCFv1" | "ccfv1" | "v1" | 1:
                return "ccf_2015"
            case "2016" | "CCFv2" | "ccfv2" | "v2" | 2:
                return "ccf_2016"
            case "2017" | "CCFv3" | "ccfv3" | "v3" | 3:
                return "ccf_2017"
            case "2022" | "CCFv4" | "ccfv4" | "v4" | 4:
                return "ccf_2022"
            case _:
                raise ValueError(f"Unrecognised Allen atlas version: '{version}'")

    def _get_unannoted_bg_regions(self) -> list[int]:
        bg_atlas_name = _get_brainglobe_name(self.name)
        atlas_meshes_dir = Path.home()/".brainglobe"/bg_atlas_name/"meshes"
        if not atlas_meshes_dir.exists():
            raise ValueError(f"BrainGlobe atlas not downloaded: '{bg_atlas_name}'")
        regions_w_annotation = [int(p.stem) for p in atlas_meshes_dir.iterdir() if p.suffix == ".obj"]
        regions_wo_annotation = _visit_dict.get_where(self.dict, "children", lambda n,d: n["id"] not in regions_w_annotation, _visit_dict.visit_dfs)
        return [region["id"] for region in regions_wo_annotation]

    def _get_unannoted_regions(self) -> tuple[list[str], str]:
        # alternative implementation: use annotation's nrrd file - https://help.brain-map.org/display/mousebrain/API#API-DownloadAtlas3-DReferenceModels
        # this way is not totally precise, because some masks files are present even if they're empty
        #
        # Regarding ccf_2022, little is known about future plans of publishing strucutral masks for resolutions lower than 10nm
        # https://community.brain-map.org/t/2022-ccfv3-mouse-atlas/2287
        url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/{self.annotation_version}/structure_masks/structure_masks_10"
        try:
            site_content = requests.get(url).content
        except requests.ConnectionError:
            print(f"WARNING: could not remove unannoted regions from {self.annotation_version} ontology")
            return [], None
        soup = BeautifulSoup(site_content, "html.parser")
        try:
            if self.annotation_version == "ccf_2022":
                regions_w_annotation = [int(link["href"].removeprefix("./").split("_")[0])
                                        for link in soup.select('a[href*=".nii.gz"]')]
            else:
                regions_w_annotation = [int(link["href"].\
                                            removeprefix("./").\
                                            removeprefix("structure_").\
                                            removesuffix(".nrrd")
                                        ) for link in soup.select('a[href*=".nrrd"]')]
        except ValueError:
            print(f"WARNING: could not remove unannoted regions from {self.annotation_version} ontology")
            return [], None
        regions_wo_annotation = _visit_dict.get_where(self.dict, "children", lambda n,d: n["id"] not in regions_w_annotation, _visit_dict.visit_dfs)
        return [region["id"] for region in regions_wo_annotation]

    def is_region(self, r: int|str, key: str="acronym", unreferenced: bool=False) -> bool:
        """
        Checks whether a region is in the ontology.

        Parameters
        ----------
        r
            A value that uniquely indentifies a brain region (e.g. its acronym).
        key
            The key in Allen's structural graph used to identify `r`.
        unreferenced
            If True, it considers as region also those structures that have no reference
            in the atlas annotation.

        Returns
        -------
        :
            True, if a region identifiable by `r` exists in the ontoloy. False otherwise.
        """
        region_tree = _visit_dict.find_subtree(self.dict, key, r, "children")
        return region_tree is not None and (unreferenced or has_reference(region_tree))

    def are_regions(self, a: Iterable, key: str="acronym", unreferenced: bool=False) -> npt.NDArray:
        """
        Checks whether each of the elements of the iterable are in the ontology.

        Parameters
        ----------
        a
            List of values that identify uniquely a brain region (e.g. their acronyms).
        key
            The key in Allen's structural graph used to identify `a`.
        unreferenced
            If True, it considers regions also those structures that have no reference
            in the atlas annotation.

        Returns
        -------
        :
            A numpy array of bools being True where the corresponding value in `a` is a region and False otherwise.

        Raises
        ------
        ValueError
            If duplicates are detected in the list of acronyms.
        """
        assert isinstance(a, Iterable), f"Expected a '{Iterable}' but get '{type(a)}'"
        a = list(a)
        s = set(a)
        if len(a) != len(s):
            raise ValueError(f"Duplicates detected in the list of '{key}' requested to be identified as brain regions")
        is_region = np.full_like(a, False, dtype=bool)
        def check_region(node, depth):
            id = node[key]
            if (unreferenced or has_reference(node)) and id in s:
                is_region[a.index(id)] = True # NOTE: if it wont work properly if 'a' has duplicates
        _visit_dict.visit_dfs(self.dict, "children", check_region)
        return is_region

    def _check_regions(self, regions: Iterable, key: str, unreferenced: bool):
        if not all(is_region:=self.are_regions(regions, key=key, unreferenced=unreferenced)):
            raise KeyError(f"Regions not found: '{key}(s)'={np.asarray(regions)[~is_region]}")

    @deprecated(since="1.1.0")
    def contains_all_children(self, parent: str, regions: Container[str]) -> bool:
        """
        Check whether a brain region contains all the given regions

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
        """
        return set(self.direct_subregions[parent]) == set(regions)

    def minimum_treecover(self,
                          acronyms: Iterable[str],
                          unreferenced: bool=False,
                          blacklisted: bool=True) -> set[str]:
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

        Returns
        -------
        :
            A list of acronyms of regions

        Examples
        --------
        >>> braian.utils.cache("ontology.json", "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        >>> brain_ontology = braian.legacy.AllenBrainOntology("ontology.json", [], version="v3")
        >>> sorted(brain_ontology.minimum_treecover(['P', 'MB', 'TH', 'MY', 'CB', 'HY']))
        {'BS', 'CB'}

        >>> sorted(brain_ontology.minimum_treecover(["RE", "Xi", "PVT", "PT", "TH"]))
        {'MTN', 'TH'}
        """
        g = self.to_igraph(unreferenced=unreferenced, blacklisted=blacklisted)
        vs = _graph_utils.minimum_treecover(g.vs.select(name_in=acronyms))
        return {v["name"] for v in vs}

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.blacklist"])
    def blacklist_regions(self,
                          regions: Iterable,
                          key: str="acronym",
                          has_reference: bool=True):
        """
        Blacklists from further analysis the given `regions` the ontology, as well as all their sub-regions.
        If the reason of blacklisting is that `regions` no longer exist in the used version of the
        Common Coordinate Framework, set `has_reference=False`

        Parameters
        ----------
        regions
            Regions to blacklist
        key
            The key in Allen's structural graph used to identify the `regions` to blacklist
        has_reference
            If `regions` exist or not in the used version of the CCF.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        """
        return self.blacklist(regions=regions, key=key, unreferenced=not has_reference)

    def blacklist(self,
                  regions: Iterable,
                  key: str="acronym",
                  unreferenced: bool=False):
        """
        Blacklists from further analysis the given `regions` the ontology, as well as all their sub-regions.
        If the reason of blacklisting is that `regions` no longer exist in the used version of the
        Common Coordinate Framework, set `has_reference=False`

        Parameters
        ----------
        regions
            Regions to blacklist
        key
            The key in Allen's structural graph used to identify the `regions` to blacklist
        unreferenced
            If `regions` exist or not in the used version of the CCF.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology
        """
        self._check_regions(regions, key=key, unreferenced=True)
        for region_value in regions:
            region_node = _visit_dict.find_subtree(self.dict, key, region_value, "children")
            _visit_dict.visit_bfs(region_node, "children", lambda n,d: set_blacklisted(n, True))
            if unreferenced:
                _visit_dict.visit_bfs(region_node, "children", lambda n,d: set_reference(n, False))
        if unreferenced:
            self.direct_subregions = self._get_all_subregions()
            self.parent_region = self._get_all_parent_areas()

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.blacklisted"])
    def get_blacklisted_trees(self,
                              unreferenced: bool=False,
                              key: str="acronym") -> list:
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

    def blacklisted(self, unreferenced: bool=False, key: str="acronym") -> list:
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
        if unreferenced:
            def select_node(n: dict, d: int):
                return is_blacklisted(n) or not has_reference(n)
        else:
            def select_node(n: dict, d: int):
                return is_blacklisted(n) and has_reference(n)
        regions = _visit_dict.non_overlapping_where(self.dict, "children", select_node, mode="bfs")
        return [region[key] for region in regions]

    def _get_unreferenced_trees(self, key: str="acronym") -> list:
        def select_node(n: dict, d: int):
            return not has_reference(n)
        regions = _visit_dict.non_overlapping_where(self.dict, "children", select_node, mode="bfs")
        return [region[key] for region in regions]

    def _mark_major_divisions(self):
        _visit_dict.add_boolean_attribute(self.dict, "children", "major_division", lambda node,d: node["acronym"] in MAJOR_DIVISIONS)

    def _add_depth_to_regions(self):
        def add_depth(node, depth):
            node["depth"] = depth
        _visit_dict.visit_bfs(self.dict, "children", add_depth)

    def select_at_depth(self, depth: int):
        """
        Select all non-overlapping brain regions at the same depth in the ontology.\\
        If a brain region is above the given depth but has no sub-regions, it is selected anyway.
        This grants that the list of [selected][braian.legacy.AllenBrainOntology.get_selected_regions] brain structures
        will be non-overlapping and comprehensive of the whole-brain, excluding eventual blacklisted regions.

        Parameters
        ----------
        depth
            The desired depth in the ontology to select

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        def is_selected(node, _depth):
            return _depth == depth or (_depth < depth and (not node["children"] or
                                                           not any(has_reference(child) for child in node["children"])))
        _visit_dict.add_boolean_attribute(self.dict, "children", "selected", is_selected)

    def select_at_structural_level(self, level: int):
        """
        Select all non-overlapping brain regions at the same structural level in the ontology.
        The _structural level_ is an attribute given to each region by Allen,
        defining different level of granularity to study the brain.
        If a brain region is above the given `level` and has no sub-regions, it is not selected.

        Parameters
        ----------
        level
            The structural level in the ontology to select

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        _visit_dict.add_boolean_attribute(self.dict, "children", "selected", lambda node,d: node["st_level"] == level)

    def select_leaves(self):
        """
        Select all the non-overlapping smallest brain regions in the ontology.
        A region is also selected if it's not the smallest possible,
        but all of its sub-regions have no reference in the atlas annotations.

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        _visit_dict.add_boolean_attribute(self.dict, "children", "selected", lambda node, d: _visit_dict.is_leaf(node, "children") or \
                                         has_reference(node) and all([not has_reference(child) for child in node["children"]]))
        # not is_blacklisted(node) and all([has_reference(child) for child in node["children"]])) # some regions (see CA1 in CCFv3) have all subregions unannoted

    def select_summary_structures(self):
        """
        Select all summary structures in the ontology.
        The list of Summary Structures is defined by Allen as a set of non-overlapping, finer divisions,
        independent of their exact depth in the tree. They are a set of brain regions often used in the literature.

        Summary Structures can be retrieved from Table S2 of this article: https://www.sciencedirect.com/science/article/pii/S0092867420304025

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        # if self.annotation_version is not None:
        # # if self.annotation_version!= "ccf_2017":
        #     raise ValueError("BraiAn does not support 'summary structures' selection_method on atlas versions different from CCFv3")
        file = braian.utils.get_resource_path("CCFv3_summary_structures.csv")
        key = "id"
        regions = pd.read_csv(file, sep="\t", index_col=0)
        self.select(regions[key].values, key=key)

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.select"])
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
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`selected`][braian.legacy.AllenBrainOntology.selected]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`partition`][braian.legacy.AllenBrainOntology.partition]
        """
        return self.select(regions, key=key)

    def select(self, regions: Iterable, key: str="acronym"):
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
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`selected`][braian.legacy.AllenBrainOntology.selected]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`partition`][braian.legacy.AllenBrainOntology.partition]
        """
        self._check_regions(regions, key=key, unreferenced=False)
        _visit_dict.add_boolean_attribute(self.dict, "children", "selected", lambda node,d: node[key] in regions)

    def add_to_selection(self, regions: Iterable, key: str="acronym"):
        """
        Add the given brain regions to the current selection in the ontology

        Parameters
        ----------
        regions
            The brain structures uniquely identified by their IDs or their acronyms.
        key
            The region identifier of the returned sibling structures.

        Raises
        ------
        KeyError
            If it can't find at least one of the `regions` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `regions`.

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        assert all(is_region:=self.are_regions(regions, key=key)), \
            f"Some regions are not part of the ontology: {np.asarray(regions)[~is_region]}"
        _visit_dict.add_boolean_attribute(self.dict, "children", "selected",
                              lambda node,d: ("selected" in node and node["selected"]) or node[key] in regions)

    def has_selection(self) -> bool:
        """
        Checks if there are structures currently selected in the ontology.

        Returns
        -------
        :
            True, if the ontology has at least one brain structure selected. Otherwise, False.

        See also
        --------
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        return "selected" in self.dict

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.selected"])
    def get_selected_regions(self, key: str="acronym") -> list:
        """
        Gets the list of the selected, non-overlapping, non-blacklisted brain structures.\\
        If a region $R$ and its subregions $R_1$, $R_2$,..., $R_n$ are selected,
        it will return only $R$.

        Parameters
        ----------
        key
            The key in Allen's structural graph used to indentify the regions

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
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`select`][braian.legacy.AllenBrainOntology.select]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`partition`][braian.legacy.AllenBrainOntology.partition]
        """
        return self.selected(key=key)

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
            This parameter is ignored.
        unreferenced
            This parameter is ignored.
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
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`select`][braian.legacy.AllenBrainOntology.select]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`partition`][braian.legacy.AllenBrainOntology.partition]
        """
        if not self.has_selection():
            return []
        regions = _visit_dict.non_overlapping_where(self.dict, "children", lambda n,d: n["selected"] and not is_blacklisted(n), mode="dfs")
        return [region[key] for region in regions]

    def unselect_all(self):
        """
        Resets the selection in the ontology

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        if "selected" in self.dict:
            _visit_dict.del_attribute(self.dict, "children", "selected")

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.partition"])
    def get_regions(self, selection_method: str) -> list[str]:
        """
        Returns a list of acronyms of non-overlapping regions based on the selection method.

        Parameters
        ----------
        selection_method
            Must be "summary structure", "major divisions", "leaves", "depth <d>" or "structural level <l>"

        Returns
        -------
        :
            A list of acronyms of brain regions

        Raises
        ------
        ValueError
            If `selection_method` is not recognised

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        """
        return self.partition(selection_method=selection_method)

    def partition(self, selection_method: str) -> list[str]:
        """
        Returns a list of non-overlapping, brain structures representing
        the whole non-blacklisted brain.

        Parameters
        ----------
        selection_method
            In order to retrieve the partition, it uses some of the available section methods.
            Must be "summary structure", "major divisions", "leaves", "depth <d>" or "structural level <l>"
        Returns
        -------
        :
            A list of brain structures uniquely identified by their acronym.

        Raises
        ------
        ValueError
            If `selection_method` is not recognised

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`selected`][braian.legacy.AllenBrainOntology.selected]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`select`][braian.legacy.AllenBrainOntology.select]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_at_structural_level`][braian.legacy.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        """
        old_selection = self.selected(key="id")
        if old_selection:
            self.unselect_all()
        match selection_method:
            case "summary structures":
                self.select_summary_structures()
            case "major divisions":
                self.select(MAJOR_DIVISIONS)
            case "smallest" | "leaves":
                # excluded_from_leaves = set(braian.MAJOR_DIVISIONS) - {"Isocortex", "OLF", "CTXsp", "HPF", "STR", "PAL"}
                # excluded_from_leaves = self.minimimum_treecover(excluded_from_leaves)
                # self.blacklist_regions(excluded_from_leaves)
                self.select_leaves()
            case s if s.startswith("depth"):
                n = selection_method.split(" ")[-1]
                try:
                    depth = int(n)
                except Exception:
                    raise ValueError("Could not retrieve the <n> parameter of the 'depth' method for 'selection_method'")
                self.select_at_depth(depth)
            case s if s.startswith("structural level"):
                n = selection_method.split(" ")[-1]
                try:
                    level = int(n)
                except Exception:
                    raise ValueError("Could not retrieve the <n> parameter of the 'structural level' method for 'selection_method'")
                self.select_at_structural_level(level)
            case _:
                raise ValueError(f"Invalid value '{selection_method}' for selection_method")
        selected_regions = self.selected()
        # print(f"You selected {len(selected_regions)} regions to plot.")
        self.select(old_selection, key="id")
        return selected_regions

    def ids_to_acronym(self, ids: Container[int], mode: Literal["breadth", "depth"]="depth") -> list[str]:
        """
        Converts the given brain region IDs into their corresponding acronyms.

        Parameters
        ----------
        ids
            The brain structures uniquely identified by their IDs.
        mode
            If None, it returns the acronyms in the same order as `ids`.
            Otherwise, it returns them in breadth-first or depth-first order.

        Returns
        -------
        :
            A list of acronyms

        Raises
        ------
        KeyError
            If it can't find at least one of the `ids` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `ids`.
        ValueError
            When `mode` has an invalid value.
        """
        match mode:
            case "breadth":
                visit_alg = _visit_dict.visit_bfs
            case "depth" | None:
                visit_alg = _visit_dict.visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth', 'depth' or None.")
        self._check_regions(ids, key="id", unreferenced=True)
        areas = _visit_dict.get_where(self.dict, "children", lambda n,d: n["id"] in ids, visit_alg)
        if mode is None:
            areas.sort(key=lambda r: list(ids).index(r["id"]))
        return [area["acronym"] for area in areas]

    def acronyms_to_id(self, acronyms: Container[str], mode: Literal["breadth", "depth"]="depth") -> list[int]:
        """
        Converts the given brain region acronyms into their corresponding IDs.

        Parameters
        ----------
        acronyms
            The brain structures uniquely identified by their acronyms.
        mode
            If None, it returns the IDs in the same order as `acronyms`.
            Otherwise, it returns them in breadth-first or depth-first order.

        Returns
        -------
        :
            A list of region IDs

        Raises
        ------
        KeyError
            If it can't find at least one of the `acronyms` in the ontology.
        ValueError
            If there is at least one brain structure that appears twice in `acronyms`.
        ValueError
            When `mode` has an invalid value.
        """
        match mode:
            case "breadth":
                visit_alg = _visit_dict.visit_bfs
            case "depth" | None:
                visit_alg = _visit_dict.visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth' and 'depth'.")
        self._check_regions(acronyms, key="acronym", unreferenced=True)
        regions = _visit_dict.get_where(self.dict, "children", lambda n,d: n["acronym"] in acronyms, visit_alg)
        if mode is None:
            regions.sort(key=lambda r: list(acronyms).index(r["acronym"]))
        return [r["id"] for r in regions]

    def get_sibiling_regions(self, region:str|int, key: str="acronym") -> list:
        """
        Get all brain regions that, combined, make the whole parent of the given `region`.

        It does not include the regions that have no reference in the atlas annotations.

        Parameters
        ----------
        region
            A region, identified by its ID or its acronym.
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            All `region`'s siblings, including itself

        Raises
        ------
        KeyError
            If the given `region` is not found in the ontology.
        """
        parents = _visit_dict.get_parents_where(self.dict,
                                               "children",
                                               lambda parent,child: child[key] == region and has_reference(child),
                                               key)
        if len(parents) != 1:
            raise KeyError(f"Parent region not found ('{key}'={region})")
        parent = parents[region]
        if parent is None: # region is the root
            return [region]
        return [sibling[key] for sibling in parent["children"] if has_reference(sibling)]
        ## Alternative implementation:
        # parent_id = find_subtree(self.dict, key, value, "children")["parent_structure_id"]
        # return [area[key] for area in find_subtree(self.dict, "id", parent_id, "children")["children"]]

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.parents"])
    def get_parent_regions(self, regions: Iterable, key: str="acronym") -> dict:
        """
        Finds, for each of the given brain regions, their parent region in the ontology.
        It accepts blacklisted and unreferenced structures, too.

        Parameters
        ----------
        regions
            The brain structures to search the parent for.
        key
            The key in Allen's structural graph used to indentify the regions

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
        parents = _visit_dict.get_parents_where(self.dict, "children", lambda parent,child: child[key] in regions, key)
        for region in regions:
            if region not in parents.keys():
                raise KeyError(f"Parent region not found ('{key}'={region})")
        return {region: (parent[key] if parent else None) for region,parent in parents.items()}

        # parents = get_parents_where(self.dict, "children", lambda parent,child: child[key] in values, key)
        # for area in values:
        #     assert area in parents.keys(), f"Can't find the parent of an area with '{key}': {area}"
        # return {area: (parent[key] if parent else None) for area,parent in parents.items()}

    def _get_all_parent_areas(self, key: str="acronym") -> dict:
        """
        Finds, for each brain region in the ontology, the corresponding parent region.
        The "root" region has no entry in the returned dictionary

        The resulting dictionary does not include unreferenced brain regions.

        Parameters
        ----------
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            A dictionary mapping region→parent
        """
        return {subregion: region for region,subregion in self._get_edges(key, unreferenced=False)}

    def _get_edges(self, key: str="id", unreferenced: bool=True) -> list[tuple]:
        assert key in ("id", "acronym", "graph_order"), "'key' parameter must be  'id', 'acronym' or 'graph_order'"
        edges = []
        _visit_dict.visit_parents(self.dict,
                                 "children",
                                 lambda region,subregion:
                                    edges.append((region[key], subregion[key]))
                                        if region is not None and (unreferenced or has_reference(subregion))
                                    else None)
        return edges

    def _get_all_subregions(self, key: str="acronym") -> dict:
        """
        returns a dictionary where all parent regions are the keys,
        and the subregions that belong to the parent are stored
        in a list as the value corresponding to the key.
        Regions with no subregions have no entries in the dictionary.

        The resulting dictionary does not include unreferenced brain regions.

        Parameters
        ----------
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            a dictionary that maps region→[subregions...]
        """
        subregions = dict()
        def add_subregions(node, depth):
            if node["children"] and has_reference(node) and any([has_reference(child) for child in node["children"]]):
                subregions[node[key]] = [child[key] for child in node["children"] if has_reference(child)]
        _visit_dict.visit_bfs(self.dict, "children", add_subregions)
        return subregions

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.subregions"])
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
        """
        return self.subregions(acronym, mode=mode,
                               blacklisted=blacklisted, unreferenced=unreferenced)

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
            This parameter is ignored.

        Raises
        ------
        KeyError
            If `region` is not found in the ontology.
        ValueError
            When `mode` has an invalid value.
        """
        match mode:
            case "breadth":
                visit_alg = _visit_dict.visit_bfs
            case "depth":
                visit_alg = _visit_dict.visit_dfs
            case _:
                raise ValueError(f"Unsupported mode '{mode}'. Available modes are 'breadth' and 'depth'.")

        attr = "acronym"
        region = _visit_dict.find_subtree(self.dict, attr, region, "children")
        if not region or (not unreferenced and not has_reference(region)):
            raise KeyError(f"Region not found ('{attr}'='{region}')")
        subregions = _visit_dict.get_all_nodes(region, "children", visit=visit_alg)
        return [subregion[attr] for subregion in subregions if (blacklisted or not is_blacklisted(subregion)) and (unreferenced or has_reference(subregion))]

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.ancestors"])
    def get_regions_above(self, acronym: str) -> list[str]:
        """
        Gets all the regions for which `acronym` is a subregion.

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
        return self.ancestors(acronym)

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
            This parameter is ignored.
        key
            This parameter is ignored.


        Returns
        -------
        :
            The whole ancestry of structures containing `region`, excluding `region` itself

        Raises
        ------
        KeyError
            If `region` is not found in the ontology.
        """
        path = []
        # if self.is_region(acronym, key="acronym", unreferenced=unreferenced):
        if region != self.dict["acronym"] and region not in self.parent_region: # if it's not the root
            raise KeyError(f"Region not found ('acronym'='{region}')")
        while region in self.parent_region.keys():
            parent = self.parent_region[region]
            path.append(parent)
            region = parent
        return path

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.partitioned"])
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
        regions = (acronym, *acronyms)
        return self.partitioned(regions, partition="major divisions")

    def partitioned(self,
                    regions: Sequence,
                    *,
                    partition: Literal["major divisions"]="major divisions",
                    key: str=None
                    ) -> OrderedDict[str|int,str|int]:
        """
        Partitions the given regions based on the specified partitioning method.

        Parameters
        ----------
        regions
            The brain structures uniquely identified by their IDs or their acronyms.
        partition
            The partitioning method to use.

            For compatibility reasons, it only accepts `"major divisions"`.
        key
            This parameter is ignored.

        Returns
        -------
        :
            An OrderedDict mapping $\\text{region}→\\text{major division}$

        Raises
        ------
        ValueError
            If there is at least one brain structure that appears twice in
            `regions` or `partition`.
        ValueError
            If `partition` is not recognised as a valid partitioning method.

        See also
        --------
        [`ancestors`][braian.legacy.AllenBrainOntology.ancestors]
        [`partition`][braian.legacy.AllenBrainOntology.partition]
        [`selected`][braian.legacy.AllenBrainOntology.selected]
        """
        if partition != "major divisions":
            raise ValueError(f"Unsupported partition: '{partition}'")
        self._check_regions(regions, key="acronym", unreferenced=False)
        # def get_region_mjd(node: dict, depth: int):
        #     if node["major_division"]:
        #         get_region_mjd.curr_mjd = node["acronym"]
        #     if node["acronym"] in acronyms:
        #         get_region_mjd.res[node["acronym"]] = get_region_mjd.curr_mjd
        # get_region_mjd.curr_mjd = None
        # get_region_mjd.res = OrderedDict()
        # _visit_dict.visit_dfs(self.dict, "children", get_region_mjd)
        # return get_region_mjd.res
        mds = OrderedDict()
        for acronym in regions:
            for ancestor in (acronym, *self.ancestors(acronym)):
                if ancestor in MAJOR_DIVISIONS:
                    mds[acronym] = ancestor
                    break
            if acronym not in mds:
                mds[acronym] = None
        return mds

    def get_layer1(self) -> list[str]:
        """
        Returns the layer 1 in the Isocortex accordingly to CCFv3

        Returns
        -------
        :
            A list of acronyms of brain regions
        """
        assert self.annotation_version == "ccf_2017", "Unsupported version of the ontology. The current ontology does not know which regions make up the layer 1"
        return ["FRP1", "MOp1", "MOs1", "SSp-n1", "SSp-bfd1", "SSp-ll1", "SSp-m1", "SSp-ul1", "SSp-tr1", "SSp-un1", "SSs1", "GU1", "VISC1", "AUDd1", "AUDp1", "AUDpo1", "AUDv1", "VISal1", "VISam1", "VISl1", "VISp1", "VISpl1", "VISpm1", "ACAd1", "ACAv1", "PL1", "ILA1", "ORBl1", "ORBm1", "ORBvl1", "AId1", "AIp1", "AIv1", "RSPagl1", "RSPd1", "RSPv1", "PTLp1", "TEa1", "PERI1", "ECT1", "PIR1", "OT1", "NLOT1", "COAa1", "COApl1", "COApm1", "PAA1", "TR1"]

    def _get_full_names(self) -> dict[str,str]:
        """
        Computes a dictionary that translates acronym of a brain region in its full name

        Returns
        -------
        :
            A dictionary that maps acronyms→name
        """
        all_nodes = _visit_dict.get_all_nodes(self.dict, "children")
        return {area["acronym"]: area["name"] for area in all_nodes}

    @deprecated(since="1.1.0", alternatives=["braian.legacy.AllenBrainOntology.colors"])
    def get_region_colors(self) -> dict[str,str]:
        """
        Gets the dictionary that maps brain structures to their hex color triplet,
        as defined by the Allen Institute.

        Returns
        -------
        :
            A dictionary that maps acronyms→color
        """
        return self.colors()
    
    def colors(self, key: Literal["id","acronym"]="acronym") -> dict[str,str]:
        """
        Gets the dictionary that maps brain structures to their hex color triplet,
        as defined by the Allen Institute.

        Parameters
        ----------
        key
            This parameter is ignored.

        Returns
        -------
        :
            A dictionary that maps region→color
        """
        all_areas = _visit_dict.get_all_nodes(self.dict, "children")
        return {area["acronym"]: "#"+area["color_hex_triplet"] for area in all_areas}

    def to_igraph(self,
                  unreferenced: bool=False,
                  blacklisted: bool=True) -> ig.Graph:
        """
        Translates the current brain ontology into an igraph directed Graph,
        where nodes are brain regions and edges region→subregion relationships

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
        def add_attributes(graph: ig.Graph):
            def visit(region, depth):
                v = graph.vs[region["graph_order"]]
                v["id"] = region["id"]
                v["name"] = region["acronym"]
                # v["full_name"] = region["name"]
                v["depth"] = depth
                # v["color"] = region["color_hex_triplet"]
                v["structural_level"] = region["st_level"]
            return visit

        g = ig.Graph(edges=self._get_edges(key="graph_order"), directed=True)
        _visit_dict.visit_bfs(self.dict, "children", add_attributes(g))
        # if self.dict was modified removing some nodes, 'graph_order' creates some empty vertices
        # in that case, we remove those vertices

        blacklisted_trees = set(self.blacklisted(unreferenced=False, key="acronym"))
        unrefs_trees = set(self._get_unreferenced_trees(key="acronym"))
        if not unreferenced:
            if not blacklisted:
                blacklisted_unrefs_trees = set(self.blacklisted(unreferenced=True, key="acronym"))
                _graph_utils.remove_branch(g, blacklisted_unrefs_trees)
            else:
                _graph_utils.remove_branch(g, unrefs_trees)
                _graph_utils.blacklist_regions(g, blacklisted_trees)
        else: # if regions unreferenced are kept in the graph, it means it will also keep the
              # blacklisted ones because unreferenced regions are obligatorily blacklisted from any analysis.
            _graph_utils.blacklist_regions(g, blacklisted_trees, unreferenced=unrefs_trees)
        if self.has_selection():
            _graph_utils.select_regions(g, self.selected())
        return g