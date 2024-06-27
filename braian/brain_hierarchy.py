# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@author: carlocastoldi
"""

import braian.utils
import igraph as ig
import json
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import requests

from bs4 import BeautifulSoup
from collections import OrderedDict
from collections.abc import Iterable, Container, Callable
from operator import xor
from plotly.colors import DEFAULT_PLOTLY_COLORS
from braian.visit_dict import *

__all__ = ["AllenBrainHierarchy", "MAJOR_DIVISIONS"]

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

def is_blacklisted(node):
    return "blacklisted" in node and node["blacklisted"]

def set_reference(node, has_reference):
    node["has_reference"] = has_reference

def has_reference(node):
    return "has_reference" in node and node["has_reference"]

class AllenBrainHierarchy:
    def __init__(self, path_to_allen_json: str, blacklisted_acronyms: Iterable=[], version=None):
        """
        Crates an ontology of brain regions based on Allen Institute's structure graphs.
        To know more where to get the structure graphs, read the
        [official guide](https://community.brain-map.org/t/downloading-an-ontologys-structure-graph/2880)
        from Allen Institute.
        However, the ontology differs from the region annotations depeding on the version of the common coordinate
        framework (CCF). This happens when a new CCF version changes a branch in the ontoloy to reflect a new scientific consesus.
        In Allen's website, the "old" brain region can be identified in grey italic text: [](https://atlas.brain-map.org/atlas)
        If you want to clean the ontology based on a particular version of the CCF, you can provide a valid value for `version`

        Parameters
        ----------
        path_to_allen_json
            The path to an Allen structural graph json
        blacklisted_acronyms
            Acronyms of branches from the onthology to exclude completely from the analysis
        version
            Must be `"CCFv1"`, `"CCFv2"`, `"CCFv3"`, `"CCFv4"` or `None`.
            The version of the Common Coordinates Framework to which sync the onthology
        """
        with open(path_to_allen_json, "r") as file:
            allen_data = json.load(file)

        self.dict = allen_data["msg"][0]
        # First label every region as "not blacklisted"
        visit_bfs(self.dict, "children", lambda n,d: set_blacklisted(n, False))
        visit_bfs(self.dict, "children", lambda n,d: set_reference(n, True))
        if blacklisted_acronyms:
            self.blacklist_regions(blacklisted_acronyms, key="acronym")
            # we don't prune, otherwise we won't be able to work with region_to_exclude (QuPath output)
            # prune_where(self.dict, "children", lambda x: x["acronym"] in blacklisted_acronyms)
        if version is not None:
            unannoted_regions, self.annotation_version = self.__get_unannoted_regions(version)
            self.blacklist_regions(unannoted_regions, key="acronym", has_reference=False)
        else:
            self.annotation_version = None

        self.__add_depth_to_regions()
        self.__mark_major_divisions()
        self.parent_region: dict[str,str] = self.__get_all_parent_areas() #: A dictionary mapping region's acronyms to the parent region. It does not have 'root'.
        self.direct_subregions: dict[str,list[str]] = self.__get_all_subregions()
        """A dictionary mappin region's acronyms to a list of direct subregions.

        Examples
        --------
        >>> braian.cache("ontology.json", "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        >>> brain_ontology = braian.AllenBrainHierarchy("ontology.json", [])
        >>> brain_ontology.direct_subregions["ACA"]
        ["ACAv", "ACAd"]                                    # dorsal and ventral part
        >>> brain_ontology.direct_subregions["ACAv"]
        ["ACAv5", "ACAv2/3", "ACAv6a", "ACAv1", "ACAv6b"]   # all layers in ventral part
        """
        self.full_name: dict[str,str] = self.__get_full_names()
        """A dictionary mapping a regions' acronym to its full name."""

    def __get_unannoted_regions(self, version):
        # alternative implementation: use annotation's nrrd file - https://help.brain-map.org/display/mousebrain/API#API-DownloadAtlas3-DReferenceModels
        # this way is not totally precise, because some masks files are present even if they're empty
        match version:
            case "2015" | "CCFv1" | "ccfv1" | "v1" | 1:
                annotation_version = "ccf_2015"
            case "2016" | "CCFv2" | "ccfv2" | "v2" | 2:
                annotation_version = "ccf_2016"
            case "2017" | "CCFv3" | "ccfv3" | "v3" | 3:
                annotation_version = "ccf_2017"
            case "2022" | "CCFv4" | "ccfv4" | "v4" | 4:
                annotation_version = "ccf_2022"
            case _:
                raise ValueError(f"Unrecognised '{version}' version")
        url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/{annotation_version}/structure_masks/structure_masks_10"
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        if annotation_version == "ccf_2022":
            regions_w_annotation = [int(link["href"][:-len(".nii.gz")].split("_")[0]) for link in soup.select('a[href*=".nii.gz"]')]
        else:
            regions_w_annotation = [int(link["href"][len("structure_"):-len(".nrrd")]) for link in soup.select('a[href*=".nrrd"]')]
        regions_wo_annotation = get_where(self.dict, "children", lambda n,d: n["id"] not in regions_w_annotation, visit_dfs)
        return [region["acronym"] for region in regions_wo_annotation], annotation_version

    def are_regions(self, a: Iterable, key="acronym") -> npt.NDArray:
        """
        Check whether each of the elements of the iterable are a brain region of the current ontology or not

        Parameters
        ----------
        a
            List of values that identify uniquely a brain region (e.g. their acronyms)
        key
            The key in Allen's structural graph used for the check

        Returns
        -------
        :
            An array of bools being True where the corresponding value in `a` is a region and False otherwise
        """
        assert isinstance(a, Iterable), f"Expected a '{Iterable}' but get '{type(a)}'"
        a = list(a)
        s = set(a)
        is_region = np.full_like(a, False)
        def check_region(node, depth):
            id = node[key]
            if id in s:
                is_region[a.index(id)] = True
        visit_dfs(self.dict, "children", check_region)
        return is_region

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
        return all(r in regions for r in self.direct_subregions[parent])

    def minimimum_treecover(self, acronyms: Iterable[str]) -> list[str]:
        """
        Returns the minimum set of regions that covers all the given regions, and not more.

        Parameters
        ----------
        acronyms
            The acronyms of the regions to cover

        Returns
        -------
        :
            A list of acronyms of regions

        Examples
        --------
        >>> braian.cache("ontology.json", "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        >>> brain_ontology = braian.AllenBrainHierarchy("ontology.json", [])
        >>> sorted(brain_ontology.minimimum_treecover(['P', 'MB', 'TH', 'MY', 'CB', 'HY']))
        ['BS', 'CB']

        >>> sorted(brain_ontology.minimimum_treecover(["RE", "Xi", "PVT", "PT", "TH"]))
        ['MTN', 'TH']
        """
        acronyms = set(acronyms)
        _regions = {parent if self.contains_all_children(parent, acronyms) else acronym
                        for acronym, parent in self.get_parent_regions(acronyms).items()}
        if acronyms == _regions:
            return list(acronyms)
        return self.minimimum_treecover(_regions)

    def blacklist_regions(self, regions: Iterable, key="acronym", has_reference=True):
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
            If `regions` exist in the used version of the CCF or not

        Raises
        ------
        ValueError
            If it can't find at least one of the `regions` in the ontology
        """
        if not all(is_region:=self.are_regions(regions, key=key)):
            raise ValueError(f"Some given regions are not recognised as part of the ontology: {np.asarray(regions)[~is_region]}")
        for region_value in regions:
            region_node = find_subtree(self.dict, key, region_value, "children")
            visit_bfs(region_node, "children", lambda n,d: set_blacklisted(n, True))
            if not has_reference:
                visit_bfs(region_node, "children", lambda n,d: set_reference(n, False))

    def get_blacklisted_trees(self, key="acronym") -> list:
        """
        Returns the biggest brain region of each branch in the ontology that was blacklisted

        Parameters
        ----------
        key
            The key in Allen's structural graph used to indentify the returned regions

        Returns
        -------
        :
            A list of blacklisted regions, each of which is identifiable with `key`
        """
        regions = non_overlapping_where(self.dict, "children", lambda n,d: is_blacklisted(n) and has_reference(n), mode="bfs")
        return [region[key] for region in regions]

    def __mark_major_divisions(self):
        add_boolean_attribute(self.dict, "children", "major_division", lambda node,d: node["acronym"] in MAJOR_DIVISIONS)

    def __add_depth_to_regions(self):
        def add_depth(node, depth):
            node["depth"] = depth
        visit_bfs(self.dict, "children", add_depth)

    def select_at_depth(self, depth: int):
        """
        Select all non-overlapping brain regions at the same depth in the ontology.
        If a brain region is above the given depth but has no sub-regions, it is selected anyway.

        Parameters
        ----------
        depth
            The desired depth in the ontology to select

        See also
        --------
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        def is_selected(node, _depth):
            return _depth == depth or (_depth < depth and not node["children"])
        add_boolean_attribute(self.dict, "children", "selected", is_selected)

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
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        add_boolean_attribute(self.dict, "children", "selected", lambda node,d: node["st_level"] == level)

    def select_leaves(self):
        """
        Select all th enon-overlapping smallest brain regions in the ontology.
        A region is also selected if it's not the smallest possible, but all of its sub-regions are blacklisted

        See also
        --------
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        add_boolean_attribute(self.dict, "children", "selected", lambda node, d: is_leaf(node, "children") or \
                              not is_blacklisted(node) and all([is_blacklisted(child) for child in node["children"]]))
                              # not is_blacklisted(node) and all([has_reference(child) for child in node["children"]])) # some regions (see CA1 in CCFv3) have all subregions unannoted

    def select_summary_structures(self):
        """
        Select all summary structures in the ontology.
        The list of Summary Structures is defined by Allen as a set of non-overlapping, finer divisions,
        independent of their exact depth in the tree. They are a set of brain regions often used in the literature.

        Summary Structures can be retrieved from Table S2 of this article: https://www.sciencedirect.com/science/article/pii/S0092867420304025

        See also
        --------
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        # if self.annotation_version is not None:
        # # if self.annotation_version!= "ccf_2017":
        #     raise ValueError("BraiAn does not support 'summary structures' selection_method on atlas versions different from CCFv3")
        file = braian.utils.get_resource_path("CCFv3_summary_structures.csv")
        key = "id"
        regions = pd.read_csv(file, sep="\t", index_col=0)
        self.select_regions(regions[key].values, key=key)

    def select_regions(self, regions: Iterable, key: str="acronym"):
        """
        Select the given regions in the ontology

        Parameters
        ----------
        regions
            The brain regions to select
        key
            The key in Allen's structural graph used to indentify the regions

        Raises
        ------
        ValueError
            If it can't find at least one of the `regions` in the ontology

        See also
        --------
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        if not all(is_region:=self.are_regions(regions, key=key)):
            raise ValueError(f"Some given regions are not recognised as part of the ontology: {np.asarray(regions)[~is_region]}")
        add_boolean_attribute(self.dict, "children", "selected", lambda node,d: node[key] in regions)

    def add_to_selection(self, regions: Iterable, key: str="acronym"):
        """
        Add the given brain regions to the current selection in the ontology

        Parameters
        ----------
        regions
            The brain regions to add to selection
        key
            The key in Allen's structural graph used to indentify the regions

        See also
        --------
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        assert all(is_region:=self.are_regions(regions, key=key)), \
            f"Some given regions are not recognised as part of the ontology: {np.asarray(regions)[~is_region]}"
        add_boolean_attribute(self.dict, "children", "selected",
                              lambda node,d: ("selected" in node and node["selected"]) or node[key] in regions)

    def get_selected_regions(self, key="acronym") -> list:
        """
        Returns a non-overlapping list of selected non-blacklisted brain regions

        Parameters
        ----------
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            A list of brain regions identified by `key`

        See also
        --------
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        if "selected" not in self.dict: return []
        regions = non_overlapping_where(self.dict, "children", lambda n,d: n["selected"] and not is_blacklisted(n), mode="dfs")
        return [region[key] for region in regions]

    def unselect_all(self):
        """
        Resets the selection in the ontology

        See also
        --------
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        [`braian.AllenBrainHierarchy.get_regions`][]
        """
        if "selected" in self.dict:
            del_attribute(self.dict, "children", "selected")

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
        [`braian.AllenBrainHierarchy.get_selected_regions`][]
        [`braian.AllenBrainHierarchy.unselect_all`][]
        [`braian.AllenBrainHierarchy.add_to_selection`][]
        [`braian.AllenBrainHierarchy.select_at_depth`][]
        [`braian.AllenBrainHierarchy.select_at_structural_level`][]
        [`braian.AllenBrainHierarchy.select_leaves`][]
        [`braian.AllenBrainHierarchy.select_summary_structures`][]
        [`braian.AllenBrainHierarchy.select_regions`][]
        """
        old_selection = self.get_selected_regions(key="id")
        if old_selection: self.unselect_all()
        match selection_method:
            case "summary structures":
                self.select_summary_structures()
            case "major divisions":
                self.select_regions(MAJOR_DIVISIONS)
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
        selected_regions = self.get_selected_regions()
        # print(f"You selected {len(selected_regions)} regions to plot.")
        self.select_regions(old_selection, key="id")
        return selected_regions

    def ids_to_acronym(self, ids: Container[int], mode="depth") -> list[str]:
        """
        Converts the given brain regions IDs into their corresponding acronyms.

        Parameters
        ----------
        ids
            The brain regions' IDs to convert
        mode
            Must be eithe "breadth" or "depth".
            The order in which the returned acronyms will be: breadth-first or depth-first

        Returns
        -------
        :
            A list of acronyms

        Raises
        ------
        ValueError
            If given `mode` is not supported

        ValueError
            If it can't find at least one of the `ids` in the ontology
        """
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth' and 'depth'.")
        if not all(is_region:=self.are_regions(ids, key="id")):
            raise ValueError(f"Some given IDs are not recognised as part of the ontology: {np.asarray(ids)[~is_region]}")
        areas = get_where(self.dict, "children", lambda n,d: n["id"] in ids, visit_alg)
        return [area["acronym"] for area in areas]

    def acronym_to_ids(self, acronyms: Container[str], mode="depth") -> list[int]:
        """
        Converts the given brain regions acronyms into ther corresponding IDs

        Parameters
        ----------
        acronyms
            the brain regons' acronyms to convert
        mode
            Must be eithe "breadth" or "depth".
            The order in which the returned acronyms will be: breadth-first or depth-first

        Returns
        -------
        :
            A list of region IDs

        Raises
        ------
        ValueError
            If given `mode` is not supported

        ValueError
            If it can't find at least one of the `ids` in the ontology
        """
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth' and 'depth'.")
        regions = get_where(self.dict, "children", lambda n,d: n["acronym"] in acronyms, visit_alg)
        return [r["id"] for r in regions]

    def get_sibiling_regions(self, region:str|int, key="acronym") -> list:
        """
        Get all brain regions that, combined, make the whole parent of the given `region`

        It does not take into account blacklisted regions

        Parameters
        ----------
        region
            A brain region
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            All `region`'s sibilings, including itself

        Raises
        ------
        ValueError
            If it can't find a parent for the given `region`
        """
        parents = get_parents_where(self.dict, "children", lambda parent,child: child[key] == region, key)
        if len(parents) != 1: raise ValueError(f"Can't find the parent of a region with '{key}': {region}")
        parent = parents[region]
        return [sibiling[key] for sibiling in parent["children"]]
        ## Alternative implementation:
        # parent_id = find_subtree(self.dict, key, value, "children")["parent_structure_id"]
        # return [area[key] for area in find_subtree(self.dict, "id", parent_id, "children")["children"]]

    def get_parent_regions(self, regions:Iterable, key:str="acronym") -> dict:
        """
        Finds, for each of the given brain regions, their parent region in the ontology

        It does not take into account blacklisted regions

        Parameters
        ----------
        regions
            The brain regions to search the parent for
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            A dicrtionary mapping region→parent

        Raises
        ------
        ValueError
            If it can't find the parent of one of the given `regions`
        """
        parents = get_parents_where(self.dict, "children", lambda parent,child: child[key] in regions, key)
        for region in regions:
            if region not in parents.keys(): raise ValueError(f"Can't find the parent of a region with '{key}': {region}")
        return {region: (parent[key] if parent else None) for region,parent in parents.items()}

        # parents = get_parents_where(self.dict, "children", lambda parent,child: child[key] in values, key)
        # for area in values:
        #     assert area in parents.keys(), f"Can't find the parent of an area with '{key}': {area}"
        # return {area: (parent[key] if parent else None) for area,parent in parents.items()}

    def __get_all_parent_areas(self, key="acronym") -> dict:
        """
        Finds, for each brain region in the ontology, the corresponding parent region.
        The "root" region has no entry in the returned dictionary

        It does not take into account blacklisted regions

        Parameters
        ----------
        key
            The key in Allen's structural graph used to indentify the regions

        Returns
        -------
        :
            A dictionary mapping region→parent
        """
        return {subregion: region for region,subregion in self.__get_edges(key)}

    def __get_edges(self, key="id") -> list[tuple]:
        assert key in ("id", "acronym", "graph_order"), "'key' parameter must be  'id', 'acronym' or 'graph_order'"
        edges = []
        visit_parents(self.dict,
                      "children",
                      lambda region,subregion:
                          None if region is None else
                          edges.append((region[key], subregion[key])))
        return edges

    def __get_all_subregions(self, key: str="acronym") -> dict:
        """
        returns a dictionary where all parent regions are the keys,
        and the subregions that belong to the parent are stored
        in a list as the value corresponding to the key.
        Regions with no subregions have no entries in the dictionary.

        It does not take into account blacklisted regions

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
            if node["children"]:
                subregions[node[key]] = [child[key] for child in node["children"]]
        visit_bfs(self.dict, "children", add_subregions)
        return subregions

    def list_all_subregions(self, acronym: str, mode: str="breadth") -> list:
        """
        Lists all subregions of a brain region, at all hierarchical levels.

        It does not take into account blacklisted regions

        Parameters
        ----------
        acronym
            The acronym of the brain region
        mode
            Must be eithe "breadth" or "depth".
            The order in which the returned acronyms will be: breadth-first or depth-first

        Returns
        -------
        :
            A list of subregions of `acronym`

        Raises
        ------
        ValueError
            If given `mode` is not supported
        ValueError
            If it can't find `acronym` in the ontology
        """
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported mode '{mode}'. Available modes are 'breadth' and 'depth'.")

        attr = "acronym"
        region = find_subtree(self.dict, attr, acronym, "children")
        if not region:
            raise ValueError(f"Can't find a region with '{attr}'='{acronym}' in the ontology")
        subregions = get_all_nodes(region, "children", visit=visit_alg)
        return [subregion[attr] for subregion in subregions]

    def get_regions_above(self, acronym: str) -> list[str]:
        """
        Lists all the regions for which `acronym` is a subregion.

        It does not take into account blacklisted regions

        Parameters
        ----------
        acronym
            Acronym of the brain region of which you want to know the regions above

        Returns
        -------
        :
            List of all regions above, excluding `acronym`
        """
        path = []
        attr = "acronym"
        while acronym in self.parent_region.keys():
            parent = self.parent_region[acronym]
            path.append(parent)
            acronym = parent
        return path

    def get_areas_major_division(self, acronym: str, *acronyms: str) -> OrderedDict[str, str]:
        """
        Finds the corresponding major division for each on the the `acronyms`.
        The returned dictionary is sorted in depth-first-search order.

        It does not take into account blacklisted regions

        Parameters
        ----------
        acronym
            Acronym of the brain region of which you want to know the corresponding major division

        Returns
        -------
        :
            An OrderedDict mapping <region acronym>→<major division>
        """
        def get_region_mjd(node: dict, depth: int):
            if node["major_division"]:
                get_region_mjd.curr_mjd = node["acronym"]
            if node["acronym"] in (acronym, *acronyms):
                get_region_mjd.res[node["acronym"]] = get_region_mjd.curr_mjd
        get_region_mjd.curr_mjd = None
        get_region_mjd.res = OrderedDict()
        visit_dfs(self.dict, "children", get_region_mjd)
        return get_region_mjd.res

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

    def __get_full_names(self) -> dict[str,str]:
        """
        Computes a dictionary that translates acronym of a brain region in its full name

        Returns
        -------
        :
            A dictionary that maps acronyms→name
        """
        all_nodes = get_all_nodes(self.dict, "children")
        return {area["acronym"]: area["name"] for area in all_nodes}

    def get_region_colors(self) -> dict[str,str]:
        """
        Computes a dictionary that translates acronym of a brain region into a hex color triplet

        Returns
        -------
        :
            A dictionary that maps acronyms→color
        """
        all_areas = get_all_nodes(self.dict, "children")
        return {area["acronym"]: "#"+area["color_hex_triplet"] for area in all_areas}

    def to_igraph(self) -> ig.Graph:
        """
        Translates the current brain ontology into an igraph directed Graph,
        where nodes are brain regions and edges region→subregion relationships

        Returns
        -------
        :
            A graph
        """
        def add_attributes(graph: ig.Graph):
            def visit(region, depth):
                v = graph.vs[region["graph_order"]]
                v["name"] = region["acronym"]
                # v["full_name"] = region["name"]
                v["depth"] = depth
                # v["color"] = region["color_hex_triplet"]
                v["structural_level"] = region["st_level"]
            return visit

        G = ig.Graph(edges=self.__get_edges(key="graph_order"))
        visit_bfs(self.dict, "children", add_attributes(G))
        # if self.dict was modified removing some nodes, 'graph_order' creates some empty vertices
        # in that case, we remove those vertices
        G.delete_vertices([v.index for v in G.vs if v.degree() == 0])
        return G

    def plot(self) -> go.Figure:
        """
        Plots the ontolofy as a tree

        Returns
        -------
        :
            A plotly Figure
        """
        G = self.to_igraph()
        graph_layout = G.layout_reingold_tilford(mode="in", root=[0])
        edges_trace = self.draw_edges(G, graph_layout, width=0.5)
        nodes_trace = self.draw_nodes(G, graph_layout, node_size=5, metrics={"Subregions": lambda vs: np.asarray(vs.degree())-1})
        nodes_trace.marker.line = dict(color="black", width=0.25)
        plot_layout = go.Layout(
            title="Allen's brain region hierarchy",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, dtick=1, autorange="reversed", title="depth"),
            template="none"
        )
        return go.Figure([edges_trace, nodes_trace], layout=plot_layout)

    def draw_edges(self, G: ig.Graph, layout: ig.Layout, width: int) -> go.Scatter:
        """
        Draws a plotly Line plot of the given graph `G`, based on the given layout.
        If `G` is a directed graph, it the drawn edges are arrows

        Parameters
        ----------
        G
            A graph
        layout
            The layout used to position the nodes of the graph `G`
        width
            The width of the edges' lines

        Returns
        -------
        :
            A plotly scatter trace
        """
        edge_x = []
        edge_y = []
        for e in G.es:
            x0, y0 = layout.coords[e.source]
            x1, y1 = layout.coords[e.target]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=width, color="#888"),
            hoverinfo="none",
            mode="lines+markers" if G.is_directed() else "lines",
            showlegend=False)

        if G.is_directed():
            edges_trace.marker = dict(
                    symbol="arrow",
                    size=10,
                    angleref="previous",
                    standoff=8,
                )

        return edges_trace

    def draw_nodes(self, G: ig.Graph, layout: ig.Layout, node_size: int,
                outline_size: float=0.5, use_centrality: bool=False, centrality_metric: str=None,
                use_clustering: bool=False,
                metrics: dict[str,Callable[[ig.VertexSeq],Iterable[float]]]={"degree": ig.VertexSeq.degree}) -> go.Scatter:
        """
        Draws a plotly Scatter plot of the given graph `G`, based on the given layout.

        Parameters
        ----------
        G
            A graph
        layout
            The layout used to position the nodes of the graph `G`
        node_size
            The size of the region nodes
        outline_size
            the size of the region nodes' outlines
        use_centrality
            If true, it colors the regions nodes based on the attribute defined in `centrality_metric` of each `G` vertex
            If false, it uses the corresponding brain region color.
        centrality_metric
            The name of the attribute used if `use_centrality=True`
        use_clustering
            If true, it colors the regions nodes outlines based on the `cluster` attribute of each `G` vertex
            If false, it uses the corresponding brain region color.
        metrics
            A dictionary that defines M additional information for the vertices of graph `G`.
            The keys are title of an additional metric, while the values are functions that
            take a `igraph.VertexSeq` and spits a value for each vertex.

        Returns
        -------
        :
            A plotly scatter trace

        Raises
        ------
        ValueError
            If `use_centrality=True`, but the vertices of `G` have no attribute as defined in `centrality_metric`
        ValueError
            If `use_clustering=True`, but the vertices of `G` have no `cluster` attribute
        """
        colors = self.get_region_colors()
        nodes_color = []
        outlines_color = []
        if use_clustering:
            if "cluster" not in G.vs.attributes():
                raise ValueError("No clustering is made on the provided connectome")
            get_outline_color = lambda v: DEFAULT_PLOTLY_COLORS[v["cluster"] % len(DEFAULT_PLOTLY_COLORS)]
        else:
            get_outline_color = lambda v: colors[v["name"]]
        if use_centrality and (centrality_metric is None or centrality_metric not in G.vs.attributes()):
            raise ValueError("If you want to plot the centrality, you must also specify a nodes' attribute in 'centrality_metric'")
        for v in G.vs:
            if v.degree() > 0:
                outline_color = get_outline_color(v)
                node_color = v[centrality_metric] if use_centrality else colors[v["name"]]
            elif "is_undefined" in v.attributes() and v["is_undefined"]:
                outline_color = 'rgb(140,140,140)'
                node_color = '#A0A0A0'
            else:
                outline_color = 'rgb(150,150,150)'
                node_color = '#CCCCCC'
            nodes_color.append(node_color)
            outlines_color.append(outline_color)

        customdata, hovertemplate = self.nodes_hover_info(G, title_dict=metrics)
        nodes_trace = go.Scatter(
            x=[coord[0] for coord in layout.coords],
            y=[coord[1] for coord in layout.coords],
            mode="markers",
            name="",
            marker=dict(symbol="circle",
                        size=node_size,
                        color=nodes_color,
                        line=dict(color=outlines_color, width=outline_size)),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False
        )
        return nodes_trace

    def nodes_hover_info(self, G: ig.Graph,
                         title_dict: dict[str,Callable[[ig.VertexSeq],Iterable[float]]]={}
                         ) -> tuple[npt.NDArray,str]:
        """
        Computes the information when hovering over a vertex of the graph `G`.
        It allows to add additional information based on given functions.
        Returns a tuple where the first element is an matrix of custom data,
        while the second element is a hover template.
        Both are used by plotly to modify information when hovering points in a Scatter plot

        Parameters
        ----------
        G
            A graph with N vertices
        title_dict
            A dictionary that defines M additional information for the vertices of graph `G`.
            The keys are title of an additional metric, while the values are functions that
            take a `igraph.VertexSeq` and spits a value for each vertex.

        Returns
        -------
        :
            A customdata M×N matrix and a hovertemplate
        """
        customdata = []
        hovertemplates = []
        i = 0
        # Add vertices' attributes
        for attr in G.vs.attributes():
            match attr:
                case "name":
                    customdata.extend((
                        G.vs["name"],
                        [self.full_name[acronym] for acronym in G.vs["name"]]
                    ))
                    hovertemplates.extend((
                        f"Region: <b>%{{customdata[{i}]}}</b>",
                        f"<i>%{{customdata[{i+1}]}}</i>"
                    ))
                    i += 2
                case "upper_region":
                    customdata.extend((
                        G.vs["upper_region"],
                        [self.full_name[acronym] for acronym in G.vs["upper_region"]]
                    ))
                    hovertemplates.append(f"Major Division: %{{customdata[{i}]}} (%{{customdata[{i+1}]}})")
                    i += 2
                case _:
                    if attr.lower() in (t.lower() for t in title_dict.keys()):
                        # If one of the additional information wants to overload an attribute (e.g. 'depth')
                        # then skip it
                        continue
                    customdata.append(G.vs[attr])
                    attribute_title = attr.replace("_", " ").title()
                    hovertemplates.append(f"{attribute_title}: %{{customdata[{i}]}}")
                    i += 1
        # Add additional information
        for custom_title, fun in title_dict.items():
            customdata.append(fun(G.vs))
            hovertemplates.append(f"{custom_title.title()}: %{{customdata[{i}]}}")
            i += 1

        hovertemplate = "<br>".join(hovertemplates)
        hovertemplate += "<extra></extra>"
        # customdata=np.hstack((old_customdata.customdata, np.expand_dims(<new_data>, 1))), # update customdata
        return np.stack(customdata, axis=-1), hovertemplate