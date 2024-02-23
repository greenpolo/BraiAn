#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:04:28 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""

import igraph as ig
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

from bs4 import BeautifulSoup
from collections import OrderedDict
from operator import xor
from plotly.colors import DEFAULT_PLOTLY_COLORS
from .visit_dict import *

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

RE_TOT_ID = 181*100
RE_TOT_ACRONYM = "REtot"

def set_blacklisted(node, is_blacklisted):
    node["blacklisted"] = is_blacklisted

def is_blacklisted(node):
    return "blacklisted" in node and node["blacklisted"]

class AllenBrainHierarchy:
    def __init__(self, path_to_allen_json, blacklisted_acronyms=[], version=None):
        with open(path_to_allen_json, "r") as file:
            allen_data = json.load(file)
        
        self.dict = allen_data["msg"][0]
        # First label every region as "not blacklisted"
        visit_bfs(self.dict, "children", lambda n,d: set_blacklisted(n, False))
        if blacklisted_acronyms:
            self.blacklist_regions(blacklisted_acronyms, key="acronym")
            # we don't prune, otherwise we won't be able to work with region_to_exclude (QuPath output)
            # prune_where(self.dict, "children", lambda x: x["acronym"] in blacklisted_acronyms)
        if version is not None:
            unannoted_regions = self.__get_unannoted_regions(version)
            self.blacklist_regions(unannoted_regions, key="acronym")

        self.add_depth_to_regions()
        self.mark_major_divisions()
        self.parent_region = self.get_all_parent_areas()
        self.direct_subregions = self.get_all_subregions()
        self.full_name = self.get_full_names()

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
        return [region["acronym"] for region in regions_wo_annotation]

    def contains_all_children(self, regions: list[str], parent: str):
        return all(r in regions for r in self.direct_subregions[parent])

    def minimimum_treecover(self, regions: list[str]) -> list[str]:
        # returns the minimum set of regions that covers all the given regions and not more
        regions = set(regions)
        _regions = {parent if self.contains_all_children(regions, parent) else region
                        for region, parent in self.get_parent_areas(regions).items()}
        if regions == _regions:
            return list(regions)
        return self.minimimum_treecover(_regions)

    def blacklist_regions(self, blacklisted_regions, key="acronym"):
        # find every region to-be-blacklisted, and blacklist all its tree
        for region_value in blacklisted_regions:
            blacklisted_region = find_subtree(self.dict, key, region_value, "children")
            if not blacklisted_region:
                raise ValueError(f"Can't find a region with '{key}'='{blacklisted_region}' to blacklist in Allen's Brain")
            visit_bfs(blacklisted_region, "children", lambda n,d: set_blacklisted(n, True))

    def mark_major_divisions(self):
        add_boolean_attribute(self.dict, "children", "major_division", lambda node,d: node["acronym"] in MAJOR_DIVISIONS)

    def add_depth_to_regions(self):
        def add_depth(node, depth):
            node["depth"] = depth
        visit_bfs(self.dict, "children", add_depth)

    def select_at_depth(self, level):
        def is_selected(node, depth):
            return depth == level or (depth < level and not node["children"])
        add_boolean_attribute(self.dict, "children", "selected", is_selected)

    def select_at_structural_level(self, level):
        add_boolean_attribute(self.dict, "children", "selected", lambda node,d: node["st_level"] == level)

    def select_leaves(self):
        add_boolean_attribute(self.dict, "children", "selected", lambda node, d: is_leaf(node, "children") or \
                              not is_blacklisted(node) and all([is_blacklisted(child) for child in node["children"]])) # some regions (see CA1 in CCFv3) are have all subregions unannoted

    def select_regions(self, acronyms):
        add_boolean_attribute(self.dict, "children", "selected", lambda node,d: node["acronym"] in acronyms)

    # Meant for selecting Summary Structures.
    # Summary Structures is a list of non-overlapping, finer divisions, independent of their exact depth in the tree
    # i.e., they're brain regions that are often used in the literature.
    # They can be retrieved from Table S2 of the following paper:
    # https://www.sciencedirect.com/science/article/pii/S0092867420304025
    def select_from_csv(self, file, key="id", include_nre_tot=False):
        regions = pd.read_csv(file, sep="\t", index_col=0)
        if include_nre_tot:
            # regions = regions[~regions["acronym"].isin(("PR", "RE", "Xi", "RH"))]
            regions = regions[~regions["acronym"].isin(("RE", "Xi", "RH"))]
            regions = pd.concat((regions, pd.DataFrame(pd.Series(dict(id=RE_TOT_ID, acronym=RE_TOT_ACRONYM))).T), ignore_index=True)
        add_boolean_attribute(self.dict, "children", "selected", lambda n,d: n[key] in regions[key].values)

    def add_to_selection(self, acronyms):
        add_boolean_attribute(self.dict, "children", "selected",
                              lambda node,d: ("selected" in node and node["selected"]) or node["acronym"] in acronyms)

    def get_selected_regions(self, key="acronym"):
        assert "selected" in self.dict, "No area is selected."
        regions = non_overlapping_where(self.dict, "children", lambda n,d: n["selected"] and not is_blacklisted(n), mode="dfs")
        return [region[key] for region in regions]

    def unselect_all(self):
        if "selected" in self.dict:
            del_attribute(self.dict, "children", "selected")

    def ids_to_acronym(self, ids, mode="depth"):
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth' and 'depth'.")
        areas = get_where(self.dict, "children", lambda n,d: n["id"] in ids, visit_alg)
        return [area["acronym"] for area in areas]

    def acronym_to_ids(self, acronyms, mode="depth"):
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth' and 'depth'.")
        areas = get_where(self.dict, "children", lambda n,d: n["acronym"] in acronyms, visit_alg)
        return [area["id"] for area in areas]

    def get_sibiling_areas(self, acronym=None, id=None) -> list:
        assert xor(acronym is not None, id is not None), "You must specify one of 'acronym' and 'id' parameters"
        key, value = ("acronym", acronym) if acronym is not None else ("id", id)
        parent = get_parents_where(self.dict, "children", lambda parent,child: child[key] == value, key)
        assert len(parent) == 1, f"Can't find the parent of an area with '{key}': {value}"
        return [sibiling[key] for sibiling in parent[value]["children"]]
        ## Alternative implementation:
        # parent_id = find_subtree(self.dict, key, value, "children")["parent_structure_id"]
        # return [area[key] for area in find_subtree(self.dict, "id", parent_id, "children")["children"]]

    def get_parent_areas(self, acronyms=[], ids=[]) -> dict:
        assert xor(len(acronyms)>0, len(ids)>0), "You must specify one of 'acronyms' and 'ids' lists"
        key, values = ("acronym", acronyms) if acronyms else ("id", ids)
        parents = get_parents_where(self.dict, "children", lambda parent,child: child[key] in values, key)
        for area in values:
            assert area in parents.keys(), f"Can't find the parent of an area with '{key}': {area}"
        return {area: (parent[key] if parent else None) for area,parent in parents.items()}

    def get_all_parent_areas(self, attr="acronym"):
        '''
        returns a dictionary where all the subregions' attribute (default: 'acronyms') are the keys,
        and the parent region's acronym is the corresponding value.
        'root' region has not entry in the dictionary.
        '''
        return {subregion: region for region,subregion in self.get_edges(attr)}
    
    def get_edges(self, attr="id"):
        assert attr in ("id", "acronym", "graph_order"), "'attr' parameter must be  'id', 'acronym' or 'graph_order'"
        edges = []
        visit_parents(self.dict,
                      "children",
                      lambda region,subregion:
                          None if region is None else
                          edges.append((region[attr], subregion[attr])))
        return edges

    def get_all_subregions(self, attr="acronym"):
        '''
        returns a dictionary where all parent regions are the keys,
        and the subregions that belong to the parent are stored
        in a list as the value corresponding to the key.
        Regions with no subregions have no entries in the dictionary.

        Example:
        subregions["ACA"] = ["ACAv", "ACAd"] # (dorsal and ventral part)
        subregions["ACAv"] = ["ACAv5", "ACAv2/3", "ACAv6a", "ACAv1", "ACAv6b"] # all layers in ventral part
        '''
        subregions = dict()
        def add_subregions(node, depth):
            if node["children"]:
                subregions[node[attr]] = [child[attr] for child in node["children"]]
        visit_bfs(self.dict, "children", add_subregions)
        return subregions

    def list_all_subregions(self, region_acronym, mode="breadth"):
        '''
        This function lists all subregions of 'region_acronym' at all hierarchical levels.
        
        Inputs
        ------
            region_acronym (str)
            Acronym of the region of which you want to know the subregions.
            
        Output
        ------
            subregions (list)
            List of all subregions at all hierarchical levels, including 'region_acronym'.
        '''
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported mode '{mode}'. Available modes are 'breadth' and 'depth'.")

        attr = "acronym"
        region = find_subtree(self.dict, attr, region_acronym, "children")
        if not region:
            raise ValueError(f"Can't find a region with '{attr}'='{region_acronym}' in Allen's Brain")
        subregions = get_all_nodes(region, "children", visit=visit_alg)
        return [subregion[attr] for subregion in subregions]

    def get_regions_above(self, region_acronym):
        '''
        This function lists all the regions for which 'region_acronym' is a subregion.
        
        Inputs
        ------
            region_acronym (str)
            Acronym of the region of which you want to know the regions above.
            
        Output
        ------
            subregions (list)
            List of all regions above, excluding 'region_acronym'.
        '''
        path = []
        attr = "acronym"
        while region_acronym in self.parent_region.keys():
            parent = self.parent_region[region_acronym]
            path.append(parent)
            region_acronym = parent
        return path

    def get_areas_major_division(self, *acronyms, sorted=False) -> dict:
        def get_region_mjd(node: dict, depth: int):
            if node["major_division"]:
                get_region_mjd.curr_mjd = node["acronym"]
            if node["acronym"] in acronyms:
                get_region_mjd.res[node["acronym"]] = get_region_mjd.curr_mjd
        get_region_mjd.curr_mjd = None
        get_region_mjd.res = OrderedDict() if sorted else dict()
        visit_dfs(self.dict, "children", get_region_mjd)
        return get_region_mjd.res

    def get_layer1(self) -> list[str]:
        # NOTE: this is the layer1 as defined as in CCFv3
        # TODO: should check whether the current instance's version is CCFv3
        return ["FRP1", "MOp1", "MOs1", "SSp-n1", "SSp-bfd1", "SSp-ll1", "SSp-m1", "SSp-ul1", "SSp-tr1", "SSp-un1", "SSs1", "GU1", "VISC1", "AUDd1", "AUDp1", "AUDpo1", "AUDv1", "VISal1", "VISam1", "VISl1", "VISp1", "VISpl1", "VISpm1", "ACAd1" "ACAv1", "PL1", "ILA1", "ORBl1", "ORBm1", "ORBvl1", "AId1", "AIp1", "AIv1", "RSPagl1", "RSPd1", "RSPv1", "PTLp1", "TEa1", "PERI1", "ECT1", "PIR1", "OT1", "NLOT1", "COAa1", "COApl1", "COApm1", "PAA1", "TR1"]

    def get_full_names(self):
        '''
        returns a dictionary that translates acronym of a region in its full name.
        '''
        all_nodes = get_all_nodes(self.dict, "children")
        return {area["acronym"]: area["name"] for area in all_nodes}

    def get_region_colors(self):
        all_areas = get_all_nodes(self.dict, "children")
        return {area["acronym"]: "#"+area["color_hex_triplet"] for area in all_areas}

    def to_igraph(self):
        def add_attributes(graph: ig.Graph):
            def visit(region, depth):
                v = graph.vs[region["graph_order"]]
                v["name"] = region["acronym"]
                v["Depth"] = depth
            return visit

        G = ig.Graph(edges=self.get_edges(attr="graph_order"))
        visit_bfs(self.dict, "children", add_attributes(G))
        return G

    def plot(self):
        G = self.to_igraph()
        graph_layout = G.layout_reingold_tilford(mode="in", root=[0])
        edges_trace = self.draw_edges(G, graph_layout, width=0.5)
        nodes_trace = self.draw_nodes(G, graph_layout, node_size=5)
        plot_layout = go.Layout(
            title="Allen's brain region hierarchy",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, dtick=1, autorange="reversed")
        )
        return go.Figure([edges_trace, nodes_trace], layout=plot_layout)

    def draw_edges(self, G: ig.Graph, layout: ig.Layout, width: int):
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
    
    def draw_nodes(self, G: ig.Graph, graph_layout: ig.Layout, node_size: int,
               outline_size=0.5, use_centrality=False, centrality_metric: str=None,
               use_clustering=False):
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

        customdata, hovertemplate = self.nodes_hover_info(G, title_dict={"Degree": ig.VertexSeq.degree})
        nodes_trace = go.Scatter(
            x=[coord[0] for coord in graph_layout.coords],
            y=[coord[1] for coord in graph_layout.coords],
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
    
    def nodes_hover_info(self, G: ig.Graph, title_dict: dict={}):
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
                    if attr in title_dict:
                        fun = title_dict[attr]
                        customdata.append(fun(G.vs))
                    else:
                        customdata.append(G.vs[attr])
                    hovertemplates.append(f"{attr}: %{{customdata[{i}]}}")
                    i += 1
        # Add additional information
        # fun is expected to be a function that takes a VertexSeq and spits a value for each vertex.
        for attr_title, fun in title_dict.items():
            if attr_title in G.vs.attributes():
                continue
            customdata.append(fun(G.vs))
            hovertemplates.append(f"{attr_title}: %{{customdata[{i}]}}")
            i += 1

        hovertemplate = "<br>".join(hovertemplates)
        hovertemplate += "<extra></extra>"
        # customdata=np.hstack((old_customdata.customdata, np.expand_dims(<new_data>, 1))), # update customdata
        return np.stack(customdata, axis=-1), hovertemplate