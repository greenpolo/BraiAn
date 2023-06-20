#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:04:28 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""

import copy
import json
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests

from bs4 import BeautifulSoup
from operator import xor
from networkx.drawing.nx_pydot import graphviz_layout
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
    def __init__(self, path_to_allen_json, blacklisted_acronyms=[], use_literature_reuniens=False, version=None):
        with open(path_to_allen_json, "r") as file:
            allen_data = json.load(file)
        
        self.dict = allen_data["msg"][0]
        if use_literature_reuniens:
            self.__use_literature_reuniens()
        self.use_literature_reuniens = use_literature_reuniens
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
    
    def __use_literature_reuniens(self):
        dor_pm = find_subtree(self.dict, "acronym", "DORpm", "children")
        # med = find_subtree(dor_pm, "acronym", "MED", "children")
        mtn = find_subtree(dor_pm, "acronym", "MTN", "children")
        ilm = find_subtree(dor_pm, "acronym", "ILM", "children")
        
        # pr = find_subtree(med, "acronym", "PR", "children")
        re = find_subtree(mtn, "acronym", "RE", "children")
        xi = find_subtree(mtn, "acronym", "Xi", "children")
        rh = find_subtree(ilm, "acronym", "RH", "children")
        # 'graph_order' is not set/updated. At the moment is not used, so it's not a problem
        re_tot = {
            "id": RE_TOT_ID,
            "atlas_id": 212*100,
            "ontology_id": 2,
            "acronym": RE_TOT_ACRONYM,
            "name": "Nucleus reuniens (literature)",
            "color_hex_triplet": "FF909F",
            # "graph_order":
            "st_level": 7,
            "hemisphere_id": 3,
            "parent_structure_id": 856,
            # "children": [pr, re, xi, rh]
            "children": [re, xi, rh]
        }
        add_child(dor_pm, re_tot, "children")
        # remove_child(med, pr, "children")
        remove_child(mtn, re, "children")
        remove_child(mtn, xi, "children")
        remove_child(ilm, rh, "children")
    
    def __get_unannoted_regions(self, version):
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
        url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/{annotation_version}/structure_meshes/"
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        regions_w_annotation = [int(link["href"][:-len(".obj")]) for link in soup.select('a[href*=".obj"]')]
        regions_wo_annotation = get_where(self.dict, "children", lambda n,d: n["id"] not in regions_w_annotation, visit_dfs)
        return [region["acronym"] for region in regions_wo_annotation]
    
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

    def get_selected_regions(self, key="acronym", mode="depth"):
        assert "selected" in self.dict, "No area is selected."
        match mode:
            case "breadth":
                visit_alg = visit_bfs
            case "depth":
                visit_alg = visit_dfs
            case _:
                raise ValueError(f"Unsupported '{mode}' mode. Available modes are 'breadth' and 'depth'.")
        areas = get_where(self.dict,
                            "children",
                            lambda n,d: n["selected"] and not is_blacklisted(n),
                            visit_alg)
        return [area[key] for area in areas]
    
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
        assert attr == "acronym" or attr == "id", "'attr' parameter must either be 'acronym' or 'id'"
        all_areas = get_all_nodes(self.dict, "children")
        all_areas_acronyms = [node[attr] for node in all_areas if node["acronym"] != "root"]
        return self.get_parent_areas(**{attr+"s": all_areas_acronyms})

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

    def get_areas_major_division(self, *acronyms) -> dict:
        regions_md = {}
        for region_acronym in acronyms:
            if region_acronym in MAJOR_DIVISIONS:
                regions_md[region_acronym] = region_acronym
                continue
            regions_above = self.get_regions_above(region_acronym)
            above_md = [r for r in regions_above if r in MAJOR_DIVISIONS]
            regions_md[region_acronym] = above_md[0] if above_md else None
        return regions_md

    def get_full_names(self):
        '''
        returns a dictionary that translates acronym of a region in its full name.
        '''
        all_nodes = get_all_nodes(self.dict, "children")
        return {area["acronym"]: area["name"] for area in all_nodes}
    
    def get_region_colors(self):
        all_areas = get_all_nodes(self.dict, "children")
        return {area["acronym"]: "#"+area["color_hex_triplet"] for area in all_areas}

    
    def to_nx_attributes(self, attributes):
        # due to a bug of networkx it's not possible to call an attribute 'name'.
        # However we have a 'name' attribute in Allen's JSON.
        # For this reason we change it to 'region_name'
        if "region_name" in attributes:
            def change_attr_name(d, old_attr, new_attr):
                d[new_attr] = d[old_attr]
                del d[old_attr]
            brain_dict = copy.deepcopy(self.dict)
            visit_bfs(brain_dict, "children", lambda node,_: change_attr_name(node, "name", "region_name"))
        else:
            brain_dict = self.dict

        all_areas = get_all_nodes(brain_dict, "children")
        attributes_dict = {area["id"]: {attribute: area[attribute] for attribute in attributes} for area in all_areas}
        return attributes_dict

    def get_nx_graph(self):
        G = nx.Graph()
        edges = self.get_all_parent_areas(attr="id")
        G.add_edges_from(edges.items())
        
        # Add attributes to the regions
        attribute_columns = ["acronym", "region_name", "color_hex_triplet", "depth"]
        attrs = self.to_nx_attributes(attribute_columns)
        nx.set_node_attributes(G, attrs)
        #for col_name in attribute_columns:
        #    nx.set_node_attributes(G, self.df[col_name].to_dict(), col_name)
        return G

    def plot_plotly_graph(self):
        '''
        This function plots the brain hierarchy using plotly.
        G = networkx graph
        pos = node positions (as a dictionary)
        '''
        
        G = self.get_nx_graph()
        if not(hasattr(self, "pos")):
            print("Calculating node positions...")
            self.nx_node_pos = graphviz_layout(G, prog="dot")
            self.nx_node_pos = {int(n):p for n,p in self.nx_node_pos.items()}
            nx.set_node_attributes(G, self.nx_node_pos, name="pos")

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]["pos"]
            x1, y1 = G.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines")

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=[],
                size=5,
                line_width=.1))

        node_colors = []
        node_text = []
        for node_id in G.nodes():
            # Number of connections as color
            node_colors.append("#"+G.nodes()[node_id]["color_hex_triplet"])
            # Region name as text to show
            node_text.append(G.nodes()[node_id]["region_name"] + " (" +
                             G.nodes()[node_id]["acronym"] + "), " + 
                             "level = " + str(G.nodes()[node_id]["depth"])) 

        node_trace.marker.color = node_colors
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title="Hierarchy of brain regions",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        return fig