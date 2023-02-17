#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:04:28 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""

import pandas as pd
import json
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import plotly.graph_objects as go

from operator import xor, itemgetter
from visit_dict import *

class AllenBrainHierarchy:
    def __init__(self, path_to_allen_json, blacklisted_acronyms=[]):
        with open(path_to_allen_json, 'r') as file:
            allen_data = json.load(file)
        
        self.dict = allen_data["msg"][0]
        if blacklisted_acronyms:
            self.blacklist_regions(blacklisted_acronyms)
            # prune_where(self.dict, "children", lambda x: x["acronym"] in blacklisted_acronyms)

        self.add_depth_to_regions()
        self.edges_dict = self.get_all_parent_areas()
        self.tree_dict = self.get_all_subregions()
        self.brain_region_dict = self.get_full_names()
    
    def blacklist_regions(self, blacklisted_acronyms):
        attr = "acronym"
        def set_blacklisted(node, is_blacklisted):
            node["is_blacklisted"] = is_blacklisted
        # First label every region as 'not blacklisted'
        visit_bfs(self.dict, "children", lambda n,d: set_blacklisted(n, False))
        # Then find every region to-be-blacklisted, and blacklist all its tree
        for blacklisted_acronym in blacklisted_acronyms:
            blacklisted_region = find_subtree(self.dict, attr, blacklisted_acronym, "children")
            visit_bfs(blacklisted_region, "children", lambda n,d: set_blacklisted(n, True))

    def add_depth_to_regions(self):
        def add_depth(node, depth):
            node["depth"] = depth
        visit_bfs(self.dict, "children", add_depth)

    def select_at_depth(self, level):
        def is_selected(node, depth):
            return not node["is_blacklisted"] and (
                depth == level or \
                (depth < level and not node["children"])
            )
        areas = get_where(self.dict, "children", is_selected)
        return [area["acronym"] for area in areas]

    def select_at_structural_level(self, level):
        areas = get_where(self.dict, "children", lambda node,d: node["st_level"] == level)
        return [area["acronym"] for area in areas]
    
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
        subregions['ACA'] = ['ACAv', 'ACAd'] # (dorsal and ventral part)
        subregions['ACAv'] = ['ACAv5', 'ACAv2/3', 'ACAv6a', 'ACAv1', 'ACAv6b'] # all layers in ventral part
        '''
        subregions = dict()
        def add_subregions(node, depth):
            if node["children"]:
                subregions[node[attr]] = [child[attr] for child in node["children"]]
        visit_bfs(self.dict, "children", add_subregions)
        return subregions
    
    def list_all_subregions(self, region_acronym):
        '''
        This function lists all subregions of 'region_acronym' at all hierarchical levels.
        
        Inputs
        ------
            region_acronym (str)
            Acronym of the region of which you want to know the subregions.
            
        Output
        ------
            subregions (list)
            List of all subregions at all hierarchical levels.
        '''
        attr = "acronym"
        region = find_subtree(self.dict, attr, region_acronym, "children")
        subregions = get_all_nodes(region, "children")
        return [subregion[attr] for subregion in subregions]


    def get_full_names(self):
        '''
        returns a dictionary that translates acronym of a region in its full name.
        '''
        all_nodes = get_all_nodes(self.dict, "children")
        return {area["acronym"]: area["name"] for area in all_nodes}

    
    def to_nx_attributes(self, attrs):
        all_nodes = get_all_nodes(self.dict, "children")
        {node["id"]: itemgetter(*attrs)(node) for node in all_nodes}

    def get_nx_graph(self):
        G = nx.Graph()
        edges = self.get_all_parent_areas(attr="id")
        G.add_edges_from(edges.items())
        
        # Add attributes to the regions
        attribute_columns = ['acronym','name','color_hex_triplet','depth']
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
        if not(hasattr(self, 'pos')):
            print('Calculating node positions...')
            self.pos = graphviz_layout(G, prog='dot')
            nx.set_node_attributes(G, self.pos, 'pos')

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=[],
                size=5,
                line_width=.1))

        node_colors = []
        node_text = []
        for node_id in G.nodes():
            # Number of connections as color
            node_colors.append('#'+G.nodes()[node_id]['color_hex_triplet'])
            # Region name as text to show
            node_text.append(G.nodes()[node_id]['region_name'] + ' (' +
                             G.nodes()[node_id]['acronym'] + '), ' + 
                             'level = ' + str(G.nodes()[node_id]['distance_from_root'])) 

        node_trace.marker.color = node_colors
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Hierarchy of brain regions',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        return fig