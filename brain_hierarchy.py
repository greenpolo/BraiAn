#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:04:28 2022

@author: lukasvandenheuvel
"""

import pandas as pd
import numpy as np
import json
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import plotly.graph_objects as go


def belongs_to_exclude(node_id, G, to_exclude_ids):
    # See if the node ID is a subregion of regions to exclude. 
    shortest_path = nx.shortest_path(G, node_id, 997) # 997 is root
    return len(set.intersection(set(shortest_path), set(to_exclude_ids))) > 0

class AllenBrainHierarchy:

    def __init__(self, path_to_allen_json):
        # Load json file
        with open(path_to_allen_json,) as file:
            allen_data = json.load(file)

        # Unpack json
        count = 0
        while True:
            record = ['msg'] + ['children']*count
            df = pd.json_normalize(allen_data,record_path=record)
            if count==0:
                allen_df = df.copy()
            else:
                allen_df = pd.concat([allen_df, df])
            count += 1

            if df.empty:
                break

        allen_df = allen_df.rename(columns={'name':'region_name'}).set_index('id')
        allen_df['id'] = allen_df.index # also make an id column, this will come in handy 
        
        self.df = allen_df
        G = self.get_graph()
        self.edges_dict = self.get_edges_dict()
        self.tree_dict = self.edges_to_tree()
        self.brain_region_dict = dict(zip(self.df.acronym, self.df.region_name))
        
    def get_graph(self):
        # create graph from rows with a parent 
        clean_df = self.df[self.df.parent_structure_id.values != None]
        
        # Load graph as nx object
        G = nx.from_pandas_edgelist(clean_df, 'id', 'parent_structure_id', 
                                    create_using = nx.DiGraph())

        # Remove the node called 'None'
        # G.remove_node(None) # commented out since it was removed with clean_df

        # add distance to the root to the Allen dict
        self.df['distance_from_root'] = self.df.id.apply(lambda x: nx.shortest_path_length(G, source=x, target=997))
        
        # Add attributes to the regions
        attribute_columns = ['acronym','region_name','color_hex_triplet','distance_from_root']
        for col_name in attribute_columns:
            nx.set_node_attributes(G, self.df[col_name].to_dict(), col_name)
        
        self.graph = G
        return self.graph        

    def list_regions_to_analyze(self, level, to_exclude):
        '''
        Output which regions to analyze as a list
        '''
        # Set regions to exclude from their branches
        to_exclude_ids = [self.df[self.df.acronym==region].index.item() for region in to_exclude]
        self.df['to_exclude'] = self.df.id.apply(lambda x: belongs_to_exclude(x,self.graph,to_exclude_ids))
        # Find the regions upstream of the level which have no children
        to_keep = self.df.apply(lambda row: (row['distance_from_root']<level and len(row['children'])==0), axis=1)
        # Add the regions of the level itself
        to_keep = (to_keep ^ (self.df.distance_from_root==level))
        # Remove regions to exlclude
        to_keep = (to_keep & (self.df.to_exclude==False))
        return list(self.df[to_keep].acronym.values)

    
    def get_edges_dict(self):
        edges_dict = {}
        for edge in self.graph.edges:
            edges_dict[self.df.loc[edge[0],'acronym']] = self.df.loc[edge[1],'acronym']
        return edges_dict
    
    def edges_to_tree(self):
        '''
        Converts a dictionary with edges to a tree dictionary.
        In a tree dictionary, all parent regions are the keys,
        and the subregions that belong to the parent are stored
        in a list as the value corresponding to the key.

        Example:
        tree_dict['ACA'] = ['ACAv', 'ACAd'] # (dorsal and ventral part)
        tree_dict['ACAv'] = ['ACAv5', 'ACAv2/3', 'ACAv6a', 'ACAv1', 'ACAv6b'] # all layers in ventral part
        '''

        # Convert the edges into a tree
        tree_dict = {}
        for child, parent in self.edges_dict.items():
            if parent in tree_dict:
                tree_dict[parent].append(child)
            else:
                tree_dict[parent] = [child]

        return tree_dict

    def plot_plotly_graph(self):
        '''
        This function plots the brain hierarchy using plotly.
        G = networkx graph
        pos = node positions (as a dictionary)
        '''
        
        if not(hasattr(self, 'pos')):
            print('Calculating node positions...')
            self.pos = graphviz_layout(self.graph, prog='dot')
            nx.set_node_attributes(self.graph, self.pos, 'pos')

        G = self.graph
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