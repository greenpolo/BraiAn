#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 22:28:08 2022

@author: lukasvandenheuvel
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px

class PLS:
    '''
    This class facilitates partial least squares analysis on the BLA and IL results.
    The PLS objects has the following properties:
    - X (pd dataframe): the brain activity matrix with animals as rows and brain regions as columns
    - y (pd dataframe): the data groups (i.e., targets), for each animal. '1' means IL, '0' means BLA.
    - u (numpy array): saliences of Y (group matrix)
    - s (numpy array): singular values of the correlation matrix
    - v (numpy array): saliences of X (brain activity matrix)
    - Lx (pd dataframe): latent variables of X, i.e. projection of X on v.
    - Ly (pd dataframe): latent variables of Y, i.e. projection of Y on u.
    '''
    
    def __init__(self, group_1_results, group_2_results, group_1_names, group_2_names, regions, tracer, normalization):
        
        # Fill a data matrix
        animal_list = list(group_1_names.union(group_2_names)) #group_1_names + group_2_names
        animal_list.sort()
        data = pd.DataFrame(index=regions+['group'], columns=animal_list)

        for animal in group_1_names:
            data.loc[regions,animal] = group_1_results.swaplevel(axis=0).loc[(animal,regions),(tracer,normalization)].reset_index(level=0, drop=True)
            data.loc['group',animal] = True

        for animal in group_2_names:
            data.loc[regions,animal] = group_2_results.swaplevel(axis=0).loc[(animal,regions),(tracer,normalization)].reset_index(level=0, drop=True)
            data.loc['group',animal] = False

        self.X = data.loc[regions].T.dropna(axis='columns').astype('float64', copy=False)
        self.y = data.loc['group'].T.astype('bool', copy=False)
        self.u, self.s, self.v = self.partial_least_squares_correlation(self.X,self.y)
        
        self.Lx = self.X @ self.v
        self.Ly = pd.get_dummies(self.y) @ self.u
        
    def partial_least_squares_correlation(self,X,y):
    
        Y = pd.get_dummies(y)
        
        num_animals,num_groups = Y.shape
        

        # Compute M = diag{1.T * Y}.inv * Y.T * X (the average for each group)
        M = np.linalg.inv(np.diag(np.ones(num_animals).dot(Y))).dot( Y.T.dot(X) ).astype('float')
        
        # Mean-center M to get R
        R = M - np.ones((num_groups,1)).dot( np.ones((1,num_groups)).dot(M) ) / num_groups
        
        # SVD
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        
        return u,s,vh.T
    
    def bootstrap_salience_scores(self,rank,num_bootstrap):

        u_bootstrap = np.expand_dims(np.zeros(self.u.shape), axis=2).repeat(num_bootstrap,axis=2)
        v_bootstrap = np.expand_dims(np.zeros(self.v.shape), axis=2).repeat(num_bootstrap,axis=2)

        data = pd.concat([self.X,self.y],axis=1)
        for i in range(0,num_bootstrap):
            data_sampled = data.sample(n=data.shape[0], replace=True)
            X_sampled = data_sampled.drop('group',axis=1)
            y_sampled = data_sampled['group']
            u_bootstrap[:,:,i],s,v_bootstrap[:,:,i] = self.partial_least_squares_correlation(X_sampled,y_sampled)

        v_salience = self.v / v_bootstrap.std(axis=2)
        u_salience = self.u / u_bootstrap.std(axis=2)

        self.v_salience_scores = pd.DataFrame(v_salience[:,0:rank], index=self.X.columns)
        self.u_salience_scores = pd.DataFrame(u_salience[:,0:rank])
    
        return self.u_salience_scores,self.v_salience_scores
    
    def randomly_permute_singular_values(self,num_permutations):
    
        singular_values = np.expand_dims(np.zeros(self.s.shape), axis=0).repeat(num_permutations,axis=0)
        count = 0
        for i in range(num_permutations):
            random_index = np.arange(self.X.shape[0])
            np.random.shuffle(random_index)
            
            X_perm = pd.DataFrame( self.X.to_numpy().astype('float') )
            y_perm = pd.Series( self.y.to_numpy()[random_index].astype('float') )
            
            
            if np.array_equal(y_perm.to_numpy(), self.y.to_numpy()):
                
                continue

            u_random, singular_values[count,:], vh_random = self.partial_least_squares_correlation(X_perm,y_perm)
            count += 1
            
        self.singular_values = singular_values[:count,:]
        return self.s,self.singular_values
    
    def plot_salience_scores(self, threshold, output_path, file_title, brain_region_dict,
                             fig_width=300, fig_height=500):

        to_plot = self.v_salience_scores[(self.v_salience_scores[0]>threshold) | (self.v_salience_scores[0]<-threshold)]
        fig = px.bar(to_plot.reset_index(), x=0, y='index')
        fig.update_layout(
            width=fig_width, height=fig_height,
            xaxis=dict(
                title = 'Salience score'
            ),
        )
        #salient_regions = list(to_plot.index.values)
        fig.show()

        # Save figure as PNG
        if not(os.path.exists(output_path)):
            os.mkdir(output_path)
        output_file = os.path.join(output_path, file_title)
        fig.write_image(output_file)
        return fig, to_plot