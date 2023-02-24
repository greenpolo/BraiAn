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
from .animal_group import AnimalGroup

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
    
    def __init__(self, group_1: AnimalGroup, group_2: AnimalGroup, regions: list[str], normalization: str) -> None:
        assert group_1.is_comparable(group_2), "Group 1 and Group 2 are not comparable!\n\
Please check that you're reading two groups that normalized on the same brain regions and on the same marker."
        assert normalization in group_1.get_normalization_methods(), f"normalization method '{normalization}' not found.\n\
Available normalizations methods are: {group_1.get_normalization_methods()}"
        assert normalization in group_2.get_normalization_methods(), f"normalization method '{normalization}' not found.\n\
Available normalizations methods are: {group_2.get_normalization_methods()}"
        # Fill a data matrix
        group_1_animals = group_1.get_animals()
        group_2_animals = group_2.get_animals()
        animal_list = list(group_1_animals.union(group_2_animals))
        animal_list.sort()
        data = pd.DataFrame(index=regions+["group"], columns=animal_list)

        for animal in group_1_animals:
            data.loc[regions,animal] = group_1.select(regions, animal)[normalization]
            data.loc["group",animal] = True

        for animal in group_2_animals:
            data.loc[regions,animal] = group_2.select(regions, animal)[normalization]
            data.loc["group",animal] = False

        self.X = data.loc[regions].T.dropna(axis="columns").astype("float64", copy=False)
        self.y = data.loc["group"].T.astype("bool", copy=False)
        self.u, self.s, self.v = self.partial_least_squares_correlation(self.X,self.y)
        
        self.Lx = self.X @ self.v
        self.Ly = pd.get_dummies(self.y) @ self.u
        
    def partial_least_squares_correlation(self,X,y):
    
        Y = pd.get_dummies(y)
#        print ("\n X:")
#        print (X)
#        print("\n y:")
#        print (y)
#        print ("\n gtdummies Y:")
#        print(Y)
        num_animals,num_groups = Y.shape
#        print("\n num_animals:")
#        print(num_animals)
#        print("\n num_groups:")
#        print(num_groups)

        # Compute M = diag{1.T * Y}.inv * Y.T * X (the average for each group)
        M = np.linalg.inv(np.diag(np.ones(num_animals).dot(Y))).dot( Y.T.dot(X) ).astype("float")
#        print("\n M:")
#        print(M)
        # Mean-center M to get R
        R = M - np.ones((num_groups,1)).dot( np.ones((1,num_groups)).dot(M) ) / num_groups
#        print("\n other matrix to subtract to m:")
#        print(num_animals)
#        print("\n R:")
#        print(R)
        # SVD
        u, s, vh = np.linalg.svd(R, full_matrices=False)
#        print("\n u:")
#        print(u)
#        print("\n s:")
#        print(s)
        return u,s,vh.T
    
    def bootstrap_salience_scores(self,rank,num_bootstrap):

        u_bootstrap = np.expand_dims(np.zeros(self.u.shape), axis=2).repeat(num_bootstrap,axis=2)
        v_bootstrap = np.expand_dims(np.zeros(self.v.shape), axis=2).repeat(num_bootstrap,axis=2)

        data = pd.concat([self.X,self.y],axis=1)
        for i in range(0,num_bootstrap):
            data_sampled = data.sample(n=data.shape[0], replace=True)
            X_sampled = data_sampled.drop("group",axis=1)
            y_sampled = data_sampled["group"]
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
            
            X_perm = pd.DataFrame( self.X.to_numpy().astype("float") )
            y_perm = pd.Series( self.y.to_numpy()[random_index].astype("float") )
#            print ("y_perm after i="+ str(i)+": \n")
#            print (y_perm)
#            print ("\n x_perm after i="+ str(i)+": \n")
#            print (X_perm)
            
            if np.array_equal(y_perm.to_numpy(), self.y.to_numpy()):
#                print ("in continue lop after after i="+ str(i)+": \n")
                continue

#            print("\n for i=" + str(i))
            u_random, singular_values[count,:], vh_random = self.partial_least_squares_correlation(X_perm,y_perm)
            count += 1
            
        self.singular_values = singular_values[:count,:]
        return self.s,self.singular_values
    
    def plot_salience_scores(self, threshold, output_path, file_title,
                             fig_width=300, fig_height=500):

        to_plot = self.v_salience_scores[(self.v_salience_scores[0]>threshold) | (self.v_salience_scores[0]<-threshold)]
        fig = px.bar(to_plot.reset_index(), x=0, y="index")
        fig.update_layout(
            width=fig_width, height=fig_height,
            xaxis=dict(
                title = "Salience score"
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