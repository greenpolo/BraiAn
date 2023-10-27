#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 22:28:08 2022

@author: lukasvandenheuvel
"""

import pandas as pd
import numpy as np
import plotly.express as px
from functools import reduce

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
    
    def __init__(self, regions: list[str], group1: AnimalGroup, group2: AnimalGroup, *groups: AnimalGroup, marker=None) -> None:
        groups = [group1, group2, *groups]
        if marker is None:
            if any(len(g.markers) > 1 for g in groups):
                raise ValueError("PLS of AnimalGroups with multiple markers isn't implemented yet")
            else:
                marker = group1.markers[0]
        assert all(group1.is_comparable(g) for g in groups[1:]), "Group 1 and Group 2 are not comparable!\n\
Please check that you're reading two groups that normalized on the same brain regions and on the same marker."
        # Fill a data matrix
        animal_list = [a for g in groups for a in g.get_animals()]
        if len(animal_list) != len(set(animal_list)):
            print("WARNING: some animal(s) are in multiple AnimalGroups")
        animal_list.sort()
        data = pd.DataFrame(index=regions+["group"], columns=animal_list)

        for group in groups:
            data.loc[regions,group.get_animals()] = group.select(regions).to_pandas(marker)
            data.loc["group",group.get_animals()] = group.name

        self.X = data.loc[regions].T.dropna(axis="columns", how="any").astype("float64", copy=False)
        self.Y = pd.get_dummies(data.loc["group"].T)
        self.u, self.s, self.v = self.partial_least_squares_correlation(self.X,self.Y)
        
        self.Lx = self.X @ self.v
        self.Ly = self.Y @ self.u
        
    def partial_least_squares_correlation(self,X,Y):
    
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
        M = np.linalg.inv(np.diag(np.ones(num_animals) @ Y)) @ (Y.T @ X).astype("float")
#        print("\n M:")
#        print(M)
        # Mean-center M to get R
        R = M - np.ones((num_groups,1)) @ ( np.ones((1,num_groups)) @ M) / num_groups
#        print("\n other matrix to subtract to m:")
#        print(num_animals)
#        print("\n R:")
#        print(R)
        # SVD
        u, s, vh = np.linalg.svd(R, full_matrices=False) # self.X, retrieved from AnimalGroup, must have no NaN [dropna(how='any')]. If it does the PLS cannot be computed in the other regions as well
#        print("\n u:")
#        print(u)
#        print("\n s:")
#        print(s)
        return u,s,vh.T
    
    def bootstrap_salience_scores(self,rank,num_bootstrap):
        u_bootstrap = np.expand_dims(np.zeros(self.u.shape), axis=2).repeat(num_bootstrap,axis=2)
        v_bootstrap = np.expand_dims(np.zeros(self.v.shape), axis=2).repeat(num_bootstrap,axis=2)

        num_animals = self.X.shape[0]
        Y_np = self.Y.to_numpy()
        X_np = self.X.to_numpy()
        for i in range(0,num_bootstrap):
            while True:
                sample = np.random.randint(0, num_animals, num_animals)
                Y_sampled = Y_np[sample]
                if Y_sampled.any(axis=0).all():
                    # make sure the sample has at least one animal for each group
                    break
            X_sampled = X_np[sample]
            u_bootstrap[:,:,i],s,v_bootstrap[:,:,i] = self.partial_least_squares_correlation(X_sampled,Y_sampled)

        v_salience = self.v / v_bootstrap.std(axis=2)
        u_salience = self.u / u_bootstrap.std(axis=2)

        self.v_salience_scores = pd.DataFrame(v_salience[:,0:rank], index=self.X.columns)
        self.u_salience_scores = pd.DataFrame(u_salience[:,0:rank])
    
        return self.u_salience_scores,self.v_salience_scores
    
    def randomly_permute_singular_values(self,num_permutations):
    
        singular_values = np.expand_dims(np.zeros(self.s.shape), axis=0).repeat(num_permutations,axis=0)
        X_np = self.X.to_numpy()
        Y_np = self.Y.to_numpy()
        count = 0
        for i in range(num_permutations):
            random_index = np.arange(self.X.shape[0])
            np.random.shuffle(random_index)
            
            #X_perm = X_np[random_index,:]
            Y_perm = Y_np[random_index,:]
#            print ("y_perm after i="+ str(i)+": \n")
#            print (y_perm)
#            print ("\n x_perm after i="+ str(i)+": \n")
#            print (X_perm)
            
            if np.array_equal(Y_perm, Y_np):
#                print ("in continue lop after after i="+ str(i)+": \n")
                continue

#            print("\n for i=" + str(i))
            u_random, singular_values[count,:], vh_random = self.partial_least_squares_correlation(X_np, Y_perm)
            count += 1
            
        self.singular_values = singular_values[:count,:]
        return self.s,self.singular_values
    
    def above_threshold(self, threshold, group=0):
        return self.v_salience_scores[group][self.v_salience_scores[group].abs() > threshold]