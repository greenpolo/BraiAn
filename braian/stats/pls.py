# SPDX-FileContributor: Lukas van den Heuvel <https://github.com/lukasvandenheuvel>
import numpy as np
import pandas as pd
import scipy.stats

from braian import AnimalGroup, BrainData
from collections.abc import Sequence

__all__ = [
    "PLS",
    "pls_regions_salience"
]

"""
Created on Wed Mar  9 22:28:08 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""
class PLS:
    '''
    This class facilitates mean-centered task Partial Least Squares Correlation on brain-wide results.
    The PLS objects has the following properties:
    - X (pd dataframe): the brain activity matrix with animals as rows and brain regions as columns
    - y (pd dataframe): the data groups (i.e., targets), for each animal
    - u (numpy array): saliences of Y (group matrix). rows: group profiles that best characterize R
    - s (numpy array): singular values of the correlation matrix
    - v (numpy array): saliences of X (brain activity matrix). rows: brain regions that best characterize R
    - Lx (pd dataframe): latent variables of X, i.e. projection of X on v. AKA "brain scores"
    - Ly (pd dataframe): latent variables of Y, i.e. projection of Y on u. AKA "group scores"
    '''
    def __init__(self, regions: Sequence[str],
                 group1: AnimalGroup, group2: AnimalGroup, *groups: AnimalGroup,
                 marker: str|Sequence[str]=None) -> None:
        groups = [group1, group2, *groups]
        if marker is None:
            if any(len(g.markers) > 1 for g in groups):
                raise ValueError("You have to specify the marker to compute the analysis on."+\
                                 "PLS of AnimalGroups with multiple markers isn't implemented yet.")
            marker = [group1.markers[0]]*len(groups)
        if isinstance(marker, str): # isinstance(marker, Sequence) and not
            marker = [marker]*len(groups)
        assert len(groups) == len(marker), "The number of given 'marker' should be the same as the number of groups."
        assert all(group1.is_comparable(g) for g in groups[1:]), "Group 1 and Group 2 are not comparable!\n"+\
                                                                 "Please check that you're reading two groups that normalized "+\
                                                                 "on the same brain regions and on the same marker."
        # Fill a data matrix
        animal_list = [f"{a}_{i}" for i,g in enumerate(groups) for a in g.get_animals()]
        animal_list.sort()
        data = pd.DataFrame(index=regions+["group"], columns=animal_list)

        for i,(group,_marker) in enumerate(zip(groups, marker)):
            selected_data = group.select(regions).to_pandas(_marker)
            selected_data.columns = selected_data.columns.str.cat((str(i),)*selected_data.shape[1], sep="_")
            data.loc[regions,selected_data.columns] = selected_data
            data.loc["group",selected_data.columns] = group.name+"_"+str(i)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.X = data.loc[regions].T.dropna(axis="columns", how="any").astype("float64", copy=False)
        self.Y = pd.get_dummies(data.loc["group"].T)
        self.u, self.s, self.v = self.partial_least_squares_correlation(self.X,self.Y)
        # self.u = pd.DataFrame(self.u, index=self.Y.columns)
        # self.v = pd.DataFrame(self.v, index=self.X.columns)

        self.Lx = self.X @ self.v
        self.Ly = self.Y @ self.u

        self.v_salience_scores = None
        self.u_salience_scores = None
        self.singular_values = None

    def partial_least_squares_correlation(self,X,Y):
        num_animals,num_groups = Y.shape
        # Compute M = diag{1.T * Y}.inv * Y.T * X (the average for each group)
        M = np.linalg.inv(np.diag(np.ones(num_animals) @ Y)) @ (Y.T @ X).astype("float")
        # R := matrix of the deviations of the groups to their grand mean
        # Mean-center M to get R
        R = M - np.ones((num_groups,1)) @ ( np.ones((1,num_groups)) @ M) / num_groups
        # SVD
        u, s, vh = np.linalg.svd(R, full_matrices=False) # self.X, retrieved from AnimalGroup, must have no NaN [dropna(how='any')]. If it does the PLS cannot be computed in the other regions as well
        return u,s,vh.T

    def bootstrap_salience_scores(self, num_bootstrap, seed=None):
        if seed is not None:
            np.random.seed(seed)
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

        self.v_salience_scores = pd.DataFrame(v_salience, index=self.X.columns)
        self.u_salience_scores = pd.DataFrame(u_salience, index=self.Y.columns)

        return self.u_salience_scores,self.v_salience_scores

    def randomly_permute_singular_values(self,num_permutations, seed=None):
        if seed is not None:
            np.random.seed(seed)
        singular_values = np.expand_dims(np.zeros(self.s.shape), axis=0).repeat(num_permutations,axis=0)
        X_np = self.X.to_numpy()
        Y_np = self.Y.to_numpy()
        count = 0
        for i in range(num_permutations):
            random_index = np.arange(self.X.shape[0])
            np.random.shuffle(random_index)

            #X_perm = X_np[random_index,:]
            Y_perm = Y_np[random_index,:]

            if np.array_equal(Y_perm, Y_np):
                continue

            u_random, singular_values[count,:], vh_random = self.partial_least_squares_correlation(X_np, Y_perm)
            count += 1

        self.singular_values = singular_values[:count,:]

    def test_null_hypothesis(self):
        n_permutations,_ = self.singular_values.shape
        return (self.singular_values > self.s).sum(axis=0)/n_permutations

    def above_threshold(self, threshold, component=1):
        return self.v_salience_scores[component-1][self.v_salience_scores[component-1].abs() > threshold]

    @staticmethod
    def norm_threshold(p: float, two_tailed=True) -> float:
        assert p > 0 and p < 1
        return scipy.stats.norm.ppf(1-p/2 if two_tailed else 1-p)


def pls_regions_salience(group1: AnimalGroup, group2: AnimalGroup,
                         selected_regions: list[str], marker=None,
                         n_bootstrap=5000, fill_nan=True, seed=None) -> BrainData|dict[str,BrainData]:
    """
    Computes [PLS][braian.stats.PLS] between two groups with the same markers.\\
    It estimates the standard error of the regions' saliences [by bootstrap][braian.stats.PLS.bootstrap_salience_scores].

    NOTE: it assumes that the respective latent variables of the two groups are generalisable
    by [permutation test][braian.stats.PLS.randomly_permute_singular_values]. If they were not,
    the resulting salience scores would not be reliable.

    Parameters
    ----------
    group1
        _description_
    group2
        _description_
    selected_regions
        _description_
    marker
        _description_
    n_bootstrap
        _description_
    fill_nan
        _description_
    seed
        _description_

    Returns
    -------
    :
        _description_
    """
    markers = group1.markers if marker is None else (marker,)
    salience_scores = dict()
    for m in markers:
        pls = PLS(selected_regions, group1, group2, marker=m)
        # pls.randomly_permute_singular_values(PLS_N_PERMUTATION, seed=seed)
        # p = pls.test_null_hypothesis()[0]
        pls.bootstrap_salience_scores(num_bootstrap=n_bootstrap, seed=seed)
        v = pls.v_salience_scores[0].copy()
        if fill_nan:
            v_ = pd.Series(np.nan, index=selected_regions)
            v_[v.index] = v
            v = v_
        brain_data = BrainData(v, f"{group1.name}+{group2.name}", "pls_salience", "z-score")
        if len(markers) == 1:
            return brain_data
        salience_scores[m] = brain_data
    return salience_scores