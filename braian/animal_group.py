import os
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from itertools import product, chain
from functools import reduce
from typing import Self

from braian.brain_data import BrainData
from braian.ontology import AllenBrainOntology
from braian.animal_brain import AnimalBrain
from braian.utils import save_csv

__all__ = ["AnimalGroup", "PLS"]

def common_regions(animals: list[AnimalBrain]) -> list[str]:
    all_regions = [set(brain.regions) for brain in animals]
    return list(reduce(set.__or__, all_regions))

def have_same_regions(animals: list[AnimalBrain]) -> bool:
    regions = animals[0].regions
    all_regions = [set(brain.regions) for brain in animals]
    return len(reduce(set.__and__, all_regions)) ==  len(regions)

class AnimalGroup:
    def __init__(self, name: str, animals: list[AnimalBrain], merge_hemispheres=False,
                 brain_ontology: AllenBrainOntology=None, fill_nan: bool=True) -> None:
        self.name = name
        # if not animals or not brain_ontology:
        #     raise ValueError("You must specify animals: list[AnimalBrain] and brain_ontology: AllenBrainOntology.")
        self.n = len(animals)
        assert self.n > 0, "Inside the group there must be at least one animal."
        assert all([marker in animals[0].markers for brain in animals[1:] for marker in brain.markers]), "All AnimalBrain composing the group must use the same markers."
        assert all([brain.mode == animals[0].mode for brain in animals]), "All AnimalBrains of a group must be hava been processed the same way."
        is_split = animals[0].is_split
        assert all(is_split == brain.is_split for brain in animals), "All AnimalBrains of a group must either have spit hemispheres or not."
        if is_split and merge_hemispheres:
            merge = AnimalBrain.merge_hemispheres
        else:
            merge = lambda brain: brain
        if brain_ontology is not None:
            sort = lambda brain: brain.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=False)
        elif fill_nan:
            regions = common_regions(animals)
            sort = lambda brain: brain.select_from_list(regions, fill_nan=True, inplace=False)
        elif have_same_regions(animals):
            sort = lambda brain: brain
        else:
            # now BrainGroup.regions, which returns the regions of the first animal, is correct
            raise ValueError("Cannot set fill_nan=False and brain_ontology=None if all animals of the group don't have the same brain regions.")
        self.animals: list[AnimalBrain] = [sort(merge(brain)) for brain in animals] # brain |> merge |> analyse |> sort
        self.mean = self._update_mean()

    @property
    def metric(self) -> str:
        return self.animals[0].mode

    @property
    def is_split(self) -> bool:
        return self.animals[0].is_split

    @property
    def markers(self) -> npt.NDArray[np.str_]:
        return np.asarray(self.animals[0].markers)

    def markers_corr(self, marker1: str, marker2: str, other: Self=None) -> BrainData:
        if other is None:
            other = self
        else:
            assert self.metric == other.metric
        corr = self.to_pandas(marker1).corrwith(other.to_pandas(marker2), method="pearson", axis=1)
        return BrainData(corr, self.name, str(self.metric)+f"-corr (n={self.n})", f"corr({marker1}, {marker2})")

    def pls_regions(self, other: Self, selected_regions: list[str], marker=None,
                    n_bootstrap=5000, fill_nan=True, seed=None) -> dict[str,BrainData]:
        markers = self.markers if marker is None else (marker,)
        salience_scores = dict()
        for m in markers:
            pls = PLS(selected_regions, self, other, marker=m)
            # pls.randomly_permute_singular_values(PLS_N_PERMUTATION, seed=seed)
            # p = pls.test_null_hypothesis()[0]
            pls.bootstrap_salience_scores(num_bootstrap=n_bootstrap, seed=seed)
            v = pls.v_salience_scores[0].copy()
            if fill_nan:
                v_ = pd.Series(np.nan, index=selected_regions)
                v_[v.index] = v
                v = v_
            brain_data = BrainData(v, f"{self.name}+{other.name}", "pls_salience", "z-score")
            if len(markers) == 1:
                return brain_data
            salience_scores[m] = brain_data
        return salience_scores
    
    def __str__(self) -> str:
        return f"AnimalGroup('{self.name}', metric={self.metric}, n={self.n})"

    def _update_mean(self) -> dict[str, BrainData]:
        return {marker: BrainData.mean(*[brain[marker] for brain in self.animals], name=self.name) for marker in self.markers}

    def combine(self, op, **kwargs) -> dict[str, BrainData]:
        """
        _summary_

        Parameters
        ----------
        op
            _description_
        **kwargs
            Other keyword arguments are passed to [`BrainData.reduce`][braian.BrainData.reduce].

        Returns
        -------
        :
            _description_
        """
        return {marker: BrainData.reduce(*[brain[marker] for brain in self.animals], op=op, name=self.name, **kwargs) for marker in self.markers}

    def to_pandas(self, marker=None, units=False) -> pd.DataFrame:
        if marker in self.markers:
            df = pd.concat({brain.name: brain[marker].data for brain in self.animals}, join="outer", axis=1)
            df.columns.name = str(self.metric)
            if units:
                a = self.animals[0]
                df.rename(columns={marker: f"{marker} ({a[marker].units})"}, inplace=True)
            return df
        df = {"area": pd.concat({brain.name: brain.areas.data for brain in self.animals}, join="outer", axis=0)}
        for marker in self.markers:
            all_animals = pd.concat({brain.name: brain[marker].data for brain in self.animals}, join="outer", axis=0)
            df[marker] = all_animals
        df = pd.concat(df, join="outer", axis=1)
        df = df.reorder_levels([1,0], axis=0)
        ordered_indices = product(self.regions, [animal.name for animal in self.animals])
        df = df.reindex(ordered_indices)
        df.columns.name = str(self.metric)
        if units:
            a = self.animals[0]
            df.rename(columns={col: f"{col} ({a[col].units if col != 'area' else a.areas.units})" for col in df.columns}, inplace=True)
        return df
    
    def sort_by_ontology(self, brain_ontology: AllenBrainOntology, fill_nan=True, inplace=True) -> None:
        if not inplace:
            return AnimalGroup(self.name, self.animals, brain_ontology=brain_ontology, fill_nan=fill_nan)
        else:
            for brain in self.animals:
                brain.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=True)
            return self
    
    def get_animals(self) -> list[str]:
        return [brain.name for brain in self.animals]

    @property
    def regions(self) -> list[str]:
        # NOTE: all animals of the group are expected to have the same regions!
        # if not have_same_regions(animals):
        #     # NOTE: if the animals of the AnimalGroup were not sorted by ontology, the order is not guaranteed to be significant
        #     print(f"WARNING: animals of {self} don't have the same brain regions. "+\
        #           "The order of the brain regions is not guaranteed to be significant. It's better to first call sort_by_ontology()")
        #     return list(reduce(set.union, all_regions))
        #     # return set(chain(*all_regions))
        return self.animals[0].regions

    def merge_hemispheres(self, inplace=False) -> Self:
        animals = [AnimalBrain.merge_hemispheres(brain) for brain in self.animals]
        if not inplace:
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self.animals = animals
            self.mean = self._update_mean()
            return self

    def is_comparable(self, other) -> bool:
        if not isinstance(other, AnimalGroup):
            return False
        return set(self.markers) == set(other.markers) and \
                self.is_split == other.is_split and \
                self.metric == other.metric # and \
                # set(self.regions) == set(other.regions)
    
    def select(self, regions: list[str], fill_nan=False, inplace=False) -> Self:
        animals = [brain.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for brain in self.animals]
        if not inplace:
            # self.metric == animals.metric -> no self.metric.analyse(brain) is computed
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self.animals = animals
            self.mean = self._update_mean()
            return self

    def select_animal(self, animal_name: str) -> AnimalBrain:
        return next((brain for brain in self.animals if brain.name == animal_name))

    def get_units(self, marker=None) -> str:
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self.animals[0].get_units(marker)

    def to_csv(self, output_path, file_name, overwrite=False) -> None:
        df = self.to_pandas(units=True)
        save_csv(df, output_path, file_name, overwrite=overwrite, index_label=(df.columns.name, None))

    @staticmethod
    def from_pandas(df: pd.DataFrame, group_name: str) -> Self:
        animals = [AnimalBrain.from_pandas(df.xs(animal_name, level=1), animal_name) for animal_name in df.index.unique(1)]
        return AnimalGroup(group_name, animals, fill_nan=False)

    @staticmethod
    def from_csv(group_name, root_dir, file_name) -> Self:
        # # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=0, index_col=[0,1])
        df.columns.name = df.index.names[0]
        df.index.names = (None, None)
        return AnimalGroup.from_pandas(df, group_name)

"""
Created on Wed Mar  9 22:28:08 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""
# pyls.meancentered_pls(pls.X.sort_index(key=lambda s: [int(g) for g in s.str[-1]] ),
#                       groups=[5,6], n_cond=1, mean_centering=0, p_perm=10_000, n_boot=10_000,
#                       split=0, rotate=False, seed=42)
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
    def __init__(self, regions: list[str], group1: AnimalGroup, group2: AnimalGroup, *groups: AnimalGroup,
                 marker=None, markers=None) -> None:
        groups = [group1, group2, *groups]
        if marker is None:
            if markers is not None:
                assert len(groups) == len(markers), "You need to pass only one marker OR as many markers as the number of groups!"
            elif any(len(g.markers) > 1 for g in groups):
                raise ValueError("PLS of AnimalGroups with multiple markers isn't implemented yet")
            else:
                markers = [group1.markers[0]]*len(groups)
        else:
            assert all(marker in g.markers for g in groups), f"Missing marker '{marker}' in at least on group!"
            markers = [marker]*len(groups)
        assert all(group1.is_comparable(g) for g in groups[1:]), "Group 1 and Group 2 are not comparable!\n\
Please check that you're reading two groups that normalized on the same brain regions and on the same marker."
        # Fill a data matrix
        animal_list = [f"{a}_{i}" for i,g in enumerate(groups) for a in g.get_animals()]
        animal_list.sort()
        data = pd.DataFrame(index=regions+["group"], columns=animal_list)

        for i,(group,_marker) in enumerate(zip(groups, markers)):
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
