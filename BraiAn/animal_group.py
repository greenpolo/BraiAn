import os
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from itertools import product, chain
from functools import reduce
from typing import Self

from .brain_data import BrainData
from .brain_hierarchy import AllenBrainHierarchy
from .brain_metrics import BrainMetrics
from .animal_brain import AnimalBrain, BrainMetrics
from .utils import save_csv

def common_regions(animals: list[AnimalBrain]) -> list[str]:
    all_regions = [set(brain.get_regions()) for brain in animals]
    return list(reduce(set.__or__, all_regions))

def have_same_regions(animals: list[AnimalBrain]) -> bool:
    regions = animals[0].get_regions()
    all_regions = [set(brain.get_regions()) for brain in animals]
    return len(reduce(set.__and__, all_regions)) ==  len(regions)

class AnimalGroup:
    def __init__(self, name: str, animals: list[AnimalBrain], metric: BrainMetrics, merge_hemispheres=False,
                 brain_onthology: AllenBrainHierarchy=None, fill_nan=True, **kwargs) -> None:
        self.name = name
        # if not animals or not brain_onthology:
        #     raise ValueError("You must specify animals: list[AnimalBrain] and brain_onthology: AllenBrainHierarchy.")
        assert len(animals) > 0, "Inside the group there must be at least one animal."
        assert all([marker in animals[0].markers for brain in animals[1:] for marker in brain.markers]), "All AnimalBrain composing the group must use the same markers."
        self.metric = BrainMetrics(metric)
        assert all([brain.mode == animals[0].mode for brain in animals]), "All AnimalBrains of a group must be hava been processed the same way."
        self.n = len(animals)
        self.is_split = animals[0].is_split
        assert all(self.is_split == brain.is_split for brain in animals), "All AnimalBrains of a group must either have spit hemispheres or not."
        if self.is_split and merge_hemispheres:
            merge = AnimalBrain.merge_hemispheres
        else:
            merge = lambda brain: brain
        if animals[0].mode != self.metric:
            analyse = lambda brain: self.metric.analyse(brain, **kwargs)
        else:
            analyse = lambda brain: brain
        if brain_onthology is not None:
            sort = lambda brain: brain.sort_by_onthology(brain_onthology, fill=fill_nan, inplace=False)
        elif fill_nan:
            regions = common_regions(animals)
            sort = lambda brain: brain.select_from_list(regions, fill=True, inplace=False)
        elif have_same_regions(animals):
            sort = lambda brain: brain
        else:
            # now BrainGroup.get_regions(), which returns the regions of the first animal, is correct
            raise ValueError("Cannot set fill_nan=False and brain_onthology=None if all animals of the group don't have the same brain regions.")
        self.animals: list[AnimalBrain] = [sort(analyse(merge(brain))) for brain in animals]
        self.markers: npt.NDArray[np.str_] = np.asarray(self.animals[0].markers)
        self.mean = self._update_mean()
    
    def markers_corr(self, marker1: str, marker2: str) -> BrainData:
        corr = self.to_pandas(marker1).corrwith(self.to_pandas(marker2), method="pearson", axis=1)
        return BrainData(corr, self.name, str(self.metric)+f"-corr (n={self.n})", f"corr({marker1}, {marker2})")

    def pls_regions(self, other: Self, selected_regions: list[str], marker=None,
                    n_permutations=5000, n_bootstrap=5000, fill_nan=True) -> dict[str,BrainData]:
        markers = self.markers if marker is None else (marker,)
        salience_scores = dict()
        for m in markers:
            pls = PLS(selected_regions, self, other, marker=m)
            pls.randomly_permute_singular_values(n_permutations)
            pls.bootstrap_salience_scores(rank=1, num_bootstrap=n_bootstrap)
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
        return {marker: BrainData.merge(*[brain[marker] for brain in self.animals], op=op, name=self.name, **kwargs) for marker in self.markers}

    def to_pandas(self, marker=None, units=False) -> pd.DataFrame:
        if marker in self.markers:
            df = pd.concat({brain.name: brain.markers_data[marker].data for brain in self.animals}, join="outer", axis=1)
            df.columns.name = str(self.metric)
            if units:
                a = self.animals[0]
                df.rename(columns={marker: f"{marker} ({a[marker].units})"}, inplace=True)
            return df
        df = {"area": pd.concat({brain.name: brain.areas.data for brain in self.animals}, join="outer", axis=0)}
        for marker in self.markers:
            all_animals = pd.concat({brain.name: brain.markers_data[marker].data for brain in self.animals}, join="outer", axis=0)
            df[marker] = all_animals
        df = pd.concat(df, join="outer", axis=1)
        df = df.reorder_levels([1,0], axis=0)
        ordered_indices = product(self.get_regions(), [animal.name for animal in self.animals])
        df = df.reindex(ordered_indices)
        df.columns.name = str(self.metric)
        if units:
            a = self.animals[0]
            df.rename(columns={col: f"{col} ({a[col].units if col != 'area' else a.areas.units})" for col in df.columns}, inplace=True)
        return df
    
    def sort_by_onthology(self, brain_onthology: AllenBrainHierarchy, fill=True, inplace=True) -> None:
        if not inplace:
            return AnimalGroup(self.name, self.animals, metric=self.metric, brain_onthology=brain_onthology, fill_nan=fill)
        else:
            for brain in self.animals:
                brain.sort_by_onthology(brain_onthology, fill=fill, inplace=True)
            return self
    
    def get_animals(self) -> list[str]:
        return [brain.name for brain in self.animals]

    def get_regions(self) -> list[str]:
        # NOTE: all animals of the group are expected to have the same regions!
        # if not have_same_regions(animals):
        #     # NOTE: if the animals of the AnimalGroup were not sorted by onthology, the order is not guaranteed to be significant
        #     print(f"WARNING: animals of {self} don't have the same brain regions. "+\
        #           "The order of the brain regions is not guaranteed to be significant. It's better to first call sort_by_onthology()")
        #     return list(reduce(set.union, all_regions))
        #     # return set(chain(*all_regions))
        return self.animals[0].get_regions()

    def merge_hemispheres(self, inplace=False) -> Self:
        animals = [AnimalBrain.merge_hemispheres(brain) for brain in self.animals]
        if not inplace:
            return AnimalGroup(self.name, animals, metric=self.metric, brain_onthology=None, fill_nan=False)
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
                # set(self.get_regions()) == set(other.get_regions())
    
    def select(self, regions: list[str], fill_nan=False, inplace=False) -> Self:
        animals = [brain.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for brain in self.animals]
        if not inplace:
            # self.metric == animals.metric -> no self.metric.analyse(brain) is computed
            return AnimalGroup(self.name, animals, metric=self.metric, brain_onthology=None, fill_nan=False)
        else:
            self.animals = animals
            self.mean = self._update_mean()
            return self

    def select_animal(self, animal_name: str) -> AnimalBrain:
        return next((brain for brain in self.animals if brain.name == animal_name))

    def remove_smaller_subregions(self, *args, **kwargs) -> None:
        for brain in self.animals:
            brain.remove_smaller_subregions(*args, **kwargs)
        self.mean = self._update_mean()

    def get_units(self, marker=None) -> str:
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self.animals[0].get_units(marker)

    def get_plot_title(self, marker=None) -> str:
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get the plot title for marker '{marker}'!"
        match self.metric:
            case BrainMetrics.DENSITY:
                return f"[#{marker} / area]"
            case BrainMetrics.PERCENTAGE:
                return f"[#{marker} / brain]"
            case BrainMetrics.RELATIVE_DENSITY:
                return f"[#{marker} / area] / [{marker} (brain) / area (brain)]"
            case _:
                raise ValueError(f"Don't know the appropriate title for {self.metric}")

    def to_csv(self, output_path, file_name, overwrite=False) -> None:
        df = self.to_pandas(units=True)
        save_csv(df, output_path, file_name, overwrite=overwrite, index_label=(df.columns.name, None))

    @staticmethod
    def from_pandas(group_name, df: pd.DataFrame) -> Self:
        animals = [AnimalBrain.from_pandas(animal_name, df.xs(animal_name, level=1)) for animal_name in df.index.unique(1)]
        return AnimalGroup(group_name, animals, df.columns.name, fill_nan=False)

    @staticmethod
    def from_csv(group_name, root_dir, file_name) -> Self:
        # # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=0, index_col=[0,1])
        df.columns.name = df.index.names[0]
        df.index.names = (None, None)
        return AnimalGroup.from_pandas(group_name, df)

"""
Created on Wed Mar  9 22:28:08 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""

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
        num_animals,num_groups = Y.shape
        # Compute M = diag{1.T * Y}.inv * Y.T * X (the average for each group)
        M = np.linalg.inv(np.diag(np.ones(num_animals) @ Y)) @ (Y.T @ X).astype("float")
        # Mean-center M to get R
        R = M - np.ones((num_groups,1)) @ ( np.ones((1,num_groups)) @ M) / num_groups
        # SVD
        u, s, vh = np.linalg.svd(R, full_matrices=False) # self.X, retrieved from AnimalGroup, must have no NaN [dropna(how='any')]. If it does the PLS cannot be computed in the other regions as well
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
            
            if np.array_equal(Y_perm, Y_np):
                continue

            u_random, singular_values[count,:], vh_random = self.partial_least_squares_correlation(X_np, Y_perm)
            count += 1
            
        self.singular_values = singular_values[:count,:]
        return self.s,self.singular_values

    def above_threshold(self, threshold, group=0):
        return self.v_salience_scores[group][self.v_salience_scores[group].abs() > threshold]

    @staticmethod
    def norm_threshold(nsigma=2) -> float:
        # returns the μ ± (nsigma)σ of the normal
        return scipy.stats.norm.ppf([0.6826, 0.9545, 0.9973])[nsigma]