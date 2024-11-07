# SPDX-FileContributor: Lukas van den Heuvel <https://github.com/lukasvandenheuvel>
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats

from braian import AnimalGroup, BrainData
from collections.abc import Sequence
from numbers import Number

__all__ = [
    "PLS",
    "pls_regions_salience"
]

"""
Created on Wed Mar  9 22:28:08 2022

@author: lukasvandenheuvel
@author: carlocastoldi
"""
# pyls.meancentered_pls(pls.X.sort_index(key=lambda s: [int(g) for g in s.str[-1]] ),
#                       groups=[5,6], n_cond=1, mean_centering=0, p_perm=10_000, n_boot=10_000,
#                       split=0, rotate=False, seed=42)
class PLS:
    """
    This class facilitates mean-centered task Partial Least Squares Correlation on brain-wide results.\
    This statistical tool analyzes the relationship between brain activity and experimental design.
    It should be used when the $I$ observations are structured into $N$ `groups` or `markers`.\
    PLSC searches for latent variables (i.e., [$L_X$][braian.stats.PLS.Lx] and [$L_Y$][braian.stats.PLS.Ly])
    that express the largest amount of information common to both $X$ and $Y$, respectively the brain
    activity and the groups matrices.

    The implementation follows a [tutorial from Krishnan et al., 2011](https://doi.org/10.1016/j.neuroimage.2010.07.034);
    for better understanding this method, we _strongly_ suggest reading it.
    
    _NOTE_: if a region is missing from at least one observation, it wont' be taken into account in the analysis.
    """
    def __init__(self, regions: Sequence[str],
                 group1: AnimalGroup, group2: AnimalGroup, *groups: AnimalGroup,
                 marker: str|Sequence[str]=None) -> None:
        """
        Parameters
        ----------
        regions
            The acronyms of brain regions to take into account in the relationship between brain
            activities and experimental design.
        group1
            The first cohort to take into account.
        group2
            The second cohort to take into account.
        *groups
            Any other cohort to take into account.
        marker
            The marker whose activity has to be studied.\
            If multiple markers are given, they'll be coupled with the respective `group`.

        Raises
        ------
        ValueError
            If `marker` is empty, and the given `groups` don't have a single marker to choose from.
        """        
        groups = [group1, group2, *groups]
        if marker is None or (isinstance(marker, Sequence) and len(marker) < 1):
            if any(len(g.markers) > 1 for g in groups):
                raise ValueError("You have to specify the marker to compute the analysis on."+\
                                 "PLS of AnimalGroups with multiple markers isn't implemented yet.")
            marker = [group1.markers[0]]*len(groups)
        if isinstance(marker, str):
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
        self.X: pd.DataFrame = data.loc[regions].T.dropna(axis="columns", how="any").astype("float64", copy=False)
        self.Y: pd.DataFrame = pd.get_dummies(data.loc["group"].T)
        self._u, self._s, self._v = self._mean_centered_task(self.X,self.Y)

        self.Ly: pd.DataFrame = self.Y @ self._u
        """
        The _group scores_, projection of the observed groups composition on [u][braian.stats.PLS.u].
        Depending on the experimental design (e.g. the number of cohorts), there may be projections on
        multiple axes. In order to choose which _latent variables_ to keep, you should
        [generalize the results][braian.stats.PLS.random_permutation] and
        [test the null hypothesis][braian.stats.PLS.test_null_hypothesis].
        """
        self.Lx: pd.DataFrame = self.X @ self._v
        """
        The _brain scores_, projection of the brain activity observations on [v][braian.stats.PLS.v].
        Depending on the experimental design (e.g. the number of cohorts), there may be projections on
        multiple axes. In order to choose which _latent variables_ to keep, you should
        [generalize the results][braian.stats.PLS.random_permutation] and
        [test the null hypothesis][braian.stats.PLS.test_null_hypothesis].
        """

        self._s_sampling_distribution: pd.DataFrame = None
        self._v_salience_scores: pd.DataFrame = None
        self._u_salience_scores: pd.DataFrame = None

    @property
    def u(self) -> pd.DataFrame:
        """
        Group profiles that best characterize $R$, the matrix of the deviations of the groups to their grand mean.
        """
        return pd.DataFrame(self._u, index=self.Y.columns)

    @property
    def s(self) -> npt.ArrayLike:
        """
        Singular values of $R$, the matrix of the deviations of the groups to their gran mean.
        """
        return self._s

    @property
    def v(self) -> pd.DataFrame:
        """
        Brain regions profiles that best characterize $R$, the matrix of the deviations of the groups to their grand mean.
        """
        return pd.DataFrame(self._v, index=self.X.columns)

    @property
    def s_sampling_distribution(self) -> pd.DataFrame:
        """
        The sampling distribution of the singular values, result of a random permutation
        of the [singular values][braian.stats.PLS.s]. Each row is a single _permutation sample_.
        """
        if self._s_sampling_distribution is None:
            raise ValueError(f"Random permutation missing for {self}. Call random_permutation() first.")
        return self._s_sampling_distribution

    @property
    def v_salience_scores(self) -> pd.DataFrame:
        """
        The normalised [v scores][braian.stats.PLS.v] with bootstrapping.
        """
        if self._v_salience_scores is None:
            raise ValueError(f"Region scores are not yet normalized. Call bootstrap_salience_scores first.")
        return self._v_salience_scores

    @property
    def u_salience_scores(self) -> pd.DataFrame:
        """
        The normalised [u scores][braian.stats.PLS.u] with bootstrapping.
        """
        if self._u_salience_scores is None:
            raise ValueError(f"Group scores are not yet normalized. Call bootstrap_salience_scores first.")
        return self._u_salience_scores

    def _mean_centered_task(self, X: pd.DataFrame, Y: pd.DataFrame) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        num_animals,num_groups = Y.shape
        # Compute M = diag{1.T * Y}.inv * Y.T * X (the average for each group)
        M = np.linalg.inv(np.diag(np.ones(num_animals) @ Y)) @ (Y.T @ X).astype("float")
        # R := matrix of the deviations of the groups to their grand mean
        # Mean-center M to get R
        R = M - np.ones((num_groups,1)) @ ( np.ones((1,num_groups)) @ M) / num_groups
        # SVD
        u, s, vh = np.linalg.svd(R, full_matrices=False) # self.X, retrieved from AnimalGroup, must have no NaN [dropna(how='any')]. If it does the PLS cannot be computed in the other regions as well
        return u, s, vh.T

    def n_components(self) -> int:
        """
        Returns the number of components of the current PLS.
        The number of components is determined by the number of group/marker compararisons of the PLS.
        For example, if the current `PLS` is comparing just two groups, it has only 1 component.

        Returns
        -------
        :
            The maximum number of components of the current PLS.
        """
        return self.v.shape[1]+1

    def random_permutation(self, n: int, seed: Number=None):
        """
        Randomly shuffles to which group each brain is part of, and uses this _permutation
        sample_ to compute the mean-centered task PLSC. This process is repeated `n` times.\

        The resulting [_sampling distribution_][braian.stats.PLS.s_sampling_distribution]
        of the singular values can then be used to generalize the results (i.e. salient scores)
        of current PLS as a [null hypothesis test][braian.stats.PLS.test_null_hypothesis].

        Parameters
        ----------
        n
            The number of permutations done to create the sampling distribution.
        seed
            A random seed.
        """
        if seed is not None:
            np.random.seed(seed)
        singular_values = s = np.zeros((n, *self._s.shape))
        X_np = self.X.to_numpy()
        Y_np = self.Y.to_numpy()
        count = 0
        for i in range(n):
            random_index = np.arange(self.X.shape[0])
            np.random.shuffle(random_index)

            #X_perm = X_np[random_index,:]
            Y_perm = Y_np[random_index,:]

            if np.array_equal(Y_perm, Y_np):
                continue

            u_random, singular_values[count,:], vh_random = self._mean_centered_task(X_np, Y_perm)
            count += 1

        self._s_sampling_distribution = singular_values[:count,:]

    def bootstrap_salience_scores(self, n: int, seed: Number=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""
        Identifies the regions $r$ and groups $g$ that are _stable_ by assigning them a score akin to Z-score.
        $$
        \\frac {v_r} {\\hat \\sigma(v_r)} \\text {  and  } \\frac {u_g} {\\hat \\sigma(u_g)}
        $$
        This normalization relies on a set of _bootstrap samples_ to make an estimator of
        salience standard error, $\hat \sigma(v_r)$ and $\hat \sigma(v_r)$.\
        This is achieved through repeatedly drawing samples _with replacement_ from the original dataset.
        Within each sample, the brains' group remains unchanged while the composition of such groups may change.
        Mean-centered task PLS is computed on each sample, effectively creating a large number
        of $u$ and $v$ salience scores samples.
        
        _NOTE_: any interpretation of the resulting normalized scores should be coupled with the
        result of a [permutation test][braian.stats.PLS.random_permutation].

        Parameters
        ----------
        n
            The size of the set of bootstrap samples, used to compute the standard error.
        seed
            A random seed.
        """
        if seed is not None:
            np.random.seed(seed)
        u_bootstrap = np.zeros((*self._u.shape, n))
        v_bootstrap = np.zeros((*self._v.shape, n))

        num_animals = self.X.shape[0]
        Y_np = self.Y.to_numpy()
        X_np = self.X.to_numpy()
        for i in range(0, n):
            while True:
                sample = np.random.randint(0, num_animals, num_animals)
                Y_sampled = Y_np[sample]
                if Y_sampled.any(axis=0).all():
                    # make sure the sample has at least one animal for each group
                    break
            X_sampled = X_np[sample]
            u_bootstrap[:,:,i], s, v_bootstrap[:,:,i] = self._mean_centered_task(X_sampled, Y_sampled)

        v_salience = self._v / v_bootstrap.std(axis=2)
        u_salience = self._u / u_bootstrap.std(axis=2)

        self._v_salience_scores = pd.DataFrame(v_salience, index=self.X.columns)
        self._u_salience_scores = pd.DataFrame(u_salience, index=self.Y.columns)

        return self._u_salience_scores, self._v_salience_scores

    def test_null_hypothesis(self) -> float:
        """
        Tests the null hypothesis on the [sampling distribution of the singular values][braian.stats.PLS.s_sampling_distribution].

        Returns
        -------
        :
            A p-value for each latent variable/component.
        """
        n_permutations,_ = self.s_sampling_distribution.shape
        return (self.s_sampling_distribution > self._s).sum(axis=0)/n_permutations

    def above_threshold(self, threshold: float, component: int=1) -> pd.DataFrame:
        """
        Get the `component`-th regions salience scores that are above `threshold`.

        Parameters
        ----------
        threshold
            A Z-score value. See [`to_zscore`][braian.stats.PLS.to_zscore]
        component
            The n-th component (or [latent variable][braian.stats.PLS.Lx]) of the salience scores
            on which to apply the filter. It cannot be less than 1.

        Returns
        -------
        :
            The list of brain regions, along with the relative score, that have a salience above `threshold`.
        """
        assert 1 <= component < self.n_components(), f"The 'component' must be between 1 and {self.n_components()}."
        return self.v_salience_scores[component-1][self.v_salience_scores[component-1].abs() > threshold]

    @staticmethod
    def to_zscore(p: float, two_tailed: bool=True) -> float:
        """
        Given a probability in null-hypothesis significance testing, it computes the equivalent
        [Z-score](https://en.wikipedia.org/wiki/Standard_score)

        Parameters
        ----------
        p
            The probability.
        two_tailed
            Whether the `p` corresponds to a two-tailed or one-tailed test.

        Returns
        -------
        :
            _description_
        """
        assert p > 0 and p < 1
        return scipy.stats.norm.ppf(1-p/2 if two_tailed else 1-p)


def pls_regions_salience(group1: AnimalGroup, group2: AnimalGroup,
                         selected_regions: list[str], marker: str=None,
                         n_bootstrap: int=5000, component: int=1,
                         fill_nan=True, seed=None,
                         test_h0=True, p_value=0.05, n_permutation: int=5000) -> BrainData|dict[str,BrainData]:
    """
    Computes [PLS][braian.stats.PLS] between two groups with the same markers.\\
    It estimates the standard error of the regions' saliences [by bootstrap][braian.stats.PLS.bootstrap_salience_scores].

    NOTE: it assumes that the `component`-th latent variable is generalisable
    by [permutation test][braian.stats.PLS.random_permutation]. If they were not,
    the resulting salience scores would not be reliable.

    Parameters
    ----------
    group1
        The first cohort to analyze.
    group2
        The second cohort to analyze.
    selected_regions
        The acronyms of brain regions to take into account in the relationship between brain
        activities and experimental design.
    marker
        The marker whose activity has to be studied.\
        If `None`, a separate `PLS` will be computed on each marker in the groups independently.
    n_bootstrap
        The `n` paremeter in [`bootstrap_salience_scores`][braian.stats.PLS.bootstrap_salience_scores].
    component
        The n-th component (or [latent variable][braian.stats.PLS.Lx]) of the salience scores.
    fill_nan
        Whether to fill with [`NA`][pandas.NA] the scores of those regions for which the salience is not
        computable (e.g. if brain data is missing in at least one brain of the groups).
    seed
        A random seed.

    Returns
    -------
    :
        A `BrainData` of the regions salience scores based on `marker` activity.\
        If `marker=None` and the groups have multiple markers, it returns a dictionary
        mapping each marker into the respective regions salience.
    """    
    markers = group1.markers if marker is None else (marker,)
    salience_scores = dict()
    for m in markers:
        pls = PLS(selected_regions, group1, group2, marker=m)
        assert 1 <= component < pls.n_components(), f"The 'component' must be between 1 and {pls.n_components()}."
        if test_h0:
            pls.random_permutation(n_permutation, seed=seed)
            p = pls.test_null_hypothesis()[component-1]
            if p > p_value:
                print(f"WARNING: H0 test is {p}>{p_value} on component {component}! "+\
                      " Make sure to consider this when using the corresponding salience scores.")
        pls.bootstrap_salience_scores(n=n_bootstrap, seed=seed)
        v = pls.v_salience_scores[component-1].copy()
        if fill_nan:
            v_ = pd.Series(np.nan, index=selected_regions)
            v_[v.index] = v
            v = v_
        brain_data = BrainData(v, f"{group1.name}+{group2.name}", "pls_salience", "z-score")
        if len(markers) == 1:
            return brain_data
        salience_scores[m] = brain_data
    return salience_scores