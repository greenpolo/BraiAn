import numpy as np
import numpy.typing as npt
import pandas as pd

from collections.abc import Callable, Iterable, Sequence
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Self

from braian.animal_brain import AnimalBrain, SliceMetrics
from braian.brain_data import BrainData
from braian.ontology import AllenBrainOntology
from braian.sliced_brain import SlicedBrain
from braian.utils import save_csv

__all__ = ["AnimalGroup", "SlicedGroup"]

def _common_regions(animals: list[AnimalBrain]) -> list[str]:
    all_regions = [set(brain.regions) for brain in animals]
    return list(reduce(set.__or__, all_regions))

def _have_same_regions(animals: list[AnimalBrain]) -> bool:
    regions = animals[0].regions
    all_regions = [set(brain.regions) for brain in animals]
    return len(reduce(set.__and__, all_regions)) ==  len(regions)

class AnimalGroup:
    def __init__(self, name: str, animals: Sequence[AnimalBrain], hemisphere_distinction: bool=True,
                 brain_ontology: AllenBrainOntology=None, fill_nan: bool=True) -> None:
        """
        Creates an experimental cohort from a set of `AnimalBrain`.\\
        In order for a cohort to be valid, it must consist of brains with
        the same type of data (i.e. [metric][braian.AnimalBrain.metric]),
        the same [markers][braian.AnimalBrain.markers] and
        the data must all be hemisphere-aware or not (i.e. [`AnimalBrain.is_split`][braian.AnimalBrain.is_split]).

        Data for regions missing in one animal but present in others will be always
        filled with [`NA`][pandas.NA].

        Parameters
        ----------
        name
            The name of the cohort.
        animals
            The animals part of the group.
        hemisphere_distinction
            If False, it merges, for each region, the data from left/right hemispheres into a single value.
        brain_ontology
            The ontology to which the brains' data was registered against.
            If specified, it sorts the data in depth-first search order with respect to `brain_ontology`'s hierarchy.
        fill_nan
            If True, it sets the value to [`NA`][pandas.NA] for all the regions missing
            from the data but present in `brain_ontology`.

        See also
        --------
        [`AnimalBrain.merge_hemispheres`][braian.AnimalBrain.merge_hemispheres]
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        [`AnimalBrain.sort_by_ontology`][braian.AnimalBrain.sort_by_ontology]
        [`BrainData.sort_by_ontology`][braian.BrainData.sort_by_ontology]
        [`AnimalBrain.select_from_list`][braian.AnimalBrain.select_from_list]
        [`AnimalBrain.select_from_ontology`][braian.AnimalBrain.select_from_ontology]
        """
        self.name = name
        """The name of the group."""
        # if not animals or not brain_ontology:
        #     raise ValueError("You must specify animals: list[AnimalBrain] and brain_ontology: AllenBrainOntology.")
        assert len(animals) > 0, "A group must be made of at least one animal." # TODO: should we enforce a statistical signficant n? E.g. MIN=4
        _all_markers = {marker for brain in animals for marker in brain.markers}
        assert all(marker in brain.markers for marker in _all_markers for brain in animals), "All AnimalBrain in a group must have the same markers."
        assert all(brain.metric == animals[0].metric for brain in animals[1:]), "All AnimalBrains in a group must be have the same metric."
        is_split = animals[0].is_split
        assert all(is_split == brain.is_split for brain in animals), "All AnimalBrains of a group must either have spit hemispheres or not."

        if is_split and not hemisphere_distinction:
            merge = AnimalBrain.merge_hemispheres
        else:
            merge = lambda brain: brain

        if _have_same_regions(animals):
            fill = lambda brain: brain
        else:
            regions = _common_regions(animals)
            fill: Callable[[AnimalBrain], AnimalBrain] = lambda brain: brain.select_from_list(regions, fill_nan=True, inplace=False)

        if brain_ontology is None:
            sort = lambda brain: brain
        else:
            def sort(brain: AnimalBrain) -> AnimalBrain:
                return brain.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=False)

        self._animals: list[AnimalBrain] = [sort(fill(merge(brain))) for brain in animals] # brain |> merge |> fill |> sort -- OLD: brain |> merge |> analyse |> sort
        self._mean: dict[str, BrainData] = self._update_mean()

    @property
    def n(self) -> int:
        """The size of the group."""
        return len(self._animals)

    @property
    def metric(self) -> str:
        """The metric used to compute the brains' data."""
        return self._animals[0].metric

    @property
    def is_split(self) -> bool:
        """Whether the data of the current `AnimalGroup` makes a distinction between right and left hemisphere."""
        return self._animals[0].is_split

    @property
    def markers(self) -> npt.NDArray[np.str_]:
        """The name of the markers for which the current `AnimalGroup` has data."""
        return np.asarray(self._animals[0].markers)

    @property
    def regions(self) -> list[str]:
        """The list of region acronyms for which the current `AnimalGroup` has data."""
        # NOTE: all animals of the group are expected to have the same regions!
        #       This should be enforced during AnimalGroup construction
        # if not _have_same_regions(animals):
        #     # NOTE: if the animals of the AnimalGroup were not sorted by ontology, the order is not guaranteed to be significant
        #     print(f"WARNING: animals of {self} don't have the same brain regions. "+\
        #           "The order of the brain regions is not guaranteed to be significant. It's better to first call sort_by_ontology()")
        #     return list(reduce(set.union, all_regions))
        #     # return set(chain(*all_regions))
        return self._animals[0].regions

    @property
    def animals(self) -> list[AnimalBrain]:
        """The brains making up the current group."""
        return list(self._animals)

    @property
    def mean(self) -> dict[str, BrainData]:
        """The mean between each brain of the group, for each region."""
        return dict(self._mean)

    def get_animals(self) -> list[str]:
        """
        Returns
        -------
        :
            The names of the animals part of the current group.
        """
        return [brain.name for brain in self._animals]

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"AnimalGroup('{self.name}', metric={self.metric}, n={self.n})"

    def _update_mean(self) -> dict[str, BrainData]:
        # NOTE: skipna=True does not work with FloatingArrays (what BrainData uses)
        #       https://github.com/pandas-dev/pandas/issues/59965
        return {marker: BrainData.mean(*[brain[marker] for brain in self._animals], name=self.name, skipna=True)
                for marker in self.markers}

    def reduce(self, op: Callable[[pd.DataFrame], pd.Series], **kwargs) -> dict[str, BrainData]:
        """
        Applies a reduction to all animals of the group, for each region
        and for each marker.

        Parameters
        ----------
        op
            A function that maps a `DataFrame` into a `Series`. It must include an `axis` parameter.
        **kwargs
            Other keyword arguments are passed to [`BrainData.reduce`][braian.BrainData.reduce].

        Returns
        -------
        :
            Brain data for each marker of the group, result of the the folding.
        """
        return {marker: BrainData.reduce(
                            *[brain[marker] for brain in self._animals],
                            op=op, name=self.name, **kwargs
                            ) for marker in self.markers}

    def is_comparable(self, other: Self) -> bool:
        """
        Tests whether two `AnimalGroup` are comparable for an analysis,
        i.e. they have the same markers, the same metric and both either operate on brains
        hemisphere-aware or not.

        Parameters
        ----------
        other
            The other group to compare with the current one.

        Returns
        -------
        :
            True if the current group and `other` are comparable. False otherwise.
        """
        if not isinstance(other, AnimalGroup):
            return False
        return set(self.markers) == set(other.markers) and \
                self.is_split == other.is_split and \
                self.metric == other.metric # and \
                # set(self.regions) == set(other.regions)

    def select(self, regions: Sequence[str], fill_nan=False, inplace=False) -> Self:
        """
        Filters the data from a given list of regions.

        Parameters
        ----------
        regions
            The acronyms of the regions to select from the data.
        fill_nan
            If True, the regions missing from the current data are filled with [`NA`][pandas.NA].
            Otherwise, if the data from some regions are missing, they are ignored.
        inplace
            If True, it applies the filtering to the current instance.

        Returns
        -------
        :
            A group with data filtered accordingly to the given `regions`.
            If `inplace=True` it returns the same instance.

        See also
        --------
        [`AnimalBrain.select_from_ontology`][braian.AnimalBrain.select_from_ontology]
        """
        animals = [brain.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for brain in self._animals]
        if not inplace:
            # self.metric == animals.metric -> no self.metric.analyse(brain) is computed
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self._animals = animals
            self._mean = self._update_mean()
            return self

    def __getitem__(self, animal_name: str) -> AnimalBrain:
        """

        Parameters
        ----------
        animal_name
            The name of an animal part of the group

        Returns
        -------
        :
            The corresponding `AnimalBrain` in the current group.

        Raises
        ------
        TypeError
            If `animal_name` is not a string.
        KeyError
            If no brain with `animal_name` was found in the group.
        """
        if not isinstance(animal_name, str):
            raise TypeError("AnimalGroup animals are identified by strings")
        try:
            return next(brain for brain in self._animals if brain.name == animal_name)
        except StopIteration:
            raise KeyError(f"'{animal_name}'")

    def apply(self, f: Callable[[AnimalBrain], AnimalBrain],
              brain_ontology: AllenBrainOntology=None, fill_nan: bool=False) -> Self:
        """
        Applies a function to each animal of the group and creates a new `AnimalGroup`.
        Especially useful when applying some sort of metric to the brain data.

        Parameters
        ----------
        f
            A function that maps an `AnimalBrain` into another `AnimalBrain`.
        brain_ontology
            The ontology to which the brains' data was registered against.\\
            If specified, it sorts the data in depth-first search order with respect to brain_ontology's hierarchy.
        fill_nan
            If True, it sets the value to [`NA`][pandas.NA] for all the regions missing from the data but present in `brain_ontology`.

        Returns
        -------
        :
            A group with the data of each animal changed accordingly to `f`.
        """
        return AnimalGroup(self.name,
                           [f(a) for a in self._animals],
                           hemisphere_distinction=False,
                           brain_ontology=brain_ontology,
                           fill_nan=fill_nan)

    def get_units(self, marker: str|None=None) -> str:
        """
        Returns the units of measurment of a marker.

        Parameters
        ----------
        marker
            The marker to get the units for. It can be omitted, if the current brain has only one marker.

        Returns
        -------
        :
            A string representing the units of measurement of `marker`.
        """
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self._animals[0].get_units(marker)

    def sort_by_ontology(self, brain_ontology: AllenBrainOntology,
                         fill_nan=True, inplace=True) -> None:
        """
        Sorts the data in depth-first search order with respect to `brain_ontology`'s hierarchy.

        Parameters
        ----------
        brain_ontology
            The ontology to which the current data was registered against.
        fill_nan
            If True, it sets the value to [`NA`][pandas.NA] for all the regions in
            `brain_ontology` missing in the current `AnimalBrain`.
        inplace
            If True, it applies the sorting to the current instance.

        Returns
        -------
        :
            A brain with data sorted accordingly to `brain_ontology`.
            If `inplace=True` it returns the same instance.
        """
        if not inplace:
            return AnimalGroup(self.name, self._animals, brain_ontology=brain_ontology, fill_nan=fill_nan)
        else:
            for brain in self._animals:
                brain.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=True)
            return self

    def merge_hemispheres(self, inplace=False) -> Self:
        """
        Creates a new `AnimalGroup` from the current instance with no hemisphere distinction.

        Parameters
        ----------
        inplace
            If True, it applies the sorting to the current instance.

        Returns
        -------
        :
            A new [`AnimalGroup`][braian.AnimalGroup] with no hemisphere distinction.
            If `inplace=True` it returns the same instance.

        See also
        --------
        [`AnimalBrain.merge_hemispheres`][braian.AnimalBrain.merge_hemispheres]
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        """
        animals = [AnimalBrain.merge_hemispheres(brain) for brain in self._animals]
        if not inplace:
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self._animals = animals
            self._mean = self._update_mean()
            return self

    def to_pandas(self, marker: str=None, units: bool=False) -> pd.DataFrame:
        """
        Constructs a `DataFrame` with data from the current group.

        Parameters
        ----------
        marker
            If specified, it includes data only from the given marker.
        units
            Whether to include the units of measurement in the `DataFrame` index.

        Returns
        -------
        :
            A  $m×n$ `DataFrame`.\\
            If `marker` is specified, $m=\\#regions$ and $n=\\#brains$.\\
            Otherwise, $m=\\#regions⋅\\#brains$ and $n=\\#markers+1$, as it contains
            the size of the regions as well.
            In the latter case, the index of the `DataFrame` has two levels:
            the acronyms of the regions and the name of the animal in the group.

            If a region is missing in some animals, the corresponding row is [`NA`][pandas.NA]-filled.
        """
        if marker in self.markers:
            df = pd.concat({brain.name: brain[marker].data for brain in self._animals}, join="outer", axis=1)
            df.columns.name = str(self.metric)
            if units:
                a = self._animals[0]
                df.rename(columns={marker: f"{marker} ({a[marker].units})"}, inplace=True)
            return df
        df = {"area": pd.concat({brain.name: brain.areas.data for brain in self._animals}, join="outer", axis=0)}
        for marker in self.markers:
            all_animals = pd.concat({brain.name: brain[marker].data for brain in self._animals}, join="outer", axis=0)
            df[marker] = all_animals
        df = pd.concat(df, join="outer", axis=1)
        df = df.reorder_levels([1,0], axis=0)
        ordered_indices = product(self.regions, [animal.name for animal in self._animals])
        df = df.reindex(ordered_indices)
        df.columns.name = str(self.metric)
        if units:
            a = self._animals[0]
            df.rename(columns={col: f"{col} ({a[col].units if col != 'area' else a.areas.units})" for col in df.columns}, inplace=True)
        return df

    def to_csv(self, output_path: Path|str, sep: str=",", overwrite: bool=False) -> str:
        """
        Write the current `AnimalGroup` to a comma-separated values (CSV) file in `output_path`.

        Parameters
        ----------
        output_path
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        sep
            Character to treat as the delimiter.
        overwrite
            If True, it overwrite any conflicting file in `output_path`.

        Returns
        -------
        :
            The file path to the saved CSV file.
        
        Raises
        ------
        FileExistsError
            If `overwrite=False` and there is a conflicting file in `output_path`.

        See also
        --------
        [`from_csv`][braian.AnimalGroup.from_csv]
        """
        df = self.to_pandas(units=True)
        file_name = f"{self.name}_{self.metric}.csv"
        return save_csv(df, output_path, file_name, overwrite=overwrite, sep=sep, index_label=(df.columns.name, None))

    @staticmethod
    def from_pandas(df: pd.DataFrame, group_name: str) -> Self:
        """
        Creates an instance of [`AnimalGroup`][braian.AnimalGroup] from a `DataFrame`.

        Parameters
        ----------
        df
            A [`to_pandas`][braian.AnimalGroup.to_pandas]-compatible `DataFrame`.
        group_name
            The name of the group associated to the data in `df`.

        Returns
        -------
        :
            An instance of `AnimalGroup` that corresponds to the data in `df`.

        See also
        --------
        [`to_pandas`][braian.AnimalGroup.to_pandas]
        """
        animals = [AnimalBrain.from_pandas(df.xs(animal_name, level=1), animal_name) for animal_name in df.index.unique(1)]
        return AnimalGroup(group_name, animals, fill_nan=False)

    @staticmethod
    def from_csv(filepath: Path|str, name: str, sep: str=",") -> Self:
        """
        Reads a comma-separated values (CSV) file into `AnimalGroup`.

        Parameters
        ----------
        filepath
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        name
            Name of the group associated to the data.
        sep
            Character or regex pattern to treat as the delimiter.

        Returns
        -------
        :
            An instance of `AnimalGroup` that corresponds to the data in the CSV file

        See also
        --------
        [`to_csv`][braian.AnimalGroup.to_csv]
        """
        df = pd.read_csv(filepath, sep=sep, header=0, index_col=[0,1])
        df.columns.name = df.index.names[0]
        df.index.names = (None, None)
        return AnimalGroup.from_pandas(df, name)

    @staticmethod
    def to_prism(marker, brain_ontology: AllenBrainOntology,
                 group1: Self, group2: Self, *groups: Self) -> pd.DataFrame:
        """
        Prepares the marker data from multiple groups in a table structure that is convenient
        to analyse with statistical applications such as Prism by GraphPad, JASP or OriginPro.

        Parameters
        ----------
        marker
            The marker used to compare all groups.
        brain_ontology
            The ontology to which the groups' data was registered against.
        group1
            The first group to include in the export.
        group2
            The second group to include in the export.
        *groups
            Any other number of groups to include in the export.

        Returns
        -------
        :
            A `DataFrame` where rows are brain regions, columns are animals from each group.

        Raises
        ------
        ValueError
            If the given groups are not [comparable][braian.AnimalGroup.is_comparable].
        """
        groups = [group1, group2, *groups]
        if not all(group1.is_comparable(g) for g in groups[1:]):
            raise ValueError("The AnimalGroups are not comparable! Please check that all groups work on the same kind of data (i.e. markers, hemispheres and metric)")
        df = pd.concat({g.name: g.to_pandas(marker) for g in groups}, axis=1)
        major_divisions = brain_ontology.get_corresponding_md(*df.index)
        df["major_divisions"] = [major_divisions[region] for region in df.index]
        df.set_index("major_divisions", append=True, inplace=True)
        return df

class SlicedGroup:
    @staticmethod
    def from_qupath(name: str, brain_names: Iterable[str],
                    qupath_dir: Path|str,
                    brain_ontology: AllenBrainOntology,
                    ch2marker: dict[str,str],
                    exclude_parents: bool,
                    results_subdir: str="results",
                    results_suffix: str="_regions.tsv",
                    exclusions_subdir: str="regions_to_exclude",
                    exclusions_suffix: str="_regions_to_exclude.txt") -> Self:
        """
        Creates an experimental cohort from the section files exported with QuPath. 

        Parameters
        ----------
        name
            The name of the cohort.
        brain_names
            The names of the animals part of the group.
        qupath_dir
            The path to where all the reports of the brains' sections were saved from QuPath.
        brain_ontology
            An ontology against whose version the brains were aligned.
        ch2marker
            A dictionary mapping QuPath channel names to markers.
        exclude_parents
            `exclude_parent_regions` from [`BrainSlice.exclude_regions`][braian.BrainSlice.exclude_regions].
        results_subdir
            The name of the subfolder in `qupath_dir/brain_name` that contains all cell counts files of each brain section.\\
            It can be `None` if no subfolder is used.
        results_suffix
            The suffix used to identify cell counts files saved in `results_subdir`. It includes the file extension.
        exclusions_subdir
            The name of the subfolder in `qupath_dir/brain_name` that contains all regions to exclude from further
            analysis of each brain section.\\
            It can be `None` if no subfolder is used.
        exclusions_suffix
            The suffix used to identify exclusion files saved in `results_subdir`. It includes the file extension.

        Returns
        -------
        :
            A group made of sliced brain data.

        See also
        --------
        [`SlicedBrain.from_qupath`][braian.SlicedBrain.from_qupath]
        """
        sliced_brains = []
        for brain_name in brain_names:
            sliced_brain = SlicedBrain.from_qupath(brain_name, qupath_dir/brain_name, brain_ontology,
                                                   ch2marker, exclude_parents,
                                                   results_subdir, results_suffix,
                                                   exclusions_subdir, exclusions_suffix)
            sliced_brains.append(sliced_brain)
        return SlicedGroup(name, sliced_brains, brain_ontology)

    def __init__(self, name: str, animals: Iterable[SlicedBrain],
                 brain_ontology: AllenBrainOntology) -> None:
        """
        Creates an experimental cohort from a set of `SlicedBrain`.\\
        It is meant to help keeping organised raw data coming multiple sections per-animal.

        Parameters
        ----------
        name
            The name of the cohort.
        animals
            The animals part of the group.
        brain_ontology
            The ontology to which the brains' data was registered against.
        """
        self._name = str(name)
        self._animals = tuple(animals)
        self._brain_ontology = brain_ontology

    @property
    def name(self) -> str:
        """The name of the sliced group."""
        return self._name

    @property
    def animals(self) -> tuple[SlicedBrain]:
        """The brains making up the current sliced group."""
        return self._animals

    @property
    def n(self) -> int:
        """The size of the sliced group."""
        return len(self._animals)

    def get_animals(self) -> list[str]:
        """
        Returns
        -------
        :
            The names of the animals part of the current sliced group.
        """
        return [brain.name for brain in self._animals]

    def to_group(self, metric: SliceMetrics,
                 min_slices: int, densities: bool,
                 hemisphere_distinction: bool, validate: bool) -> AnimalGroup:
        """
        Aggrecates the data from all sections of each [`SlicedBrain`][braian.SlicedBrain]
        into [`AnimalBrain`][braian.AnimalBrain] and organises them into the corresponding
        [`AnimalGroup`][braian.AnimalGroup].

        Parameters
        ----------
        metric
            The metric used to reduce sections data from the same region into a single value.
        min_slices
            The minimum number of sections for a reduction to be valid. If a region has not enough sections, it will disappear from the dataset.
        densities
            If True, it computes the reduction on the section density (i.e., marker/area) instead of doing it on the raw cell counts.
        hemisphere_distinction
            If False, it merges, for each region, the data from left/right hemispheres into a single value.
        validate
            If True, it validates each region in each brain, checking that they are
            present in the brain region ontology against which the brains were alligned.

        Returns
        -------
        :
            A group with the values from sections of the same animals aggregated.

        See also
        --------
        [`AnimalBrain.from_slices`][braian.AnimalBrain.from_slices]
        """
        brains = []
        for sliced_brain in self._animals:
            brain = AnimalBrain.from_slices(sliced_brain, metric, min_slices=min_slices, hemisphere_distinction=hemisphere_distinction, densities=densities)
            brains.append(brain)
        ontology = self._brain_ontology if validate else None
        return AnimalGroup(self._name, brains, hemisphere_distinction=True, brain_ontology=ontology, fill_nan=not validate)