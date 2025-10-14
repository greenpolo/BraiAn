from braian import AnimalBrain, AnimalGroup, AtlasOntology, BrainHemisphere, SlicedBrain, SlicedGroup, SlicedMetric
from braian.utils import _compatibility_check, deprecated, multiindex_from_columns, multiindex_to_columns, save_csv
from collections.abc import Iterable, Callable
from pathlib import Path
from typing import Any, Self

import pandas as pd

__all__ = ["Experiment", "SlicedExperiment"]

class Experiment:
    def __init__(self, name: str, group1: AnimalGroup, group2: AnimalGroup,
                 *groups: AnimalGroup) -> None:
        """
        Creates an experiment from the data of two or more [`AnimalGroups`][braian.AnimalGroup].

        Parameters
        ----------
        name
            The name of the experiment.
        group1
            The first group of the experiment.
        group2
            The second group of the experiment.
        *groups
            Any other group of the experiment.

        Raises
        ------
        ValueError
            If the groups don't have the same metric.
        """
        self._name = str(name)
        self._groups = (group1, group2, *groups)
        _compatibility_check(self._groups)

    @property
    def atlas(self) -> str:
        """The name of the atlas used to align the brain data."""
        return self._groups[0].atlas

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._name

    @property
    def groups(self) -> tuple[AnimalGroup]:
        """The groups in the experiment."""
        return self._groups

    @property
    def n(self) -> int:
        """The number of groups in the experiment."""
        return len(self._groups)

    @property
    def metric(self) -> str:
        """The metric of the brain quantifications."""
        return self._groups[0].metric

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return self._groups[0].is_split

    @property
    def markers(self) -> str:
        """The name of the markers for which the `AnimalGroup` has data."""
        return self._groups[0].markers

    @property
    def hemispheres(self) -> tuple[BrainHemisphere]|tuple[BrainHemisphere,BrainHemisphere]:
        """The hemispheres for which the `AnimalGroup` has data."""
        return self._groups[0].hemispheres

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"Experiment('{self.name}', groups={self.n}, metric={self.metric}, is_split={self.is_split})"

    def __iter__(self) -> Iterable[AnimalGroup]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __getattr__(self, name: str) -> AnimalGroup:
        """
        Get a specific group in the experiment by accessing it with an
        attribute named like the name of the group.

        Parameters
        ----------
        name
            The name of the group

        Returns
        -------
        :
            The group in the experiment having the same name as `name`.

        Raises
        ------
        AttributeError
            If no group with `name` was found in the experiment.
        """
        for g in self._groups:
            if g.name.lower() == name.lower():
                return g
        raise AttributeError(f"Unknown group named '{name.lower()}'")

    def __contains__(self, animal_name: str) -> bool:
        """
        Parameters
        ----------
        animal_name
            The name of an animal.

        Returns
        -------
        :
            True, if the experiment contains an animal with `animal_name`.
            False, otherwise.
        """
        return any(animal_name in group for group in self._groups)

    def __getitem__(self, val: str|int) -> AnimalBrain:
        """

        Parameters
        ----------
        val
            The name or the index of a `AnimalBrain` of the sliced experiment.

        Returns
        -------
        : AnimalGroup
            The corresponding group in the experiment, if `val` is an `int`.
        : AnimalBrain
            The corresponding brain in the experiment, if `val is a `str`.

        Raises
        ------
        TypeError
            If `val` is not a string nor an int.
        IndexError
            If `val` is an int index out of bound.
        KeyError
            If no brain named `val` was found in the experiment.
        """
        if isinstance(val, int):
            return self._groups[val]
        if not isinstance(val, str):
            raise TypeError("Experiment's animals are identified by strings or int")
        for group in self._groups:
            try:
                return group[val]
            except KeyError:
                pass
        raise KeyError(f"{val}")

    @deprecated(since="1.1.0",
                params=["hemisphere_distinction", "brain_ontology", "fill_nan"])
    def apply(self,
              f: Callable[[AnimalBrain], AnimalBrain],
              *fs: Callable[[AnimalBrain], AnimalBrain],
              hemisphere_distinction: bool=True,
              brain_ontology: AtlasOntology=None, fill_nan: bool=False) -> Self:
        """
        Applies a function `f` (or a series of functions `fs`) to each animal of the groups
        of the experiment and creates a new `Experiment`.\\
        Especially useful when applying some sort of metric to the brain data.

        Parameters
        ----------
        f
            A function that maps an `AnimalBrain` into another `AnimalBrain`.
        *fs
            Any number of functions that are going to be applied _after_ `f`.

        Returns
        -------
        :
            An experiment with the data of each animal changed accordingly to `f`.
        """
        groups = [g.apply(f, *fs) for g in self._groups]
        return Experiment(self._name, *groups)

    def to_pandas(self, *,
                  marker: str=None, units: bool=False,
                  missing_as_nan: bool=False,
                  hemisphere_as_value: bool=False,
                  hemisphere_as_str: bool=False) -> pd.DataFrame:
        """
        Constructs a `DataFrame` with data from the current experiment.

        Parameters
        ----------
        marker
            If specified, it includes data only from the given marker.
        units
            Whether to include the units of measurement in the `DataFrame` columns.
            Available only when `marker=None`.
        missing_as_nan
            If True, it converts missing values [`NA`][pandas.NA] as [`NaN`][numpy.nan].
            Note that if the corresponding brain data is integer-based, it converts them to float.
        hemisphere_as_value
            If True and `legacy=False`, it converts the regions' hemisphere to the corresponding value (i.e. 0, 1 or 2)
        hemisphere_as_str
            If True and `legacy=False`, it converts the regions' hemisphere to the corresponding string (i.e. "both", "left", "right")

        Returns
        -------
        :
            A  $m×n$ `DataFrame`, where $m=|regions|$ and $n=|groups|\\times|brains|\\times(|markers|+1)$.\\
            If `marker` is not None, $n=|groups|\\times|brains|$.

            The `DataFrame` is indexed, on the columns, by three levels: the
            [groups][braian.Experiment.groups], the [animals][braian.AnimalGroup.animals] and the
            [markers][braian.AnimalBrain.markers].

            If a region is missing in some groups/animals, the corresponding row is [`NA`][pandas.NA]-filled.

        See also
        --------
        [`from_pandas`][braian.Experiment.from_pandas]
        [`AnimalBrain.to_pandas`][braian.AnimalBrain.to_pandas]
        [`AnimalGroup.to_pandas`][braian.AnimalGroup.to_pandas]
        """
        df = pd.concat(
            {group.name: group.to_pandas(marker=marker, units=units, missing_as_nan=missing_as_nan,
                                        legacy=False, hemisphere_as_value=hemisphere_as_value,
                                        hemisphere_as_str=hemisphere_as_str)
                for group in self._groups},
            join="outer", axis=1)
        if marker is None:
            df.columns.names = (self.name, None, self.metric)
        else:
            df.columns.names = (self.name, self.metric)
        return df

    def to_csv(self, path: Path|str, sep: str=",",
               overwrite: bool=False) -> str:
        """
        Writes the experiment's data to a comma-separated values (CSV) file in `path`.

        Parameters
        ----------
        path
            Any valid string path is acceptable. It also accepts any [os.PathLike][].

            If the path isn't of a `.csv` file, it creates a file in `path` named
            `<`[`name`][braian.Experiment.name]`>_<`[`metric`][braian.Experiment.metric]`>.csv`.
        sep
            Character to treat as the delimiter.
        overwrite
            If True, it overwrite any conflicting file in `path`.

        Returns
        -------
        :
            The file path to the saved CSV file.

        Raises
        ------
        FileExistsError
            If `overwrite=False` and there is a conflicting file in `path`.

        See also
        --------
        [`from_csv`][braian.Experiment.from_csv]
        [`to_pandas`][braian.Experiment.to_pandas]
        [`AnimalBrain.to_csv`][braian.AnimalBrain.to_csv]
        [`AnimalGroup.to_csv`][braian.AnimalGroup.to_csv]
        """
        df = self.to_pandas(units=True, hemisphere_as_str=True)
        output_path = Path(path)
        if (file_name:=output_path.name.lower()).endswith(".csv"):
            path = output_path.parent
        else:
            file_name = f"{self.name}_{self.metric}.csv"
        multiindex_to_columns(df, inplace=True)
        return save_csv(df, path, file_name, overwrite=overwrite, sep=sep, index=False)

    def from_pandas(df: pd.DataFrame,
                    *,
                    ontology: AtlasOntology,
                    name: str=None) -> Self:
        """
        Creates an instance of [`Experiment`][braian.Experiment] from a `DataFrame`.

        Parameters
        ----------
        df
            A [`to_pandas`][braian.Experiment.to_pandas]-compatible `DataFrame`.
        ontology
            The ontology of the atlas used to align the brain data in `df`.
        name
            The name of the group associated with the data in `df`,
            if you want to overwrite the one stated in the first columns' name.

        Returns
        -------
        :
            An instance of `Experiment` that corresponds to the data in `df`.

        Raises
        ------
        UnknownBrainRegionsError
            If `df` contains regions not present in `ontology`.

        See also
        --------
        [`to_pandas`][braian.Experiment.to_pandas]
        [`AnimalBrain.from_pandas`][braian.AnimalBrain.from_pandas]
        [`AnimalGroup.from_pandas`][braian.AnimalGroup.from_pandas]
        """
        if name is None:
            name = df.columns.names[0]
        groups = df.columns.unique(0)
        name = df.columns.names[0]
        groups = [AnimalGroup.from_pandas(
                    df=df[group_name],
                    name=group_name, ontology=ontology, legacy=False)
                for group_name in groups]
        return Experiment(name, *groups)

    @staticmethod
    def from_csv(filepath: Path|str,
                 *,
                 ontology: AtlasOntology,
                 name: str=None,
                 sep: str=",") -> Self:
        """
        Reads a comma-separated values (CSV) file into `Experiment`.

        Parameters
        ----------
        filepath
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        ontology
            The ontology of the atlas used to align the brain data in `filepath`.
        name
            Name of the group associated  with the data in `filepath`,
            if you want to overwrite the one with which it was saved.
        sep
            Character or regex pattern to treat as the delimiter.

        Returns
        -------
        :
            An instance of `Experiment` that corresponds to the data in the CSV file.

        Raises
        ------
        UnknownBrainRegionsError
            If the data in `filepath` contains regions not present in `ontology`.

        See also
        --------
        [`from_csv`][braian.from_csv]
        [`Experiment.to_csv`][braian.Experiment.to_csv]
        [`AnimalBrain.from_csv`][braian.AnimalBrain.from_csv]
        [`AnimalGroup.from_csv`][braian.AnimalGroup.from_csv]
        """
        df = pd.read_csv(filepath, sep=sep, header=[0,1,2,3])
        multiindex_from_columns(df, index_col=[0,1], inplace=True)
        return Experiment.from_pandas(df, name=name, ontology=ontology)

class SlicedExperiment:
    def __init__(self, name: str, group1: SlicedGroup, group2: SlicedGroup,
                 *groups: Iterable[SlicedGroup]) -> None:
        """
        Creates an experiment from the data of two or more [`SlicedGroups`][braian.SlicedGroup].

        Parameters
        ----------
        name
            The name of the sliced experiment.
        group1
            The first group of the sliced experiment.
        group2
            The second group of the sliced experiment.
        *groups
            Any other group of the sliced experiment.
        """
        self._name: str = str(name)
        self._groups: tuple[SlicedGroup] = (group1, group2, *groups)
        _compatibility_check(self._groups, check_metrics=False, check_hemispheres=False)

    @property
    def atlas(self) -> str:
        """The name of the atlas used to align the brain data."""
        return self._groups[0].atlas

    @property
    def name(self) -> str:
        """The name of the sliced experiment."""
        return self._name

    @property
    def groups(self) -> tuple[SlicedGroup]:
        """The `SlicedGroup`s in the sliced experiment."""
        return self._groups

    @property
    def n(self) -> int:
        """The number of groups in the experiment."""
        return len(self._groups)

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return self._groups[0].is_split

    @property
    def markers(self) -> str:
        """The name of the markers for which the `SlicedGroup` has data."""
        return self._groups[0].markers

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"SlicedExperiment('{self.name}', groups={self.n}, is_split={self.is_split})"


    def __iter__(self) -> Iterable[AnimalGroup]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __getattr__(self, name: str) -> Any:
        """
        Get a specific group in the sliced experiment by accessing it with an attribute named like the name of the group.

        Parameters
        ----------
        name
            The name of the group

        Returns
        -------
        :
            The group in the sliced experiment having the same name as `name`.

        Raises
        ------
        AttributeError
            If no group with `name` was found in the current sliced experiment.
        """
        for g in self._groups:
            if g.name.lower() == name.lower():
                return g
        raise AttributeError(f"Uknown group named '{name.lower()}'")

    @deprecated(since="1.1.0", alternatives=["braian.SlicedExperiment.reduce"])
    def to_experiment(self, metric: SlicedMetric,
                      min_slices: int, densities: bool,
                      hemisphere_distinction: bool, validate: bool) -> Experiment:
        """
        Aggregates the data from all sections of each [`SlicedBrain`][braian.SlicedBrain]
        into [`SlicedGroup`][braian.SlicedGroup] and organises them into the corresponding
        [`Experiment`][braian.Experiment].

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
            An experiment with the values from sections of the same animals aggregated.

        See also
        --------
        [`SlicedGroup.to_group`][braian.SlicedGroup.to_group]
        """
        return self.reduce(metric=metric, min_slices=min_slices, densities=densities)

    def reduce(self,
               metric: SlicedMetric,
               *,
               min_slices: int,
               densities: bool) -> Experiment:
        """
        Aggregates the data from all sections of each [`SlicedBrain`][braian.SlicedBrain]
        into [`SlicedGroup`][braian.SlicedGroup] and organises them into the corresponding
        [`Experiment`][braian.Experiment].

        Parameters
        ----------
        metric
            The metric used to reduce sections data from the same region into a single value.
        min_slices
            The minimum number of sections for a reduction to be valid. If a region has not enough sections, it will disappear from the dataset.
        densities
            If True, it computes the reduction on the section density (i.e., marker/area) instead of doing it on the raw cell counts.

        Returns
        -------
        :
            An experiment with the values from sections of the same animals aggregated.

        See also
        --------
        [`SlicedBrain.reduce`][braian.SlicedBrain.reduce]
        [`SlicedGroup.reduce`][braian.SlicedGroup.reduce]
        """
        groups = [g.reduce(metric, min_slices=min_slices, densities=densities) for g in self._groups]
        return Experiment(self.name, *groups)

    def __contains__(self, animal_name: str) -> bool:
        """
        Parameters
        ----------
        animal_name
            The name of an animal.

        Returns
        -------
        :
            True, if the experiment contains an animal with `animal_name`.
            False, otherwise.
        """
        return any(animal_name in group for group in self._groups)

    def __getitem__(self, val: str|int) -> SlicedGroup|SlicedBrain:
        """

        Parameters
        ----------
        val
            The name or the index of a `SlicedBrain` of the sliced experiment.

        Returns
        -------
        : SlicedGroup
            The corresponding group in the sliced experiment, if `val` is an `int`.
        : SlicedBrain
            The corresponding group in the sliced experiment, if `val is a `str`.

        Raises
        ------
        TypeError
            If `val` is not a string nor an int.
        IndexError
            If `val` is an int index out of bound.
        KeyError
            If no brain named `val` was found in the experiment.
        """
        if isinstance(val, int):
            return self._groups[val]
        if not isinstance(val, str):
            raise TypeError("Experiment's animals are identified by strings or int")
        for group in self._groups:
            try:
                return group[val]
            except KeyError:
                pass
        raise KeyError(f"{val}")

    def apply(self,
              f: Callable[[SlicedBrain], SlicedBrain],
              *fs: Callable[[AnimalBrain], AnimalBrain],) -> Self:
        """
        Applies a function `f` (or a series of functions `fs`) to each animal of the groups
        of the experiment and creates a new `SlicedExperiment`.

        Parameters
        ----------
        f
            A function that maps an `SlicedBrain` into another `SlicedBrain`.
        *fs
            Any number of functions that are going to be applied _after_ `f`.

        Returns
        -------
        :
            An experiment with the data of each animal changed accordingly to `f`.
        """
        groups = [g.apply(f, *fs) for g in self._groups]
        return SlicedExperiment(self._name, *groups)