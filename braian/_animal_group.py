import numpy as np
import numpy.typing as npt
import pandas as pd

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Self

from braian import AnimalBrain, AtlasOntology, BrainData, BrainHemisphere, SlicedBrain, SlicedMetric
from braian.utils import _compatibility_check, _same_regions, deprecated, merge_ordered, save_csv

__all__ = ["AnimalGroup", "SlicedGroup"]

def _combined_regions(animals: list[AnimalBrain]) -> dict[BrainHemisphere,list[str]]:
    common_regions = dict()
    for hemi in animals[0].hemispheres:
        common_regions[hemi] = merge_ordered(*[brain.hemiregions[hemi] for brain in animals], raises=False)
        # all_regions = [set(brain.hemiregions[hemi]) for brain in animals]
        # common_regions[hemi] = list(functools.reduce(set.__or__, all_regions))
    return common_regions

def _have_same_regions(brains: list[AnimalBrain]) -> bool:
    for hem,regions in brains[0].hemiregions.items():
        if not _same_regions(regions, map(lambda b: b.hemiregions[hem], brains[1:])):
            return False
    return True

class AnimalGroup:
    @deprecated(since="1.1.0",
                params=["hemisphere_distinction", "brain_ontology", "fill_nan"],
                alternatives=dict(
                    hemisphere_distinction="braian.merge_hemispheres"
                ))
    def __init__(self, name: str, animals: Sequence[AnimalBrain], hemisphere_distinction: bool=True,
                 brain_ontology: AtlasOntology=None, fill_nan: bool=False) -> None:
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
        [`AnimalBrain.sort_by_ontology`][braian.AnimalBrain.sort_by_ontology]
        [`BrainData.sort_by_ontology`][braian.BrainData.sort_by_ontology]
        [`AnimalBrain.select`][braian.AnimalBrain.select]
        """
        self.name = name
        """The name of the group."""
        # if not animals or not brain_ontology:
        #     raise ValueError("You must specify animals: list[AnimalBrain] and brain_ontology: AtlasOntology.")
        assert len(animals) > 0, "A group must be made of at least one animal." # TODO: should we enforce a statistical signficant n? E.g. MIN=4
        _compatibility_check(animals)

        no_update = lambda b: b  # noqa: E731

        if _have_same_regions(animals):
            fill = no_update
        else:
            hemiregions = _combined_regions(animals)
            def fill(brain: AnimalBrain) -> AnimalBrain:
                for hemi,combined_regions in hemiregions.items():
                    brain = brain.select(combined_regions, fill_nan=True, inplace=False,
                                         hemisphere=hemi, select_other_hemisphere=True)
                return brain

        self._animals: list[AnimalBrain] = [fill(brain) for brain in animals] # OLD: brain |> merge |> analyse |> sort
        self._hemimean: dict[str, tuple[BrainData]|tuple[BrainData,BrainData]] = self._update_mean()

    @property
    def n(self) -> int:
        """The size of the group."""
        return len(self._animals)

    @property
    def metric(self) -> str:
        """The metric of the brain quantifications."""
        return self._animals[0].metric

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return self._animals[0].is_split

    @property
    def markers(self) -> list[str]:
        """The name of the markers for which the `AnimalGroup` has data."""
        return self._animals[0].markers

    @property
    def hemispheres(self) -> tuple[BrainHemisphere]|tuple[BrainHemisphere,BrainHemisphere]:
        """The hemispheres for which the `AnimalGroup` has data."""
        return self._animals[0].hemispheres

    @property
    def regions(self) -> list[str]:
        """The list of region acronyms for which the current `AnimalGroup` has data."""
        # NOTE: assumes that all brains in the group were synchronised in __init__
        # and never touched again outside of the group
        return self._animals[0].regions

    @property
    def hemiregions(self) -> dict[BrainHemisphere, list[str]]:
        """
        The dictionary that maps the hemispheres to the list of region acronyms for which the current `AnimalGroup` has data.
        """
        # NOTE: assumes that all brains in the group were synchronised in __init__
        # and never touched again outside of the group
        # if not _have_same_regions(animals):
        #     # NOTE: if the animals of the AnimalGroup were not sorted by ontology, the order is not guaranteed to be significant
        #     print(f"WARNING: animals of {self} don't have the same brain regions. "+\
        #           "The order of the brain regions is not guaranteed to be significant. It's better to first call sort_by_ontology()")
        #     return list(functools.reduce(set.union, all_regions))
        #     # return set(chain(*all_regions))
        return self._animals[0].hemiregions

    @deprecated(since="1.1.0", alternatives=["braian.AnimalGroup.units"])
    def get_units(self, marker: str|None=None) -> str:
        return self.units(marker)

    def units(self, marker: str|None=None) -> str:
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
        return self._animals[0].units(marker)

    @property
    def animals(self) -> list[AnimalBrain]:
        """The brains making up the current group."""
        return list(self._animals)

    @property
    def mean(self) -> dict[str, BrainData]:
        """The mean between each brain of the group, for each marker and for each region."""
        if self.is_split:
            raise ValueError("Cannot get a single marker mean for all brain regions because the group is split between left and right hemispheres."+\
                             "Use AnimalGroup.hemimean.")
        return {marker: hemimeans[0] for marker,hemimeans in self._hemimean.items()}

    @property
    def hemimean(self) -> dict[str,tuple[BrainData,BrainData]|tuple[BrainData]]:
        """
        The mean between each brain of the group, for each marker and for each region.
        If the AnimalGroup has data for two hemispheres, the dictionary maps to tuples of size two.
        """
        return dict(self._hemimean)

    @property
    def animal_names(self) -> list[str]:
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
        return f"AnimalGroup('{self.name}', brains={self.n}, metric={self.metric}, is_split={self.is_split})"

    def _update_mean(self) -> dict[str, tuple[BrainData]|tuple[BrainData,BrainData]]:
        # NOTE: skipna=True does not work with FloatingArrays (what BrainData uses)
        #       https://github.com/pandas-dev/pandas/issues/59965
        #       braindata regions with np.nan values resulting from a computation (e.g division by zero)
        #       'corrupt' the whole result into becoming a pd.NA (not a np.nan).
        return {marker:
            tuple(BrainData.mean(*[brain[marker,hemi] for brain in self._animals],
                                 name=self.name, skipna=True)
                  for hemi in self.hemispheres)
            for marker in self.markers}

    def reduce(self,
               op: Callable[[pd.DataFrame], pd.Series],
               op_name: str=None,
               same_units: bool=True,
               same_hemisphere: bool=True,
               **kwargs) -> dict[str, BrainData]:
        """
        Applies a reduction on each brain structure, between all the brains in the group,
        and for each marker.

        If `op` is [`pd.DataFrame.mean`][pandas.DataFrame.mean], it is equivalent to
        [`mean`][braian.AnimalGroup.mean] and [`hemimean`][braian.AnimalGroup.hemimean].

        Parameters
        ----------
        op
            A function that maps a `DataFrame` into a `Series`. It must include an `axis` parameter.
        op_name
            The name of the reduction function. If not specified, it uses `op` name.
        same_units
            Whether it should enforce the same units of measurement for all `BrainData`.
        same_hemisphere
            Whether it should enforce the same hemisphere for all `BrainData`.
        **kwargs
            Other keyword arguments are passed to `op`.

        Returns
        -------
        :
            Brain data for each marker of the group, result of the the folding.

        Examples
        --------
        >>> import braian.config
        >>> import pandas as pd
        >>> config = braian.config.BraiAnConfig("../config_example.yml")
        >>> g = config.experiment_from_csv().groups[0]
        >>> reduction = g.reduce(pd.DataFrame.mean, skipna=True)
        >>> [(hemiredux.data == hemimean.data).all()
        >>>  for marker in g.markers
        >>>  for hemiredux,hemimean in zip(reduction[marker],g.hemimean[marker])]
        [np.True_, np.True_, np.True_, np.True_, np.True_, np.True_]

        >>> gm = g.merge_hemispheres()
        >>> reduction = gm.reduce(pd.DataFrame.mean, skipna=True)
        >>> [(reduction[marker].data == gm.mean[marker].data).all()
        >>>  for marker in gm.markers]
        [np.True_, np.True_, np.True_]
        """
        if self.is_split:
            return {marker:
                tuple(BrainData.reduce(*[brain[marker,hemi] for brain in self._animals],
                                        name=self.name, op=op, op_name=op_name,
                                        same_units=same_units, same_hemisphere=same_hemisphere,
                                        **kwargs)
                      for hemi in self.hemispheres)
                for marker in self.markers}
        return {marker: BrainData.reduce(*[brain[marker] for brain in self._animals],
                                         name=self.name, op=op, op_name=op_name,
                                         same_units=same_units, same_hemisphere=same_hemisphere,
                                         **kwargs)
                for marker in self.markers}

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

    def select(self, regions: Sequence[str]|AtlasOntology,
               fill_nan: bool=False, inplace: bool=False,
               hemisphere: BrainHemisphere|str|int=BrainHemisphere.BOTH,
               select_other_hemisphere: bool=False) -> Self:
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
        hemisphere
            If not [`BOTH`][braian.BrainHemisphere] and the brains [are split][braian.AnimalGroup.is_split],
            it only selects the brain regions from the given hemisphere.
        select_other_hemisphere
            If True and `hemisphere` is not [`BOTH`][braian.BrainHemisphere], it also selects the opposite hemisphere.\
            If False, it deselect the opposite hemisphere.

        Returns
        -------
        :
            A group with data filtered accordingly to the given `regions`.
            If `inplace=True` it returns the same instance.

        See also
        --------
        [`AnimalBrain.select`][braian.AnimalBrain.select]
        """
        animals = [brain.select(regions,
                                fill_nan=fill_nan,
                                inplace=inplace,
                                hemisphere=hemisphere,
                                select_other_hemisphere=select_other_hemisphere)
                   for brain in self._animals]
        if not inplace:
            # self.metric == animals.metric -> no self.metric.analyse(brain) is computed
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self._animals = animals
            self._hemimean = self._update_mean()
            return self

    def __iter__(self) -> Iterable[AnimalBrain]:
        return iter(self._animals)

    def __len__(self) -> int:
        return self.n

    def __contains__(self, name: str) -> bool:
        if not isinstance(name, str):
            return False
        for brain in self._animals:
            if brain.name == name:
                return True
        return False

    def __getitem__(self, val: str|int) -> AnimalBrain:
        """

        Parameters
        ----------
        val
            The name or the index of a `AnimalBrain` of the group.

        Returns
        -------
        :
            The corresponding `AnimalBrain` in the group.

        Raises
        ------
        TypeError
            If `val` is not a string nor an int.
        IndexError
            If `val` is an int index out of bound.
        KeyError
            If no brain named `val` was found in the group.
        """
        if isinstance(val, int):
            return self._animals[val]
        if not isinstance(val, str):
            raise TypeError("AnimalGroup's animals are identified by strings or int")
        try:
            return next(brain for brain in self._animals if brain.name == val)
        except StopIteration:
            pass
        raise KeyError(f"{val}")

    @deprecated(since="1.1.0",
                params=["hemisphere_distinction", "brain_ontology", "fill_nan"])
    def apply(self, f: Callable[[AnimalBrain], AnimalBrain],
              hemisphere_distinction: bool=True,
              brain_ontology: AtlasOntology=None, fill_nan: bool=False) -> Self:
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
        animals = [f(a) for a in self._animals]
        return AnimalGroup(name=self.name,
                           animals=animals)

    def sort_by_ontology(self, brain_ontology: AtlasOntology,
                         fill_nan: bool=True, inplace: bool=True) -> None:
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
            If `inplace=True` it modifies and returns the same instance.

        See also
        --------
        [`AnimalBrain.merge_hemispheres`][braian.AnimalBrain.merge_hemispheres]
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        """
        animals = [brain.merge_hemispheres() for brain in self._animals]
        if not inplace:
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self._animals = animals
            self._hemimean = self._update_mean()
            return self

    def to_pandas(self, marker: str=None, units: bool=False,
                  missing_as_nan: bool=False,
                  legacy: bool=False,
                  hemisphere_as_value: bool=False,
                  hemisphere_as_str: bool=False) -> pd.DataFrame:
        """
        Constructs a `DataFrame` with data from the current group.

        Parameters
        ----------
        marker
            If specified, it includes data only from the given marker.
        units
            Whether to include the units of measurement in the `DataFrame` index.
            Available only when `marker=None`.
        missing_as_nan
            If True, it converts missing values [`NA`][pandas.NA] as [`NaN`][numpy.nan].
            Note that if the corresponding brain data is integer-based, it converts them to float.
        legacy
            If True, it distinguishes hemispheric data by appending 'Left:' or 'Right:' in front of brain region acronyms.
        hemisphere_as_value
            If True and `legacy=False`, it converts the regions' hemisphere to the corresponding value (i.e. 0, 1 or 2)
        hemisphere_as_str
            If True and `legacy=False`, it converts the regions' hemisphere to the corresponding string (i.e. "both", "left", "right")

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
        if marker is None:
            df = pd.concat(
                {brain.name: brain.to_pandas(marker=None, units=units, missing_as_nan=missing_as_nan,
                                             legacy=legacy, hemisphere_as_value=hemisphere_as_value,
                                             hemisphere_as_str=hemisphere_as_str)
                 for brain in self._animals},
                join="outer", axis=0)
            hemiregions = self.hemiregions
            if legacy:
                if self.is_split:
                    regions_L = list("Left: "+pd.Index(hemiregions[BrainHemisphere.LEFT]))
                    regions_R = list("Right: "+pd.Index(hemiregions[BrainHemisphere.RIGHT]))
                    hemiregions = regions_L+regions_R
                else:
                    hemiregions = self.regions
                index_sorted = pd.MultiIndex.from_product((hemiregions, self.animal_names))
            else:
                index_tupled = [(hemi.name.lower() if hemisphere_as_str else
                                 hemi.value if hemisphere_as_value else
                                 hemi,region,animal)
                                for hemi in hemiregions
                                for region in hemiregions[hemi]
                                for animal in self.animal_names]
                index_sorted = pd.MultiIndex.from_tuples(index_tupled, names=("hemisphere","acronym","animal"))
            df = df.reorder_levels((1,0) if legacy else (1,2,0), axis=0).reindex(index_sorted)
        else:
            df = pd.concat(
                {brain.name: brain.to_pandas(marker=marker, units=False, missing_as_nan=missing_as_nan,
                                             legacy=legacy, hemisphere_as_value=hemisphere_as_value,
                                             hemisphere_as_str=hemisphere_as_str)
                 for brain in self._animals},
                join="outer", axis=1)
            df = df.xs(marker, level=1, axis=1)
        df.columns.name = str(self.metric)
        return df

    def to_csv(self, output_path: Path|str, sep: str=",",
               overwrite: bool=False, legacy: bool=False) -> str:
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
        legacy
            If True, it distinguishes hemispheric data by appending 'Left:' or 'Right:' in front of brain region acronyms.

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
        df = self.to_pandas(units=True, legacy=legacy, hemisphere_as_str=True)
        if legacy:
            labels = (df.columns.name, None)
        if not legacy:
            labels = (df.columns.name, None, None)

        file_name = f"{self.name}_{self.metric}.csv"
        return save_csv(df, output_path, file_name, overwrite=overwrite, sep=sep, index_label=labels)

    @staticmethod
    @deprecated(since="1.1.0",
                params=["group_name"],
                alternatives=dict(animal_name="name"))
    def from_pandas(df: pd.DataFrame,
                    *,
                    name: str,
                    ontology: AtlasOntology,
                    legacy: bool=False,
                    group_name: str=None) -> Self:
        """
        Creates an instance of [`AnimalGroup`][braian.AnimalGroup] from a `DataFrame`.

        Parameters
        ----------
        df
            A [`to_pandas`][braian.AnimalGroup.to_pandas]-compatible `DataFrame`.
        name
            The name of the group associated with the data in `df`.
        ontology
            The ontology of the atlas used to align the brain data in `df`.
        legacy
            If `df` distinguishes hemispheric data by appending 'Left:' or 'Right:' in front of brain region acronyms.
        group_name
            The name of the group associated  with the data in `df`.

        Returns
        -------
        :
            An instance of `AnimalGroup` that corresponds to the data in `df`.

        See also
        --------
        [`to_pandas`][braian.AnimalGroup.to_pandas]
        [`AnimaBrain.from_pandas`][braian.AnimaBrain.from_pandas]
        """
        if group_name is not None:
            name = group_name
        brains_level = 1 if legacy else 2
        animals = [AnimalBrain.from_pandas(
                        df=df.xs(animal_name, axis=0, level=brains_level),
                        name=animal_name, ontology=ontology, legacy=legacy)
                   for animal_name in df.index.unique(brains_level)]
        return AnimalGroup(name, animals, fill_nan=False)

    @staticmethod
    def from_csv(filepath: Path|str,
                 *,
                 name: str,
                 ontology: AtlasOntology,
                 sep: str=",",
                 legacy: bool=False) -> Self:
        """
        Reads a comma-separated values (CSV) file into `AnimalGroup`.

        Parameters
        ----------
        filepath
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        name
            Name of the group associated  with the data in `filepath`.
        ontology
            The ontology of the atlas used to align the brain data in `filepath`.
        sep
            Character or regex pattern to treat as the delimiter.
        legacy
            If the CSV distinguishes hemispheric data by appending 'Left:' or 'Right:' in front of brain region acronyms.

        Returns
        -------
        :
            An instance of `AnimalGroup` that corresponds to the data in the CSV file

        See also
        --------
        [`to_csv`][braian.AnimalGroup.to_csv]
        [`AnimaBrain.from_csv`][braian.AnimaBrain.from_csv]
        [`Experiment.from_brain_csv`][braian.Experiment.from_brain_csv]
        [`Experiment.from_group_csv`][braian.Experiment.from_group_csv]
        """
        df = pd.read_csv(filepath, sep=sep, header=0, index_col=[0,1] if legacy else [0,1,2])
        df.columns.name = df.index.names[0]
        df.index.names = (None,)*(2 if legacy else 3)
        return AnimalGroup.from_pandas(df, name=name, ontology=ontology, legacy=legacy)

    @staticmethod
    def to_prism(marker, brain_ontology: AtlasOntology,
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
        if len(hems:=df.index.unique(level=0)) == 1 and hems[0] is BrainHemisphere.BOTH:
            df.index = df.index.get_level_values(1)
            regions = df.index
        else:
            regions = df.index.get_level_values(1)
        major_divisions = brain_ontology.partitioned(regions.unique(), partition="major divisions", key="acronym")
        df["major_divisions"] = [major_divisions[region] for region in regions]
        df.set_index("major_divisions", append=True, inplace=True)
        return df

class SlicedGroup:
    @staticmethod
    @deprecated(since="1.1.0", params=["exclude_parents"],
                message="Quantifications in ancestor regions are now completely removed too. "+\
                        "If you want the old behaviour, you can momentarily use braian.BrainSlice.exclude with ancestors=False")
    def from_qupath(name: str, brain_names: Iterable[str],
                    qupath_dir: Path|str,
                    brain_ontology: AtlasOntology,
                    ch2marker: dict[str,str],
                    *,
                    exclude_ancestors_layer1: bool,
                    exclude_parents: bool=None,
                    check: bool=False,
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
        exclude_ancestors_layer1
            `ancestors_layer1` from [`BrainSlice.exclude`][braian.BrainSlice.exclude].
        exclude_parents
            `exclude_parents` from [`BrainSlice.exclude`][braian.BrainSlice.exclude].
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
            sliced_brain = SlicedBrain.from_qupath(name=brain_name,
                                                   animal_dir=qupath_dir/brain_name,
                                                   brain_ontology=brain_ontology,
                                                   ch2marker=ch2marker,
                                                   exclude_ancestors_layer1=exclude_ancestors_layer1,
                                                   results_subdir=results_subdir, results_suffix=results_suffix,
                                                   exclusions_subdir=exclusions_subdir, exclusions_suffix=exclusions_suffix)
            sliced_brains.append(sliced_brain)
        return SlicedGroup(name, sliced_brains, brain_ontology)

    @deprecated(since="1.1.0", params=["brain_ontology"])
    def __init__(self,
                 name: str,
                 animals: Iterable[SlicedBrain],
                 brain_ontology: AtlasOntology) -> None:
        """
        Creates an experimental cohort from a set of `SlicedBrain`.\\
        It is meant to help keeping organised raw data coming multiple sections per-animal.

        Parameters
        ----------
        name
            The name of the cohort.
        animals
            The animals part of the group.
        """
        self._name = str(name)
        self._animals = tuple(animals)
        _compatibility_check(self._animals, check_metrics=False, check_hemispheres=False)

    @property
    def name(self) -> str:
        """The name of the sliced group."""
        return self._name

    @property
    def animals(self) -> tuple[SlicedBrain]:
        """The brains making up the current sliced group."""
        return self._animals

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return self._animals[0].is_split

    @property
    def markers(self) -> npt.NDArray[np.str_]:
        """The name of the markers for which the current `SlicedGroup` has data."""
        return np.asarray(self._animals[0].markers)

    @property
    def n(self) -> int:
        """The size of the sliced group."""
        return len(self._animals)

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"SlicedGroup('{self.name}', brains={self.n}, split={self.is_split})"

    def __iter__(self) -> Iterable[AnimalBrain]:
        return iter(self._animals)

    def __len__(self) -> int:
        return self.n

    def __contains__(self, name: str) -> bool:
        if not isinstance(name, str):
            return False
        for brain in self._animals:
            if brain.name == name:
                return True
        return False

    def __getitem__(self, val: str|int) -> SlicedBrain:
        """

        Parameters
        ----------
        val
            The name or the index of a `SlicedBrain` of the sliced group.

        Returns
        -------
        :
            The corresponding `SlicedBrain` in the sliced group.

        Raises
        ------
        TypeError
            If `val` is not a string nor an int.
        IndexError
            If `val` is an int index out of bound.
        KeyError
            If no brain named `val` was found in the sliced group.
        """
        if isinstance(val, int):
            return self._animals[val]
        if not isinstance(val, str):
            raise TypeError("SlicedGroup's animals are identified by strings or int")
        try:
            return next(brain for brain in self._animals if brain.name == val)
        except StopIteration:
            pass
        raise KeyError(f"{val}")

    @property
    def animal_names(self) -> list[str]:
        """
        Returns
        -------
        :
            The names of the animals part of the current sliced group.
        """
        return [brain.name for brain in self._animals]

    def region(self,
               region: str,
               *,
               metric: str,
               hemisphere: BrainHemisphere=BrainHemisphere.BOTH,
               as_density: bool=False) -> pd.DataFrame:
        """
        Extracts all values of a brain region from all [`slices`][braian.SlicedBrain.slices] in the group.
        If the group [is split][braian.SlicedGroup.is_split], the resulting DataFrame
        may contain two values for the same brain slice.

        Parameters
        ----------
        region
            A brain structure identified by its acronym.
        metric
            The metric to extract from the `SlicedGroup`.
            It can either be `"area"` or any value in [`SlicedGroup.markers`][braian.SlicedGroup.markers].
        hemisphere
            The hemisphere of the brain region to extract. If [`BOTH`][braian.BrainHemisphere]
            and the group [is split][braian.SlicedGroup.is_split], it may return both hemispheric values
            of the region.
        as_density
            If `True`, it retrieves the values as densities (i.e. marker/area).

        Returns
        -------
        :
            A `DataFrame` with:

            * a [`pd.MultiIndex`][pandas.MultiIndex] of _brain_, _slice_ and _hemisphere_,
            revealing the brain, the name of the slice and the [hemisphere][braian.BrainHemisphere]
            from which the region data was extracted;

            * a `metric` column, revealing the value for the specified brain region.

            If there is no data for `region`, it returns an empty `DataFrame`.
        """
        region_marker_ = {b.name: b.region(region, hemisphere=hemisphere,
                                              metric=metric, as_density=as_density)
                             for b in self._animals}
        region_marker = {name: data for (name,data) in region_marker_.items() if len(data) != 0}
        if len(region_marker) == 0:
            assert self.n > 0, "SlicedGroups should have at least one animal"
            region_marker = next(iter(region_marker_.values()))
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=("brain", *region_marker.index.names)),
                columns=[metric])
        region_marker = pd.concat(region_marker) # fails if region_marker is empty
        region_marker.index.names = ["brain", *region_marker.index.names[1:]]
        return region_marker

    def apply(self, f: Callable[[SlicedBrain], SlicedBrain]) -> Self:
        """
        Applies a function to each animal of the group and creates a new `SlicedGroup`.

        Parameters
        ----------
        f
            A function that maps an `SlicedBrain` into another `SlicedBrain`.

        Returns
        -------
        :
            A group with the data of each animal changed accordingly to `f`.
        """
        animals = [f(a) for a in self._animals]
        return SlicedGroup(name=self._name, animals=animals)

    @deprecated(since="1.1.0", params=["hemisphere_distinction", "validate"])
    def to_group(self, metric: SlicedMetric,
                 min_slices: int, densities: bool,
                 hemisphere_distinction: bool, validate: bool) -> AnimalGroup:
        """
        Aggregates the data from all sections of each [`SlicedBrain`][braian.SlicedBrain]
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
        [`SlicedBrain.reduce`][braian.SlicedBrain.reduce]
        [`SlicedExperiment.reduce`][braian.SlicedExperiment.reduce]
        """
        return self.reduce(metric=metric, min_slices=min_slices, densities=densities)

    def reduce(self,
               metric: SlicedMetric,
               *,
               min_slices: int=0,
               densities: bool=False) -> AnimalGroup:
        """
        Aggregates the data from all sections of each [`SlicedBrain`][braian.SlicedBrain]
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

        Returns
        -------
        :
            A group with the values from sections of the same animals aggregated.

        See also
        --------
        [`SlicedBrain.reduce`][braian.SlicedBrain.reduce]
        [`SlicedExperiment.reduce`][braian.SlicedExperiment.reduce]
        """
        brains = [b.reduce(metric=metric, min_slices=min_slices, densities=densities) for b in self._animals]
        return AnimalGroup(self._name, brains)