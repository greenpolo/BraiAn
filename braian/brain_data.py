import itertools
import numpy as np
import pandas as pd
import re
from collections.abc import Collection, Iterable, Sequence
from typing import Self, Callable
from numbers import Number

from braian.deflector import deflect
from braian import AllenBrainOntology

__all__ = ["BrainData"]

class UnkownBrainRegionsError(Exception):
    def __init__(self, unknown_regions: Iterable[str]):
        super().__init__(f"The following regions are unknown to the given brain ontology: '"+"', '".join(unknown_regions)+"'")

def extract_acronym(region_class: str) -> str:
    """
    Extracts the region acronym from a QuPath's PathClass assigned by ABBA.
    Example: "Left: AVA" becomes "AVA".

    Parameters
    ----------
    region_class
        The name of QuPath PathClass identifying a brain region with hemisphere distinction.

    Returns
    -------
    :
        The acronym of the corresponding brain region.\\
        If `region_class` has no hemisphere distinction, it returns a copy of `region_class`.
    """
    acronym = re.compile("[Left|Right]: (.+)").findall(region_class)
    if len(acronym) == 0:
        # the region's class didn't distinguish between left|right hemispheres
        return str(region_class)
    return acronym[0]

def _hemisphere_name(hem: str) -> str:
    match hem.lower():
        case "left" | "l":
            return "Left"
        case "right" | "r":
            return "Right"
        case _:
            raise ValueError(f"Unrecognised hemisphere '{hem}'!")

def _is_split_left_right(index: pd.Index) -> bool:
    return (index.str.startswith("Left: ", na=False) | \
            index.str.startswith("Right: ", na=False)).all()

def _split_index(regions: Iterable[str]) -> list[str]:
    """
    Splits between left/right hemisphere the given list of acronyms.
    """
    return [": ".join(t) for t in itertools.product(("Left", "Right"), regions)]

def _sort_by_ontology(data: pd.DataFrame|pd.Series,
                      brain_ontology: AllenBrainOntology,
                      fill=False, fill_value=np.nan) -> pd.DataFrame|pd.Series:
    """Sorts a DataFrame/Series by depth-first search in the given ontology.
    The index of the data is the name of the regions.
    If fill=True, it adds data for the missing regions"""
    all_regions = brain_ontology.list_all_subregions("root", mode="depth")
    if _is_split_left_right(data.index):
        all_regions = _split_index(all_regions)
    if len(unknown_regions:=data.index[~data.index.isin(all_regions)]) > 0:
        raise UnkownBrainRegionsError(unknown_regions)
    if not fill:
        all_regions = np.array(all_regions)
        all_regions = all_regions[np.isin(all_regions, data.index)]
    # NOTE: if fill_value=np.nan -> converts dtype to float
    sorted = data.reindex(all_regions, copy=False, fill_value=fill_value)
    return sorted

class BrainData(metaclass=deflect(on_attribute="data", arithmetics=True, container=True)):
    @staticmethod
    def reduce(first: Self, *others: Self,
              op: Callable[[pd.DataFrame],pd.Series]=pd.DataFrame.mean,
              name=None, op_name=None,
              same_units=True,
              **kwargs) -> Self:
        """
        Reduces two (or more) `BrainData` into a single one based on a given function.\\
        It fails if the given data don't all have the same metric.

        Parameters
        ----------
        first
            The first data to reduce.
        *others
            Any number of additional brain data to reduce.
        op
            A function that maps a `DataFrame` into a `Series`. It must include an `axis` parameter.
        name
            The name of the resulting BrainData.\\
            If not specified, it builds a name joining all given data names.
        op_name
            The name of the reduction function. If not specified, it uses `op` name.
        same_units
            Whether it should enforce the same units of measurement for all `BrainData`.
        **kwargs
            Other keyword arguments are passed to `op`.

        Returns
        -------
        :
            A new `BrainData` result of the reduction of all the given data.
        """
        assert all([first.metric == other.metric for other in others]),\
            f"Merging must be done between BrainData of the same metric, instead got {[first.metric, *[other.metric for other in others]]}!"
        if same_units:
            assert all([first.units == other.units for other in others]),\
                f"Merging must be done between BrainData of the same units, {[first.units, *[other.units for other in others]]}!"
        if name is None:
            name = ":".join([first.data_name, *[other.data_name for other in others]])
        if op_name is None:
            op_name = op.__name__
        data: pd.Series = op(pd.concat([first.data, *[other.data for other in others]], axis=1), axis=1, **kwargs)
        return BrainData(data, name, f"{first.metric}:{op_name} (n={len(others)+1})", first.units)

    @staticmethod
    def mean(*data: Self, **kwargs) -> Self:
        """
        Computes the mean for each brain region between all `data`.

        Parameters
        ----------
        *data
            The `BrainData` to average.
        **kwargs
            Other keyword arguments are passed to [`BrainData.reduce`][braian.BrainData.reduce].

        Returns
        -------
        :
            The mean of all `data`.
        """
        assert len(data) > 0, "You must provide at least one BrainData object."
        return BrainData.reduce(*data, op=pd.DataFrame.mean, same_units=True, **kwargs)

    @staticmethod
    def minimum(*data: Self, **kwargs) -> Self:
        """
        Computes the minimum value for each brain region between all `data`.

        Parameters
        ----------
        *data
            The `BrainData` to search the minimum from.
        **kwargs
            Other keyword arguments are passed to [`BrainData.reduce`][braian.BrainData.reduce].

        Returns
        -------
        :
            The minimum value of all `data`.
        """
        assert len(data) > 0, "You must provide at least one BrainData object."
        return BrainData.reduce(*data, op=pd.DataFrame.min, same_units=False, **kwargs)

    @staticmethod
    def maximum(*data: Self, **kwargs) -> Self:
        """
        Computes the maximum value for each brain region between all `data`.

        Parameters
        ----------
        *data
            The `BrainData` to search the maximum from.
        **kwargs
            Other keyword arguments are passed to [`BrainData.reduce`][braian.BrainData.reduce].

        Returns
        -------
        :
            The maximum value of all `data`.
        """
        assert len(data) > 0, "You must provide at least one BrainData object."
        return BrainData.reduce(*data, op=pd.DataFrame.max, same_units=False, **kwargs)

    RAW_TYPE: str = "raw"
    """The identifier used to specify the nature of raw data as 'metric' attribute in `BrainData`."""

    def __init__(self, data: pd.Series, name: str, metric: str, units: str,
                 brain_ontology: AllenBrainOntology|None=None, fill_nan=False) -> None:
        """
        This class is the base structure for managing any data that associates values to brain regions.\
        You can access its interanal representation through [`BrainData.data`][braian.BrainData.data].

        Parameters
        ----------
        data
            A pandas Series associating brain region acronyms (i.e. the index) to brain data (i.e. the values).
        name
            A name identifying `data`.
        metric
            The metric used to extract `data`. If they no metric was previosuly used, use [`RAW_TYPE`][braian.BrainData.RAW_TYPE].
        units
            The units of measurment of the values in `data`.
        brain_ontology
            The ontology against which the extracted data was aligned.
            It is used to check that all `data` is attributable to a region in the ontology and to sort it accordingly.\\
            If left empty, no check or sorting is performed.
        fill_nan
            If ontology is not `None`, it fills with [`NA`][pandas.NA] the value of the regions in `ontology` and missing from `data`.

        See also
        --------
        [`sort_by_ontology`][braian.BrainData.sort_by_ontology]
        """        # convert to nullable type
        self.data: pd.Series = data.copy().convert_dtypes()
        """The internal representation of the current brain data."""
        self.is_split: bool = _is_split_left_right(self.data.index)
        """Whether the data of the current `BrainData` makes a distinction between right and left hemisphere."""
        self.data_name: str = str(name) # data_name
        """The name of the current `BrainData`."""
        self.data.name = self.data_name
        self.metric: str = str(metric)
        """The name of the metric used to compute the data.
        Equals to [`RAW_TYPE`][braian.BrainData.RAW_TYPE] if no previous normalization was preformed.
        """
        if units is not None:
            self.units = str(units)
            """The units of measurement of the current `BrainData`."""
        else:
            self.units = ""
            print(f"WARNING: {self} has no units")
        if brain_ontology is not None:
            self.sort_by_ontology(brain_ontology, fill_nan, inplace=True)

    @property
    def regions(self) -> list[str]:
        """The list of region acronyms for which the current instance records data."""
        return list(self.data.index)

    def __str__(self) -> str:
        return f"BrainData(name={self.data_name}, metric={self.metric})"

    def sort_by_ontology(self, brain_ontology: AllenBrainOntology,
                         fill_nan=False, inplace=False) -> Self:
        """
        Sorts the data in depth-first search order with respect to `brain_ontology`'s hierarchy.

        Parameters
        ----------
        brain_ontology
            The ontology to which the current data was registered against.
        fill_nan
            If True, it sets the value to [`NaN`][pandas.NA] for all the regions in
            `brain_ontology` missing in the current `AnimalBrain`.
        inplace
            If True, it applies the sorting to the current instance.

        Returns
        -------
        :
            Brain data sorted accordingly to `brain_ontology`.
            If `inplace=True` it returns the same instance.
        """
        data = _sort_by_ontology(self.data, brain_ontology, fill=fill_nan, fill_value=pd.NA)\
                # .convert_dtypes() # no need to convert to IntXXArray/FloatXXArray as self.data should already be
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self

    def root(self, hemisphere: str=None) -> float:
        """
        Retrieves the value associated to the whole brain.

        Parameters
        ----------
        hemisphere
            Anything between "left", "L", "right" and "R".\\
            If the current `BrainData` is split, it defines for which hemisphere to retrieve the data.
            Otherwise, this parameter is ignored.

        Returns
        -------
        :
            The value of the root.

        Raises
        ------
        ValueError
            If `hemisphere` was not specified but the current `BrainData` is split.
        ValueError
            If there is no data for the 'root' brain region.
        """
        acronym = "root"
        if self.is_split:
            if hemisphere is None:
                raise ValueError(f"You have to specify the hemisphere of '{acronym}' you want!")
            acronym = f"{_hemisphere_name(hemisphere)}: {acronym}"
        if acronym not in self.data:
            raise ValueError(f"No data for '{acronym}' in {self}!")
        return self.data[acronym]

    def min(self, skipna: bool=True, skiinf: bool=False) -> float:
        """
        Parameters
        ----------
        skipna
            If True, it does not consider [`NA`][pandas.NA] values.
        skiinf
            If True, it does not consider infinite values.

        Returns
        -------
        :
            The smallest value in the current `BrainData`.
        """
        # TODO: skipna does not work in current pandas
        #       https://github.com/pandas-dev/pandas/issues/59965
        # return self.data[self.data != np.inf].min(skipna=skipna)
        if skipna:
            data = self.data[~np.isnan(self.data)]
        else:
            data = self.data
        if skiinf:
            data = data[data != np.inf]
        return data.min()

    def max(self, skipna: bool=True, skiinf: bool=False) -> float:
        """
        Parameters
        ----------
        skipna
            If True, it does not consider [`NA`][pandas.NA] values.
        skiinf
            If True, it does not consider infinite values.

        Returns
        -------
        :
            The biggest value in the current `BrainData`.
        """
        # TODO: skipna does not work in current pandas
        #       https://github.com/pandas-dev/pandas/issues/59965
        # return self.data[self.data != np.inf].max(skipna=skipna)
        if skipna:
            data = self.data[~np.isnan(self.data)]
        else:
            data = self.data
        if skiinf:
            data = data[data != np.inf]
        return data.max(skipna=skipna)

    def remove_region(self, region: str, *regions: str, inplace: bool=False, fill_nan: bool=False) -> Self:
        """
        Removes one or multiple regions from the current `BrainData`.

        Parameters
        ----------
        region, *regions
            Acronyms of the brain regions to remove.
        inplace
            If True, it removes the region(s) from the current instance.
        fill_nan
            If True, instead of removing the region(s), it sets their value to [`NA`][pandas.NA]

        Returns
        -------
        :
            Brain data with `regions` removed.
            If `inplace=True` it returns the same instance.
        """
        data = self.data.copy() if not inplace else self.data
        regions = [region, *regions]
        if fill_nan:
            data[regions] = pd.NA
        else:
            data = data[~data.index.isin(regions)]
        
        if inplace:
            self.data = data
            return self
        else:
            return braian.BrainData(data, name=self.data_name, metric=self.metric, units=self.units)

    def set_regions(self, brain_regions: Collection[str],
                    brain_ontology: AllenBrainOntology,
                    fill: Number|Collection[Number]=pd.NA,
                    overwrite: bool=False, inplace: bool=False) -> Self:
        """
        Assign a new value to the given `brain_regions`. It checks that each of the given
        brain region exists in the given `brain_ontology`.

        Parameters
        ----------
        brain_regions
            The acronym of the brain regions to set the value for.
        brain_ontology
            The ontology to which the current data was registered against.
        fill
            If a number, it sets the same value for all `brain_regions`.\\
            If a collection the same length as `brain_region`, it sets each brain region to the respective value in `fill`.
        overwrite
            If False, it fails if `brain_regions` contains region acronyms for which a value is already assigned.
        inplace
            If True, it sets the regions for the current instance.

        Returns
        -------
        :
            Brain data with `brain_regions` added.
            If `inplace=True` it returns the same instance.

        Raises
        ------
        ValueError
            if `fill` is a collection of different length than `brain_regions`.
        UnkownBrainRegionsError
            if any of `brain_regions` is missing in `brain_ontology`.
        """
        if isinstance(fill, Collection):
            brain_regions = np.asarray(brain_regions)
            if len(fill) != len(brain_regions):
                raise ValueError("'fill' argument requires a collection of the same length as 'brain_regions'")
        else:
            assert pd.isna(fill) or isinstance(fill, (int, float, np.number)), "'fill' argument must either be a collection or a number"
            fill = itertools.repeat(fill)
        if not all(are_regions := brain_ontology.are_regions(brain_regions, "acronym")):
            unknown_regions = brain_regions[~are_regions]
            raise UnkownBrainRegionsError(unknown_regions)
        data = self.data.copy() if not inplace else self.data
        for region,value in zip(brain_regions, fill):
            if not overwrite and region in data.index:
                continue
            data[region] = value
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self

    def missing_regions(self) -> list[str]:
        """
        Return the acronyms of the brain regions with missing data.

        Returns
        -------
        :
            The acronyms of the brain regions with missing data.
        """
        return list(self.data[self.data.isna()].index)

    def select_from_list(self, brain_regions: Sequence[str], fill_nan=False, inplace=False) -> Self:
        """
        Filters the data from a given list of regions.

        Parameters
        ----------
        brain_regions
            The acronyms of the regions to select from the data.
        fill_nan
            If True, the regions missing from the current data are filled with [`NA`][pandas.NA].
            Otherwise, if the data from some regions are missing, they are ignored.
        inplace
            If True, it applies the filtering to the current instance.

        Returns
        -------
        :
            A brain data filtered accordingly to the given `brain_regions`.
            If `inplace=True` it returns the same instance.
        """
        if fill_nan:
            data = self.data.reindex(index=brain_regions, fill_value=pd.NA)
        elif not (unknown_regions:=np.isin(brain_regions, self.data.index)).all():
            unknown_regions = np.array(brain_regions)[~unknown_regions]
            raise ValueError(f"Can't find some regions in {self}: '"+"', '".join(unknown_regions)+"'!")
        else:
            data = self.data[self.data.index.isin(brain_regions)]
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self

    def select_from_ontology(self, brain_ontology: AllenBrainOntology,
                              *args, **kwargs) -> Self:
        """
        Filters the data from a given ontology, accordingly to a non-overlapping list of regions
        previously selected in `brain_ontology`.\\
        It fails if no selection method was called on the ontology.

        Parameters
        ----------
        brain_ontology
            The ontology to which the current data was registered against.
        *args, **kwargs
            Other arguments are passed to [`select_from_list`][braian.BrainData.select_from_list].

        Returns
        -------
        :
            A brain data filtered accordingly to the given ontology selection.
        See also
        --------
        [`AllenBrainOntology.get_selected_regions`][braian.AllenBrainOntology.get_selected_regions]
        [`AllenBrainOntology.unselect_all`][braian.AllenBrainOntology.unselect_all]
        [`AllenBrainOntology.add_to_selection`][braian.AllenBrainOntology.add_to_selection]
        [`AllenBrainOntology.select_at_depth`][braian.AllenBrainOntology.select_at_depth]
        [`AllenBrainOntology.select_at_structural_level`][braian.AllenBrainOntology.select_at_structural_level]
        [`AllenBrainOntology.select_leaves`][braian.AllenBrainOntology.select_leaves]
        [`AllenBrainOntology.select_summary_structures`][braian.AllenBrainOntology.select_summary_structures]
        [`AllenBrainOntology.select_regions`][braian.AllenBrainOntology.select_regions]
        [`AllenBrainOntology.get_regions`][braian.AllenBrainOntology.get_regions]
        """
        selected_allen_regions = brain_ontology.get_selected_regions()
        selectable_regions = set(self.data.index).intersection(set(selected_allen_regions))
        return self.select_from_list(list(selectable_regions), *args, **kwargs)

    def merge_hemispheres(self) -> Self:
        """
        Creates a new `BrainData` by merging the hemispheric data of the current instance.

        Returns
        -------
        :
            A new [`BrainData`][braian.BrainData] with no hemisphere distinction.
            If the caller is already merged, it return the same instance with no changes.
        """
        if not self.is_split:
            return self
        if self.metric not in (BrainData.RAW_TYPE, "sum", "count_slices"):
            raise ValueError(f"Cannot properly merge '{self.metric}' BrainData from left/right hemispheres into a single region!")
        corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in self.data.index]
        data = self.data.groupby(corresponding_region).sum(min_count=1)
        return BrainData(data, name=self.data_name, metric=self.metric, units=self.units)