import itertools
import numpy as np
import pandas as pd
from collections.abc import Collection, Iterable, Sequence
from enum import Enum
from typing import Callable, Literal, Self
from numbers import Number

from braian import AtlasOntology, SlicedMetric
from braian._deflector import deflect
from braian.utils import _compatibility_check_bd, classproperty, deprecated

__all__ = [
    "extract_legacy_hemispheres",
    "BrainData",
    "BrainHemisphere",
    "UnknownBrainRegionsError",
    "InvalidRegionsHemisphereError",
]

class UnknownBrainRegionsError(Exception):
    def __init__(self,
                 unknown_regions: Iterable[str],
                 ontology: AtlasOntology):
        super().__init__(self.message(unknown_regions, ontology.name, ))

    @staticmethod
    def message(unknown_structures: Iterable[str], ontology: str) -> str:
        return f"Regions unknown in '{ontology}' atlas: '"+"', '".join(unknown_structures)+"'"

class InvalidRegionsHemisphereError(Exception):
    def __init__(self, context=None):
        context = f"'{context}': " if context is not None else ""
        super().__init__(f"{context}Error occurred extracting the brain hemispheres."+\
        " Some rows are in the form '{Left|Right}: <region acronym>', while others are not.")

class BrainHemisphere(Enum):
    """
    Enum used to tag [brain data][braian.BrainData] as belonging to a certain hemisphere of the brain.

    Attributes
    ----------
    MERGED
        The associated brain region data concerns the whole brain, with no hemisphere distinction.
    LEFT
        The associated brain region data concerns the _left_ hemisphere only.
    RIGHT
        The associated brain region data concerns the _right_ hemisphere only.
    """
    MERGED = 0
    LEFT = 1    # same as brainglobe_atlasapi.core.Atlas.left_hemisphere_value
    RIGHT = 2   # same as brainglobe_atlasapi.core.Atlas.right_hemisphere_value
    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        match value.lower():
            case "merged" | "m":
                return BrainHemisphere.MERGED
            case "left" | "l":
                return BrainHemisphere.LEFT
            case "right" | "r":
                return BrainHemisphere.RIGHT
            case _:
                return None

def extract_legacy_hemispheres(data: pd.DataFrame, reindex: bool=False, inplace: bool=False):
    match_groups = data.index.str.extract(r"((Left|Right): )?(.+)")
    # the above regex extracts 3 groups:
    #  0) '(Left|Right): '
    #  1) '(Left|Right)'
    #  2) '<region_name>'
    match_groups.set_index(data.index, inplace=True)
    # if (unknown_classes:=match_groups[2] != data["Name"]).any():
    #     raise ValueError("Unknown regions: '"+"', '".join(match_groups.index[unknown_classes])+"'")
    if not inplace:
        data = data.copy()
    if match_groups[1].isna().all():
        data["hemisphere"] = BrainHemisphere.MERGED.value
    else:
        try:
            data["hemisphere"] = match_groups[1].map(lambda s: BrainHemisphere(s).value)
        except ValueError:
            raise InvalidRegionsHemisphereError()
    if reindex:
        data.index = pd.MultiIndex.from_arrays((data["hemisphere"],match_groups[2]), names=("hemisphere","acronym"))
        data.drop(columns="hemisphere", inplace=True)
    return data

class BrainData(metaclass=deflect(on_attribute="data", arithmetics=True, container=True)):
    @staticmethod
    def reduce(first: Self, *others: Self,
               op: Callable[[pd.DataFrame],pd.Series]=pd.DataFrame.mean,
               name: str=None, op_name: str=None,
               same_units: bool=True,
               same_hemisphere: bool=True,
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
        same_hemisphere
            Whether it should enforce the same hemisphere for all `BrainData`.
        **kwargs
            Other keyword arguments are passed to `op`.

        Returns
        -------
        :
            A new `BrainData` result of the reduction of all the given data.

        Raises
        ------
        ValueError
            If at least one of the `BrainData` has a different
            [metric][braian.BrainData.metric], [units][braian.BrainData.units] or
            [hemisphere][braian.BrainData.hemisphere] from the others, and
            `same_units` or `same_hemisphere` are `True, respectively.
        """
        # NOTE: skipna=True does not work with FloatingArrays (what BrainData uses)
        #       https://github.com/pandas-dev/pandas/issues/59965
        #       braindata regions with np.nan values resulting from a computation (e.g division by zero)
        #       'corrupt' the whole result into becoming a pd.NA (not a np.nan).
        _compatibility_check_bd(tuple((first, *others)),
                                check_atlas=True,
                                check_metrics=True,
                                check_hemisphere=same_hemisphere)
        if same_units and not all([first.units == other.units for other in others]):
            msg = f"Merging must be done between BrainData of the same units, {[first.units, *[other.units for other in others]]}!"
            raise ValueError(msg)
        hemisphere = first.hemisphere
        if name is None:
            name = ":".join([first.name, *[other.name for other in others]])
        if op_name is None:
            op_name = op.__name__
        # TODO: add min_count attribute to set to NaN those values result of a reduction of less than min_count BrainData
        # join = "inner" if _min_count == 0 else "outer"
        # data = pd.concat([first.data, *[other.data for other in others]], axis=1, join=join)
        data = pd.concat([first.data, *[other.data for other in others]], axis=1)
        redux: pd.Series = op(data, axis=1, **kwargs)
        return BrainData(redux,
                         name=name,
                         metric=f"{first.metric}:{op_name} (n={len(others)+1})",
                         units=first.units,
                         hemisphere=hemisphere,
                         ontology=first.atlas, check=False)

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

    @classproperty
    def RAW_METRIC(cls) -> str:
        """
        Metric used to identify `raw` data that are result of a direct regional quantification.

        It was created to support results of volumetric dataset analyses, such as
        [ClearMap2](https://clearanatomics.github.io/ClearMapDocumentation/).
        """
        return "raw"

    @staticmethod
    def is_raw(metric: str) -> bool:
        """
        Test whether the given string can be associated to a raw metric or not.\\
        Brain data are considered _raw_, if they are a direct—or indirect—quantification within a brain region
        (e.g., cell counts). An _indirect_ raw quantification is the result of a [reduction][braian.reduce]
        across sections.

        Parameters
        ----------
        metric
            A string representing the name of a metric.

        Returns
        -------
        :
            True, if the given string is associated to a raw metric. Otherwise, False.

        See also
        --------
        [`reduce`][braian.reduce]
        [`SlicedMetric.raw`][braian.SlicedMetric.raw]
        """
        try:
            return SlicedMetric(metric).raw
        except ValueError:
            return metric == BrainData.RAW_METRIC

    @deprecated(since="1.1.0",
                params=["brain_ontology", "fill_nan"],
                alternatives=dict(brain_ontology="atlas"))
    def __init__(self, data: pd.Series,
                 *,
                 name: str, metric: str, units: str, hemisphere: BrainHemisphere|str,
                 ontology: AtlasOntology|str, check: bool=True,
                 brain_ontology: AtlasOntology|None=None, fill_nan=False) -> None:
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
            The metric used to extract `data`.

            If `data` is a direct regional quantification (e.g., it's the result of
            [ClearMap2](https://clearanatomics.github.io/ClearMapDocumentation/)),
            use [`RAW_METRIC`][braian.BrainData.RAW_METRIC].
        units
            The units of measurment of the values in `data`.
        hemisphere
            The brain hemisphere to which the `data` is referring to.
        ontology
            The atlas used to align the brain data.
            It is used to check that all regions in `data` exist in the ontology.

            If `check=False`, it can be a string.
        check
            Whether to check or not that all regions in `data` exist in the `ontology`.
        brain_ontology
            The atlas used to align the brain data.
            It is used to check that all regions in `data` exist in the ontology.

            If `check=False`, it can be a string.
        fill_nan
            If ontology is not `None`, it fills with [`NA`][pandas.NA] the value of the regions in `ontology` and missing from `data`.

        See also
        --------
        [`sort_by_ontology`][braian.BrainData.sort_by_ontology]
        """        # convert to nullable type
        if brain_ontology is not None:
            ontology = brain_ontology
        self.data: pd.Series = data.copy().convert_dtypes()
        """The internal representation of the current brain data."""
        self.hemisphere = BrainHemisphere(hemisphere)
        """The brain hemisphere to which the `data` is referring to."""
        self.name: str = str(name) # data_name
        self.metric: str = str(metric)
        if units is not None:
            self._units = str(units)
        else:
            self._units = ""
            print(f"WARNING: {self} has no units")
        if check:
            if not isinstance(ontology, AtlasOntology):
                raise TypeError(type(ontology))
            if (missing:=~ontology.are_regions(self.data.index)).any():
                raise UnknownBrainRegionsError(self.data.index[missing], ontology)
        self._atlas = str(ontology) if isinstance(ontology, str) else ontology.name

    @property
    def name(self) -> str:
        """The name of the current `BrainData`."""
        # if 'data.name' is different, it means that the object was created by the @deflector
        # and it copied the old '_name', but overwrote 'data' (and thus 'data.name' too)
        if self.data.name != self._name:
            self.data.name = self._name
        return self.data.name

    @name.setter
    def name(self, val: str):
        # need a private attribute that gets copied during @deflections
        # it's not actually used
        self._name = val
        self.data.name = self._name

    @property
    def atlas(self) -> str:
        """The name of the atlas used to align the brain data."""
        return str(self._atlas)

    @property
    def metric(self) -> str:
        """The metric of the per-region quantifications."""
        return str(self._metric)

    @metric.setter
    def metric(self, value: str):
        # AnimalGroup.to_csv uses the 'hemisphere' column to store the name and metric.
        # if the name of the metric was set to a value compatible with BrainHemisphere, it could conflict (?)
        try:
            BrainHemisphere(value)
        except ValueError:
            self._metric = value
            return
        raise ValueError(f"Incompatible name for a metric: '{value}'")

    @property
    def units(self) -> str:
        """The units of measurement of the `BrainData`."""
        return self._units

    @property
    def regions(self) -> list[str]:
        """The list of region acronyms for which the current instance records data."""
        return list(self.data.index)

    def __str__(self) -> str:
        return f"BrainData(name='{self.name}', hemisphere={self.hemisphere.name}, metric='{self._metric}', units='{self._units}')"

    def __repr__(self) -> str:
        return str(self)

    @deprecated(since="1.1.0", alternatives=["braian.BrainData.sort"])
    def sort_by_ontology(self, brain_ontology: AtlasOntology,
                         fill_nan: bool=False, inplace: bool=False) -> Self:
        return self.sort(brain_ontology, mode="depth",
                         blacklisted=False, unreferenced=False,
                         fill_nan=fill_nan, inplace=inplace)

    def sort(self, ontology: AtlasOntology,
             *, mode: Literal["depth","width"]="depth",
             blacklisted: bool=True, unreferenced: bool=False,
             fill_nan: bool=False, inplace: bool=False) -> Self:
        """
        Sorts the data in depth-first order or width-first, based on the
        atlas hierarchical ontology.\\
        If `fill_nan=True`, any data whose region is missing from the `ontology` will be removed.

        Parameters
        ----------
        ontology
            The ontology of the atlas to which the data was registered.
        mode
            The mode in which to visit the hierarchy of the atlas ontology, which dictates
            how to sort a linearised list of regions.
        blacklisted
            If `True`, it fills the data with `fill_value` also in correspondance to
            structures that are blacklisted in the ontology.
        unreferenced
            If `True`, it fills the data with `fill_value` also in correspondance to
            structures that have no reference in the atlas annotations.
        fill_nan
            If True, it fills the data with [`NA`][pandas.NA] corresponding
            to the regions missing, but present in `ontology`.
        inplace
            If True, it sorts and returns the instance in place.

        Returns
        -------
        :
            The data sorted according to `ontology`.

            If `inplace=True` it returns the same instance.

        Raises
        ------
        ValueError
            If `ontology` is incompatibile with the atlas to which
            the brain data were registered to.
        KeyError
            If the data contains values for regions that are not found in the ontology,
            either because they are missing or because they are blacklisted or unreferenced
            and `blacklisted=False` or `unreferenced=False`.
        ValueError
            When `mode` has an invalid value.

        See also
        --------
        [`sort`][braian.sort]
        [`AnimalBrain.sort`][braian.AnimalBrain.sort]
        [`AtlasOntology.sort`][braian.AtlasOntology.sort]
        """
        if self._atlas != ontology.name:
            raise ValueError(f"Incompatibile atlas ontology: '{ontology.name}'")
        data = ontology.sort(self.data, mode=mode,
                             blacklisted=blacklisted, unreferenced=unreferenced,
                             fill=fill_nan, fill_value=pd.NA, key="acronym")#\
                # .convert_dtypes() # no need to convert to IntXXArray/FloatXXArray as self.data should already be
        if not inplace:
            return BrainData(data,
                             name=self.name,
                             metric=self._metric, units=self._units,
                             hemisphere=self.hemisphere,
                             ontology=self.atlas, check=False)
        else:
            self.data = data
            return self

    @property
    def root(self) -> float:
        """
        Retrieves the value associated to the whole brain.

        Raises
        ------
        ValueError
            If there is no data for the 'root' brain region.
        """
        acronym = "root"
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
            return BrainData(data,
                             name=self.name,
                             metric=self._metric, units=self._units,
                             hemisphere=self.hemisphere,
                             ontology=self.atlas, check=False)

    def set_regions(self, brain_regions: Collection[str],
                    brain_ontology: AtlasOntology,
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
        brain_regions = np.asarray(brain_regions)
        if isinstance(fill, Collection):
            if len(fill) != len(brain_regions):
                raise ValueError("'fill' argument requires a collection of the same length as 'brain_regions'")
        else:
            assert pd.isna(fill) or isinstance(fill, (int, float, np.number)), "'fill' argument must either be a collection or a number"
            fill = itertools.repeat(fill)
        if (missing:=~brain_ontology.are_regions(brain_regions)).any():
            raise UnknownBrainRegionsError(brain_regions[missing], brain_ontology)
        data = self.data.copy() if not inplace else self.data
        for region,value in zip(brain_regions, fill):
            if not overwrite and region in data.index:
                continue
            data[region] = value
        if not inplace:
            return BrainData(data,
                             name=self.name,
                             metric=self._metric, units=self._units,
                             hemisphere=self.hemisphere,
                             ontology=self.atlas, check=False)
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

    def _select_from_list(self, regions: Sequence[str], *, fill_nan: bool, inplace: bool) -> Self:
        if fill_nan:
            data = self.data.reindex(index=regions, fill_value=pd.NA)
        elif not (unknown_regions:=np.isin(regions, self.data.index)).all(): # TODO: valutare se ignorare o cambiare le docstrings
            unknown_regions = np.array(regions)[~unknown_regions]
            raise ValueError(f"Can't find some regions in {self}: '"+"', '".join(unknown_regions)+"'!")
        else:
            data = self.data[self.data.index.isin(regions)]
        if not inplace:
            return BrainData(data,
                             name=self.name,
                             metric=self._metric, units=self._units,
                             hemisphere=self.hemisphere,
                             ontology=self.atlas, check=False)
        else:
            self.data = data
            return self

    def select_from_list(self, regions: Sequence[str],
                          *,
                          ontology: AtlasOntology,
                          fill_nan: bool=False, inplace: bool=False) -> Self:
        """
        Filters the data from a given list of regions.

        Parameters
        ----------
        regions
            The acronyms of the regions to select from the data.
        ontology
            The ontology of the atlas used to align the brain data.
            It is used to check that all of `regions` exist.
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

        Raises
        ------
        UnknownBrainRegionsError
            If `df` contains regions not present in `ontology`
        """
        if ontology.name != self.atlas:
            raise ValueError(f"Incompatible atlas: data used '{self.atlas}' but '{ontology.name}' was used")
        regions = np.array(regions)
        if (missing:=~ontology.are_regions(regions)).any():
            raise UnknownBrainRegionsError(regions[missing], ontology)
        return self._select_from_list(regions, fill_nan=fill_nan, inplace=inplace)

    def select_from_data(self, other: Self,
                          *,
                          fill_nan: bool=False,
                          inplace: bool=False) -> Self:
        if self.atlas != other.atlas: # makes no sense to use _compatibility_check_bd, every other check would be False
            raise ValueError(f"Incompatible atlas: data used '{self.atlas}' but '{other.name}' was used")
        return self._select_from_list(other.data.index, fill_nan=fill_nan, inplace=inplace)

    def select_from_ontology(self,
                             brain_ontology: AtlasOntology,
                             *,
                             fill_nan: bool=False,
                             inplace: bool=False) -> Self:
        """
        Filters the data from a given ontology, accordingly to a non-overlapping list of regions
        previously selected in `brain_ontology`.\\
        It fails if no selection method was called on the ontology.

        Parameters
        ----------
        brain_ontology
            The ontology to which the current data was registered against.
        fill_nan
            If True, the regions missing from the current data are filled with [`NA`][pandas.NA].
            Otherwise, if the data from some regions are missing, they are ignored.
        inplace
            If True, it applies the filtering to the current instance.

        Returns
        -------
        :
            A brain data filtered accordingly to the given ontology selection.
        See also
        --------
        [`AtlasOntology.selected`][braian.AtlasOntology.selected]
        [`AtlasOntology.unselect_all`][braian.AtlasOntology.unselect_all]
        [`AtlasOntology.add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`AtlasOntology.select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`AtlasOntology.select_leaves`][braian.AtlasOntology.select_leaves]
        [`AtlasOntology.select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        [`AtlasOntology.select_regions`][braian.AtlasOntology.select]
        [`AtlasOntology.get_regions`][braian.AtlasOntology.partition]
        """
        assert brain_ontology.has_selection(), "No selection found in the given ontology."
        selected_allen_regions = brain_ontology.selected()
        if not fill_nan:
            selectable_regions = set(self.data.index).intersection(set(selected_allen_regions))
        else:
            selectable_regions = selected_allen_regions
        return self._select_from_list(list(selectable_regions), fill_nan=fill_nan, inplace=inplace)

    @deprecated(since="1.1.0", alternatives=["braian.BrainData.merge"])
    def merge_hemispheres(self, other: Self) -> Self:
        return self.merge(other)

    def merge(self, other: Self) -> Self:
        """
        Creates a new `BrainData` by merging two hemispheric data into one.

        Parameters
        ----------
        other
            The brain data from the other hemisphere.

        Returns
        -------
        :
            A new [`BrainData`][braian.BrainData] with no hemisphere distinction.

        Raises
        ------
        ValueError
            If at least one the given `BrainData` is already merged.
        ValueError
            If the given `BrainData` metric is not suitable for being merged.
        ValueError
            If the given `BrainData` are not compatible between each other.

        See also
        --------
        [`merge.merge_hemispheres`][braian.merge_hemispheres]
        [`AnimalBrain.merge_hemispheres`][braian.AnimalBrain.merge_hemispheres]
        [`BrainSlice.merge_hemispheres`][braian.BrainSlice.merge_hemispheres]
        [`SlicedBrain.merge_hemispheres`][braian.SlicedBrain.merge_hemispheres]
        """
        _compatibility_check_bd((self, other), min_count=2,
                                check_hemisphere=False)
        if self.hemisphere is BrainHemisphere.MERGED or other.hemisphere is BrainHemisphere.MERGED:
            raise ValueError("Data already have no distinction between right/left hemispheres")
        if self.hemisphere == other.hemisphere:
            raise ValueError(f"Incompatible brain data: '{self}' and '{other}' have the same hemisphere ('{self.hemisphere}', '{other.hemisphere}')")
        if self._metric not in (BrainData.RAW_METRIC, "sum", "count_slices"):
            # NOTE: can't use is_raw(_metric) because not all raw metrics can be summed
            raise ValueError(f"Cannot merge hemispheric '{self._metric}' data")
        if self._units != other._units:
            raise ValueError(f"Incompatible brain data: '{self}' and '{other}' have different units ('{self._units}', '{other._units}')")
        data = self.data.add(other.data, fill_value=0)
        return BrainData(data,
                         name=self.name,
                         metric=self._metric, units=self._units,
                         hemisphere=BrainHemisphere.MERGED,
                         ontology=self.atlas, check=False)