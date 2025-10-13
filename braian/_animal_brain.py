import copy
import functools
import pandas as pd
import re

from collections.abc import Sequence, Callable
from pathlib import Path
from typing import Generator, Self

from braian import AtlasOntology, BrainData, BrainHemisphere, SlicedMetric, UnknownBrainRegionsError
from braian._brain_data import extract_legacy_hemispheres #, sort_by_ontology
from braian.utils import _compatibility_check_bd, deprecated, merge_ordered, save_csv

__all__ = ["AnimalBrain"]

def _combined_regions(*bd: BrainData) -> list[str]:
    return {
        hem: merge_ordered(*[_bd for _bd in bd if _bd.hemisphere is hem])
        for hem in BrainHemisphere
    }

def _to_legacy_index(bd: BrainData) -> pd.Series:
    if bd.hemisphere is not BrainHemisphere.BOTH:
        return (bd.hemisphere.name.capitalize()+": ")+bd.data.index
    return bd.data.index

class AnimalBrain:
    @staticmethod
    @deprecated(since="1.1.0", alternatives=["braian.SlicedBrain.reduce"])
    def from_slices(sliced_brain,   #: SlicedBrain,
                    metric: SlicedMetric|str=SlicedMetric.SUM,
                    min_slices: int=0,
                    hemisphere_distinction: bool=True,
                    densities: bool=False) -> Self:
        """
        Crates a cohesive [`AnimalBrain`][braian.AnimalBrain] from data coming from brain sections.

        Parameters
        ----------
        sliced_brain
            A sectioned brain.
        metric
            The metric used to reduce sections data from the same region into a single value.
        min_slices
            The minimum number of sections for a reduction to be valid.
            If a region has not enough sections, it will disappear from the dataset.
        hemisphere_distinction
            if False and `sliced_brain` is split between left/right hemispheres,
            it first merges, for each section, the hemispheric data.
        densities
            If True, it computes the reduction on the section density (i.e., marker/area)
            instead of doing it on the raw cell counts.

        Returns
        -------
        :
            An `AnimalBrain`.

        Raises
        ------
        EmptyBrainError
            when `sliced_brain` has not enough sections or when `min_slices` filters out all brain regions.
        """
        b = sliced_brain.reduce(metric=metric,
                                min_slices=min_slices,
                                densities=densities)
        if not hemisphere_distinction:
            return b.merge_hemispheres()
        return b

    @deprecated(since="1.1.0", params=["raw"])
    def __init__(self,
                 markers_data: dict[str,BrainData|tuple[BrainData,BrainData]],
                 sizes: BrainData|tuple[BrainData,BrainData],
                 raw: bool=False) -> None:
        """
        Associates [`BrainData`][braian.BrainData] coming from a single subject,
        for each marker and for each brain region.

        Parameters
        ----------
        markers_data
            A dictionary that associates the name of a marker to a `BrainData`
        sizes
            A `BrainData` with the size of the subject's brain regions.\
            A tuple of two `BrainData` is required if the associated data is split between right and left [hemispheres][braian.BrainHemisphere].
        raw
            Whether the data can be considered _raw_ (e.g., contains simple cell positive counts) or not.
        """
        assert len(markers_data) > 0 and sizes is not None, "You must provide both a dictionary of BrainData (markers) and an additional BrainData for the size of each region"
        markers_data = {m: (ds,) if isinstance(ds, BrainData) else tuple(sorted(ds, key=lambda d: d.hemisphere.value))
                        for m,ds in markers_data.items()}
        are_merged = [m.hemisphere is BrainHemisphere.BOTH for ms in markers_data.values() for m in ms]
        assert all(are_merged) or not any(are_merged), "You must provide BrainData that has merged hemispheres for all markers or for none."
        if all(are_merged):
            if isinstance(sizes, BrainData):
                sizes = (sizes,)
            assert (isinstance(sizes, tuple) and len(sizes) == 1 and\
                all(map(lambda x: type(x) is BrainData, sizes))), "'sizes' shoud be a BrainData object."
        else:
            assert isinstance(sizes, tuple) and len(sizes) == 2 and\
                 all(map(lambda x: type(x) is BrainData, sizes)), "'sizes' shoud be a tuple of one or two BrainData objects."
        self._markers: tuple[str] = tuple(markers_data.keys())
        self._markers_data: dict[str,tuple[BrainData]|tuple[BrainData,BrainData]] = markers_data
        self._sizes: tuple[BrainData]|tuple[BrainData,BrainData] = sizes
        _compatibility_check_bd(self._sizes, check_atlas=False,
                                check_metrics=True, check_hemisphere=False, check_regions=False)
        for i,hemisizes in enumerate(self._sizes):
            # compatibility between hemidata
            hemidata = [md[i] if isinstance(md, tuple) else md
                       for md in self._markers_data.values()]
            #check sizes and metrics have the same regions and hemispheres
            _compatibility_check_bd([hemisizes, *hemidata], check_atlas=True,
                                    check_metrics=False, check_regions=True)
            # check markers have the same metric
            _compatibility_check_bd(hemidata, check_atlas=False,
                                    check_metrics=True, check_regions=False)
        assert all([m.data_name == self.name for ms in markers_data.values() for m in ms]), "All markers' BrainData must be from the same animal!"
        assert all([m.metric == self.metric for ms in markers_data.values() for m in ms]), "All markers' BrainData must have the same metric!"
        return

    @property
    def atlas(self) -> str:
        """The name of the atlas used to align the brain data."""
        return self._sizes[0].atlas

    @property
    def hemispheres(self) -> tuple[BrainHemisphere]|tuple[BrainHemisphere,BrainHemisphere]:
        """The hemispheres for which the `AnimalBrain` has data."""
        return tuple(s.hemisphere for s in self._sizes)

    @property
    def sizes(self) -> BrainData:
        """
        The data corresponding to the size of each brain region of an `AnimalBrain`.
        Only available if the brain is not split between left and right hemispheres.
        """
        if self.is_split:
            raise ValueError("Cannot get a single size for all brain regions because the brain is split between left and right hemispheres."+\
                             "Use AnimalBrain.hemisizes.")
        return self._sizes[0]

    @property
    def hemisizes(self) -> tuple[BrainData,BrainData]|tuple[BrainData]:
        """
        The data corresponding to the size of each brain region of an `AnimalBrain`.
        If the AnimalBrain has data for two hemispheres, it is a tuple of size two.
        Else, it is a tuple of size one.
        """
        return self._sizes

    @property
    @deprecated(since="1.0.3", alternatives=["braian.AnimalBrain.sizes"])
    def areas(self) -> BrainData:
        return self.sizes

    @property
    def markers(self) -> list[str]:
        """The name of the markers for which an `AnimalBrain` has data."""
        return list(self._markers)

    @property
    def metric(self) -> str:
        """The metric of the per-region quantifications."""
        return self._markers_data[self._markers[0]][0].metric

    @property
    def raw(self) -> bool:
        """
        Whether the data can be considered raw or not.\\
        Brain data are considered _raw_, if they are a direct—or indirect—quantification within a brain region
        (e.g., cell counts). An _indirect_ raw quantification is the result of a [reduction][braian.reduce]
        across sections.

        See also
        --------
        [`reduce`][braian.reduce]
        [`SlicedMetric.raw`][braian.SlicedMetric.raw]
        """
        return BrainData.is_raw(self.metric)

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return len(self._sizes) == 2

    @property
    def name(self) -> str:
        """The name of the animal."""
        return self._sizes[0].data_name
        # return self._markers_data[self._markers[0]][0].data_name

    @property
    def regions(self) -> list[str]:
        """
        The list of region acronyms for which an `AnimalBrain` has data.\\
        Only available if the brain is not split between left and right hemispheres.
        """
        # assumes sizes' and all markers' BrainData are synchronized
        if self.is_split:
            raise ValueError("Cannot get a single list for all brain regions because the brain is split between left and right hemispheres."+\
                             "Use AnimalBrain.hemiregions.")
        return self._sizes[0].regions

    @property
    def hemiregions(self) -> dict[BrainHemisphere,list[str]]:
        """
        The brain regions corresponding to each possible [`BrainHemisphere`][braian.BrainHemisphere].
        """
        return {s.hemisphere: s.regions for s in self._sizes}

    @deprecated(since="1.1.0", alternatives=["braian.AnimalBrain.units"])
    def get_units(self, marker:str|None=None) -> str:
        return self.units(marker)

    def units(self, marker:str|None=None) -> str:
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
        if len(self._markers) == 1:
            marker = self._markers[0]
        assert marker is not None, "Missing marker to get units of."
        assert marker in self._markers, f"Could not get units of marker '{marker}'!"
        return self._markers_data[marker][0].units

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AnimalBrain(name='{self.name}', metric={self.metric}, is_split={self.is_split})"

    def __getitem__(self, key) -> BrainData:
        """
        Get the [`BrainData`][braian.BrainData] associated to `marker`.
        Fails if there is no data for the the given marker

        Parameters
        ----------
        key
            The marker to extract the data for.

        Returns
        -------
        :
            The data associated to maker `key`.
        """
        if isinstance(key,str):
            if self.is_split:
                raise ValueError("Cannot get a single marker data for all brain regions because the brain is split between left and right hemispheres."+\
                                 "You should also specify a braian.BrainHemisphere.")
            return self._markers_data[key][0] # there is no hemisphere distinction
        elif isinstance(key,tuple) and len(key) == 2 and isinstance(key[0],str):
            marker,hemi = key
            hemi = BrainHemisphere(hemi)
            if not self.is_split and hemi is not BrainHemisphere.BOTH:
                ValueError(f"The brain has no hemisphere distinction. You cannot select data for '{hemi.name}' hemisphere.")
            try:
                return next(d for d in self._markers_data[marker] if d.hemisphere is hemi)
            except StopIteration | KeyError:
                pass
            msg = f"Cannot find '{marker}' data"
            if self.is_split:
                msg += f" for {hemi.name} hemisphere"
            raise KeyError(msg+".")
        raise TypeError(f"Unknown marker data selection: '{key}'")

    def remove_region(self, region: str, *regions, fill_nan: bool=True,
                      hemisphere: BrainHemisphere=BrainHemisphere.BOTH) -> None:
        """
        Removes the data from all the given regions

        Parameters
        ----------
        region
            The acronyms of the regions to exclude from the data.
        fill_nan
            If True, instead of removing the regions completely, it fills their value to [`NA`][pandas.NA].
        hemisphere
            If `BOTH`, it completely removes all `regions` from the brain.\
            Else, it remove only the `regions` from the specified hemisphere.
        """
        regions = (region, *regions)
        hemisphere = BrainHemisphere(hemisphere)
        for hemidata in self._markers_data.values():
            for data in hemidata:
                if data.hemisphere is not hemisphere:
                    continue
                data.remove_region(*regions, inplace=True, fill_nan=fill_nan)
        for sizes in self._sizes:
            if sizes.hemisphere is not hemisphere:
                continue
            sizes.remove_region(*regions, inplace=True, fill_nan=fill_nan)

    def remove_missing(self, kind: str="same") -> None:
        """
        Removes the regions for which there is no data about the size.

        Parameters
        ----------
        kind
            Choice to select which missing brain regions to remove:
                * `'same'`: removes only from one hemisphere the brain missing regions missing from that same hemisphere;
                * `'both'`: removes only the brain regions missing _from both_ hemispheres;
                * `'any'`:  removes from all hemispheres the brain regions missing from _any_ of the two hemispheres.

            If the brain has no hemisphere distinction, the result will be the same.
        """
        match kind:
            case "same":
                for sizes in self._sizes:
                    missing_regions = sizes.missing_regions()
                    if len(missing_regions) > 0:
                        self.remove_region(*missing_regions, fill_nan=False, hemisphere=sizes.hemisphere)
            case "both":
                missing_regions = set()
                for sizes in self._sizes:
                    missing_regions &= set(sizes.missing_regions())
                if len(missing_regions) > 0:
                    self.remove_region(*missing_regions, fill_nan=False, hemisphere=BrainHemisphere.BOTH)
            case "any":
                missing_regions = set()
                for sizes in self._sizes:
                    missing_regions |= set(sizes.missing_regions())
                if len(missing_regions) > 0:
                    self.remove_region(*missing_regions, fill_nan=False, hemisphere=BrainHemisphere.BOTH)
            case _:
                raise ValueError(f"Unknown kind '{kind}'.")

    def sort_by_ontology(self, brain_ontology: AtlasOntology,
                         fill_nan: bool=False, inplace: bool=False) -> Self:
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
        # TODO: should we add an option to sync hemiregions? And markers?

        # NOTE: this adds the regions present in only one of the two hemispheres
        # if self.is_split:
        #     combined_regions = _combined_regions(self[self._markers[0], 1], self[self._markers[0], 2])
        #     combined_regions = sort_by_ontology(combined_regions, atlas_ontology, fill=False)
        #     combined_hemidata = {m: tuple(hemidata.select_from_list(combined_regions, fill_nan=True, inplace=False) for hemidata in data)
        #                         for m,data in self._markers_data.items()}
        #     combined_hemisizes = tuple(hemisize.select_from_list(combined_regions, fill_nan=True, inplace=False) for hemisize in self.hemisizes)
        markers_data = {marker: tuple(m_data.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=inplace) for m_data in hemidata)
                        for marker, hemidata in self._markers_data.items()}
        sizes = tuple(s.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=inplace) for s in self._sizes)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, sizes=sizes)
        else:
            self._markers_data = markers_data
            self._sizes = sizes
            return self

    def _apply(self,
               hemidata: tuple[BrainData,BrainData],
               f1: Callable[[BrainData],Callable],
               f2: Callable[[BrainData],Callable],
               hemisphere: BrainHemisphere):
        return tuple(f1(data) if data.hemisphere is hemisphere else f2(data)
                     for data in hemidata)

    @deprecated(since="1.1.0", alternatives=["braian.AnimalBrain.select"])
    def select_from_list(self, regions: Sequence[str], *, ontology: AtlasOntology, **kwargs):
        return self.select(ontology, regions=regions, **kwargs)

    @deprecated(since="1.1.0", alternatives=["braian.AnimalBrain.select"])
    def select_from_ontology(self, brain_ontology: AtlasOntology, **kwargs):
        return self.select(brain_ontology, **kwargs)

    def _select(self, regions: Sequence[str],
                *,
                fill_nan: bool=False, inplace: bool=False,
                hemisphere: BrainHemisphere=BrainHemisphere.BOTH,
                select_other_hemisphere: bool=False,
                select_list: Callable[...,BrainData]=BrainData.select_from_list,
                **kwargs) -> Self:
        hemisphere = BrainHemisphere(hemisphere)
        if not self.is_split and hemisphere is not BrainHemisphere.BOTH:
            raise ValueError("You cannot select only one hemisphere because the brain data is merged between left and right hemispheres.")
        if regions is None:
            def f1(bd: BrainData):
                return bd.select_from_ontology(regions, fill_nan=fill_nan, inplace=inplace)
        else:
            def f1(bd: BrainData):
                return select_list(bd, regions, fill_nan=fill_nan, inplace=inplace, **kwargs)

        if hemisphere is BrainHemisphere.BOTH:
            f2 = f1
        elif select_other_hemisphere:
            def f2(bd: BrainData):
                return bd
        else:
            def f2(bd: BrainData):
                return bd._select_from_list([], fill_nan=fill_nan, inplace=inplace)

        markers_data = {marker: self._apply(hemidata, f1, f2, hemisphere=hemisphere)
                        for marker, hemidata in self._markers_data.items()}
        sizes = self._apply(self._sizes, f1, f2, hemisphere=hemisphere)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, sizes=sizes)
        else:
            self._markers_data = markers_data
            self._sizes = sizes
            return self

    def select(self, ontology: AtlasOntology,
               *,
               regions: Sequence[str]=None,
               fill_nan: bool=False, inplace: bool=False,
               hemisphere: BrainHemisphere=BrainHemisphere.BOTH,
               select_other_hemisphere: bool=False) -> Self:
        """
        Filters the data from based on a non-overlapping list of regions selected
        in the [`ontology`][braian.AtlasOntology].\\
        If the ontology has no [active selection][braian.AtlasOntology.has_selection], it fails.

        Alternatively, if `regions` is not `None`, it filters the data on the provided `regions`.
        the data accordingly to a non-overlapping list of regions previously selected.

        Parameters
        ----------
        ontology
            An ontology with an [active selection][braian.AtlasOntology.has_selection].

            If `regions` is provided too, it is used only to check that `regions` exist in the `ontology`
        regions
            The acronyms of the regions to select in the data.
        fill_nan
            If True, the regions missing from the current data are filled with [`NA`][pandas.NA].
            Otherwise, if the data from some regions are missing, they are not added.
        inplace
            If True, it applies the filtering to the current instance.
        hemisphere
            If not [`BOTH`][braian.BrainHemisphere] and the brain [is split][braian.AnimalBrain.is_split],
            it only selects the brain regions from the given hemisphere.
        select_other_hemisphere
            If True and `hemisphere` is not [`BOTH`][braian.BrainHemisphere], it also selects the opposite hemisphere.\
            If False, it deselect the opposite hemisphere.

        Returns
        -------
        :
            A brain with data filtered accordingly to the given `regions`.
            If `inplace=True` it returns the same instance.

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
        return self._select(ontology=ontology, regions=regions, fill_nan=fill_nan, inplace=inplace,
                            hemisphere=hemisphere, select_other_hemisphere=select_other_hemisphere,
                            select_list=BrainData.select_from_list)

    def merge_hemispheres(self) -> Self:
        """
        Creates a new `AnimalBrain` from the current instance with no hemisphere distinction.

        Returns
        -------
        :
            A new [`AnimalBrain`][braian.AnimalBrain] with no hemisphere distinction.
            If it is already merged, it return the same instance with no changes.

        See also
        --------
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        """
        if not self.is_split:
            raise ValueError("Data already have no distinction between right/left hemispheres")
        sizes = (self._sizes[0].merge(self._sizes[1]),)
        markers_data = {m: (hemidata1.merge(hemidata2),) for m, (hemidata1,hemidata2) in self._markers_data.items()}
        return AnimalBrain(markers_data=markers_data, sizes=sizes)

    def to_pandas(self, marker: str=None, units: bool=False,
                  missing_as_nan: bool=False,
                  legacy: bool=False,
                  hemisphere_as_value: bool=False,
                  hemisphere_as_str: bool=False) -> pd.DataFrame:
        """
        Converts the current `AnimalBrain` to a DataFrame. T

        Parameters
        ----------
        marker
            If specified, it includes data only from the given marker.
        units
            Whether the columns should include the units of measurement or not.
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
            A DataFrame where the rows are the brain regions, the first column is the size
            of the regions, while the other columns contains the data for each marker.
            The columns' name is the name of the metric used.

        See also
        --------
        [`from_pandas`][braian.AnimalBrain.from_pandas]
        """
        brain_dict = dict()
        hemisizes = sorted(self._sizes, key=lambda bd: bd.hemisphere.value) # first LEFT and then RIGHT hemispheres
        size_col = f"size ({self._sizes[0].units})" if units else "size"
        if legacy:
            index = functools.reduce(lambda i1,i2: i1.union(i2, sort=False), map(_to_legacy_index, self._sizes))
        else:
            hemiregions = self.hemiregions
            hemiregions = [(hemi.name.lower() if hemisphere_as_str else
                            hemi.value if hemisphere_as_value else
                            hemi,region)
                           for hemi in hemiregions
                           for region in hemiregions[hemi]]
            index = pd.MultiIndex.from_tuples(hemiregions, names=("hemisphere","acronym"))
        hemisizes = pd.concat((bd.data for bd in hemisizes))
        hemisizes.index = index
        brain_dict[size_col] = hemisizes
        if marker is not None:
            assert marker in self.markers, f"No data data available for marker '{marker}'."
            markers_data = ((marker, self._markers_data[marker]),)
        else:
            markers_data = self._markers_data.items()
        for marker, hemidata in markers_data:
            hemidata = sorted(hemidata, key=lambda bd: bd.hemisphere.value) # first LEFT and then RIGHT hemispheres
            marker_col = f"{marker} ({hemidata[0].units})" if units else marker
            hemidata = pd.concat((bd.data for bd in hemidata))
            hemidata.index = index # NOTE: exploits the assumtion that the markers regions are sync'd with the sizes'.
            brain_dict[marker_col] = hemidata
        data = pd.concat(brain_dict, axis=1)
        data.columns.name = str(self.metric)
        if missing_as_nan:
            data = data.astype(float)
        return data

    def to_csv(self, output_path: Path|str, sep: str=",",
               overwrite: bool=False, legacy: bool=False) -> str:
        """
        Write the current `AnimalBrain` to a comma-separated values (CSV) file in `output_path`.

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
        [`from_csv`][braian.AnimalBrain.from_csv]
        """
        df = self.to_pandas(units=True, legacy=legacy)
        if not legacy:
            df.index = df.index.map(lambda i: (i[0].name.lower(), *i[1:]))
        file_name = f"{self.name}_{self.metric}.csv"
        index_label = df.columns.name if legacy else (df.columns.name, None)
        return save_csv(df, output_path, file_name, overwrite=overwrite, sep=sep, index_label=index_label)

    @staticmethod
    @deprecated(since="1.1.0", alternatives=["braian.BrainData.is_raw"])
    def is_raw(metric: str) -> bool:
        """
        Test whether the given string can be associated to a raw metric or not.

        Parameters
        ----------
        metric
            A string representing the name of a metric.

        Returns
        -------
        :
            True, if the given string is associated to a raw metric. Otherwise, False.
        """
        return BrainData.is_raw(metric)

    @staticmethod
    @deprecated(since="1.1.0",
                params=["animal_name"],
                alternatives=dict(animal_name="name"))
    def from_pandas(df: pd.DataFrame,
                    *,
                    name: str,
                    ontology: AtlasOntology,
                    legacy: bool=False,
                    animal_name: str=None) -> Self:
        """
        Creates an instance of [`AnimalBrain`][braian.AnimalBrain] from a `DataFrame`.

        Parameters
        ----------
        df
            A [`to_pandas`][braian.AnimalBrain.to_pandas]-compatible `DataFrame`.
        name
            The name of the animal associated  with the data in `df`.
        ontology
            The ontology of the atlas used to align the brain data in `df`.
        legacy
            If `df` distinguishes hemispheric data by appending 'Left:' or 'Right:' in front of brain region acronyms.
        animal_name
            The name of the animal associated  with the data in `df`.

        Returns
        -------
        :
            An instance of `AnimalBrain` that corresponds to the data in `df`.

        Raises
        ------
        UnknownBrainRegionsError
            If `df` contains regions not present in `ontology`.

        See also
        --------
        [`to_pandas`][braian.AnimalBrain.to_pandas]
        [`AnimalGroup.from_pandas`][braian.AnimalGroup.from_pandas]
        """
        if animal_name is not None:
            name = animal_name
        if legacy:
            assert not isinstance(df.index, pd.MultiIndex), \
                "Legacy dataframes are expected to have a simple Index of the acronyms and the hemishere prepended. "+\
                f"Instead got a '{type(df.index)}'"
            df = extract_legacy_hemispheres(df, reindex=True, inplace=False)
        else:
            assert isinstance(df.index, pd.MultiIndex), \
                "The dataframe is expected to have a two-level MultiIndex (hemisphere,acronym). "+\
                f"Instead, got a '{type(df.index)}'"
            df = df.copy()
        if isinstance(metric:=df.columns.name, str):
            metric = str(df.columns.name)
        markers_data = dict()
        sizes = None
        df.index = df.index.map(lambda i: (BrainHemisphere(i[0]),i[1]))
        hemispheres = sorted(df.index.unique(level=0), key=lambda hem: hem.value)
        regions = df.index.unique(level=1)
        if (missing:=~ontology.are_regions(regions)).any():
            raise UnknownBrainRegionsError(regions[missing], ontology)
        regex = r'(.+) \((.+)\)$'
        pattern = re.compile(regex)
        for column, data in df.items():
            # extracts column name and units from the column's name. E.g. 'size (mm²)' -> ('size', 'mm²')
            matches = re.findall(pattern, column)
            column, units = matches[0] if len(matches) == 1 else (column, None)
            if column == "area" or column == "size": # braian <= 1.0.3 called sizes "area"
                sizes = tuple(
                    BrainData(data[hem], name=name,
                              metric=metric, units=units, hemisphere=hem,
                              ontology=ontology.name, check=False) # no need to check because we did it earlier for all the DataFrame
                    for hem in hemispheres)
            else: # it's a marker
                markers_data[column] = tuple(
                    BrainData(data[hem], name=name,
                              metric=metric, units=units, hemisphere=hem,
                              ontology=ontology.name, check=False)
                    for hem in hemispheres)
        return AnimalBrain(markers_data=markers_data, sizes=sizes)

    @staticmethod
    def from_csv(filepath: Path|str,
                 *,
                 name: str,
                 ontology: AtlasOntology,
                 sep: str=",",
                 legacy: bool=False) -> Self:
        """
        Reads a comma-separated values (CSV) file into `AnimalBrain`.

        Parameters
        ----------
        filepath
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        name
            Name of the animal associated  with the data in `filepath`.
        ontology
            The ontology of the atlas used to align the brain data in `filepath`.
        sep
            Character or regex pattern to treat as the delimiter.
        legacy
            If the CSV distinguishes hemispheric data by appending 'Left:' or 'Right:' in front of brain region acronyms.

        Returns
        -------
        :
            An instance of `AnimalBrain` that corresponds to the data in the CSV file.

        Raises
        ------
        UnknownBrainRegionsError
            If the data in `filepath` contains regions not present in `ontology`.

        See also
        --------
        [`to_csv`][braian.AnimalBrain.to_csv]
        [`AnimalGroup.from_csv`][braian.AnimalGroup.from_csv]
        [`Experiment.from_brain_csv`][braian.Experiment.from_brain_csv]
        [`Experiment.from_group_csv`][braian.Experiment.from_group_csv]
        """
        # read CSV
        # filename = f"{name}.csv" if metric is None else f"{name}_{str(metric)}.csv"
        df = pd.read_csv(filepath, sep=sep, header=0, index_col=0 if legacy else [0,1])
        if df.index.name == "Class":
            # is old csv
            raise ValueError("Trying to read an AnimalBrain from an outdated formatted .csv. Please re-run the analysis from the SlicedBrain!")
        df.columns.name = df.index.names[0] # good whether it's legacy or not
        df.index.names = (None,)*(1 if legacy else 2)
        return AnimalBrain.from_pandas(df, name, ontology=ontology, legacy=legacy)

def _extract_name_and_units(ls) -> Generator[str, None, None]:
    regex = r'(.+) \((.+)\)$'
    pattern = re.compile(regex)
    for s in ls:
        matches = re.findall(pattern, s)
        assert len(matches) == 1, f"Cannot find units in column '{s}'"
        yield matches[0]