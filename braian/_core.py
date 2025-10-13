from braian import AtlasOntology,\
                   BrainSlice, SlicedBrain, SlicedGroup, SlicedExperiment, SlicedMetric,\
                   BrainData, AnimalBrain, AnimalGroup, Experiment
from typing import Literal

__all__ = ["reduce", "merge_hemispheres"]

def reduce(d: SlicedBrain|SlicedGroup|SlicedExperiment,
           metric: SlicedMetric,
           min_slices: int=1,
           densities: bool=False) -> AnimalBrain|AnimalGroup|Experiment:
    """
    Aggregates the data from all [`SlicedBrain`][braian.SlicedBrain] into the
    corresponding [`AnimalBrain`][braian.AnimalBrain].

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
    : AnimalBrain
        If `d` is a `SlicedBrain`
    : AnimalGroup
        If `d` is a `SlicedGroup`
    : Experiment
        If `d` is a `SlicedExperiment`

    See also
    --------
    [`SlicedBrain.reduce`][braian.SlicedBrain.reduce]
    [`SlicedGroup.reduce`][braian.SlicedGroup.reduce]
    [`SlicedExperiment.reduce`][braian.SlicedExperiment.reduce]
    """
    if isinstance(d, (SlicedBrain, SlicedGroup, SlicedExperiment)):
        return d.reduce(metric=metric,
                        min_slices=min_slices,
                        densities=densities)
    raise TypeError(type(d))

def sort(d: BrainData|AnimalBrain|AnimalGroup|Experiment, ontology: AtlasOntology,
         *, mode: Literal["depth","width"]="depth",
         blacklisted: bool=True, unreferenced: bool=False,
         fill_nan: bool=False, inplace: bool=False) -> BrainData|AnimalBrain|AnimalGroup|Experiment:
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
    [`BrainData.sort`][braian.BrainData.sort]
    [`AnimalBrain.sort`][braian.AnimalBrain.sort]
    [`AtlasOntology.sort`][braian.AtlasOntology.sort]
    """
    if isinstance(d, (BrainData, AnimalBrain)):
        return d.sort(ontology=ontology, mode=mode,
                      blacklisted=blacklisted, unreferenced=unreferenced,
                      fill_nan=fill_nan, inplace=inplace)
    elif inplace: # d: AnimalGroup | Experiment
        for d_ in d:
            sort(d_, ontology, mode=mode,
                 blacklisted=blacklisted, unreferenced=unreferenced,
                 fill_nan=fill_nan, inplace=True)
        # if it's just a sorting, it should not need any update on the side of AnimalGroup and Experiment
        # what about AnimalGroup.hemimeans, tho?
        return d
    else:
        return d.apply(lambda b: b.sort(ontology, mode=mode,
                                        blacklisted=blacklisted, unreferenced=unreferenced,
                                        fill_nan=fill_nan, inplace=inplace))

# def merge(d):
def merge_hemispheres(d: BrainSlice|SlicedBrain|SlicedGroup|SlicedExperiment|\
                        AnimalBrain|AnimalGroup|Experiment):
    """
    Merges data from left and right hemisheres into a single value, by sum.

    Returns
    -------
    :
        The regional data from `d` with no distinction between hemispheres.

    Raises
    ------
    ValueError
        If `d` is not [split][braian.AnimalBrain.is_split] between right and left hemispheres.

    See also
    --------
    [`BrainData.merge`][braian.BrainData.merge]
    [`AnimalBrain.merge`][braian.AnimalBrain.merge_hemispheres]
    [`BrainSlice.merge`][braian.BrainSlice.merge_hemispheres]
    [`SlicedBrain.merge`][braian.SlicedBrain.merge_hemispheres]
    """
    if isinstance(d, (SlicedBrain, AnimalBrain, BrainSlice)):
        return d.merge_hemispheres()
    elif isinstance(d, (SlicedGroup, AnimalGroup, SlicedExperiment, Experiment)):
        return d.apply(merge_hemispheres)
    raise TypeError(type(d))