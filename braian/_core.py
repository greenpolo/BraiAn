from braian import BrainSlice, SlicedBrain, SlicedGroup, SlicedExperiment, SlicedMetric,\
                    AnimalBrain, AnimalGroup, Experiment

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
    hemisphere_distinction
        If False, it merges, for each region, the data from left/right hemispheres into a single value.
    validate
        If True, it validates each region in each brain, checking that they are
        present in the brain region ontology against which the brains were alligned.
    brain_ontology
        The ontology to which the brains' data was registered against. Used if `validate=True`, else it can be None.

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

# def merge(d):
def merge_hemispheres(d: BrainSlice|SlicedBrain|SlicedGroup|SlicedExperiment|\
                        AnimalBrain|AnimalGroup|Experiment):
    if isinstance(d, (SlicedBrain, AnimalBrain, BrainSlice)):
        return d.merge_hemispheres()
    elif isinstance(d, (SlicedGroup, AnimalGroup, SlicedExperiment, Experiment)):
        return d.apply(merge_hemispheres)
    raise TypeError(type(d))
