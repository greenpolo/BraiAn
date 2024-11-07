import enum

from braian.brain_data import BrainData
from braian.animal_brain import AnimalBrain
from braian.animal_group import AnimalGroup
from collections.abc import Callable

# NOTE: some arithmetic operations (e.g. division by zero) are not correctly converted to pd.NA,
# which results in having BrainData filled with np.nan.
# https://github.com/pandas-dev/pandas/issues/59961

__all__ = ["density",
           "percentage",
           "relative_density",
           "fold_change",
           "diff_change",
           "markers_overlap",
           "markers_jaccard_index",
           "markers_similarity_index",
           "markers_overlap_coefficient",
           # "markers_chance_level",
           "markers_difference",
           "markers_correlation"
]

class BrainMetrics(enum.Enum):
    DENSITY = enum.auto()
    PERCENTAGE = enum.auto()
    RELATIVE_DENSITY = enum.auto()
    OVERLAPPING = enum.auto()
    JACCARD_INDEX = enum.auto()
    SIMILARITY_INDEX = enum.auto()
    OVERLAP_COEFFICIENT = enum.auto()
    CHANCE_LEVEL = enum.auto()
    MARKER_DIFFERENCE = enum.auto()
    FOLD_CHANGE = enum.auto()
    DIFF_CHANGE = enum.auto()

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        match value.lower():
            case "density" | "dens" | "d":
                return BrainMetrics.DENSITY
            case "percentage" | "perc" | "%" | "p":
                return BrainMetrics.PERCENTAGE
            case "relative_density" | "relativedensity" | "relative density" | "rd":
                return BrainMetrics.RELATIVE_DENSITY
            case "overlaps" | "overlap" | "overlapping":
                return BrainMetrics.OVERLAPPING
            case "jaccard" | "jaccard_index" | "jaccard index":
                return BrainMetrics.JACCARD_INDEX
            case "similarity" | "similarity index" | "similarity_index" | "similarityindex" | "sim":
                return BrainMetrics.SIMILARITY_INDEX
            case "szymkiewicz_simpson" | "overlap_coeff" | "overlap_coefficient" | "overlap coefficient":
                return BrainMetrics.OVERLAP_COEFFICIENT
            case "chance" | "chance_level" | "chance level":
                return BrainMetrics.CHANCE_LEVEL
            case "mdiff" | "marker_difference" | "marker difference":
                return BrainMetrics.MARKER_DIFFERENCE
            case "fold_change" | "fold change":
                return BrainMetrics.FOLD_CHANGE
            case "diff_change" | "diff change":
                return BrainMetrics.DIFF_CHANGE
            case _:
                return None

def _enforce_rawdata(brain: AnimalBrain):
    if not brain.raw:
        raise ValueError(f"This metric cannot be computed for AnimalBrains whose data is not raw (metric={brain.metric}).")

def density(brain: AnimalBrain) -> AnimalBrain:
    r"""
    For each region $r$ of `brain`, it computes the density $D(m)$ for each marker $m$:
    $$
    D(m_r) : \\frac {m_r} {size_r}
    $$
    with $m_r$ being the raw number of $m$ detections in region $r$.

    Parameters
    ----------
    brain
        The brain to compute density on.

    Returns
    -------
    :
        A new brain with density data.

    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    """
    _enforce_rawdata(brain)
    markers_data = dict()
    for marker in brain.markers:
        data = brain[marker] / brain.areas
        data.metric = "density"
        data.units = f"{marker}/{brain.areas.units}"
        markers_data[marker] = data
    return AnimalBrain(markers_data=markers_data, areas=brain.areas, raw=False)

def percentage(brain: AnimalBrain) -> AnimalBrain:
    r"""
    For each region $r$ of `brain`, it computes the percentage $P(m)$ for each marker $m$
    detection compared to brain-wide $m$ counts:
    $$
    P(m_r) : \\frac {m_r} {m_{root}}
    $$
    with $m_r$ being the raw number of $m$ detections in region $r$.

    Parameters
    ----------
    brain
        The brain to compute percentage on.

    Returns
    -------
    :
        A new brain with percentage data.

    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    """
    _enforce_rawdata(brain)
    if brain.is_split:
        hems = ("L", "R")
    else:
        hems = (None,)
    markers_data = dict()
    for marker in brain.markers:
        brainwide_cell_counts = sum((brain[marker].root(hem) for hem in hems))
        data = brain[marker] / brainwide_cell_counts
        data.metric = "percentage"
        data.units = f"{marker}/{marker} in root"
        markers_data[marker] = data
    return AnimalBrain(markers_data=markers_data, areas=brain.areas, raw=False)

def relative_density(brain: AnimalBrain) -> AnimalBrain:
    r"""
    For each region $r$ of `brain`, it computes the density fold change
    of each marker $m$ compared to brain-wide marker density:
    $$
    RD(m_r) : \\frac {m_r/size_r} {m_{root}/size_{root}}
    $$
    with $m_r$ being the raw number of $m$ detections in region $r$.

    Parameters
    ----------
    brain
        The brain to compute relative density on.

    Returns
    -------
    :
        A new brain with relative density data.

    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    """
    _enforce_rawdata(brain)
    if brain.is_split:
        hems = ("L", "R")
    else:
        hems = (None,)
    markers_data = dict()
    for marker in brain.markers:
        brainwide_area = sum((brain.areas.root(hem) for hem in hems))
        brainwide_cell_counts = sum((brain[marker].root(hem) for hem in hems))
        data = (brain[marker] / brain.areas) / (brainwide_cell_counts / brainwide_area)
        data.metric = "relative_density"
        data.units = f"{marker} density/root {marker} density"
        markers_data[marker] = data
    return AnimalBrain(markers_data=markers_data, areas=brain.areas, raw=False)

def _group_change(brain: AnimalBrain, group: AnimalGroup,
                  metric: str, fun: Callable[[BrainData,BrainData],BrainData],
                  symbol: str) -> AnimalBrain:
    assert brain.is_split == group.is_split, "Both AnimalBrain and AnimalGroup must either have the hemispheres split or not"
    assert set(brain.markers) == set(group.markers), "Both AnimalBrain and AnimalGroup must have the same markers"
    # assert brain.metric == group.metric == BrainMetrics.DENSITY, f"Both AnimalBrain and AnimalGroup must be on {BrainMetrics.DENSITY}"
    # assert set(brain.regions) == set(group.regions), f"Both AnimalBrain and AnimalGroup must be on the same regions"

    markers_data = dict()
    for marker,this in brain.markers_data.items():
        data = fun(this, group.mean[marker])
        data.metric = str(metric)
        data.units = f"{marker} {str(brain.metric)}{symbol}{group.name} {str(group.metric)}"
        markers_data[marker] = data
    return AnimalBrain(markers_data=markers_data, areas=brain.areas, raw=False)

def fold_change(brain: AnimalBrain, group: AnimalGroup) -> AnimalBrain:
    """
    For each brain region in `brain`, compute the
    [fold change](https://en.wikipedia.org/wiki/Fold_change) of its markers with respect to `group`'s mean.

    Parameters
    ----------
    brain
        The brain for which to compute the fold change.
    group
        The group whose mean is the basis of the fold change.

    Returns
    -------
    :
        A new brain with fold change data.

    See also
    --------
    [`diff_change`][braian.stats.diff_change]
    """
    return _group_change(brain, group, "fold_change", lambda animal,group: animal/group, "/")

def diff_change(brain: AnimalBrain, group: AnimalGroup) -> AnimalBrain:
    """
    For each brain region in `brain`, compute the difference between its markers and the `group`'s mean.

    Parameters
    ----------
    brain
        The brain for which to compute the difference.
    group
        The group whose mean is subtracted.

    Returns
    -------
    :
        A new brain of the difference of `brain` from `group`'s mean.

    See also
    --------
    [`fold_change`][braian.stats.fold_change]
    """
    return _group_change(brain, group, "diff_change", lambda animal,group: animal-group, "-")

def markers_overlap(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    r"""
    For each region, it computes the ratio of positive cells and double positive counts;
    for both `marker1` and `marker2`:
    $$
    O(m_1,m_{1,2}) : \\frac {m_1} {m_{1,2}}
    $$
    $$
    O(m_2,m_{1,2}) : \\frac {m_2} {m_{1,2}}
    $$
    with $m_{1,2}$ being the number of detections being `marker1` _and_ `marker2` positive.

    Parameters
    ----------
    brain
        The brain for which to compute the markers overlapping rate.
    marker1
        The first overlapping marker.
    marker2
        The second overlapping marker.

    Returns
    -------
    :
        A new brain of the overlapping rate for `marker1` and `marker2`.

    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    ValueError
        If `brain` does not have any raw data of `marker1` or `marker2` countings
    ValueError
        If `brain` does not contain any raw data of the `marker1` and
        `marker2` double positive countings.
    """
    _enforce_rawdata(brain)
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    try:
        both = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in brain.markers)
    except StopIteration as e:
        raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
    overlaps = dict()
    for m in (marker1, marker2):
        # TODO: clipping overlaps to 100% because of a bug with the QuPath script that counts overlapping cells as belonging to different regions
        overlaps[m] = (brain[both] / brain[m]).clip(upper=1)
        overlaps[m].metric = "overlaps"
        overlaps[m].units = f"({marker1}+{marker2})/{m}"
    return AnimalBrain(markers_data=overlaps, areas=brain.areas, raw=False)

def markers_jaccard_index(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    r"""
    For each region, it computes the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
    measuring the similarity between two markers activity and the respective double positivity.
    $$
    J(m_1,m_2) : \\frac {m_{1,2}} {m_1+m_2-m_{1,2}}
    $$
    with $m_{1,2}$ being the number of detections being `marker1` _and_ `marker2` positive.

    Parameters
    ----------
    brain
        The brain for which to compute the markers Jaccard index.
    marker1
        The first overlapping marker.
    marker2
        The second overlapping marker.

    Returns
    -------
    :
        A new brain of the Jaccard index for `marker1` and `marker2`.

    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    ValueError
        If `brain` does not have any raw data of `marker1` or `marker2` countings
    ValueError
        If `brain` does not contain any raw data of the `marker1` and
        `marker2` double positive countings.
    """
    _enforce_rawdata(brain)
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    try:
        overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in brain.markers)
    except StopIteration as e:
        raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
    similarities = brain[overlapping] / (brain[marker1]+brain[marker2]-brain[overlapping])
    similarities.metric = "jaccard_index"
    similarities.units = f"({marker1}∩{marker2})/({marker1}∪{marker2})"
    return AnimalBrain(markers_data={overlapping: similarities}, areas=brain.areas, raw=False)

def markers_similarity_index(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    # computes an index of normalized similarity we developed
    r"""
    For each region, it computes an index of similarity between two markers activity
    and the respective double positivity; it is defined as:
    $$
    S(m_1,m_2) : \\frac {m_{1,2}^2} {m_1 \\cdot m_2}
    $$
    with $m_{1,2}$ being the number of detections being `marker1` _and_ `marker2` positive.

    _NOTE_: $S(m_1,m_2) = 1 \iff m_1 = m_2 = m_{1,2}$. \
    Additionally, if either $m1$ or $m2$ is zero, it goes to infinite.

    Parameters
    ----------
    brain
        The brain for which to compute the markers similarity index.
    marker1
        The first overlapping marker.
    marker2
        The second overlapping marker.

    Returns
    -------
    :
        A new brain of the similarity index between `marker1` and `marker2`.

    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    ValueError
        If `brain` does not have any raw data of `marker1` or `marker2` countings
    ValueError
        If `brain` does not contain any raw data of the `marker1` and
        `marker2` double positive countings.
    """
    _enforce_rawdata(brain)
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    try:
        overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in brain.markers)
    except StopIteration as e:
        raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
    # NOT normalized in (0,1)
    # similarities = brain[overlapping] / (brain[marker1]*brain[marker2]) * brain.areas
    # NORMALIZED
    similarities = brain[overlapping]**2 / (brain[marker1]*brain[marker2])
    similarities.metric = "similarity_index"
    similarities.units = f"({marker1}∩{marker2})²/({marker1}×{marker2})"
    return AnimalBrain(markers_data={overlapping: similarities}, areas=brain.areas, raw=False)

def markers_overlap_coefficient(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    r"""
    For each region, it computes the overlapping coefficient (or
    [Szymkiewicz–Simpson coefficient](https://en.wikipedia.org/wiki/Overlap_coefficient)),
    an index of similarity between two markers activity and the respective double positivity;
    it is defined as:
    $$
    S(m_1,m_2) : \\frac {m_{1,2}} {\\min({m_1}, {m_2})}
    $$
    with $m_{1,2}$ being the number of detections being `marker1` _and_ `marker2` positive.
    Parameters
    ----------
    brain
        The brain for which to compute the markers overlapping coefficient.
    marker1
        The first overlapping marker.
    marker2
        The second overlapping marker.

    Returns
    -------
    :
        A new brain of the overlapping coefficient between `marker1` and `marker2`.
    Raises
    ------
    ValueError
        If `brain` does not contain raw data of marker countings.
    ValueError
        If `brain` does not have any raw data of `marker1` or `marker2` countings
    ValueError
        If `brain` does not contain any raw data of the `marker1` and
        `marker2` double positive countings.
    """
    # computes Szymkiewicz–Simpson coefficient
    _enforce_rawdata(brain)
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    try:
        overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in brain.markers)
    except StopIteration as e:
        raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
    overlap_coeffs = brain[overlapping] / BrainData.minimum(brain[marker1], brain[marker2])
    overlap_coeffs.metric = "overlap_coefficient"
    overlap_coeffs.units = f"({marker1}∩{marker2})/min({marker1},{marker2})"
    return AnimalBrain(markers_data={overlapping: overlap_coeffs}, areas=brain.areas, raw=False)

# def markers_chance_level(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
#     # This chance level is good only if the used for the fold change.
#     #
#     # It is similar to our Similarity Index, as it is derived from its NOT normalized form.
#     # ideally it would use the #DAPI instead of the area, as that would give an interval
#     # which is easier to work with.
#     # However, when:
#     #  * the DAPI is not available AND
#     #  * we're interested in the difference of fold change between groups
#     # we can ignore the DAPI count it simplifies during the rate group1/group2
#     # thus the use case of this index.
#     #
#     # since the areas/DAPI simplifies only when they are ~comparable between animals,
#     # we force the AnimalBrain to be a result of MEAN of SlicedBrain, not SUM of SlicedBrain
#     _enforce_rawdata(brain)
#     for m in (marker1, marker2):
#         if m not in brain.markers:
#             raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
#     try:
#         overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in brain.markers)
#     except StopIteration as e:
#         raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
#     chance_level = brain[overlapping] / (brain[marker1]*brain[marker2])
#     chance_level.metric = "chance_level"
#     chance_level.units = f"({marker1}∩{marker2})/({marker1}×{marker2})"
#     return AnimalBrain(markers_data={overlapping: chance_level}, areas=brain.areas, raw=False)

def markers_difference(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    """
    For each brain region in `brain`, compute the difference between two markers.

    Parameters
    ----------
    brain
        The brain for which to compute the difference.
    marker1
        The first marker to subtract.
    marker2
        The second marker to subtract.

    Returns
    -------
    :
        A new brain of the difference between `marker1` and `marker2`.
    """
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    diff = brain[marker1] - brain[marker2]
    diff.metric = "marker_difference"
    diff.units = f"{brain[marker1].units}-{brain[marker2].units}"
    return AnimalBrain(markers_data={f"{marker1}+{marker2}": diff}, areas=brain.areas, raw=False)

def markers_correlation(marker1: str, marker2: str,
                        group: AnimalGroup, other: AnimalGroup=None,
                        method: str="pearson") -> BrainData:
    """
    For each brain region in `group`, compute the correlation between two markers
    within all animals in the cohort.

    Parameters
    ----------
    marker1
        The first marker to correlate.
    marker2
        The second marker to correlate.
    group
        The group from which all animals are taken to compute the correlation.
    other
        If specified, it uses data from `other`'s `marker2`.
    method
        Any method accepted by [`DataFrame.corrwith`][pandas.DataFrame.corrwith].

    Returns
    -------
    :
        Brain data of the correlation between `marker1` and `marker2`.
    """
    if other is None:
        other = group
    else:
        assert group.metric == other.metric, "Both groups must have the same metric."
    corr = group.to_pandas(marker1).corrwith(other.to_pandas(marker2), method=method, axis=1)
    return BrainData(corr, group.name, str(group.metric)+f"-corr (n={group.n})", f"corr({marker1}, {marker2})")