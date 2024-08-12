import enum

from braian.brain_data import BrainData
from braian.animal_brain import AnimalBrain
from braian.animal_group import AnimalGroup

__all__ = ["density",
           "percentage",
           "relative_density",
           "fold_change",
           "diff_change",
           "markers_overlap",
           "markers_jaccard_index",
           "markers_similarity_index",
           "markers_overlap_coefficient",
           "markers_chance_level",
           "markers_difference"]

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
        raise ValueError(f"Cannot compute densities for AnimalBrains whose data is not raw (mode={brain.mode}).")

def density(brain: AnimalBrain) -> AnimalBrain:
    _enforce_rawdata(brain)
    markers_data = dict()
    for marker in brain.markers:
        data = brain[marker] / brain.areas
        data.metric = "density"
        data.units = f"{marker}/{brain.areas.units}"
        markers_data[marker] = data
    return AnimalBrain(markers_data=markers_data, areas=brain.areas, raw=False)

def percentage(brain: AnimalBrain) -> AnimalBrain:
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

def _group_change(brain: AnimalBrain, group: AnimalGroup, metric, fun: callable, symbol: str) -> AnimalBrain:
    assert brain.is_split == group.is_split, "Both AnimalBrain and AnimalGroup must either have the hemispheres split or not"
    assert set(brain.markers) == set(group.markers), "Both AnimalBrain and AnimalGroup must have the same markers"
    # assert brain.mode == group.metric == BrainMetrics.DENSITY, f"Both AnimalBrain and AnimalGroup must be on {BrainMetrics.DENSITY}"
    # assert set(brain.regions) == set(group.regions), f"Both AnimalBrain and AnimalGroup must be on the same regions"
    
    markers_data = dict()
    for marker,this in brain.markers_data.items():
        data = fun(this, group.mean[marker])
        data.metric = str(metric)
        data.units = f"{marker} {str(brain.mode)}{symbol}{group.name} {str(group.metric)}"
        markers_data[marker] = data
    return AnimalBrain(markers_data=markers_data, areas=brain.areas, raw=False)

def fold_change(brain: AnimalBrain, group: AnimalGroup) -> AnimalBrain:
    return _group_change(group, BrainMetrics.FOLD_CHANGE, lambda animal,group: animal/group, "/")

def diff_change(brain: AnimalBrain, group: AnimalGroup) -> AnimalBrain:
    return _group_change(group, BrainMetrics.DIFF_CHANGE, lambda animal,group: animal-group, "-")

def markers_overlap(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
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
    # computes Jaccard's index
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
    # NOTE: if either marker1 or marker2 is zero, it goes to infinite
    # computes an index of normalized similarity we developed
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

def markers_chance_level(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    # This chance level is good only if the used for the fold change.
    #
    # It is similar to our Similarity Index, as it is derived from its NOT normalized form.
    # ideally it would use the #DAPI instead of the area, as that would give an interval
    # which is easier to work with.
    # However, when:
    #  * the DAPI is not available AND
    #  * we're interested in the difference of fold change between groups
    # we can ignore the DAPI count it simplifies during the rate group1/group2
    # thus the use case of this index.
    #
    # since the areas/DAPI simplifies only when they are ~comparable between animals,
    # we force the AnimalBrain to be a result of MEAN of SlicedBrain, not SUM of SlicedBrain
    _enforce_rawdata(brain)
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    try:
        overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in brain.markers)
    except StopIteration as e:
        raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
    chance_level = brain[overlapping] / (brain[marker1]*brain[marker2])
    chance_level.metric = "chance_level"
    chance_level.units = f"({marker1}∩{marker2})/({marker1}×{marker2})"
    return AnimalBrain(markers_data={overlapping: chance_level}, areas=brain.areas, raw=False)

def markers_difference(brain: AnimalBrain, marker1: str, marker2: str) -> AnimalBrain:
    for m in (marker1, marker2):
        if m not in brain.markers:
            raise ValueError(f"Marker '{m}' is unknown in '{brain.name}'!")
    diff = brain[marker1] - brain[marker2]
    diff.metric = "marker_difference"
    diff.units = f"{brain[marker1].units}-{brain[marker2].units}"
    return AnimalBrain(markers_data={f"{marker1}+{marker2}": diff}, areas=brain.areas, raw=False)