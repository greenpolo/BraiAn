# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pandas as pd
from braian.brain_hierarchy import AllenBrainHierarchy
from braian.animal_group import AnimalGroup

__all__ = ["as_prism_data"]

def as_prism_data(brain_ontology: AllenBrainHierarchy,
                  group1: AnimalGroup, group2: AnimalGroup, *groups: AnimalGroup,
                  marker=None):
    groups = [group1, group2, *groups]
    if any(len(g.markers) > 1 for g in groups):
        if marker is None:
            raise ValueError("Exporting AnimalGroups with multiple markers as Prism data isn't implemented yet")
    elif marker is None:
        markers = {g.markers[0] for g in groups}
        assert len(markers) == 1, "All AnimalGroups must have the same marker!"
        marker = group1.markers[0]
    if not all(group1.is_comparable(g) for g in groups[1:]):
        raise ImportError("The AnimalGroups are not comparable! Please check that all groups work on the same kind of data (i.e. markers, hemispheres and metric)")
    df = pd.concat({g.name: g.to_pandas(marker) for g in groups}, axis=1)
    major_divisions = brain_ontology.get_areas_major_division(*df.index)
    df["major_divisions"] = [major_divisions[region] for region in df.index]
    df.set_index("major_divisions", append=True, inplace=True)
    return df