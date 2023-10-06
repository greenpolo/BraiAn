import pandas as pd
from .brain_hierarchy import AllenBrainHierarchy
from .animal_group import AnimalGroup

def as_prism_data(normalization, group1: AnimalGroup, group2: AnimalGroup, AllenBrain: AllenBrainHierarchy):
    if len(group1.markers) > 1 or len(group2.markers) > 1:
        raise ValueError("Exporting AnimalGroups with multiple markers as Prism data isn't implemented yet")
    if not group1.is_comparable(group2):
        raise ImportError("Group 1 and Group 2 are not comparable! Please check that both groups are counting the same marker")
    groups = [group.name for group in (group1, group2) for _ in group.get_animals()]
    animals = [animal for group in (group1, group2) for animal in sorted(list(group.get_animals()))]
    df = pd.DataFrame(columns=pd.MultiIndex.from_arrays([groups, animals]))
    for group in (group1, group2):
        for animal in group.data.index.unique(1):
            df[group.name, animal] = group.data.xs(animal, axis=0, level=1, drop_level=True)[group.markers[0]][normalization]
    major_divisions = AllenBrain.get_areas_major_division(*df.index)
    df["major_divisions"] = [major_divisions[region] for region in df.index]
    df.set_index("major_divisions", append=True, inplace=True)
    return df