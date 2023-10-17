import pandas as pd
from .brain_hierarchy import AllenBrainHierarchy
from .animal_group import AnimalGroup

def as_prism_data(normalization, group1: AnimalGroup, group2: AnimalGroup, AllenBrain: AllenBrainHierarchy):
    if len(group1.markers) > 1 or len(group2.markers) > 1:
        raise ValueError("Exporting AnimalGroups with multiple markers as Prism data isn't implemented yet")
    if not group1.is_comparable(group2):
        raise ImportError("Group 1 and Group 2 are not comparable! Please check that both groups are counting the same marker")
    df = pd.concat({group1.name: group1.to_pandas(group1.markers[0]), group2.name: group2.to_pandas(group2.markers[0])}, axis=1)
    major_divisions = AllenBrain.get_areas_major_division(*df.index)
    df["major_divisions"] = [major_divisions[region] for region in df.index]
    df.set_index("major_divisions", append=True, inplace=True)
    return df