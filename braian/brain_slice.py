import os
import pandas as pd
from .brain_hierarchy import AllenBrainHierarchy
import copy

class BrainSlice:
    def __init__(self, AllenBrain: AllenBrainHierarchy, csv_file: str, excluded_regions_file: str,
                    name: str, area_key: str, tracer_key: str, marker_key, area_units="µm2") -> None:
        self.name = name
        with open(excluded_regions_file, mode="r", encoding="utf-8") as file:
            excluded_regions = file.readlines()
        excluded_regions = [line.strip() for line in excluded_regions]

        try:
            data = self.read_csv_data(csv_file)
        except Exception as e:
            if os.stat(csv_file).st_size == 0:
                print(f"Empty file: {csv_file}")
            else:
                print(f"Could not read file: {csv_file}\n")
            raise e

        self.data = pd.DataFrame(data, columns=[area_key, tracer_key])
        self.data.rename(columns={area_key: 'area', tracer_key: marker_key}, inplace=True)
        #@assert (df.area > 0).all()
        self.data = self.data[self.data['area'] > 0]
                    
        # Take care of regions to be excluded
        self.exclude_regions(excluded_regions, AllenBrain)
        match area_units:
            case "µm2":
                self.area_µm2_to_mm2()
            case "mm2":
                pass
            case _:
                raise ValueError("A brain slice's area can only be expressed in µm² or mm²!")
    
    def read_csv_data(self, csv_file) -> pd.DataFrame:
        data = pd.read_csv(csv_file, sep="\t").drop_duplicates()
        if data['Num Detections'].count() == 0:
            print(f"The file {csv_file} only contains NaNs!")
        
        # There is one region (the full slice) with Name=='Root' and Class==NaN.
        # We remove it. As we want the distinction between hemispheres
        data['Class'] = data['Class'].fillna('wholebrain')
        data = data.set_index('Class')
        data = data.drop('wholebrain', axis=0)
 
        return data
    
    def exclude_regions(self, excluded_regions, AllenBrain) -> None:
        '''
        Take care of regions to be excluded from the analysis.
        If a region is to be excluded, 2 things must happen:
        (1) The cell counts of that region must be subtracted from all
            its parent regions.
        (2) The region must disappear from the data, together with all 
            its daughter regions.
        '''

        for reg_hemi in excluded_regions:
            if ": " not in reg_hemi:
                raise SyntaxError("region_to_exclude of "+self.name+" file is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'")
            hemi = reg_hemi.split(': ')[0]
            reg = reg_hemi.split(': ')[1]

            # Step 1: subtract counting results of the regions to be excluded
            # from their parent regions.
            regions_above = AllenBrain.get_regions_above(reg)
            for region in regions_above:
                row = hemi+': '+region
                # Subtract the counting results from the parent region.
                # Use fill_value=0 to prevent "3-NaN=NaN".
                if row in self.data.index and reg_hemi in self.data.index:
                    self.data.loc[row] = self.data.loc[row].subtract(self.data.loc[reg_hemi], fill_value=0)

            # Step 2: Remove the regions that should be excluded
            # together with their daughter regions.
            subregions = AllenBrain.list_all_subregions(reg)
            for subreg in subregions:
                row = hemi+': '+subreg
                if row in self.data.index:
                    self.data = self.data.drop(row)
    
    def area_µm2_to_mm2(self) -> None:
        self.data.area = self.data.area * 1e-06

    def add_marker_density(self, marker) -> None:
        '''
        Adds a 'density' column to the BrainSlice
        '''
        if f"{marker}_density" not in self.data.columns:
            self.data[f"{marker}_density"] = self.data[marker] / self.data["area"]


def find_region_abbreviation(region_class):
    '''
    This function finds the region abbreviation
    by splitting the class value in the table.
    Example: 'Left: AVA' becomes 'AVA'.
    '''
    try: # try to split the class
        region_abb = region_class.split(': ')[1]
    except: # if splitting gives an error, don't split
        region_abb = str(region_class)
        
    return region_abb

def merge_slice_hemispheres(brain_slice) -> BrainSlice:
    '''
    Function takes as input a BrainSlice. Each row represents a left/right part of a region.
    
    The output is a dataframe with each column being the sum of the two hemispheres
    '''
    slice = copy.copy(brain_slice)
    corresponding_region = [find_region_abbreviation(region) for region in slice.data.index]
    slice.data = slice.data.groupby(corresponding_region, axis=0).sum(min_count=1)
    return slice