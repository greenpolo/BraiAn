import os
import pandas as pd
from .brain_hierarchy import AllenBrainHierarchy
import copy

global MODE_PathAnnotationObjectError
global MODE_ExcludedRegionNotRecognisedError
MODE_PathAnnotationObjectError = "print"
MODE_ExcludedRegionNotRecognisedError = "print"

class BrainSliceFileError(Exception):
    def __init__(self, animal=None, file=None, *args: object) -> None:
        self.animal_name = animal
        self.file_path = file
        super().__init__(*args)
class ExcludedRegionsNotFoundError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}' - could not read the expected regions_to_exclude: {self.file_path}"
class EmptyResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}' - empty file: {self.file_path}"
class NanResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}' - NaN-filled file: {self.file_path}"
class InvalidResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}' - could not read results file: {self.file_path}"
class MissingResultsColumnError(BrainSliceFileError):
    def __init__(self, column, *args: object) -> None:
        self.column = column
        super().__init__(*args)
    def __str__(self):
        return f"Animal '{self.animal_name}' - column '{self.column}' is missing in file: {self.file_path}"
class InvalidRegionsHemisphereError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}' - results file {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"
class InvalidExcludedRegionsHemisphereError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}' - regions_to_exclude file {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"


class BrainSlice:
    def __init__(self, AllenBrain: AllenBrainHierarchy, csv_file: str, excluded_regions_file: str,
                    animal:str, name: str, area_key: str, tracer_key: str, marker_key, area_units="µm2") -> None:
        self.animal = animal
        self.name = name
        self.marker = marker_key
        
        data = self.read_results_data(csv_file)
        excluded_regions = self.read_regions_to_exclude(excluded_regions_file)
        self.check_columns(data, [area_key, tracer_key], csv_file)
        self.data = pd.DataFrame(data, columns=[area_key, tracer_key])
        self.data.rename(columns={area_key: "area", tracer_key: marker_key}, inplace=True)
        #@assert (df.area > 0).all()
        self.data = self.data[self.data["area"] > 0]
                    
        # Take care of regions to be excluded
        self.exclude_regions(excluded_regions, AllenBrain)
        match area_units:
            case "µm2":
                self._area_µm2_to_mm2_()
            case "mm2":
                pass
            case _:
                raise ValueError("A brain slice's area can only be expressed in µm² or mm²!")
    def read_regions_to_exclude(self, file_path) -> list[str]:
        try:
            with open(file_path, mode="r", encoding="utf-8") as file:
                excluded_regions = file.readlines()
        except FileNotFoundError:
            raise ExcludedRegionsNotFoundError(animal=self.animal, file=file_path)
        to_exclude = [line.strip() for line in excluded_regions]
        # return [region for region in to_exclude if region == ""] # if we want to allow having empty lines in _regions_to_exclude.txt
        return to_exclude
    
    def read_results_data(self, csv_file) -> pd.DataFrame:
        data = self.read_csv_file(csv_file)
        self.check_columns(data, ["Name", "Class", "Num Detections",], csv_file)

        if data["Num Detections"].count() == 0:
            raise NanResultsError(animal=self.animal, file=csv_file)

        data = self.clean_rows(data, csv_file)

        # There may be one region/row with Name == "Root" and Class == NaN indicating the whole slice.
        # We remove it. As we want the distinction between hemispheres
        match (data["Class"].isnull()).sum():
            case 0:
                data = data.set_index("Class")
            case 1:
                data["Class"] = data["Class"].fillna("wholebrain")
                data = data.set_index("Class")
                data = data.drop("wholebrain", axis=0)
            case _:
                raise InvalidResultsError(animal=self.animal, file=csv_file)

        self.check_hemispheres(data, csv_file)
        return data

    def read_csv_file(self, csv_file) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_file, sep="\t").drop_duplicates()
        except Exception as e:
            if os.stat(csv_file).st_size == 0:
                raise EmptyResultsError(animal=self.animal, file=csv_file)
            else:
                raise InvalidResultsError(animal=self.animal, file=csv_file)
    
    def clean_rows(self, data, csv_file):
        if (data["Name"] == "Exclude").any():
            # some rows have the Name==Exclude because the cell counting script was run AFTER having done the exclusions
            data = data.loc[data["Name"] != "PathAnnotationObject"]
        if (data["Name"] == "PathAnnotationObject").any() and \
            not (data["Name"] == "PathAnnotationObject").all():
            global MODE_PathAnnotationObjectError
            if MODE_PathAnnotationObjectError != "silent":
                print(f"WARNING: there are rows with column 'Name'=='PathAnnotationObject' in animal '{self.animal}', file: {csv_file}\n\
\tPlease, check on QuPath that you selected the right Class for every exclusion.")
            data = data.loc[data["Name"] != "PathAnnotationObject"]
        if len(data) == 0:
            raise EmptyResultsError(animal=self.animal, file=csv_file)
        return data

    def check_columns(self, data, columns, csv_file) -> bool:
        for column in columns:
            if column not in data.columns:
                raise MissingResultsColumnError(animal=self.animal, file=csv_file, column=column)
        return True

    def check_hemispheres(self, data, csv_file) -> bool:
        if (data.index.str.startswith("Left: ", na=False) |
            data.index.str.startswith("Right: ", na=False)).sum() != 0:
            InvalidRegionsHemisphereError(csv_file)
        return True
    
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
                if MODE_ExcludedRegionNotRecognisedError != "silent":
                    print(f"WARNING: Class '{reg_hemi}' is not recognised as a brain region. It was skipped from the regions_to_exclude in animal '{self.animal}', file: {self.name}_regions_to_exclude.txt")
                    continue
                elif MODE_ExcludedRegionNotRecognisedError == "error":
                    raise InvalidExcludedRegionsHemisphereError(animal=self.animal, file=f"{self.name}_regions_to_exclude.txt")
            hemi = reg_hemi.split(": ")[0]
            reg = reg_hemi.split(": ")[1]

            # Step 1: subtract counting results of the regions to be excluded
            # from their parent regions.
            regions_above = AllenBrain.get_regions_above(reg)
            for region in regions_above:
                row = hemi+": "+region
                # Subtract the counting results from the parent region.
                # Use fill_value=0 to prevent "3-NaN=NaN".
                if row in self.data.index and reg_hemi in self.data.index:
                    self.data.loc[row] = self.data.loc[row].subtract(self.data.loc[reg_hemi], fill_value=0)

            # Step 2: Remove the regions that should be excluded
            # together with their daughter regions.
            subregions = AllenBrain.list_all_subregions(reg)
            for subreg in subregions:
                row = hemi+": "+subreg
                if row in self.data.index:
                    self.data = self.data.drop(row)
    
    def _area_µm2_to_mm2_(self) -> None:
        self.data.area = self.data.area * 1e-06

    def add_density(self) -> None:
        '''
        Adds a 'density' column to the BrainSlice
        '''
        if f"{self.marker}_density" not in self.data.columns:
            self.data[f"{self.marker}_density"] = self.data[self.marker] / self.data["area"]


def find_region_abbreviation(region_class):
    '''
    This function finds the region abbreviation
    by splitting the class value in the table.
    Example: "Left: AVA" becomes "AVA".
    '''
    try: # try to split the class
        region_abb = region_class.split(": ")[1]
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