import copy
import os
import pandas as pd
import re
from typing import Self

from .brain_data import is_split_left_right, extract_acronym
from .brain_hierarchy import AllenBrainHierarchy

global MODE_PathAnnotationObjectError
global MODE_ExcludedRegionNotRecognisedError
MODE_PathAnnotationObjectError = "print"
MODE_RegionsWithNoCountError = "silent"
MODE_ExcludedRegionNotRecognisedError = "print"

class BrainSliceFileError(Exception):
    def __init__(self, slice=None, file=None, *args: object) -> None:
        self.animal_name = slice.animal
        self.slice_name = slice.name
        self.file_path = file
        super().__init__(*args)
class ExcludedRegionsNotFoundError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - could not read the expected regions_to_exclude: {self.file_path}"
class ExcludedAllRegionsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - the corresponding regions_to_exclude excludes everything: {self.file_path}"
class EmptyResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - empty file: {self.file_path}"
class NanResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - NaN-filled file: {self.file_path}"
class InvalidResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - could not read results file: {self.file_path}"
class MissingResultsColumnError(BrainSliceFileError):
    def __init__(self, column, **kargs: object) -> None:
        self.column = column
        super().__init__(**kargs)
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - column '{self.column}' is missing in file: {self.file_path}"
class RegionsWithNoCountError(BrainSliceFileError):
    def __init__(self, tracer, regions, **kargs: object) -> None:
        self.tracer = tracer
        self.regions = regions
        super().__init__(**kargs)
    def __str__(self) -> str:
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - there are {len(self.regions)} region(s) with no count of tracer '{self.tracer}' in file: {self.file_path}"
class InvalidRegionsHemisphereError(BrainSliceFileError): 
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - results file {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"
class InvalidExcludedRegionsHemisphereError(BrainSliceFileError):
    def __str__(self):
        return f"Animal '{self.animal_name}', slice '{self.slice_name}' - regions_to_exclude file {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"


class BrainSlice:
    def __init__(self, brain_onthology: AllenBrainHierarchy, csv_file: str,
                 excluded_regions_file: str, exclude_parent_regions: bool,
                 animal:str, name: str, area_key: str,
                 tracers_key: list[str], markers_key: list[str], area_units="µm2") -> None:
        self.animal = animal
        self.name = name
        if isinstance(tracers_key, str):
            tracers_key = [tracers_key]
        if isinstance(markers_key, str):
            markers_key = [markers_key]
        assert len(tracers_key) == len(markers_key), f"The number of tracers ({len(tracers_key)}) differs from the number of markers ({len(markers_key)})"

        data = self.read_results_data(csv_file)
        excluded_regions = self.read_regions_to_exclude(excluded_regions_file)
        self.check_columns(data, [area_key, *tracers_key], csv_file)
        self.data = pd.DataFrame(data, columns=[area_key, *tracers_key])
        self.is_split = is_split_left_right(self.data.index)
        if not self.is_split:
            raise InvalidRegionsHemisphereError(slice=self, file=csv_file)
        self.data.rename(columns={area_key: "area"} | dict(zip(tracers_key, markers_key)), inplace=True)
        #assert (df.area > 0).all()
        assert (self.data["area"] > 0).any(), f"All region areas are zero or NaN for animaly={self.animal} slice={self.name}"
        self.data = self.data[self.data["area"] > 0]
        self.check_zero_rows(csv_file, markers_key)

        # Take care of regions to be excluded
        try:
            self.exclude_regions(excluded_regions, brain_onthology, exclude_parent_regions)
        except Exception:
            raise Exception(f"Animal '{self.animal}': failed to exclude regions for in slice '{self.name}'")
        if len(self.data) == 0:
            raise ExcludedAllRegionsError(slice=self, file=excluded_regions_file)
        match area_units:
            case "µm2" | "um2":
                self._area_µm2_to_mm2_()
            case "mm2":
                pass
            case _:
                raise ValueError("A brain slice's area can only be expressed in µm² or mm²!")
        self.markers_density = BrainSlice._get_marker_density(self.data, markers_key)
    
    @staticmethod
    def _get_marker_density(df: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
        return df[markers].div(df["area"], axis=0)

    def read_regions_to_exclude(self, file_path: str) -> list[str]:
        try:
            with open(file_path, mode="r", encoding="utf-8") as file:
                excluded_regions = file.readlines()
        except FileNotFoundError:
            raise ExcludedRegionsNotFoundError(slice=self, file=file_path)
        to_exclude = [line.strip() for line in excluded_regions]
        # return [region for region in to_exclude if region == ""] # if we want to allow having empty lines in _regions_to_exclude.txt
        return to_exclude
    
    def read_results_data(self, csv_file: str) -> pd.DataFrame:
        data = self.read_csv_file(csv_file)
        if "Class" in data.columns:
            raise ValueError("You are analyising results file exported with QuPath <0.5.x. Such files are no longer supported!")
        self.check_columns(data, ["Name", "Classification", "Num Detections",], csv_file)

        if data["Num Detections"].count() == 0:
            raise NanResultsError(slice=self, file=csv_file)

        data = self.clean_rows(data, csv_file)

        # There may be one region/row with Name == "Root" and Class == NaN indicating the whole slice.
        # We remove it. As we want the distinction between hemispheres
        match (data["Classification"].isnull()).sum():
            case 0:
                data = data.set_index("Classification")
            case 1:
                data["Classification"] = data["Classification"].fillna("wholebrain")
                data = data.set_index("Classification")
                data = data.drop("wholebrain", axis=0)
            case _:
                raise InvalidResultsError(slice=self, file=csv_file)
        data.index.name = None
        return data

    def read_csv_file(self, csv_file: str) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_file, sep="\t").drop_duplicates()
        except Exception as e:
            if os.stat(csv_file).st_size == 0:
                raise EmptyResultsError(slice=self, file=csv_file)
            else:
                raise InvalidResultsError(slice=self, file=csv_file)
    
    def clean_rows(self, data: pd.DataFrame, csv_file: str):
        if (data["Name"] == "Exclude").any():
            # some rows have the Name==Exclude because the cell counting script was run AFTER having done the exclusions
            data = data.loc[data["Name"] != "Exclude"]
        if (data["Name"] == "PathAnnotationObject").any() and \
            not (data["Name"] == "PathAnnotationObject").all():
            global MODE_PathAnnotationObjectError
            if MODE_PathAnnotationObjectError != "silent":
                print(f"WARNING: there are rows with column 'Name'=='PathAnnotationObject' in animal '{self.animal}', file: {csv_file}\n\
\tPlease, check on QuPath that you selected the right Class for every exclusion.")
        data = data.loc[data["Name"] != "PathAnnotationObject"]
        if len(data) == 0:
            raise EmptyResultsError(slice=self, file=csv_file)
        return data

    def check_columns(self, data, columns, csv_file) -> bool:
        for column in columns:
            if column not in data.columns:
                raise MissingResultsColumnError(slice=self, file=csv_file, column=column)
        return True
    
    def check_zero_rows(self, csv_file: str, markers: list[str]) -> bool:
        for marker in markers:
            zero_rows = self.data[marker] == 0
            if sum(zero_rows) > 0:
                err = RegionsWithNoCountError(slice=self, file=csv_file,
                            tracer=marker, regions=self.data.index[zero_rows].to_list())
                if MODE_RegionsWithNoCountError == "error":
                    raise err
                elif MODE_RegionsWithNoCountError == "print":
                    print(err)
                return False
        return True

    
    def exclude_regions(self,
                        excluded_regions: list[str],
                        brain_onthology: AllenBrainHierarchy,
                        exclude_parent_regions: bool) -> None:
        '''
        Take care of regions to be excluded from the analysis.
        If a region is to be excluded, 2 things must happen:
        (1) if exclude_parent_regions==False, the cell counts of that
            region must be subtracted from all its parent regions, 
        (2) The region must disappear from the data, together with all 
            its daughter regions.
            If exclude_parent_regions==True, all the parent regions must
            disappear too!
        NOTE: if a region of layer1 was *explicitaly* excluded, it won't
        impact (i.e. remove) the parent regions!
        This decision was taken because often layer1 is mis-aligned and with
        few detection. We don't want to delete too much data and we reckon
        this exception does not impact too much on the data
        '''
        layer1 = set(brain_onthology.get_layer1())
        for reg_hemi in excluded_regions:
            if ": " not in reg_hemi:
                if MODE_ExcludedRegionNotRecognisedError != "silent":
                    print(f"WARNING: Class '{reg_hemi}' is not recognised as a brain region. It was skipped from the regions_to_exclude in animal '{self.animal}', file: {self.name}_regions_to_exclude.txt")
                    continue
                elif MODE_ExcludedRegionNotRecognisedError == "error":
                    raise InvalidExcludedRegionsHemisphereError(slice=self, file=f"{self.name}_regions_to_exclude.txt")
            hemi, reg = reg_hemi.split(": ")

            # Step 1: subtract counting results of the regions to be excluded
            # from their parent regions.
            regions_above = brain_onthology.get_regions_above(reg)
            for region in regions_above:
                row = hemi+": "+region
                # Subtract the counting results from the parent region.
                # Use fill_value=0 to prevent "3-NaN=NaN".
                if row in self.data.index:
                    if exclude_parent_regions and reg not in layer1:
                        self.data.drop(row, inplace=True)
                    elif reg_hemi in self.data.index:
                        self.data.loc[row] = self.data.loc[row].subtract(self.data.loc[reg_hemi], fill_value=0)

            # Step 2: Remove the regions that should be excluded
            # together with their daughter regions.
            subregions = brain_onthology.list_all_subregions(reg)
            for subreg in subregions:
                row = hemi+": "+subreg
                if row in self.data.index:
                    self.data.drop(row, inplace=True)
    
    def _area_µm2_to_mm2_(self) -> None:
        self.data.area = self.data.area * 1e-06

    @staticmethod
    def merge_hemispheres(brain_slice: Self) -> Self:
        '''
        Function takes as input a BrainSlice. Each row represents a left/right part of a region.
        
        The output is a dataframe with each column being the sum of the two hemispheres
        '''
        if not brain_slice.is_split:
            return brain_slice
        slice = copy.copy(brain_slice)
        corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in slice.data.index]
        slice.data = slice.data.groupby(corresponding_region).sum(min_count=1)
        markers = [c for c in slice.data.columns if c != "area"]
        slice.markers_density = BrainSlice._get_marker_density(slice.data, markers)
        slice.is_split = False
        return slice