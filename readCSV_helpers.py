# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from brain_hierarchy import AllenBrainHierarchy

import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from itertools import product

#%%
def get_image_names_in_folder(path):
    '''
    Returns a list of all files in the directory
    that have a '.txt' extension in them. It removes '_LEFT' and '_RIGHT' from the names.
    '''
    all_files = os.listdir(path)
    # Filter txt files, and remove the '_regions.txt' extension
    txt_files = [f.replace('_regions.txt', '') for f in all_files if '_regions.txt' in f]
    # Get rid of '_LEFT' and '_RIGHT':
    files = []
    for f in txt_files:
        if '_LEFT' in f:
            files.append(f.replace('_LEFT', ''))
        elif '_RIGHT' in f:
            files.append(f.replace('_RIGHT', ''))
        else:
            files.append(f)
            
    # Remove doubles and sort:
    files = list(dict.fromkeys(files))
    files.sort()
    
    return files

#%%
def remove_hemisphere(data, hemisphere):
    '''
    This function removes all regions specific to
    a certain hemisphere (either 'Left' or 'Right')
    from a dataframe.
    
    Inputs
    ------
    data (pandas dataframe)
    A dataframe with the counting results of a slice.
    
    hemisphere (string)
    Choose 'Left' or 'Right'.
    '''
    
    if not(hemisphere=='Left' or hemisphere=='Right'):
        raise ValueError('Hemisphere should be either "Left" or "Right"!')
    
    curr_regs = data.index.tolist()
    for region in curr_regs:
        if hemisphere in region:
            data = data.drop(region, axis=0)
    
    return data

#%% 
def import_txt_file_as_dataframe(path_to_txt, hemisphere):
    '''
    This function reads a txt file into a pandas dataframe.
    It does some additional processing steps to make the handling
    of the data easier in the next steps. These steps are:
    - Create a class name for the Root region called ROOT.
    - Replace NaN values with 0.
    - Convert the Class column to the index of the dataframe.
    '''
    data = pd.read_table(path_to_txt).drop_duplicates()
    img_name = data.loc[0,'Image Name']
    (data['Num Detections'].count() == 0) and print('The txt file for '+img_name+' only contains NaNs!')
    
    # There is one region (the full slice) called 'Root', which has the class 'NaN'
    # associated to it. Replace this NaN with the more descriptive class root.
    data['Class'] = data['Class'].fillna('wholeroot')

    # Set the region classes as the column index
    data = data.set_index('Class')
    
    # Now remove the 'wholeroot'. We'll use the seperate hemispheres.
    data = data.drop('wholeroot', axis=0)
    
    # If a hemisphere is specified, remove the other hemisphere from the dataframe
    if (hemisphere == 'Left'):
        data = remove_hemisphere(data, 'Right')
    elif (hemisphere == 'Right'):
        data = remove_hemisphere(data, 'Left')
    
    return data,img_name

#%%
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

#%%
def list_regions_to_exclude_from_file(path_to_exclusion_file):
    '''
    Read the csv file containing the regions to exclude for each image,
    and summarize the information in a dictionary.
    The exclusions_file is to be initialized with initExclusionFile
    '''
    exclude_df = pd.read_csv(path_to_exclusion_file, sep=r'[;,]', index_col='Image Name', engine='python')
    exclude_dict = {}

    for img in exclude_df.index:
        
        exclude_dict[img] = []
        to_exclude = exclude_df['Regions to Exclude (Regions may not overlap!)'].loc[img]
        if type(to_exclude) == str: # if there are regions to exclude
            for region in to_exclude.split('/ '):
                
                # If no hemisphere was specified, add both hemispheres:
                if not 'Right' in region and not 'Left' in region:
                    exclude_dict[img].append('Right: '+region)
                    exclude_dict[img].append('Left: '+region)
                # If left or right hemisphere was specified:
                else:
                    exclude_dict[img].append(region)
                
    return exclude_dict

#%%
def list_regions_to_exclude(path_to_animal):
    '''
    Read the txt files containing the regions to exclude for each image,
    and summarize the information in a dictionary.
    '''
    path_to_exlusion_files = os.path.join(path_to_animal, 'regions_to_exclude')
    if not(os.path.exists(path_to_exlusion_files)):
        print('WARNING: did not find regions to exclude in '+path_to_animal)
        return {}

    all_files = os.listdir(path_to_exlusion_files)
    # Filter txt files
    txt_files = [f for f in all_files if '.txt' in f]
    
    exclude_dict = {}
    for file_name in txt_files:
        # Extract the image name from the txt file name 
        # (just remove the _to_exclude extension
        img_name = file_name.replace('_to_exclude', '')
        exclude_dict[img_name] = []
        path_to_exclude_file = os.path.join(path_to_exlusion_files, file_name)
        with open(path_to_exclude_file) as file:
            lines = file.readlines()
        
        # Remove '\n' from all lines
        lines = [l.replace('\n', '') for l in lines]

        # Add regions to exclude_dict
        for region_to_exclude in lines:
            exclude_dict[img_name].append(region_to_exclude) 
    
    return exclude_dict
    
#%%
def exclude_regions(df, regs_to_exclude, AllenBrain):
    '''
    Take care of regions to be excluded from the analysis.
    If a region is to be excluded, 2 things must happen:
    (1) The cell counts of that region must be subtracted from all
        its parent regions.
    (2) The region must disappear from the data, together with all 
         its daughter regions.
    '''

    for reg_hemi in regs_to_exclude:
        if ": " not in reg_hemi:
            raise SyntaxError("A region_to_exclude file is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'")
        hemi = reg_hemi.split(': ')[0]
        reg = reg_hemi.split(': ')[1]

        # Step 1: subtract counting results of the regions to be excluded
        # from their parent regions.
        regions_above = AllenBrain.get_regions_above(reg)
        for region in regions_above:
            row = hemi+': '+region
            # Subtract the counting results from the parent region.
            # Use fill_value=0 to prevent "3-NaN=NaN".
            if row in df.index and reg_hemi in df.index:
                df.loc[row] = df.loc[row].subtract(df.loc[reg_hemi], fill_value=0 )

        # Step 2: Remove the regions that should be excluded
        # together with their daughter regions.
        subregions = AllenBrain.list_all_subregions(reg)
        for subreg in subregions:
            row = hemi+': '+subreg
            if row in df.index:
                df = df.drop(row)
            
    return df

#%%
def load_cell_counts(root, exclude_dict, AllenBrain, area_key, tracer_key, marker_key):
    '''
    Function to load cell counts, stored in .csv files in the 'root' directory,
    as Pandas dataframes.
    '''
    
    # Get the image names present in root (e.g. "Image_01.vsi - 10x_01")
    # and the names of all files present in root (e.g. "Image_01.vsi - 10x_01 LEFT_regions.txt")
    img_names = get_image_names_in_folder(root)
    file_names = os.listdir(root)
    
    # Init dicts and lists to store data
    df_list = []         # list of all slice dataframes
    
    # Loop through the image names
    for f in img_names:
        # The following variables will be used to find out whether we have seperate files
        # for seperate hemispheres, or just one file containing both hemispheres.
        fname = f + '_regions.txt'
        fname_left = f + '_LEFT' + '_regions.txt'
        fname_right = f + '_RIGHT' + '_regions.txt'
        both_hemi = False
        right_hemi = False
        left_hemi = False
        regs_to_exclude = []

        # Read text file into a Pandas dataframe
        if fname_left in file_names: # if we have img_name LEFT_regions.txt in folder
            left_hemi = True
            path = os.path.join(root, fname_left)
            data_left,img_name_left = import_txt_file_as_dataframe(path, 'Left')
            if fname_left in exclude_dict.keys():
                regs_to_exclude = regs_to_exclude + exclude_dict[fname_left]
        if fname_right in file_names: # if we have img_name RIGHT_regions.txt in folder
            right_hemi = True
            path = os.path.join(root, fname_right)
            data_right,img_name_right = import_txt_file_as_dataframe(path, 'Right')
            if fname_right in exclude_dict.keys():
                regs_to_exclude = regs_to_exclude + exclude_dict[fname_right]
        if fname in file_names:       # if we have img_name_regions.txt (no hemisphere specification)
            both_hemi = True
            path = os.path.join(root, fname)

            '''
            In some cases, due to problems in the slices Qpath would generate some blank txt files (with size 0B)
            I wanted the code to run even if such files were present but to print an error message in the Terminal with the directory of the file,
            so that we could check the slice in Qpath and make sure it was an unusable slice, and not an error in the Qpath processing code.
            I've thus added this try except else
            '''
            try:
                data,img_name = import_txt_file_as_dataframe(path, 'Both')
            except:
                if os.stat(path).st_size == 0:
                    print('EMPTY FILE:'+ fname+ '\n \t Directory: '+ path)
                else:
                    print('!!!!!ERROR IN FILE: '+ path)
            else:
                if fname in exclude_dict.keys():
                    regs_to_exclude = regs_to_exclude + exclude_dict[fname]

                # Check for safety: we either have ONE file for both hemispheres,
                # or (max 2) file(s) for seperate hemispheres. Else, raise and error.
                if (left_hemi and both_hemi) or (right_hemi and both_hemi):
                    raise ValueError('Either LEFT and/or RIGHT, or no hemisphere specification. But not both!')
                if not(left_hemi) and not(right_hemi) and not(both_hemi):
                    raise ValueError('Filename not found!')

                # Combine left and right, if they were both present
                if left_hemi and right_hemi:        # if we have both left and right, combine dataframes
                    data = pd.concat([data_left, data_right])
                elif left_hemi and not(right_hemi): # if we have only left, data = data_left
                    data = data_left
                elif not(left_hemi) and right_hemi: # if we have only right, data = data_right
                    data = data_right

                # Combine cell counts
                df = pd.DataFrame(data, columns=[area_key, tracer_key])
                df.rename(columns={area_key: 'area', tracer_key: marker_key}, inplace=True)
                #@assert (df.area > 0).all()
                df = df[df['area'] > 0]
                
                # Take care of regions to be excluded
                regs_to_exclude = list(dict.fromkeys(regs_to_exclude))
                df = exclude_regions(df, regs_to_exclude, AllenBrain)
                
                # Store results in dictionaries / lists
                df_list.append(df)
    
    return df_list

#%%
def merge_hemispheres(slice):
    '''
    Function takes as input a dataframe. Each row represents a left/right part of a region.
    
    The output is a dataframe with each column being the sum of the two hemispheres
    '''
    corresponding_region = [find_region_abbreviation(region) for region in slice.index]
    return slice.groupby(corresponding_region, axis=0).sum(min_count=1)

#%%
def normalize_cell_counts(brain_df, tracer):
    '''
    Do normalization of the cell counts for one tracer.
    The tracer can be any column name of brain_df, e.g. 'CFos'.
    The output will be a dataframe with three columns: 'Density', 'Percentage' and 'RelativeDensity'.
    Each row is one of the original brain regions
    '''
    
    # Init dataframe
    columns = ['Density','Percentage','RelativeDensity']
    norm_cell_counts = pd.DataFrame(np.nan, index=brain_df.index, columns=columns)

    # Get the the brainwide area and cell counts (corresponding to the root)
    brainwide_area = brain_df["area"]["root"]
    brainwide_cell_counts = brain_df[tracer]["root"]
    
    # Do the normalization for each column seperately.
    norm_cell_counts['Density'] = brain_df[tracer] / brain_df["area"]
    norm_cell_counts['Percentage'] = brain_df[tracer] / brainwide_cell_counts 
    norm_cell_counts['RelativeDensity'] = (brain_df[tracer] / brain_df["area"]) / (brainwide_cell_counts / brainwide_area)
    
    return norm_cell_counts

#%%
def save_results(results_df, output_path, filename):
    if not(os.path.exists(output_path)):
        os.mkdir(output_path)
        print('\nCreated a new results_python folder '+output_path+' \n')
    else:
        print('\n! A results_python folder already existed in root. I am overwriting previous results!\n')

    # results_df.to_csv( os.path.join(output_path, filename) )
    results_df.to_csv(os.path.join(output_path, filename), sep='\t', mode='w')

    print('Results are saved in '+output_path)
    print('\nDone!')
    return True

def read_group_slices(animal_root: str, animal_dirs: list[str], AllenBrain: AllenBrainHierarchy, \
                        area_key: str, tracer_key: str, marker_key) -> list[list[pd.DataFrame]]:
    animals_slices_paths = [os.path.join(animal_root, animal, 'results') for animal in animal_dirs]
    animals_excluded_regions = [list_regions_to_exclude(os.path.join(animal_root, animal)) for animal in animal_dirs]
    # load_cell_counts() -> list[pd.DataFrame]
    return [load_cell_counts(input_path, exluded_regions, AllenBrain, area_key, tracer_key, marker_key) for (input_path, exluded_regions) in zip(animals_slices_paths, animals_excluded_regions)]

def area_µm2_to_mm2(group) -> None:
    for slices in group:
        for slice in slices:
            slice.area = slice.area * 1e-06

# for each brain region, aggregate marker counts from all the animal's slices into one value.
# aggregation methods: sum, std, coefficient of variation
def sum_cell_counts(slices: list[pd.DataFrame]) -> pd.DataFrame:# methods: Callable[[int, int], int]):
    slices_df = pd.concat(slices)
    slices_df = slices_df.groupby(slices_df.index, axis=0).sum()
    return slices_df

def animal_cell_density(slices: list[pd.DataFrame], marker_key: str) -> pd.Series:
    slices_marker_densities = [slice[marker_key] / slice["area"] for slice in slices]
    return pd.concat(slices_marker_densities)

# https://en.wikipedia.org/wiki/Coefficient_of_variation
def coefficient_variation(x) -> np.float64:
    avg = x.mean()
    if len(x) > 1 and avg != 0:
        return x.std(ddof=1) / avg
    else:
        return 0

def reduce_brain_densities(slices: list[pd.DataFrame], marker_key: str, mode, hemisphere_distinction=False) -> pd.Series:
    match mode:
        case "mean" | "avg":
            reduction_fun = np.mean
        case "std":
            reduction_fun = np.std
        case "variation" | "cvar":
            reduction_fun = coefficient_variation
        case _:
            raise NameError("Invalid mode selected.")
    if not hemisphere_distinction:
        slices = [merge_hemispheres(slice) for slice in slices]
    marker_densities = animal_cell_density(slices, marker_key)
    reduction_per_region = marker_densities.groupby(marker_densities.index, axis=0).apply(reduction_fun)
    return reduction_per_region

def write_brains(root_output_path: str, animal_names: list[str], animal_brains: list[pd.DataFrame]) -> None:
    assert len(animal_names) == len(animal_brains),\
        f"The number of animals read and analysed ({len(animal_brains)}) differs from the numner of animals in the input group ({len(animal_names)})"
    for i in range(len(animal_names)):
        brain = animal_brains[i]
        name = animal_names[i]
        output_path = os.path.join(root_output_path, animal_names[i])
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, name+'_summed.csv')
        brain.to_csv(output_path, sep='\t', mode='w')
        print(f'Raw summed cell counts are saved to {output_path}')

def analyze(animal_names: list[str], animal_brains: list[pd.DataFrame], marker_key: str, AllenBrain: AllenBrainHierarchy) -> pd.DataFrame:
    brain = pd.concat({name: normalize_cell_counts(brain, marker_key) for name,brain in zip(animal_names, animal_brains)})
    brain = pd.concat({marker_key: brain}, axis=1)
    brain = brain.reorder_levels([1,0], axis=0)
    ordered_indices = product(AllenBrain.brain_region_dict.keys(), animal_names)
    brain = brain.reindex(ordered_indices, fill_value=np.nan)
    return brain

def plot_cv_above_threshold(brains_CV, brains_name, marker_key, cv_threshold=1) -> go.Figure: 
    fig = go.Figure()
    for i,cv in enumerate(brains_CV):
        above_threshold_filter = cv > cv_threshold
        # Scatterplot (animals)
        fig.add_trace(go.Scatter(
                            mode = 'markers',
                            y = cv[above_threshold_filter],
                            x = [i]*above_threshold_filter.sum(),
                            text = cv.index[above_threshold_filter],
                            opacity=0.7,
                            marker=dict(
                                size=7,
                                line=dict(
                                    color='rgb(0,0,0)',
                                    width=1
                                )
                            ),
                            showlegend=False
                    )
        )

    fig.update_layout(
        title = f"Coefficient of variaton of {marker_key} across brain slices > {cv_threshold}",
        
        xaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(0,len(brains_name)),
            ticktext = brains_name
        ),
        yaxis=dict(
            title = "Brain regions' CV"
        ),
        width=700, height=500
    )
    return fig