# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import numpy as np
import copy

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
def filter_uppercase_characters(string):
    '''
    This function returns all uppercase characters in the input string.
    Example: 'ACAd' returns 'ACA'.
    '''
    uppercase = [char for char in string if char.isupper()]
    return ''.join(uppercase)

#%%
def find_regions_and_classes_in_slice(data):
    '''
    This function reads a dataframe that corresponds to a brain slice,
    and returns a dictionary with the names of the classes appearing
    in the slice as keys, and the full region names as the corresponding value.
    Example: 
    region_dict['ACAd'] = 'Anterior cingulate area, dorsal part' 
    '''
    # Get the number of rows, columns in dataframe
    num_regions, num_measurements = data.shape
    # Initialize region_dict
    region_dict = {}
    # Loop through all regions
    for region_class in data.index.tolist():
        # Put the region class name (abbreviation)
        # as key in the dictionary, and the full region name as corresponding value.
        region_name = data.loc[region_class,'Name']
        region_class = find_region_abbreviation(region_class)
        region_dict[region_class] = region_name
        
    return region_dict

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
        hemi = reg_hemi.split(': ')[0]
        reg = reg_hemi.split(': ')[1]

        # Step 1: subtract counting results of the regions to be excluded
        # from their parent regions.
        child = reg
        while True:
            parent = AllenBrain.edges_dict[child]
            row = hemi+': '+parent
            # Subtract the counting results from the parent region.
            # Use fill_value=0 to prevent "3-NaN=NaN".
            if row in df.index and reg_hemi in df.index:
                df.loc[row] = df.loc[row].subtract( df.loc[reg_hemi], fill_value=0 )
            if (parent == 'root'):
                break
            child = parent # go one step further in the tree

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
    slice_regions = {}   # which regions do we have per slice?
    slice_data = {}      # what are the cell counts per slice?
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

                # Find regions in current slice
                region_dict = find_regions_and_classes_in_slice(data)

                # Combine cell counts
                df = pd.DataFrame(data, columns=[area_key, tracer_key])
                df.rename(columns={area_key: 'area', tracer_key: marker_key}, inplace=True)
                #@assert (df.area > 0).all()
                df = df[df['area'] > 0]
                
                # Take care of regions to be excluded
                regs_to_exclude = list(dict.fromkeys(regs_to_exclude))
                df = exclude_regions(df, regs_to_exclude, AllenBrain)
                
                # Store results in dictionaries / lists
                slice_regions[f] = region_dict
                slice_data[f] = df
                df_list.append(df)
    
    return df_list,slice_regions,slice_data

#%%
def init_dict(key_list, init_value):
    '''
    This function initializes a dictionary with keys as specified in key_list,
    and corresponding values in init_value.
    
    key_list: list with keys
    init_value: value to initialize with
    '''
    output_dict = {}
    for key in key_list:
        output_dict[key] = init_value
    
    return output_dict

#%%
def sort_hemispheres(data):
    '''
    Function takes as input a dataframe with only one column (the data to plot).
    The rows represent regions, with left and right seperated.
    
    The output is a dataframe with three columns:
    'Left' for each region, 'Right' for each region and the sum of the two.
    '''
    
    all_regions = data.index.to_list()
    present_regions = [find_region_abbreviation(r) for r in all_regions]
    present_regions = list(dict.fromkeys(present_regions)) # remove doubles
    
    # Put normalized counts of seperate hemispheres in seperate columns
    data_sorted = pd.DataFrame(np.nan, index=present_regions, columns=['Left', 'Right', 'Sum'])
    for region in present_regions:
        if 'Left: ' + region in all_regions:
            data_sorted['Left'][region] = data['Left: ' + region]
        if 'Right: ' + region in all_regions:
            data_sorted['Right'][region] = data['Right: ' + region]

    # Sum of left and right
    data_sorted['Sum'] = data_sorted[['Left','Right']].sum(axis=1, min_count=1)
    data_sorted = data_sorted.drop(columns = ['Left', 'Right'])
    
    return data_sorted.rename(columns={'Sum':'cell counts'})

# TODO: check if the normalization is correct
#%%
def normalize_cell_counts(brain_df, tracer):
    '''
    Do normalization of the cell counts for one tracer.
    The tracer can be any column name of brain_df, e.g. 'TVA', 'RAB', 'CTB_RAB'.
    The output will be a dataframe with three columns: 'Left', 'Right' and 'Sum'.
    The columns 'Left' and 'Right' are normalized w.r.t one hemisphere only.
    The 'Sum' column is normalized w.r.t the whole brain.
    '''
    
    # Sort the hemispheres (Get 'Left', 'Right' and 'sum' as seperate columns)
    area = sort_hemispheres(brain_df['area'])
    cell_counts = sort_hemispheres(brain_df[tracer])
    
    # Init dataframe
    columns = ['Density','Percentage','RelativeDensity']
    norm_cell_counts = pd.DataFrame(np.nan, index=cell_counts.index, columns=columns)

    # Get the the brainwide area and cell counts (corresponding to the root)
    brainwide_area = area.loc['root']
    brainwide_cell_counts = cell_counts.loc['root']
    
    # Do the normalization for each column seperately.
    norm_cell_counts['Density'] = cell_counts / area
    norm_cell_counts['Percentage'] = cell_counts / brainwide_cell_counts 
    norm_cell_counts['RelativeDensity'] = (cell_counts / area) / (brainwide_cell_counts / brainwide_area)
    
    return norm_cell_counts

#%%
def collect_and_analyze_cell_counts(root, animal_list,
                                    AllenBrain, area_key, 
                                    tracer_key, marker_key,
                                    output_path_root):

    normalizations = ['Density','Percentage','RelativeDensity']
    markers = [marker_key]

    # Load brain ontology (brain hierarchy) --------------------------------------
    brain_region_dict = AllenBrain.brain_region_dict

    # Initialize a results dataframe. --------------------------------------------
    # This is a dataframe with hierarchical columns. 
    # Hierarchy: tracer -> animal -> hemisphere.
    row_iterables = [brain_region_dict.keys(), animal_list]
    col_iterables = [markers, normalizations]
    row_multi_index = pd.MultiIndex.from_product(row_iterables)
    col_multi_index = pd.MultiIndex.from_product(col_iterables)
    results = pd.DataFrame(np.nan, index=row_multi_index, columns=col_multi_index)

    # Loop over animals, load the data and normalize counts --------------------
    for animal in animal_list:

        print(f'Importing slices in {animal}...')
        input_path = os.path.join(root, animal, 'results')

        # Load regions to exclude for this animal
        exclude_dict = list_regions_to_exclude(os.path.join(root, animal))

        # Load cell counts, excluding the regions we want to exclude
        df_list,slice_regions,slice_data = load_cell_counts(input_path, exclude_dict, AllenBrain, area_key, tracer_key, marker_key)
        print(f'Imported {str(len(df_list))} slices.')

        # Now comes the tricky part. We'll first concatenate the dataframes
        # of all slices into one big dataframe (brain_df).
        # Then, we combine the rows with the same index (=region name), and sum them.
        # That is, we sum the results (area, cell counts) per region across slices.
        brain_df = pd.concat(df_list)
        brain_df = brain_df.groupby(brain_df.index, axis=0).sum()

        # Save brain_df
        output_path = os.path.join(output_path_root, animal)
        os.makedirs(output_path, exist_ok=True)
        brain_df.to_csv(os.path.join(output_path, animal+'_cell_counts.txt'), sep='\t', mode='w')
                        
        print(f'Raw cell counts are saved to {output_path}\n')

        # Normalize the results
        for m in markers:

            # Normalize
            normalized_cell_counts = normalize_cell_counts(brain_df, m)

            # Save results per animal
            present_regions = normalized_cell_counts.index.to_list()
            for region in present_regions: # loop over all regions present

                for norm in normalizations: # loop over normalization methods
                    results.loc[(region, animal), (m, norm)] = normalized_cell_counts.loc[region, norm]
        
    return results

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