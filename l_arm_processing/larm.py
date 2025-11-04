"""
larm.py

A collection of functions for loading, processing, and manipulating l-arm data from Contrasts

@author:
    David Clemens-Sewall, WHOI

"""

import os
import re
import warnings

import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

def read_spectrum_file(filepath):
    """
    Reads a spectrum file from MSDA_XE format and puts it into a long pandas dataframe
    
    Parameters
    ----------
    filepath: str
        Path object for the spectrum file to read in
    
    Returns
    -------
    Pandas DataFrame with the relevant data from the file in long format
    
    """
    
    # Read file into a list with each line as an entry
    with open(filepath, 'r') as f:
        ls_file = f.readlines()
        
    if ls_file[0]!='[Spectrum]\n':
        raise RuntimeError('Not a spectrum file: ' + filepath)

    # Pull the attributes into a dictionary
    att_dict = {}
    for i in range(len(ls_file)):
        if ls_file[i] in ['[Spectrum]\n', '[Attributes]\n', '[END] of [Attributes]\n']:
            continue
        elif ls_file[i] == '[DATA]\n':
            break
        else:
            att_dict[ls_file[i].split('=')[0].strip()] = ls_file[i].split('=')[1].strip()
            
    # If there are no comments, return None
    if (att_dict['Comment'] == '') & (att_dict['CommentSub1'] == '') & (att_dict['CommentSub2'] == ''):
        return None
    
    # Fix typos in attributes
    if att_dict['CommentSub2'] == 'az990':
        att_dict['CommentSub2'] = 'az90'

    # Create column names for data
    col_names = [att_dict['Unit1'].split(maxsplit=2)[2].replace(' ', '_').lower(),
                 att_dict['Unit2'].split(maxsplit=2)[2].replace(' ', '_'),
                 att_dict['Unit3'].split(maxsplit=2)[2].replace(' ', '_'),
                 att_dict['Unit4'].split(maxsplit=2)[2].replace(' ', '_'),]

    # Read data in with pandas
    # Note that we skip the first row which is just the integration time
    df_in = pd.read_csv(filepath, sep='\s+', header=None, names=col_names, skiprows=i+2, nrows=len(ls_file)-i-4,
                       na_values='-NAN')

    if (att_dict['IDDevice'] in ['SAM_881D', 'SAM_8333']
       ) | ((att_dict['IDDevice'] == 'SAM_881E') and (att_dict['DateTime'][:10] == '2025-08-04')):
        if (bool(re.search(r"AL", att_dict['CommentSub3']))
            | ((att_dict['Comment'] == 'station1') & (att_dict['CommentSub1'] == 'coring') 
               & (pd.Timestamp(att_dict['DateTime']) > pd.Timestamp('2025-07-10 11:21:00'))
               & (pd.Timestamp(att_dict['DateTime']) < pd.Timestamp('2025-07-10 23:59:00')))
            | ((att_dict['Comment'] == 'station1') & (att_dict['CommentSub1'] == 'ocn_pond') 
               & (pd.Timestamp(att_dict['DateTime']) > pd.Timestamp('2025-07-10 15:43:00')))):
            df_in.rename(columns={col_names[1]: col_names[1].replace('Intensity', 'Reflected'),
                                  col_names[2]: 'Reflected_' + col_names[2],
                                  col_names[3]: 'Reflected_' + col_names[3]}, inplace=True)
        else:
            df_in.rename(columns={col_names[1]: col_names[1].replace('Intensity', 'Transmitted'),
                                  col_names[2]: 'Transmitted_' + col_names[2],
                                  col_names[3]: 'Transmitted_' + col_names[3]}, inplace=True)
    elif att_dict['IDDevice'] in ['SAM_881E', 'SAM_8182']:
        df_in.rename(columns={col_names[1]: col_names[1].replace('Intensity', 'Incident'),
                              col_names[2]: 'Incident_' + col_names[2],
                              col_names[3]: 'Incident_' + col_names[3]}, inplace=True)

    # Convert to long form
    df_long = df_in.melt(id_vars='wavelength_nm')

    # Add pressure and inclination data if present
    if att_dict['IDDevice'] in ['SAM_881D', 'SAM_8333']:
        inc_press = np.zeros(4)*np.NaN
        if att_dict['InclValid']=='1':
            inc_press[0] = float(att_dict['InclV'])
            inc_press[1] = float(att_dict['InclX'])
            inc_press[2] = float(att_dict['InclY'])
        if att_dict['PressValid']=='1':
            inc_press[3] = float(att_dict['Pressure'])
        df_long = pd.concat([df_long, 
                             pd.DataFrame(data={'wavelength_nm': [np.NaN, np.NaN, np.NaN, np.NaN],
                                          'variable': ['InclV', 'InclX', 'InclY', 'Pressure'],
                                          'value': inc_press})])

    # Add timestamp, station, location, azimuth, depth, and repetition columns
    df_long['timestamp_utc'] = pd.Timestamp(att_dict['DateTime'], tz='UTC')
    
    # Need to handle exceptions to pattern
    if att_dict['DateTime'][:10] == '2025-07-12':
        df_long['station'] = att_dict['CommentSub1'][-1]
        df_long['location'] = att_dict['CommentSub2']
        if att_dict['CommentSub3'][-2:] == 'AL':
            df_long['azimuth'] = att_dict['CommentSub3'][:-2]
        else:
            df_long['azimuth'] = att_dict['CommentSub3']
    elif (att_dict['DateTime'][:10] == '2025-07-16') and (att_dict['CommentSub1'] == 'SNpond'):
        df_long['station'] = att_dict['Comment'][-1]
        df_long['location'] = att_dict['CommentSub1']
        # Extract numbers from CommentSub2 for azimuth
        comment2_numbers = re.findall("(\d+)", att_dict['CommentSub2'])
        if len(comment2_numbers) != 1:
            df_long['azimuth'] = att_dict['CommentSub2']
        else:
            df_long['azimuth'] = comment2_numbers[0]
    else:
        df_long['station'] = att_dict['Comment'][-1]
        df_long['location'] = att_dict['CommentSub1']
        # Extract numbers from CommentSub2 for azimuth
        comment2_numbers = re.findall("(\d+)", att_dict['CommentSub2'])
        if len(comment2_numbers) > 1:
            raise RuntimeError('Comment 2 contains more than one set of numbers, may not be azimuth: '+filepath)
        elif len(comment2_numbers) == 1:
            df_long['azimuth'] = comment2_numbers[0]
        else:
            df_long['azimuth'] = att_dict['CommentSub2']
    
    df_long['type'] = 'U'
    df_long['depth'] = np.NaN
    df_long['repetition'] = int(0)

    # Set index and return
    return df_long.set_index(
        ['timestamp_utc', 'station', 'location', 'azimuth', 'type', 'depth', 'repetition', 'wavelength_nm', 'variable'])

def read_paired_spectra(filepath_pair):
    """
    Reads in a pair of coincident spectra and computes the albedo or transmittance
    
    Parameters
    ----------
    filepath_pair: tuple
        Pair of paths to coincident spectra
        
    Returns
    -------
    Pandas DataFrame containing the data from each spectra and a ratio spectrum (albedo or transmittance)
    
    """
    
    # Read in paired spectra
    fp_1, fp_2 = filepath_pair
    df_1 = read_spectrum_file(fp_1)
    df_2 = read_spectrum_file(fp_2)
    
    # If both are None, return None
    if (df_1 is None) | (df_2 is None):
        if (df_1 is None) & (df_2 is None):
            return None
        else:
            raise RuntimeError("Only one of the following is missing comments: " + fp_1 + fp_2)

    # The instruments don't actually measure exactly the same wavelengths... so we need to interpolate
    var_list = ["Incident_mW/(m^2_nm)", "Reflected_mW/(m^2_nm)", "Transmitted_mW/(m^2_nm)",
               "Incident_Status", "Reflected_Status", "Transmitted_Status"]
    wvs_interp = np.linspace(320, 950, 190)
    merge_dict = {}
    
    # Interpolate each spectrum and combine
    df_temp_1 = df_1.query('variable in @var_list & value.notna()').reset_index(level='variable').pivot(columns='variable', values='value')
    df_temp_1.dropna(inplace=True)
    for col_name in df_temp_1.columns:
        spl = CubicSpline(df_temp_1.index.get_level_values('wavelength_nm').values, df_temp_1[col_name].values)
        merge_dict[col_name] = spl(wvs_interp)
    df_temp_2 = df_2.query('variable in @var_list & value.notna()').reset_index(level='variable').pivot(columns='variable', values='value')
    df_temp_2.dropna(inplace=True)
    for col_name in df_temp_2.columns:
        spl = CubicSpline(df_temp_2.index.get_level_values('wavelength_nm').values, df_temp_2[col_name].values)
        merge_dict[col_name] = spl(wvs_interp)
    df_merge = pd.DataFrame(data=merge_dict, index=wvs_interp)
    df_merge.index.name = 'wavelength_nm'

    # Compute ratio spectrum and set type in original dataframes
    if "Reflected_mW/(m^2_nm)" in df_merge.columns:
        df_merge['Albedo'] = df_merge["Reflected_mW/(m^2_nm)"] / df_merge["Incident_mW/(m^2_nm)"]
        df_merge['Albedo_Status'] = df_merge['Reflected_Status'] + df_merge['Incident_Status']
        df_merge.drop(columns=["Reflected_mW/(m^2_nm)", "Incident_mW/(m^2_nm)", 'Reflected_Status', 'Incident_Status'],
                     inplace=True)
        df_1.index = df_1.index.set_levels(['A']*df_1.size, level='type', verify_integrity=False)
        df_2.index = df_2.index.set_levels(['A']*df_2.size, level='type', verify_integrity=False)
    elif "Transmitted_mW/(m^2_nm)" in df_merge.columns:
        df_merge['Transmittance'] = df_merge["Transmitted_mW/(m^2_nm)"] / df_merge["Incident_mW/(m^2_nm)"]
        df_merge['Transmittance_Status'] = df_merge['Transmitted_Status'] + df_merge['Incident_Status']
        df_merge.drop(columns=["Transmitted_mW/(m^2_nm)", "Incident_mW/(m^2_nm)", 'Transmitted_Status', 'Incident_Status'],
                     inplace=True)
        df_1.index = df_1.index.set_levels(['T']*df_1.size, level='type', verify_integrity=False)
        df_2.index = df_2.index.set_levels(['T']*df_2.size, level='type', verify_integrity=False)
    else:
        warnings.warn('Missing Transmitted or Reflected from ' + fp_1 + fp_2)
        return None

    # Add values for the first six elements of index
    for i in range(7):
        name = df_1.index.names[i]
        val = df_1.index.get_level_values(name).unique()
        if len(val) != 1:
            raise RuntimeError('Nonunique time, site, location, azimuth, type, depth index for: '+fp_1)
        else:
            df_merge[name] = val[0]

    # Return to long format
    df_merge = df_merge.reset_index().melt(id_vars=
        ['timestamp_utc', 'station', 'location', 'azimuth', 'type', 'depth', 'repetition', 'wavelength_nm']
        ).set_index(['timestamp_utc', 'station', 'location', 'azimuth', 'type', 'depth', 'repetition', 'wavelength_nm', 'variable'])

    # Combine all data
    return pd.concat([df_1, df_2, df_merge])

def convert_00_to_a1(data_path, project_name, matching='filename'):
    """
    Converts all of the raw data (00) in a location directory into a1.
    
    Parameters
    ----------
    data_path: str or path-like
        Path to directory where all the project directories are stored.
    project_name: str
        Name of the project to import (also the directory name)
    matching: str, optional
        How to match spectra, either 'filename' or 'time'. The default is 'filename'.
    
    Returns
    -------
    Pandas DataFrame with all data formatted into a1 long format
    
    """
    
    # Identify measurement pairs
    data_level = '00'
    ls_filenames = os.listdir(os.path.join(data_path, project_name, data_level))
    ls_lonely = []
    ls_pairs = []
    if matching == 'filename':
        for filename in ls_filenames:
            if re.search('SPECTRUM_CALIBRATED', filename):
                lonely = True
                for i in range(len(ls_lonely)):
                    other_filename = ls_lonely[i]
                    if filename[:38]==other_filename[:38]:
                        ls_pairs.append((os.path.join(data_path, project_name, data_level, filename),
                                         os.path.join(data_path, project_name, data_level, ls_lonely.pop(i))))
                        lonely = False
                        break
                if lonely:
                    ls_lonely.append(filename)
    elif matching == 'time':
        for filename in ls_filenames:
            if re.search('SPECTRUM_CALIBRATED', filename):
                lonely = True
                with open(os.path.join(data_path, project_name, data_level, filename), 'r') as f:
                    ls_file = f.readlines()
                for line in ls_file:
                    if line[:8] == 'DateTime':
                        timestamp = line.split('=')[1].strip()
                        break
                for i in range(len(ls_lonely)):
                    other_timestamp = ls_lonely[i][0]
                    if timestamp==other_timestamp:
                        ls_pairs.append((os.path.join(data_path, project_name, data_level, filename),
                                         os.path.join(data_path, project_name, data_level, ls_lonely.pop(i)[1])))
                        lonely = False
                        break
                if lonely:
                    ls_lonely.append((timestamp, filename))


    # Load each dataset and set the repetition correctly
    ls_df = []
    first = True
    for i in range(len(ls_pairs)):
        ls_df.append(read_paired_spectra(ls_pairs[i]))
        
        if (ls_df[i] is not None):
            # On the first one just copy the multiindex.
            if first:
                ls_df[i].index = ls_df[i].index.set_levels(np.ones(ls_df[i].size, dtype=np.int64), level='repetition',
                                                          verify_integrity=False)
                ind = ls_df[i].index.droplevel(['timestamp_utc', 'wavelength_nm', 'variable']).unique()
                first = False
            else:
                # Check if we have prior repetitions at this site
                new_ind = ls_df[i].index.droplevel(['timestamp_utc', 'wavelength_nm', 'variable', 'repetition']).unique()
                ind_mask = ind.droplevel('repetition').isin(new_ind) # boolean array of matching indices
                if ind_mask.any():
                    rep = ind[ind_mask].get_level_values('repetition').max() + 1
                else:
                    rep = 1
                # Update repetition values in new dataframe
                ls_df[i].index = ls_df[i].index.set_levels(rep*np.ones(ls_df[i].size, dtype=np.int64), level='repetition',
                                                          verify_integrity=False)
                # Add the new index to ind
                ind = ind.append(ls_df[i].index.droplevel(['timestamp_utc', 'wavelength_nm', 'variable']).unique())

    df_a1 = pd.concat(ls_df)
    
    return df_a1