import pandas as pd
from glob import glob
import os
from configparser import ConfigParser
# TODOs:
# - use config file to sepcify the path

config = ConfigParser()
config.read('./config.ini')
RAW_DATA_PATH = config.get('Data', 'raw_data')
INTERIM_DATA_PATH = config.get('Data', 'interim_data')
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
def list_data_files(directory_path=RAW_DATA_PATH):
    """
    Lists all CSV files in the specified directory.

    Args:
        directory_path (str): Path to the directory containing data files. 
                            Defaults to '../data/raw/MetaMotion'

    Returns:
        list: A list of file paths for all CSV files in the directory
    """
    files = glob(f"{directory_path}/*.csv")
    return sorted(files)


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
def extract_features(file):
    """
    Extracts features from a given file name.
    File example: A-bench-heavy2_MetaWear_2019-01-14T14.27.00.784_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv
    The function parses the file name to extract the following information:
    - Participant ID
    - Activity label
    - Weight level (non-numeric characters from the weight level)
    - Data type
    - Sampling rate (in Hz)
    Args:
        file (str): The file path or file name to extract features from.
    Returns:
        tuple: A tuple containing:
            - participant (str): The participant ID.
            - label (str): The activity label.
            - weight_level (str): The weight level (non-numeric part).
            - data_type (str): The type of data.
            - sampling_rate (float): The sampling rate in Hz.
    """

    file = file.split("/")[-1]
    parts = file.split("_")
    participant = parts[0].split("-")[0]
    label = parts[0].split("-")[1]
    weight_level = "".join(filter(lambda x: not x.isdigit(), parts[0].split("-")[2]))
    data_type = parts[4]
    sampling_rate = float(parts[5].replace("Hz", ""))
    return participant, label, weight_level, data_type, sampling_rate


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

def combine_data_files(files):
    # A better way of combining data files:
    # Step-1: find the same set of accelerometer and gyroscope data
    # Step-2: merge one set of accelerometer and gyroscope data into a single dataframe
    #         - add a column to identify the set of data
    # Step-3: resample the data files
    # Step-4: combine all sets of data
    """
    Reads accelerometer and gyroscope data from CSV files and combines them into two dataframes.
    
    Args:
        files (list): List of file paths to process
        
    Returns:
        tuple: A tuple containing:
            - df_acc (pd.DataFrame): Combined accelerometer data
            - df_gyr (pd.DataFrame): Combined gyroscope data
    """
    acc_data = []
    gyr_data = []
    set_acc = set_gyr = 1 # create set number column as a identifier to easier implement visualization
    for file in files:
        participant, label, weight_level, data_type, sampling_rate = extract_features(file)
        
        df = pd.read_csv(file)
        df['participant'] = participant
        df['label'] = label
        df['weight_level'] = weight_level
        
        if data_type == 'Accelerometer':
            df['set'] = set_acc
            set_acc += 1
            acc_data.append(df)
        elif data_type == 'Gyroscope':
            df['set'] = set_gyr
            set_gyr += 1
            gyr_data.append(df)

    df_acc = pd.concat(acc_data, ignore_index=True)
    df_gyr = pd.concat(gyr_data, ignore_index=True)
    
    # rename columns
    df_acc.rename(columns={'x-axis (g)': 'acc_x', 'y-axis (g)': 'acc_y', 'z-axis (g)': 'acc_z'}, inplace=True)
    df_gyr.rename(columns={'x-axis (deg/s)': 'gyr_x', 'y-axis (deg/s)': 'gyr_y', 'z-axis (deg/s)': 'gyr_z'}, inplace=True)
    
    return df_acc, df_gyr


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
# Convert epoch timestamps to datetime and set as index for both dataframes
def process_datetime_index(df_acc, df_gyr):
    """
    Converts epoch timestamps to datetime index and cleans up timestamp-related columns.
    
    Args:
        df_acc (pd.DataFrame): Accelerometer dataframe
        df_gyr (pd.DataFrame): Gyroscope dataframe
        
    Returns:
        tuple: Processed (df_acc, df_gyr) with datetime index
    """
    # Convert epoch to datetime index
    df_acc.index = pd.to_datetime(df_acc['epoch (ms)'], unit='ms')
    df_gyr.index = pd.to_datetime(df_gyr['epoch (ms)'], unit='ms')
    
    # Set index name
    df_acc.index.name = df_gyr.index.name = 'timestamp'
    
    # Drop unnecessary time-related columns
    columns_to_drop = ['epoch (ms)', 'time (01:00)', 'elapsed (s)']
    df_acc = df_acc.drop(columns_to_drop, axis=1)
    df_gyr = df_gyr.drop(columns_to_drop, axis=1)
    
    return df_acc, df_gyr



# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
def merge_datasets(df_acc, df_gyr):
    """
    Merges accelerometer and gyroscope dataframes on timestamp index.
    
    Args:
        df_acc (pd.DataFrame): Accelerometer dataframe
        df_gyr (pd.DataFrame): Gyroscope dataframe
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    return pd.merge(df_acc.iloc[:,:3], df_gyr, on='timestamp', how='outer')

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
def resample_data(df_merge, freq='200ms'):
    """
    Resamples data to specified frequency, handling numeric and categorical columns separately.
    
    Args:
        df_merge (pd.DataFrame): Merged dataframe containing both accelerometer and gyroscope data
        freq (str): Resampling frequency (default: '200ms')
        
    Returns:
        pd.DataFrame: Resampled dataframe
    """
    # Define column types
    numeric_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    categorical_cols = ['participant', 'label', 'weight_level', 'set']

    # Get unique dates
    dates = pd.Series(df_merge.index.date).unique()

    # Process each day separately
    resampled_dfs = []
    for date in dates:
        # Get data for current date
        mask = df_merge.index.date == date
        df_day = df_merge[mask]
        
        # Resample numeric and categorical columns
        df_resampled_num = df_day[numeric_cols].resample(freq).mean()
        df_resampled_cat = df_day[categorical_cols].resample(freq).last()
        
        # Combine and clean
        df_day_resampled = pd.concat([df_resampled_num, df_resampled_cat], axis=1)
        df_day_resampled = df_day_resampled.dropna()
        
        resampled_dfs.append(df_day_resampled)

    # Combine all resampled dataframes
    df_resampled = pd.concat(resampled_dfs)
    # change set number to a integer variable
    df_resampled['set'] = df_resampled['set'].astype(int)
    return df_resampled

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
def export_dataset(df, output_path=INTERIM_DATA_PATH):
    """
    Exports a DataFrame to a pickle file.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        output_path (str): Path where the pickle file will be saved
    """
    file_name = '01_processed_data.pkl'
    df.to_pickle(os.path.join(output_path, file_name))
# --------------------------------------------------------------
# # Usage example:
# files = list_data_files(directory_path=RAW_DATA_PATH) # List all data files
# df_acc, df_gyr = combine_data_files(files) # Combine data files
# df_acc, df_gyr = process_datetime_index(df_acc, df_gyr) # Process datetime index
# df_merge = merge_datasets(df_acc, df_gyr) # Merge datasets
# df_resampled = resample_data(df_merge) # Resample data
# export_dataset(df_resampled, output_path=INTERIM_DATA_PATH) # Export the dataset
# # read the pickle file
# df = pd.read_pickle('../data/interim/01_processed_data.pkl')