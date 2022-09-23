### LOAD PACKAGES ###
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import os
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np
from tabulate import tabulate
# from IPython.display import display

### DATA LOAD FUNCTION ###
def data_load(folder: str, filename: str) -> pd.DataFrame:
    
    '''
    The function loads the data and returns an appropriate data structure (Pandas DataFrame).

    Parameters:
        folder (str): path of the folder where the data are found
        filename (str): name of the data file 
    
    Returns:
        df (Pandas DataFrame): data structure with loaded data
    '''
    
    if not isinstance(folder, str):
        error_message = "folder must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(filename, str):
        error_message = "filename must be specified as a string"
        raise TypeError(error_message)
    
    else:
        filepath = os.path.join(folder, filename)
            
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.txt'):
            df = pd.read_table(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            print(f"Error in {filename}")
            print("File extension is not correct or not supported.")
            print("Please specify a one of the following: xlsx, csv, txt, json.")
        
        # drop columns with no title
        df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

        return df

### DATA INFO FUNCTION ###
def data_info(df: pd.DataFrame, threshold: int=20) -> list:
    
    """
    The function creates and displays a report containing the following info:
        1) Name of the data file
        2) DataFrame dimensions (row, columns)
        3) Categorical features (if present) and their unique values
        4) Continuous features (if present) and their descriptive statistics
        5) Missing data (if present)
        6) Duplicate values (if present)
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        threshold (int): a feature having more unique values than the threshold is considered as continuous
    
    Returns:
        identifier (list): list with identifier features
        categorical (list): list of categorical features in the DataFrame
        continuous (list): list of continuous features in the DataFrame    
    """

    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)

    elif not isinstance(threshold, int):
        error_message = "threshold must be specified as an integer number"
        raise TypeError(error_message)

    else:    
        # Number of entries and features
        
        entry_num = len(df)
        feature_num = len(df.columns)
        
        print("DIMENSIONS:")
        print(100*"-")
        print(f"Entries: {entry_num}")
        print(f"Features: {feature_num}")
        print(100*"-")
        print("\n")
            
        # Categorical and Continuous features
        
        categorical = []
        continuous = []
        identifier = []

        unique_values = []
        continuous_statistics = []

        columns = list(df.columns)

        for column in columns:
            if (len(df.loc[:,column].unique()) == entry_num) and (df[column].dtype == "object"):
                identifier.append(column)
            elif len(df.loc[:,column].unique()) >= threshold:
                continuous.append(column)
            else:
                categorical.append(column)
        
        for column in categorical:
            unique_values.append([column, df.dtypes[column], df[column].value_counts().sort_index(ascending=True).to_dict()])
        
        # continuous_statistics = round(df.describe().T, 3)
        # continuous_statistics.insert(0, "Data Type", df.dtypes, True)

        for column in continuous:
            continuous_statistics.append(
                [
                    column,
                    df.dtypes[column],
                    round(len(df[column]), 3),
                    round(np.mean(df[column]), 3),
                    round(np.std(df[column]), 3),
                    round(np.min(df[column]), 3),
                    round(np.percentile(df[column], q=25), 3),
                    round(np.median(df[column]), 3),
                    round(np.percentile(df[column], q=75), 3),
                    round(np.max(df[column]), 3)
                    ]
                    )
        
        print(f"CATEGORICAL FEATURES:")
        print(100*"-")
        
        if len(categorical)>=1:
            print(tabulate(unique_values, headers=["Features", "Data Type", "Categories & Counts"]))
        else:
            print("There are no categorical features in the dataset.")
        
        print(100*"-")
        print("\n")
        
        print(f"CONTINUOUS FEATURES:")
        print(100*"-")
        # display(continuous_statistics)
        print(tabulate(continuous_statistics, headers=["Features", "Data Type", "Count", "Mean", "Std", "Min", "25th", "Median", "75th", "Max"]))
        print(100*"-")
        print("\n")

        # Missing data
        
        missing_data = df.isnull().sum()
        missing_data_dict = dict(missing_data)
        with_na = []
        without_na = []
        
        print("MISSING DATA:")
        print(100*"-")
    
        for key, val in missing_data_dict.items():
            if val>0:
                with_na.append([key, val])
            elif val==0:
                without_na.append([key, val])
        
        if len(with_na)>=1:
            print(f"The following features have missing values:")
            print(tabulate(with_na, headers=["Features", "Missing Data"]))
            print(100*"-")
            print("\n")
            
        if len(without_na)==feature_num:
            print("There are no missing values in the dataset.")
            print(100*"-")
            print("\n")
        elif len(without_na)>=1:
            print(f"The following features do not have missing values:")
            print(tabulate(without_na, headers=["Features", "Missing Data"]))
            print(100*"-")
            print("\n")

        # Duplicate values

        print("DUPLICATE VALUES:")
        print(100*"-")

        if df.duplicated().sum() == 0:
            print("There are no duplicate entries in the dataset.")
        elif df.duplicated().sum() > 0:
            print("There following entries are duplicated.")
            print(print(df[df.duplicated()==True]))
        
        print(100*"-")
        
        return identifier, categorical, continuous