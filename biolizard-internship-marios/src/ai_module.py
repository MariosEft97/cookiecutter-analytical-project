# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import os
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np
from tabulate import tabulate
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

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
    
    if not isinstance(folder, str) or not isinstance(filename, str):
        raise TypeError
            
    filepath = os.path.join(folder, filename)
    
    df = ""
    
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
    
    df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

    return df

### DATA INFO FUNCTION ###
def data_info(df: pd.DataFrame, filename: str, threshold: int=20) -> list:
    
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
        filename (str): name of the data file
        threshold (int): a feature having more unique values than the threshold is considered as continuous
    
    Returns:
        identifier (list): list with identifier columns
        categorical (list): list of categorical features in the DataFrame
        continuous (list): list of continuous features in the DataFrame    
    """

    if not isinstance(df, pd.DataFrame) or not isinstance(filename, str) or not isinstance(threshold, int):
        raise TypeError
    
    
    # Number of entries and features
    
    entry_num = len(df)
    feature_num = len(df.columns)
    
    print(f"DATA FILE:")
    print(100*"-")
    print(f"{filename}")
    print(100*"-")
    print("\n")
    print("DIMENSIONS:")
    print(100*"-")
    print(f"Entries: {entry_num}")
    print(f"Features: {feature_num}")
    print(100*"-")
    print("\n")
    
    # print(100*"-")
    # print(f"DATA FILE: {filename}")
    # print(100*"-")
    # print(f"DIMENSIONS: {entry_num} entries \t {feature_num} features")
    # print(100*"-")
        
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
        print("\n")
    else:
        pass
        
    if len(without_na)==feature_num:
        print("There are no missing values in the dataset.")
    elif len(without_na)>=1:
        print(f"The following features do not have missing values:")
        print(tabulate(without_na, headers=["Features", "Missing Data"]))
    else:
        pass
    print(100*"-")

    print("DUPLICATE VALUES:")
    print(100*"-")

    if df.duplicated().sum() == 0:
        print("There are now duplicate entries in the dataset.")
    elif df.duplicated().sum() > 0:
        print("There following entries are duplicated.")
        print(print(df[df.duplicated()==True]))
    
    print(100*"-")
    
    return identifier, categorical, continuous

### CORRELATIONS FUNCTION ###
def correlations(df: pd.DataFrame, type: str="pearson", printout: str="matrix") -> pd.DataFrame:
    
    """
    The function creates a correlation matrix/heatmap of the continuous features in the dataset.
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        type (str): type of correlations (pearson or spearman)
        printout (str): how correlations are displayed (matrix or heatmap)
    
    Returns:
        None

    """
    if not isinstance(df, pd.DataFrame) or not isinstance(type, str) or not isinstance(printout, str):
        raise TypeError
    
    matrix = df.corr(method=type)
    type_dict = {"pearson": "Pearson", "spearman": "Spearman"}

    if printout == "matrix":
        print(f"{type_dict[type]} Correlation Matrix:")
        display(round(matrix, 3))
    elif printout == "heatmap":
        fig = px.imshow(round(matrix, 3), text_auto=True, color_continuous_scale="Viridis", title=f"{type_dict[type]} Correlation Heatmap of Continuous Features")
        fig.show()
    
    return matrix

### DATA SPLIT FUNCTION ###
def data_split(df: pd.DataFrame, target: str, method: str="tt", random_state: int=None, train_proportion: float=0.8, test_proportion: float=0.2, validation_proportion:float=0.25, stratify: str="Yes") -> pd.DataFrame:
    
    """
    The function splits the data into train-test sets or train-validation-test sets.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        target (str): target variable
        method (str): split method (tt: train-test, tvt: train-validation-test, default=tt)
        random_state (int): random state value (default=None)
        train_proportion (float): fraction of data to be used for training (0-1, default=0.8)
        test_proportion (float): fraction of data to be used for testing (0-1, default=0.2)
        validation_proportion (float): fraction of data to be used for validation (0-1, default=0.25)
        stratify (str): stratified split (Yes or No, default=Yes)

    Returns:
        X (Pandas DataFrame): data structure with predictor features
        y (Pandas DataFrame): data structure with target feature
        X_train (Pandas DataFrame): data structure with training predictor features
        y_train (Pandas DataFrame): data structure with training target feature
        X_test (Pandas DataFrame): data structure with testing predictor features
        y_test (Pandas DataFrame): data structure with testing target feature
        X_val (Pandas DataFrame): data structure with validation predictor features
        y_val (Pandas DataFrame): data structure with validation target feature
    """
    

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or not isinstance(method, str) or not isinstance(random_state, int) or not isinstance(train_proportion, float) or not isinstance(test_proportion, float) or not isinstance(validation_proportion, float) or not isinstance(stratify, str):
        raise TypeError

    X = df.drop(columns=[target])
    y = df[[target]]

    if stratify == "No":
        if method == "tt":
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, random_state=random_state)
            X_val = None
            y_val = None
        elif method == "tvt":
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, random_state=random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_proportion, random_state=random_state)
    elif stratify == "Yes":
        if method == "tt":
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, stratify=y, random_state=random_state)
            X_val = None
            y_val = None
        elif method == "tvt":
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, stratify=y, random_state=random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_proportion, stratify=y_train, random_state=random_state)
    
    return X, y, X_train, y_train, X_test, y_test, X_val, y_val

### TREAT NA FUNCTION ### (NOT USED)
def treat_na(df: pd.DataFrame, identifier: list, categorical: list, continuous:list, drop_na_rows: bool=False, impute_cutoff: float=0.5, categorical_imputer: str="mode", continuous_imputer: str="mean") -> pd.DataFrame:

    """
    The function treats missing values.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        drop_na (bool): drop rows containing missing values (True or False, default=False)
        impute_cutoff (float): if NA fraction is less or equal to the specified value, missing values are imputed otherwise the feature is removed (defaul=0.5)
        categorical_imputer (str): how categorcial missing values are imputed (mode, default=mode)
        continuous_imputer (str): how missing values are imputed (mean, median, default=mean)
    
    Returns:
        df_treat_na (Pandas DataFrame): data structure with no missing values
    """

    if not isinstance(df, pd.DataFrame) or not isinstance(identifier, list) or not isinstance(categorical, list) or not isinstance(continuous, list) or not isinstance(drop_na_rows, bool) or not isinstance(impute_cutoff, float) or not isinstance(categorical_imputer, str) or not isinstance(continuous_imputer, str):
        raise TypeError
   
    df_treat_na = df.copy(deep=True)

    for column in df_treat_na.columns:
        
        missing_fraction = df_treat_na[column].isnull().sum()/df_treat_na.shape[0]
        
        if column in identifier:
            if drop_na_rows == True:
                df_treat_na.drop(df_treat_na.loc[df_treat_na[column].isnull()].index, inplace=True)
        
        if column in continuous:
            if drop_na_rows == False:
                if missing_fraction < impute_cutoff:
                    if continuous_imputer == "mean":
                        df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].mean())
                    elif continuous_imputer == "median":
                        df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].median())
                elif missing_fraction >= impute_cutoff:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
            elif drop_na_rows == True:
                if missing_fraction < impute_cutoff:
                    df_treat_na.dropna(axis=0, subset=[column], inplace=True)
                elif missing_fraction >= impute_cutoff:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
        
        if column in categorical:
            if drop_na_rows == False:
                if missing_fraction < impute_cutoff:
                    if categorical_imputer == "mode":
                        df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].mode()[0])
                elif missing_fraction >= impute_cutoff:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
            elif drop_na_rows == True:
                if missing_fraction < impute_cutoff:
                    df_treat_na.dropna(axis=0, subset=[column], inplace=True)
                elif missing_fraction >= impute_cutoff:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
    
    return df_treat_na

### TREAT NAN FUNCTION ###
def treat_nan(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, identifier: list, categorical: list, continuous:list, target: str, drop_nan_rows: bool=False, impute_cutoff: float=0.5, categorical_imputer: str="mode", continuous_imputer: str="mean") -> pd.DataFrame:

    """
    The function treats missing values.

    Parameters:
        X_train (Pandas DataFrame): data structure with train sample (predictor features)
        y_train (Pandas DataFrame): data structure with train sample (target feature)
        X_test (Pandas DataFrame): data structure with test sample (predictor features)
        y_test (Pandas DataFrame): data structure with test sample (target feature)
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
        drop_nan_rows (bool): drop rows containing missing values (True or False, default=False)
        impute_cutoff (float): if NA fraction is less or equal to the specified value, missing values are imputed otherwise the feature is removed (defaul=0.5)
        categorical_imputer (str): how categorcial missing values are imputed (mode, default=mode)
        continuous_imputer (str): how missing values are imputed (mean, median, default=mean)
    
    Returns:
        df_treat_na (Pandas DataFrame): data structure with no missing values
    """

    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame) or not isinstance(y_test, pd.DataFrame) or not isinstance(identifier, list) or not isinstance(categorical, list) or not isinstance(continuous, list) or not isinstance(target, str) or not isinstance(drop_nan_rows, bool) or not isinstance(impute_cutoff, float) or not isinstance(categorical_imputer, str) or not isinstance(continuous_imputer, str):
        raise TypeError
   
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_treat_na = train_df.copy(deep=True)
    test_treat_na = test_df.copy(deep=True)
 
    for column in train_treat_na.columns:
        
        missing_fraction_train = train_treat_na[column].isnull().sum()/ train_treat_na.shape[0]
        missing_fraction_test = test_treat_na[column].isnull().sum()/ test_treat_na.shape[0]
        
        if column in identifier:
            if drop_nan_rows == True:
                train_treat_na.drop( train_treat_na.loc[train_treat_na[column].isnull()].index, inplace=True)
                test_treat_na.drop( test_treat_na.loc[test_treat_na[column].isnull()].index, inplace=True)
        
        if column in continuous:
            if drop_nan_rows == False:
                if missing_fraction_train < impute_cutoff and missing_fraction_test < impute_cutoff:
                    if continuous_imputer == "mean":
                        train_treat_na[column] = train_treat_na[column].fillna(train_treat_na[column].mean())
                        test_treat_na[column] = test_treat_na[column].fillna(train_treat_na[column].mean())
                    elif continuous_imputer == "median":
                        train_treat_na[column] = train_treat_na[column].fillna(train_treat_na[column].median())
                        test_treat_na[column] = test_treat_na[column].fillna(train_treat_na[column].median())
                elif missing_fraction_train >= impute_cutoff or missing_fraction_test >= impute_cutoff:
                    train_treat_na.dropna(axis=1, subset=[column], inplace=True)
                    test_treat_na.dropna(axis=1, subset=[column], inplace=True)
            elif drop_nan_rows == True:
                if missing_fraction_train < impute_cutoff and missing_fraction_test < impute_cutoff:
                    train_treat_na.dropna(axis=0, subset=[column], inplace=True)
                    test_treat_na.dropna(axis=0, subset=[column], inplace=True)
                elif missing_fraction_train >= impute_cutoff or missing_fraction_test >= impute_cutoff:
                    train_treat_na.dropna(axis=1, subset=[column], inplace=True)
                    test_treat_na.dropna(axis=1, subset=[column], inplace=True)
        
        if column in categorical:
            if drop_nan_rows == False:
                if missing_fraction_train < impute_cutoff and missing_fraction_test < impute_cutoff:
                    if categorical_imputer == "mode":
                        train_treat_na[column] = train_treat_na[column].fillna(train_treat_na[column].mode()[0])
                        test_treat_na[column] = test_treat_na[column].fillna(train_treat_na[column].mode()[0])
                elif missing_fraction_train >= impute_cutoff or missing_fraction_test >= impute_cutoff:
                    train_treat_na.dropna(axis=1, subset=[column], inplace=True)
                    test_treat_na.dropna(axis=1, subset=[column], inplace=True)
            elif drop_nan_rows == True:
                if missing_fraction_train < impute_cutoff and missing_fraction_test < impute_cutoff:
                    train_treat_na.dropna(axis=0, subset=[column], inplace=True)
                    test_treat_na.dropna(axis=0, subset=[column], inplace=True)
                elif missing_fraction_train >= impute_cutoff or missing_fraction_test >= impute_cutoff:
                    train_treat_na.dropna(axis=1, subset=[column], inplace=True)
                    test_treat_na.dropna(axis=0, subset=[column], inplace=True)
    
    # X_train_treat_na = train_treat_na.drop(columns=[target])
    # y_train_treat_na = train_treat_na[[target]]
    # X_test_treat_na = test_treat_na.drop(columns=[target])
    # y_test_treat_na = test_treat_na[[target]]
    
    return train_treat_na, test_treat_na # X_train_treat_na, y_train_treat_na, X_test_treat_na, y_test_treat_na

### TREAT DUPLICATE FUNCTION ###
def treat_duplicate(train_df: pd.DataFrame, test_df: pd.DataFrame, keep_in: str="first") -> pd.DataFrame:
    
    '''
    The function identifies and removes duplicate entries from the dataset (if present).

    Parameters:
        train_df (Pandas DataFrame): data structure train sample
        test_df (Pandas DataFrame): data structure test sample
        keep (str): which occurance of duplicate value to keep (first, last)
    
    Returns:
        train_df_treat_duplicate (Pandas DataFrame): data structure with no duplicated entries (train sample)
        test_df_treat_duplicate (Pandas DataFrame): data structure with no duplicated entries (test sample)
    '''

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame) or not isinstance(keep_in, str):
        raise TypeError

    train_df_treat_duplicate = train_df.drop_duplicates(keep=keep_in)
    test_df_treat_duplicate = test_df.drop_duplicates(keep=keep_in)

    return train_df_treat_duplicate, test_df_treat_duplicate
    
    
### TREAT OUTLIERS FUNCTION ###
def treat_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, categorical: list, continuous:list, target: str, method: str, outlier_fraction: float=0.01) -> pd.DataFrame:
    
    """
    The function identifies and removes outlying observations from the dataset (if present) using automatic outlier detection methods.

    Parameters:
        train_df (Pandas DataFrame): data structure train sample
        test_df (Pandas DataFrame): data structure test sample
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
        method (str): automatic outlier detection method (if: isolation forest, mcd: minimum covariance distance, lof: local outlier factor, svm: one-class support vector machine)
        outlier_fraction: proportion of outliers in the data set (0-0.5, default=0.01)
    
    Returns:
        train_df_treat_outliers (Pandas DataFrame): data structure with no outlying obrervations (train sample)
        test_df_treat_outliers (Pandas DataFrame): data structure with no outlying obrervations (test sample)
    """

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame) or not isinstance(identifier, list) or not isinstance(categorical, list) or not isinstance(continuous, list) or not isinstance(target, str) or not isinstance(method, str) or not isinstance(outlier_fraction, float):
        raise TypeError
    
    categorical_without_target = categorical.copy()
    categorical_without_target.remove(target)

    X_train = train_df.drop(columns=[target])
    X_train_df_identifier = X_train[identifier]
    X_train_df_categorical = X_train[categorical_without_target]
    X_train_df_continuous = X_train[continuous]
    y_train = train_df[target]


    X_test = test_df.drop(columns=[target])
    X_test_df_identifier = X_test[identifier]
    X_test_df_categorical = X_test[categorical_without_target]
    X_test_df_continuous = X_test[continuous]
    y_test = test_df[target]

    if method == "if":
        iso_train = IsolationForest(contamination=outlier_fraction)
        
        # identify outliers in the train set
        yhat_train = iso_train.fit_predict(X_train_df_continuous)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        iso_test = IsolationForest(contamination=outlier_fraction)

        # identify outliers in the test set
        yhat_test = iso_test.fit_predict(X_test_df_continuous)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]

    elif method == "mcd":
        mcd = EllipticEnvelope(contamination=outlier_fraction)
        
        # identify outliers in the train set
        yhat_train = mcd.fit_predict(X_train_df_continuous)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        # identify outliers in the test set
        yhat_test = mcd.fit_predict(X_test_df_continuous)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]

    elif method == "lof":
        lof = LocalOutlierFactor()
        
        # identify outliers in the train set
        yhat_train = lof.fit_predict(X_train_df_continuous)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        # identify outliers in the test set
        yhat_test = lof.fit_predict(X_test_df_continuous)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]
    
    elif method == "svm":
        svm = OneClassSVM(nu=outlier_fraction)
        
        # identify outliers in the train set
        yhat_train = svm.fit_predict(X_train_df_continuous)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        # identify outliers in the test set
        yhat_test = svm.fit_predict(X_test_df_continuous)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]
    
    train_df_treat_outliers = pd.concat([X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train], axis=1)
    test_df_treat_outliers = pd.concat([X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test], axis=1)

    return train_df_treat_outliers, test_df_treat_outliers




# def data_processing(df, missing=True, impute=0.5, imputer, outliers=True, normalize=False, standardize=True, encoding=False):
    
#     """
#     The function performs the preprocessing of the data:
#     1) Identifies and removes/imputes missing values.
#     2) Identifies and removes outlying values.
#     3) Normalizes/standardizes data if needed.
#     4) Changes the format/encoding of values if needed.

#     Parameters:
#         df: Pandas DataFrame
#         missing (bool): treat missing values (True or False) (default=True)
#         impute (float): if NA fraction is less or equal to the specified value missing values are imputed otherwise they are removed (defaul=0.5)
#         imputer (str): how missing values are imputed (mean, median, mode)
#         outliers (bool): treat outliers
#     """



