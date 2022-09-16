# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tabulate import tabulate
from IPython.display import display


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

    Source: https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/

    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        test_df (Pandas DataFrame): data structure with test sample
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
        method (str): automatic outlier detection method (if: isolation forest, mcd: minimum covariance distance, lof: local outlier factor, svm: one-class support vector machine)
        outlier_fraction (float): proportion of estimated outliers in the data set (0-0.5, default=0.01)
    
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
        # select outliers
        train_outlier_mask = yhat_train == -1
        train_outliers_df = pd.concat([X_train_df_identifier[train_outlier_mask], X_train_df_categorical[train_outlier_mask], X_train_df_continuous[train_outlier_mask], y_train[train_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        
        iso_test = IsolationForest(contamination=outlier_fraction)

        # identify outliers in the test set
        yhat_test = iso_test.fit_predict(X_test_df_continuous)
        # select outliers
        test_outlier_mask = yhat_test == -1
        test_outliers_df = pd.concat([X_test_df_identifier[test_outlier_mask], X_test_df_categorical[test_outlier_mask], X_test_df_continuous[test_outlier_mask], y_test[test_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]
        

    elif method == "mcd":
        mcd = EllipticEnvelope(contamination=outlier_fraction)
        
        # identify outliers in the train set
        yhat_train = mcd.fit_predict(X_train_df_continuous)
        # select outliers
        train_outlier_mask = yhat_train == -1
        train_outliers_df = pd.concat([X_train_df_identifier[train_outlier_mask], X_train_df_categorical[train_outlier_mask], X_train_df_continuous[train_outlier_mask], y_train[train_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        # identify outliers in the test set
        yhat_test = mcd.fit_predict(X_test_df_continuous)
       # select outliers
        test_outlier_mask = yhat_test == -1
        test_outliers_df = pd.concat([X_test_df_identifier[test_outlier_mask], X_test_df_categorical[test_outlier_mask], X_test_df_continuous[test_outlier_mask], y_test[test_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]
        
    elif method == "lof":
        lof = LocalOutlierFactor()
        
        # identify outliers in the train set
        yhat_train = lof.fit_predict(X_train_df_continuous)
        # select outliers
        train_outlier_mask = yhat_train == -1
        train_outliers_df = pd.concat([X_train_df_identifier[train_outlier_mask], X_train_df_categorical[train_outlier_mask], X_train_df_continuous[train_outlier_mask], y_train[train_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        # identify outliers in the test set
        yhat_test = lof.fit_predict(X_test_df_continuous)
        # select outliers
        test_outlier_mask = yhat_test == -1
        test_outliers_df = pd.concat([X_test_df_identifier[test_outlier_mask], X_test_df_categorical[test_outlier_mask], X_test_df_continuous[test_outlier_mask], y_test[test_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]
        
    elif method == "svm":
        svm = OneClassSVM(nu=outlier_fraction)
        
        # identify outliers in the train set
        yhat_train = svm.fit_predict(X_train_df_continuous)
        # select outliers
        train_outlier_mask = yhat_train == -1
        train_outliers_df = pd.concat([X_train_df_identifier[train_outlier_mask], X_train_df_categorical[train_outlier_mask], X_train_df_continuous[train_outlier_mask], y_train[train_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_train = yhat_train != -1
        X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train = X_train_df_identifier[mask_train], X_train_df_categorical[mask_train], X_train_df_continuous[mask_train], y_train[mask_train]
        
        # identify outliers in the test set
        yhat_test = svm.fit_predict(X_test_df_continuous)
        # select outliers
        test_outlier_mask = yhat_test == -1
        test_outliers_df = pd.concat([X_test_df_identifier[test_outlier_mask], X_test_df_categorical[test_outlier_mask], X_test_df_continuous[test_outlier_mask], y_test[test_outlier_mask]], axis=1)
        # select all rows that are not outliers
        mask_test = yhat_test != -1
        X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test = X_test_df_identifier[mask_test], X_test_df_categorical[mask_test], X_test_df_continuous[mask_test], y_test[mask_test]
        
    method_dict = {"if": "Isolation Forest", "mcd": "Minimum Covariance Distance", "lof": "Local Outlier Factor", "svm": "One-class Support Vector Machine"}
    print(f"The following entries are probable outliers as identified by the {method_dict[method]} technique (train set):")
    display(train_outliers_df)
    print(f"The following entries are probable outliers as identified by the {method_dict[method]} technique (test set):")
    display(test_outliers_df)


    train_df_treat_outliers = pd.concat([X_train_df_identifier, X_train_df_categorical, X_train_df_continuous, y_train], axis=1)
    test_df_treat_outliers = pd.concat([X_test_df_identifier, X_test_df_categorical, X_test_df_continuous, y_test], axis=1)

    return train_df_treat_outliers, test_df_treat_outliers

### TARGET VARIABLE BALANCE FUNCTION ###
def target_balance_check(train_df: pd.DataFrame, target: str, imbalance_fraction: float=0.5) -> None:
    
    """
    The function checks whether there is imbalance in the target feature levels.

    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        target (str): target variable
        imbalance_fraction (float): fraction of acceptable imbalance between the target feature levels (0-1, default=0.5)
    
    Returns:
        None
    """

    if not isinstance(train_df, pd.DataFrame) or not isinstance(target, str) or not isinstance(imbalance_fraction, float):
        raise TypeError
    
    
    total_count = sum(train_df[target].value_counts())

    target_feature_info = []
    level_percentages = []
    for key, value in train_df[target].value_counts().sort_index(ascending=True).to_dict().items():
        target_feature_info.append([key, value, round(value/total_count, 3)])
        level_percentages.append(value/total_count)
    print(tabulate(target_feature_info, headers=["Target Feature Levels", "Counts", "Percentages"]))

    level_percentage_diff = list(np.diff(level_percentages))
    level_percentage_diff_sorted = sorted(level_percentage_diff)
    level_percentage_diff_bool = [i>=imbalance_fraction for i in level_percentage_diff_sorted]

    if True in level_percentage_diff_bool:
        print("\n")
        print("The tagret feature levels are unbalanced.")
    else:
        print("\n")
        print("The tagret feature levels are balanced.")

### SAMPLER FUNCTION ###
def sampler(train_df: pd.DataFrame, target: str, method: str, sampling_ratios: dict, random_state: int=None) -> pd.DataFrame:
    
    """
    The uses under- or over-sampling techniques to create a balanced dataset.

    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        target (str): target variable
        method (str): under- or over-sampling technique (under or over)
        sampling_ratios (dict): dictionary containing the desired ratios of the target feature levels when sampling is done (ratios are based on the class with minimum and maximum counts in case of under- and over-sampling respectively)
        random_state (int): random state value (default=None)
    
    Returns:
        
    """

    if not isinstance(train_df, pd.DataFrame) or not isinstance(target, str) or not isinstance(method, str) or not isinstance(sampling_ratios, dict):
        raise TypeError
    
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    top_count = max(train_df[target].value_counts())
    min_count = min(train_df[target].value_counts())
    total_count = sum(train_df[target].value_counts())

    if method == "under":
        undersampling = {}
        for level, ratio in sampling_ratios.items():
            undersampling.update({level: min_count*ratio})
        rus = RandomUnderSampler(sampling_strategy=undersampling, random_state=random_state)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

        total_count = sum(y_train_rus.value_counts())
        target_feature_info = []
        level_percentages = []
        for key, value in y_train_rus.value_counts().sort_index(ascending=True).to_dict().items():
            target_feature_info.append([key, value, round(value/total_count, 3)])
            level_percentages.append(value/total_count)
        print("Balanced target feature (undersampling):")
        print(tabulate(target_feature_info, headers=["Target Feature Levels", "Counts", "Percentages"]))

        train_df_resampled = pd.concat([X_train_rus, y_train_rus], axis=1)
        
        return train_df_resampled
    
    elif method == "over":
        oversampling = {}
        for level, ratio in sampling_ratios.items():
            oversampling.update({level: top_count*ratio})
        ros = RandomOverSampler(sampling_strategy=oversampling, random_state=random_state)
        X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

        total_count = sum(y_train_ros.value_counts())
        target_feature_info = []
        level_percentages = []
        for key, value in y_train_ros.value_counts().sort_index(ascending=True).to_dict().items():
            target_feature_info.append([key, value, round(value/total_count, 3)])
            level_percentages.append(value/total_count)
        print("Balanced target feature (oversampling):")
        print(tabulate(target_feature_info, headers=["Target Feature Levels", "Counts", "Percentages"]))

        train_df_resampled = pd.concat([X_train_ros, y_train_ros], axis=1)
        
        return train_df_resampled