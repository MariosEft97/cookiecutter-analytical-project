### LOAD PACKAGES ###
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd

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