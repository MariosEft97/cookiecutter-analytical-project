# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
from sklearn.feature_selection import RFE
from boruta import BorutaPy

### FEATURE SELECTION FUNCTION ###
def feature_selection(df: pd.DataFrame, method: str) -> pd.DataFrame:

    """
    The function performs feature selection.
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        method (str): method of feature selection
    
    Returns:
        selected_df (pd.DataFrame): data structure with the only the selected features

    """
    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)

    elif not isinstance(method, str):
        error_message = "method must be specified as a string\noptions: ..."
        raise TypeError(error_message)

    else:

        return None