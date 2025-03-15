from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variable should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame):
        # YOUR CODE HERE
        X = X.copy()
        wkday_null_idx = X[X[self.variable].isnull() == True].index
        X.loc[wkday_null_idx, 'weekday'] = X.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

        X = X.drop(columns=['dteday'])
        return X

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variable should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.fill = X[self.variable].mode()[0]
        return self

    def transform(self, X: pd.DataFrame):
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable: str, mappings: dict):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variable should be a string")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    #def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        #print("####",self.variable)
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)
        return X
    
# Fix in OutlierHandler Class
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable: list):
        # YOUR CODE HERE
        self.variable = variable
        return

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self  # Fix: Return self instead of X

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        df = X.copy()
        for colm in self.variable:
            print("OutlierHandler:",colm)
            q1 = df.describe()[colm].loc['25%']
            q3 = df.describe()[colm].loc['75%']

            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            for i in df.index:
                if df.loc[i,colm] > upper_bound:
                    df.loc[i,colm]= upper_bound
                if df.loc[i,colm] < lower_bound:
                    df.loc[i,colm]= lower_bound
        return df


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable: str, encoder):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variable should be a string")

        self.variable = variable
        self.encoder  = encoder

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        #X=X.copy()
        self.encoder.fit(X[[self.variable]])
        self.enc_wkday_features = self.encoder.get_feature_names_out([self.variable])
        
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        #self.encoder.fit(X[[self.variable]])
        encoded_weekday = self.encoder.transform(X[[self.variable]])
        #enc_wkday_features = self.encoder.get_feature_names_out([self.variable])
        X[self.enc_wkday_features] = encoded_weekday
        
        # drop 'weekday' column after encoding
        X = X.drop(columns=['weekday'])
        return X