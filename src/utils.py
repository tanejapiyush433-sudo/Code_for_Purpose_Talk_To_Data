# src/utils.py

import pandas as pd
import numpy as np

def preprocess(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    value_col = numeric_cols[0]

    df['ds'] = pd.date_range(start='2024-01-01', periods=len(df), freq='s')
    df['y'] = df[value_col]

    return df[['ds', 'y']]
