import pandas as pd

import utils
import numpy as np
from Model import Model

def clean_data(df):
    df = df[['id', 'name', 'user', 'rating']].copy()
    df['rating'] = df['rating'].astype('float32')
    return df

def remove_duplicates(df):
    df = df.copy()
    df.drop_duplicates(subset=['user', 'name', 'rating'], inplace=True)
    df.drop_duplicates(subset=['user', 'name'], inplace=True, keep='last')
    df.dropna(subset=['id', 'name', 'user'], how='any', inplace=True)
    return df.copy()

def pivot_with_threshold(df, thresh):
    pivot_df = df.pivot(index='user', columns=['name'], values=['rating'])
    pivot_df.columns = pivot_df.columns.droplevel()
    pivot_df.dropna(thresh=thresh, inplace=True)
    pivot_df.fillna(value=0, inplace=True)
    return pivot_df

def normalize(df, factor=10):
    df = df.copy() / factor
    return df
