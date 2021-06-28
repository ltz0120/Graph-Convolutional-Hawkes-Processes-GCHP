"""
This script contains the function loading data from csv file with or without header.
the csv file must has the first column as the timestamps, while the other columns are features.
"""

from dev.util import logger
import numpy as np
import pandas as pd
import copy
import time
from typing import Dict


def load_csv(file, single_realization=False, categorical=False):
    '''
    :param file: str or pandas.dataframe.   string is the location of file.
    :param single_realization: if the dataset contains multiple realizations, then the first column is the number of realization (seq_no).
            or if the dataset has only one realization, then the first column will be the timestamps.
    :return: a dataframe with column name seq_no, timestamp, and features.
    '''
    if isinstance(file, str):
        logger.info('Start to load from csv file...')
        df = pd.read_csv(file, index_col=0)
    elif isinstance(file, pd.DataFrame):
        df = file.copy()
    else:
        raise TypeError('file must be a string or a pandas.dataframe')
    # TODO:  check timestamp is ordered or not.

    if single_realization:
        col_names = ['timestamp'] + ['x' + str(i) for i in np.arange(df.shape[1] - 1)]
        df.columns = col_names
        df['seq_no'] = np.zeros(df.__len__())
        df = df[['seq_no'] + col_names]
        df = df.sort_values(by='timestamp')

    else:
        col_names = ['seq_no', 'timestamp'] + ['x' + str(i) for i in np.arange(df.shape[1] - 2)]
        df.columns = col_names
        seq_dict_keys = list(df['seq_no'].drop_duplicates())
        seq_dict_values = np.arange(seq_dict_keys.__len__())
        seq_dict = dict(zip(seq_dict_keys, seq_dict_values))
        df['seq_no'] = df['seq_no'].replace(seq_dict)
        df = df.sort_values(by=['seq_no', 'timestamp'])

    if categorical:
        category_keys = list(df['x0'].drop_duplicates())
        category_values = np.arange(category_keys.__len__())
        cate_dict = dict(zip(category_keys, category_values))
        df['x0'] = df['x0'].replace(cate_dict)

    df.index = np.arange(df.__len__())

    return df
