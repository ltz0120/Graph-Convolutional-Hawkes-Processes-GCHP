from dev.util import logger
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time


class EventSampler(Dataset):
    """Load event sequences via minbatch"""

    def __init__(self, dataset, memory_size, temporal_kernel='exp', batch_size=None, kernel_bandwidth=1,
                 categorical=False):
        '''
        :param dataset: from DataReader
        :param memorysize: how many historical events remembered by each event
        :param categorical: for categorical data, convert to one hot vectors.
        :param temporal_kernel: ['exp', 'gaussian', 'linear']

        return: a kernel matrix of time, feature matrix, next time, next feature vector.
        '''
        start = time.time()
        self.categorical = categorical
        self.dataset = dataset

        self.data_len = dataset.__len__()
        if batch_size is None:
            self.batch_size = dataset.__len__() - memory_size * (np.max(dataset['seq_no']) + 1)
        else:
            self.batch_size = batch_size

        logger.info('Start initializing...')

        if categorical:
            self.cate_series = pd.Series(self.dataset['x0']).copy()
            dummy_feature = pd.get_dummies(self.cate_series, drop_first=False)
            dummy_feature.columns = ['x' + str(i) for i in np.arange(dummy_feature.shape[1])]
            self.dataset = self.dataset.drop(columns=['x0'])
            self.dataset = pd.concat([self.dataset, dummy_feature], axis=1)

        self.memory_size = memory_size
        self.temporal_kernel = temporal_kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.seq_list = []
        self.cate_list = []
        self.time_temp = np.insert(self.dataset['timestamp'].to_numpy(), 0, 0)
        self.time_temp = np.diff(self.time_temp)

        for i in np.arange(np.max(self.dataset['seq_no']) + 1):
            self.seq_list += [self.dataset[self.dataset['seq_no'] == i]]
            if categorical:
                self.cate_list += [self.cate_series[self.dataset['seq_no'] == i].to_numpy()]

        self.kernel_matrix_list = []
        self.feature_matrix_list = []
        self.time_label = []
        self.feature_label = []


        self.feature_dim = self.dataset.shape[1] - 2

        logger.info('Start transforming the input into graphs...')

        for i in np.arange(self.seq_list.__len__()):
            seq_table_temp = self.seq_list[i].copy()
            seq_table_temp = seq_table_temp.sort_values(by = 'timestamp')
            seq_table_temp.index = np.arange(seq_table_temp.__len__())
            time_diff_list = np.insert(self.seq_list[i]['timestamp'].to_numpy(), 0, 0)
            time_diff_list = np.diff(time_diff_list)

            for j in np.arange(self.seq_list[i].__len__()):

                if self.memory_size <= j < self.seq_list[i].__len__():  # filter out initial events.
                    time_list_temp = np.asmatrix([seq_table_temp.loc[(j - self.memory_size): (j-1), 'timestamp'].to_numpy()])
                    self.feature_matrix_list += [seq_table_temp.loc[(j - self.memory_size): (j-1), 'x0':].values.tolist()]

                    time_diff_matrix = np.abs(
                        time_list_temp.repeat(self.memory_size, 0) -
                        time_list_temp.transpose().repeat(self.memory_size, 1))
                    if self.temporal_kernel == 'exp':
                        kernel_temp = np.exp(-time_diff_matrix * self.kernel_bandwidth) * self.kernel_bandwidth
                    elif self.temporal_kernel == 'Gaussian':
                        kernel_temp = np.exp(-time_diff_matrix**2 / self.kernel_bandwidth)
                    elif self.temporal_kernel == 'linear':
                        kernel_temp = (np.max(time_diff_matrix) - time_diff_matrix+self.kernel_bandwidth) /\
                        (np.max(time_diff_matrix) - np.min(time_diff_matrix)+self.kernel_bandwidth)
                    elif self.temporal_kernel == 'spherical':
                        kernel_temp = 1 - 1.5 * time_diff_matrix/self.kernel_bandwidth + 0.5 * (time_diff_matrix/self.kernel_bandwidth)**3
                    else:
                        raise TypeError("input temporal_kernel must be any of ['exp', 'Gaussian', 'linear', 'spherical']")

                    diag = np.asarray(1 / np.sqrt(kernel_temp.sum(axis=0)+1e-12))[0]
                    d_mat = np.diag(diag)

                    kernel_temp = np.linalg.multi_dot([d_mat, kernel_temp, d_mat])
                    kernel_temp = d_mat.dot(kernel_temp).dot(d_mat)

                    self.kernel_matrix_list += [kernel_temp.tolist()]
                    self.time_label += [time_diff_list[j]]
                    if categorical:
                        self.feature_label += [self.cate_list[i][j]]
                    else:
                        self.feature_label += [seq_table_temp.loc[j, 'x0':]]

            if ((i+1) % 10) == 0:
                logger.info('{}/{} sequences has been transformed. Time:{:.2f}s'.
                            format((i+1), self.seq_list.__len__(), time.time() - start))

        self.out_list = []

        index__ = np.arange(start=0, stop=self.kernel_matrix_list.__len__(), step=batch_size)
        if self.kernel_matrix_list.__len__() % batch_size > 0:
            index__ = np.append(index__, self.kernel_matrix_list.__len__()-1)
        if self.categorical:
            out_temp = [(torch.Tensor(self.kernel_matrix_list[index__[i]:index__[i+1]]),
                        torch.Tensor(self.feature_matrix_list[index__[i]:index__[i+1]]),
                        torch.Tensor(self.time_label[index__[i]:index__[i+1]]).unsqueeze(1),
                        torch.LongTensor(self.feature_label[index__[i]:index__[i+1]])) for i in np.arange(index__.__len__()-1)]
        else:
            out_temp = [(torch.Tensor(self.kernel_matrix_list[index__[i]:index__[i+1]]),
                        torch.Tensor(self.feature_matrix_list[index__[i]:index__[i+1]]),
                        torch.Tensor(self.time_label[index__[i]:index__[i+1]]).unsqueeze(1),
                        torch.Tensor(self.feature_label[index__[i]:index__[i+1]])) for i in np.arange(index__.__len__()-1)]
        self.out_list = out_temp

        logger.info('In this dataset, the number of sequence = {}.'.format(len(self.seq_list)))
        logger.info('Each event is influenced by its last {} historical events.'.format(self.memory_size))

    def __len__(self):
        return len(self.out_list)

    def __getitem__(self, idx):
        return self.out_list[idx]
