"""
This script contains the model architectures for Hawkes processes.
"""

import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
import time
from dev.util import logger
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.nn import Linear
from layers import GraphConvolution
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import LSTM
from os import path
import os


class GCHP_model:
    def __init__(self, args=None):
        '''
        We provide plentiful of architectures for the model.
        :param nfeat:
        :param memory_size:
        :param categorical: boolean value
        :param dropout:
        :param device:
        :param theta: the hyper-parameter control the loss of time and feature.
        :param eta: the hyper-parameter control the loss of time and feature.
        :param record_gc: record granger causality at each step
        :param record: record the results at each epoch
        :param batch_size: set the batch size
        :param model: select the model architecture. Available architectures are GCN_one_layer, GCN_two_layer,
                    LSTM_net, LSTM_net_2layer, GCN_LSTM, GCN_TRM, TRM_net_1layer, TRM_net_2layer
        '''
        print("initialize GCHP_model for RNN")
        print(args)
        self.categorical = args['categorical']
        self.nfeat = args['nfeat']
        self.memory_size = args['memory_size']
        self.theta = args['theta']
        self.eta = args['eta']
        self.record_gc = args['record_gc']
        device_id = args['device']
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(device_id) if use_cuda else "cpu")
        self.n1hid = args['n1hid']
        self.n2hid = args['n2hid']
        self.dropout = args['dropout']
        self.record = args['record']

        print(args['model'])
        if args['model'] == 'GCN_two_layer':
            self.model = GCN_two_layer(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device = self.device)
        elif args['model'] == 'LSTM_net':
            self.model = LSTM_net(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device=self.device)
        elif args['model'] == 'LSTM_net_2layer':
            self.model = LSTM_net_2layer(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device=self.device)
        elif args['model'] == 'TRM_net_1layer':
            self.model = TRM_net_1layer(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device=self.device)
        elif args['model'] == 'TRM_net_2layer':
            self.model = TRM_net_2layer(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout,
                                        device=self.device)
        elif args['model'] == 'GCN_LSTM':
            self.model = GCN_LSTM(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device=self.device)
        elif args['model'] == 'GCN_one_layer':
            self.model = GCN_one_layer(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device=self.device)
        elif args['model'] == 'GCN_TRM':
            self.model = GCN_TRM(self.nfeat, self.n1hid, self.n2hid, self.memory_size, self.dropout, device=self.device)
        if 'dir' in args:
            self.dir = args['dir']
        else:
            self.dir = "record/"

        self.file_name = args['dataset'] + "_" + args['model'] + "_" +  args['loss_type']
        if self.record:
            if not path.isdir(self.dir):
                os.mkdir(self.dir)

            f = open(self.dir + self.file_name + ".txt", 'a')
            f.write(str(args) + '\n')
            f.close()
        self.loss_type = args['loss_type']
        self.loss_test_time_by_epoch = []
        self.loss_test_feature_by_epoch = []
        if self.record_gc:
            self.gc_list = []
        if self.device is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:0" if use_cuda else "cpu")
        else:
            self.device = args['device']

        if torch.cuda.is_available():
            self.model.to(self.device)

        self.loss_time = MSELoss(reduction='sum')

        if self.categorical:
            self.loss_feature = CrossEntropyLoss(reduction='sum')
            print("cross_entropy loss")
        else:
            self.loss_feature = MSELoss(reduction='sum')
            print("MSE loss")

        if self.categorical:
            logger.info(' Categorical features are deployed')
        else:
            logger.info(' Continuous features are deployed')
        self.feature_prediction = []
        self.feature_ground_truth = []
        self.time_prediction = []
        self.time_ground_truth = []

    def loss_time2(self, pre, label):
        return torch.mean(label / pre + torch.log(pre))

    def fit(self, train_dataloader, optimizer, epochs, scheduler=None, test_dataloader=None, verbose=True):
        t = time.time()
        self.model.train()
        optimizer.zero_grad()
        train_length = np.sum([train_dataloader.__getitem__(i)[3].__len__() for i in np.arange(train_dataloader.__len__())])
        if test_dataloader is not None:
            test_length = np.sum(
                [test_dataloader.__getitem__(i)[3].__len__() for i in np.arange(test_dataloader.__len__())])

        for epoch in np.arange(epochs):
            loss_epoch_time = 0
            loss_epoch_feature = 0
            loss_epoch_total = 0
            epoch_time = time.time()

            for idx, batch_item in enumerate(train_dataloader):
                kernel_matrix_list, feature_matrix_list, label_time, label_feature = batch_item

                optimizer.zero_grad()

                kernel_matrix_list = kernel_matrix_list.to(self.device)
                feature_matrix_list = feature_matrix_list.to(self.device)
                label_time = label_time.to(self.device)
                label_time = label_time.squeeze(1)
                label_feature = label_feature.to(self.device)

                pre_time, pre_feature = self.model(feature_matrix_list, kernel_matrix_list)
                pre_time = pre_time.squeeze(-1)+1e-7
                if self.categorical:
                    pre_feature = pre_feature.view(-1, pre_feature.shape[-1])
                    label_feature = label_feature.view(-1)
                if self.loss_type =='mse':
                    loss_train_time = self.loss_time(pre_time, label_time)/kernel_matrix_list.__len__()
                else:
                    loss_train_time = self.loss_time2(pre_time, label_time) / kernel_matrix_list.__len__()

                loss_train_feature = self.loss_feature(pre_feature, label_feature)/self.nfeat
                loss_total_train = torch.mul(loss_train_time, self.eta) + torch.mul(loss_train_feature, self.theta)
                loss_total_train.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_epoch_time += loss_train_time.item()*kernel_matrix_list.__len__()
                loss_epoch_feature += loss_train_feature.item()
                loss_epoch_total += loss_total_train.item()

            if test_dataloader is not None:

                with torch.no_grad():
                    test_time_loss = 0
                    test_feature_loss = 0

                    for test_idx, batch_item_test in enumerate(test_dataloader):
                            (kernel_matrix_list_test, feature_matrix_list_test,
                             label_time_test, label_feature_test) = batch_item_test

                            kernel_matrix_list_test = kernel_matrix_list_test.to(self.device)
                            feature_matrix_list_test = feature_matrix_list_test.to(self.device)
                            label_time_test = label_time_test.to(self.device)
                            label_feature_test = label_feature_test.to(self.device)

                            pre_time_test, pre_feature_test = self.model(feature_matrix_list_test, kernel_matrix_list_test)
                            # pre_time_test = pre_time_test.flatten()

                            test_time_loss += self.loss_time(pre_time_test, label_time_test).item()
                            if self.categorical:
                                test_feature_loss += np.sum(np.argmax(pre_feature_test.cpu().numpy(), axis=1) ==
                                                            label_feature_test.cpu().numpy())

                            else:
                                test_feature_loss += nn.functional.mse_loss(pre_feature_test, label_feature_test, reduction='sum').item()/self.nfeat
            current_epoch_time = time.time() - epoch_time

            if verbose:
                if test_dataloader is not None:
                    if self.categorical:
                        print('Epoch: {:02d}'.format(epoch + 1),
                              'loss_train_time: {:.3f}'.format(np.sqrt(loss_epoch_time/train_length)),
                              'loss_test_time: {:.3f}'.format(np.sqrt(test_time_loss/test_length)),
                              'loss_train_feature: {:.3f}'.format(np.sqrt(loss_epoch_feature/train_length)),
                              'acc_test_feature: {:.3f}%'.format(test_feature_loss/test_length*100),
                              'loss_train_total: {:.3f}'.format(np.sqrt(loss_epoch_total / train_length)),
                              # 'time: {:.2f}min'.format((time.time() - t)/60))
                              'time: {:.4f}second'.format((time.time() - t)),
                              'epoch_time: {:.4f}second'.format(current_epoch_time))
                    else:
                        print('Epoch: {:02d}'.format(epoch + 1),
                              'loss_train_time: {:.3f}'.format(np.sqrt(loss_epoch_time/train_length)),
                              'loss_test_time: {:.3f}'.format(np.sqrt(test_time_loss/test_length)),
                              'loss_train_feature: {:.3f}'.format(np.sqrt(loss_epoch_feature/train_length)),
                              'loss_test_feature: {:.3f}'.format(np.sqrt(test_feature_loss/test_length)),
                              'loss_train_total: {:.3f}'.format(np.sqrt(loss_epoch_total / train_length)),
                              # 'time: {:.2f}min'.format((time.time() - t)/60))
                              'time: {:.4f}second'.format((time.time() - t)),
                              'epoch_time: {:.4f}second'.format(current_epoch_time))
                else:
                    print('Epoch: {:03d}'.format(epoch + 1),
                          'loss_train_time: {:.3f}'.format(np.sqrt(loss_epoch_time/train_length)),
                          'loss_train_feature: {:.3f}'.format(np.sqrt(loss_epoch_feature/train_length)),
                          'time: {:.2f}s'.format(time.time() - t),
                          'epoch_time: {:.4f}second'.format(current_epoch_time))
                if self.record:
                    f = open(self.dir + self.file_name + ".txt", 'a')
                    f.write('Epoch: {:02d}'.format(epoch + 1)+
                          ', loss_train_time: {:.3f}'.format(np.sqrt(loss_epoch_time / train_length))+
                          ', loss_test_time: {:.3f}'.format(np.sqrt(test_time_loss / test_length))+
                          ', loss_train_feature: {:.3f}'.format(np.sqrt(loss_epoch_feature / train_length))+
                          ', acc_test_feature: {:.3f}%'.format(test_feature_loss / test_length * 100)+
                          ', loss_train_total: {:.3f}'.format(np.sqrt(loss_epoch_total / train_length))+
                          ', time: {:.4f}second'.format((time.time() - t))+
                          ', epoch_time: {:.4f}second'.format(current_epoch_time))
                    f.write('\n')
                    f.close()

            if test_dataloader is not None:
                self.loss_test_time_by_epoch += [np.mean(test_time_loss)]
                self.loss_test_feature_by_epoch += [np.mean(test_feature_loss)]

            if test_dataloader is not None:

                with torch.no_grad():
                    test_time_loss = 0
                    test_feature_loss = 0

                    for test_idx, batch_item_test in enumerate(test_dataloader):

                        (kernel_matrix_list_test, feature_matrix_list_test,
                         label_time_test, label_feature_test) = batch_item_test

                        kernel_matrix_list_test = kernel_matrix_list_test.to(self.device)
                        feature_matrix_list_test = feature_matrix_list_test.to(self.device)
                        label_time_test = label_time_test.to(self.device)
                        label_feature_test = label_feature_test.to(self.device)

                        pre_time_test, pre_feature_test = self.model(feature_matrix_list_test, kernel_matrix_list_test)

                        self.feature_prediction.append(np.argmax(pre_feature_test.cpu().numpy(), axis=1))
                        self.feature_ground_truth.append(label_feature_test.cpu().numpy())
                        self.time_ground_truth.append(label_time_test.cpu().numpy())
                        self.time_prediction.append(pre_time_test.cpu().numpy())

            if self.record_gc:
                self.gc_list += [self.granger_causality()]


    def predict(self, next_event_num, dataset, kernel_bandwidth=1):
        '''
        :param next_event_num: int, the number of events to be predicted.
        :param dataset: dataset from DataReader.
        :param kernel_bandwidth: same as in DataLoader
        :return:
        '''

        if dataset.__len__() < self.memory_size:
            raise ValueError('dataset has too few events.')


        pre_time_output = []
        pre_feature_output = []
        pre_df = dataset.tail(self.memory_size)

        for i in np.arange(next_event_num):
            if i > 0:
                pre_df = pre_df.append(pd.Series([0, pre_time_output[-1]] + [i for i in pre_feature_output[-1]], index=pre_df.columns), ignore_index=True)
                pre_df = pre_df.tail(self.memory_size)
            with torch.no_grad():

                time_list_temp = np.asmatrix(
                    [pre_df.loc[:, 'timestamp'].to_numpy()])
                feature_matrix_list = torch.Tensor([pre_df.loc[:, 'x0':].values.tolist()])

                time_diff_matrix = np.abs(
                    time_list_temp.repeat(self.memory_size, 0) -
                    time_list_temp.transpose().repeat(self.memory_size, 1))
                kernel_temp = np.exp(-time_diff_matrix * kernel_bandwidth) * kernel_bandwidth

                diag = np.asarray(1 / np.sqrt(kernel_temp.sum(axis=0)))[0]
                d_mat = np.diag(diag)

                kernel_temp = np.linalg.multi_dot([d_mat, kernel_temp, d_mat])
                kernel_temp = d_mat.dot(kernel_temp).dot(d_mat)
                kernel_temp = torch.Tensor(kernel_temp)
                kernel_temp = kernel_temp.unsqueeze(0)

                pre_time, pre_feature = self.model(feature_matrix_list, kernel_temp)
                pre_time = pre_time.item()
                pre_time_output += [pre_time + pre_df.tail(1).loc[:, 'timestamp'].to_numpy()[0]]
                pre_feature = pre_feature[0].numpy()
                pre_feature_output += [pre_feature]

        return pre_time_output, pre_feature_output

    def granger_causality(self):
        '''
        Print the granger_causality for categorical temporal event sequences.
        :return: a causality graph.
        '''

        kernel_meta = torch.eye(self.memory_size, requires_grad=False)
        kernel_meta = kernel_meta.to(self.device)
        kernel_meta = kernel_meta.unsqueeze(0)

        output = torch.zeros([self.nfeat, self.nfeat], requires_grad=False)

        for i in np.arange(self.nfeat):

            feature_meta = torch.zeros([self.memory_size, self.nfeat], requires_grad=False)
            feature_meta[-1, i] = 1

            feature_meta = feature_meta.to(self.device)
            self.model = self.model.to(self.device)
            self.model.dropout = 0

            feature_meta = feature_meta.unsqueeze(0)

            pre_time, pre_feature = self.model(feature_meta, kernel_meta)
            output[i, ] = F.softmax(pre_feature[0], dim=0)

        return output.detach().numpy().transpose()


class GCN_two_layer(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(GCN_two_layer, self).__init__()
        print("model name: GCN_two_layer")

        self.gc1 = GraphConvolution(nfeat, n1hid)
        self.gc2 = GraphConvolution(n1hid, n2hid)
        self.output_time = Linear((n2hid+1)*memory_size, 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear((n2hid+1)*memory_size, nfeat)
        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.dropout = dropout
        self.num_pos = memory_size


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = x.flatten(start_dim=1)
        x2 = adj[:, -1, :]
        x = torch.cat((x1, x2), 1)
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature


class GCN_one_layer(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(GCN_one_layer, self).__init__()
        print("model name: GCN_one_layer")

        self.gc1 = GraphConvolution(nfeat, n1hid)
        self.output_time = Linear((n2hid+1)*memory_size, 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear((n2hid+1)*memory_size, nfeat)
        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.dropout = dropout
        self.num_pos = memory_size

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = x.flatten(start_dim=1)
        x2 = adj[:, -1, :]
        x = torch.cat((x1, x2), 1)
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature

class GCN_LSTM(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(GCN_LSTM, self).__init__()
        print("model name: GCN_LSTM")

        self.gc1 = GraphConvolution(nfeat, n1hid)
        self.output_time = Linear((n2hid+1), 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear((n2hid+1), nfeat)

        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.lstm_layer = LSTM(n1hid, n2hid)


        self.dropout = dropout
        self.num_pos = memory_size
        self.embedding_pos = torch.nn.Embedding(num_embeddings=self.num_pos, embedding_dim=n2hid)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(0, 1)
        x, hidden_state = self.lstm_layer(x)
        x = x.transpose(0,1)
        x = x[:,-1,:]
        x1 = x.flatten(start_dim=1)
        x2 = adj[:, -1, -1].unsqueeze(-1)
        x = torch.cat((x1, x2), 1)
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature


class GCN_TRM(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(GCN_TRM, self).__init__()
        print("model name: GCN_TRM")

        self.gc1 = GraphConvolution(nfeat, n1hid)
        self.output_time = Linear((n2hid+1), 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear((n2hid+1), nfeat)

        self.output_feature.weight.data.normal_(0.1, 0.2)

        self.dropout = dropout
        self.num_pos = memory_size
        print("nfeat:", nfeat, " n1hid:", n1hid, " n2hid:", n2hid, " memory_size:", memory_size)
        nhead = 1  # adjustable
        nlayers = 1  # adjustable
        self.embedding_pos = torch.nn.Embedding(num_embeddings=self.num_pos, embedding_dim=n2hid)
        encoder_layers = TransformerEncoderLayer(n2hid, nhead, n2hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.src_mask = self._generate_square_subsequent_mask(self.num_pos).to(device)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x, self.src_mask)
        x = x.transpose(0,1)
        x = x[:,-1,:]
        x1 = x.flatten(start_dim=1)
        x2 = adj[:, -1, -1].unsqueeze(-1)
        x = torch.cat((x1, x2), 1)
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature


class TRM_net_1layer(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(TRM_net_1layer, self).__init__()
        print("model name: TRM_net_1layer ")
        self.linear_1 = Linear(nfeat, n1hid)
        self.output_time = Linear(n2hid, 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear(n2hid, nfeat)

        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.dropout = dropout
        self.num_pos = memory_size
        nhead = 1
        nlayers = 1
        self.embedding_pos = torch.nn.Embedding(num_embeddings=self.num_pos, embedding_dim=n2hid)
        encoder_layers = TransformerEncoderLayer(n2hid, nhead, n2hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.src_mask = self._generate_square_subsequent_mask(self.num_pos).to(device)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, adj):
        x = F.relu(self.linear_1(x))
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x, self.src_mask)
        x = x.transpose(0,1)
        x = x[:,-1,:]
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature

class TRM_net_2layer(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(TRM_net_2layer, self).__init__()
        print("model name: TRM_net_2layer ")
        self.linear_1 = Linear(nfeat, n1hid)
        self.output_time = Linear(n2hid, 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear(n2hid, nfeat)
        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.dropout = dropout
        self.num_pos = memory_size
        nhead = 1
        nlayers = 2
        self.embedding_pos = torch.nn.Embedding(num_embeddings=self.num_pos, embedding_dim=n2hid)
        encoder_layers = TransformerEncoderLayer(n2hid, nhead, n2hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.src_mask = self._generate_square_subsequent_mask(self.num_pos).to(device)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, adj):
        x = F.relu(self.linear_1(x))
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x, self.src_mask)
        x = x.transpose(0,1)
        x = x[:,-1,:]
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature

class LSTM_net(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(LSTM_net, self).__init__()
        print("model name: LSTM_net ")
        self.linear_1 = Linear(nfeat, n1hid)
        self.output_time = Linear(n2hid, 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear(n2hid, nfeat)
        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.lstm_layer = LSTM(n2hid, n2hid)
        self.dropout = dropout
        self.num_pos = memory_size
        nhead = 1
        nlayers = 2
        self.embedding_pos = torch.nn.Embedding(num_embeddings=self.num_pos, embedding_dim=n2hid)
        encoder_layers = TransformerEncoderLayer(n2hid, nhead, n2hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.src_mask = self._generate_square_subsequent_mask(self.num_pos).to(device)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, adj):
        x = F.relu(self.linear_1(x))
        x = x.transpose(0, 1)
        x, hidden_state = self.lstm_layer(x)
        x = x.transpose(0,1)
        x = x[:,-1,:]
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature

class LSTM_net_2layer(nn.Module):
    def __init__(self, nfeat, n1hid, n2hid, memory_size, dropout, device = None):
        super(LSTM_net_2layer, self).__init__()
        print("model name: LSTM_net_2layer ")
        self.linear_1 = Linear(nfeat, n1hid)
        self.output_time = Linear(n2hid, 1)
        self.output_time.weight.data.normal_(0.1, 0.2)
        self.output_feature = Linear(n2hid, nfeat)
        self.output_feature.weight.data.normal_(0.1, 0.2)
        self.lstm_layer = LSTM(n2hid, n2hid)
        self.lstm_layer2 = LSTM(n2hid, n2hid)

        self.dropout = dropout
        self.num_pos = memory_size
        nhead = 1
        nlayers = 2
        self.embedding_pos = torch.nn.Embedding(num_embeddings=self.num_pos, embedding_dim=n2hid)
        encoder_layers = TransformerEncoderLayer(n2hid, nhead, n2hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.src_mask = self._generate_square_subsequent_mask(self.num_pos).to(device)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x, adj):
        x = F.relu(self.linear_1(x))
        x = x.transpose(0, 1)
        x, hidden_state = self.lstm_layer(x)
        x, hidden_state = self.lstm_layer2(x)
        x = x.transpose(0,1)
        x = x[:,-1,:]
        pre_time = F.relu(self.output_time(x))
        pre_feature = self.output_feature(x)
        return pre_time, pre_feature
