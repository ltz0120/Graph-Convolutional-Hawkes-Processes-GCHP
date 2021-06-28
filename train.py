"""
This script contains the pipeline of training the model with the dataset including synthetic data,
 ATM, IPTV and Weeplace. You may select the model architecture from 'GCN_one_layer','GCN_two_layer',
  'GCN_TRM', 'GCN_LSTM'.
"""


import sys
from DataLoader__ import EventSampler
import DataReader
import torch
import time
import dev.util as util
import numpy as np
import pandas as pd
import torch.optim as optim
import argparse
import GCHP_models

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='IPTV',type=str,
                    help='You can select dataset from HawkesProcess_synthetic, ATM, Weeplace, and IPTV')
parser.add_argument('--model', default='GCN_two_layer',type=str,
                    help='We implement the models: GCN_one_layer, GCN_two_layer, GCN_TRM, and GCN_LSTM.')
parser.add_argument('--epochs', default=200, type=int, help='Set number of epochs.')
parser.add_argument('--batch_size', default=3000, type=int, help='Set batch_size.')
parser.add_argument('--memory_size', default=3, help='Set the memoty size')
parser.add_argument('--theta', default=1.0, type=float, help='The hyper-parameter theta which controls the loss of time and feature.')
parser.add_argument('--eta', default=1.0, type=float, help='The hyper-parameter eta which controls the loss of time and feature.')
parser.add_argument('--gpu', default=0, type=int, help='Select GPU device.')
parser.add_argument('--n_feature', default=0, type=int, help='Select GPU device.')
parser.add_argument('--hidden', default=32, type=int, help='Hidden dimension size.')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout ratio.')
parser.add_argument('--record', default=True, type=bool, help='Record the results at each epoch.')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay parameter.')
parser.add_argument('--loss_type', default='likelihood', type=str, help='Set the loss type. You can select likelihood or mse.')
parser.add_argument('--seed', type=int, default=5151, help='Random seed.')

def train_model(args):
    train_dataset = pd.read_csv( util.DATA_DIR + args.dataset + '_train.csv')
    test_dataset = pd.read_csv( util.DATA_DIR + args.dataset + '_test.csv')
    if args.dataset == 'Weeplace':
        train_dataset = train_dataset.drop(columns = ['lat', 'lon'])
        test_dataset = test_dataset.drop(columns = ['lat', 'lon'])
    if args.dataset == 'IPTV':
        trainset = DataReader.load_csv(train_dataset, single_realization=False, categorical=True)
        testset = DataReader.load_csv(test_dataset, single_realization=False, categorical=True)
    else:
        trainset = DataReader.load_csv(train_dataset, single_realization=False)
        testset = DataReader.load_csv(test_dataset, single_realization=False)
    train_dataloader = EventSampler(trainset, memory_size=args.memory_size, batch_size=args.batch_size,
                                    kernel_bandwidth=2, categorical=True)
    test_dataloader = EventSampler(testset, memory_size=args.memory_size, batch_size=args.batch_size,
                                   kernel_bandwidth=2, categorical=True)
    nfeat = train_dataloader.__getitem__(0)[1].shape[-1]
    params = {
        "model": args.model,
        'categorical': True,
        "memory_size": args.memory_size,
        "theta": args.theta,
        "nfeat": nfeat,
        "eta": args.eta,
        "record_gc": False,
        "record": args.record,
        "device": args.gpu,
        "n1hid": args.hidden,
        "n2hid": args.hidden,
        "dropout": args.dropout,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "dataset": args.dataset,
        "loss_type": "likelihood"  # can select "likelihood" loss or "mse" loss
    }
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    myGCHP = GCHP_models.GCHP_model(params)
    optimizer = optim.Adam(myGCHP.model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
    myGCHP.fit(train_dataloader, optimizer, epochs=params['epochs'], scheduler=scheduler, test_dataloader=test_dataloader,
               verbose=True)

if __name__ == "__main__":
    args = parser.parse_args()
    train_model(args)