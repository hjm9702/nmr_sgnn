import numpy as np
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset, Subset
from dgllife.utils import RandomSplitter


from dataset import GraphDataset
from util import collate_reaction_graphs
from model import training, inference


from gnn_models.pagtn import nmrPAGTN
from gnn_models.mpnn import nmrMPNN

from search import search

from sklearn.metrics import mean_absolute_error
import time, sys, csv


def train(args):
    fold_seed = args.fold_seed
    model = args.model
    embed_mode = args.embed_mode
    target = args.target
    edge_mode = args.edge_mode
    memo = args.memo
    

    

    data_split = [0.9, 0.1]
    batch_size = 128
    if memo:
        model_path = f'./model/{model}_{target}_{embed_mode}_{edge_mode}_{fold_seed}_{memo}.pt'
    else:
        model_path = f'./model/{model}_{target}_{embed_mode}_{edge_mode}_{fold_seed}.pt'

    

    random_seed = 27407 + fold_seed

    data = GraphDataset(target, edge_mode)
    kfold = RandomSplitter()
    data_fold = kfold.k_fold_split(data, k=10, random_state=27407, log=False)
    trainval_set, test_set = data_fold[fold_seed]

    train_set, val_set = split_dataset(trainval_set, data_split, shuffle=True, random_state=random_seed)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    
    #large_idx = [i for i in range(len(test_set)) if test_set[i][1] > 64]
    
    #large_idx = [i for i in range(len(test_set)) if test_set[i][0].n_node_w_H > 64]
    #large_test_set = Subset(test_set, large_idx)
    #large_test_loader = DataLoader(dataset=large_test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    train_y = np.hstack([inst[-2][inst[-1]] for inst in iter(train_loader.dataset)])
    train_y_mean, train_y_std = np.mean(train_y), np.std(train_y)

    node_dim = data.node_attr.shape[1]
    edge_dim = data.edge_attr.shape[1]

    if model == 'pagtn':
        net = nmrPAGTN(node_dim, edge_dim, embed_mode).cuda()
    elif model == 'mpnn':
        net = nmrMPNN(node_dim, edge_dim, embed_mode).cuda()

    print('--- data_size:', data.__len__())
    print('--- train/val/test: %d/%d/%d' %(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    print('--- model_path:', model_path)

    # training
    print('-- TRAINING')
    net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path)
    
    # inference
    test_y = np.hstack([inst[-2][inst[-1]] for inst in iter(test_loader.dataset)])
    #large_test_y = np.hstack([inst[-2][inst[-1]] for inst in iter(large_test_loader.dataset)])

    test_y_pred = inference(net, test_loader, train_y_mean, train_y_std)
    test_mae = mean_absolute_error(test_y, test_y_pred)

    #large_test_y_pred = inference(net, large_test_loader, train_y_mean, train_y_std)
    #large_test_mae = mean_absolute_error(large_test_y, large_test_y_pred)

    




    print('-- prediction RESULT')
    print('--- test MAE      ', test_mae)
    #print('--- large test MAE', large_test_mae)

    return net, test_mae