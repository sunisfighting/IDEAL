import argparse
from parsers import parameter_parser
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj, dropout_edge, add_remaining_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GAE
from model import Encoder, MLP, Model
from eval import node_classification, node_classification_wiki, node_classification_reddit, LREvaluator
from utils import tab_printer, dataset_loader, drop_feature, get_split, save_variable, edgeindex2adj, add_edge
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
import copy
import pickle
from torch.utils.tensorboard import SummaryWriter

def train(model: Model, train_loader, optimizer, args):
    model.train()
    total_loss = total_num = 0
    for sub_data in train_loader:
        optimizer.zero_grad()
        sub_data.to(model.device)
        x = sub_data.x
        edge_index = sub_data.edge_index  # edge_index 重新分配, 从0到num-1

        edge_index_1, edge_mask_1 = dropout_edge(edge_index, p=args.drop_edge_rate) #pyg_2.2.0
        edge_index_2, edge_mask_2 = dropout_edge(edge_index, p=args.drop_edge_rate)
        x_1 = drop_feature(x, args.drop_feature_rate)
        x_2 = drop_feature(x, args.drop_feature_rate)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        edge_ind = edge_index[:, edge_mask_1*edge_mask_2]
        loss = model.gc_loss(z1, z2, edge_ind, args.batch_size)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * x.shape[0]
        total_num += x.shape[0]

    return total_loss/total_num

def test(model: Model, train_loader, val_loader, test_loader, args):
    model.eval()
    zs = []
    ys = []
    node_counter = []
    for loader in [train_loader, val_loader, test_loader]:
        counter = 0
        for sub_data in loader:
            sub_data.to(model.device)
            x = sub_data.x
            edge_index = sub_data.edge_index
            z = model(x, edge_index)
            counter += sub_data.num_nodes
            zs.append(z)
            ys.append(sub_data.y)
        node_counter.append(counter)
    z, y = torch.cat(zs, dim=0), torch.cat(ys, dim=0)
    split_idx = [sum(node_counter[:i+1]) for i in range(len(node_counter))]
    LREvaluator(args.le_epoch)(z, y, multi_class=True, split_idx= split_idx)


def test_cv(model: Model, data, subgraph_loader, args):
    model.eval()
    z = model.encoder.inference(data.x, subgraph_loader)
    if args.dataset.lower() == 'wikics':
        node_classification_wiki(z, data)
    elif args.dataset.lower() in ['reddit', 'arxiv']:
        node_classification_reddit(z, data)
    else:
        node_classification(z, data.y)

if __name__ == '__main__':
    args = parameter_parser()
    tab_printer(args)
    # plt.style.use(['science', 'ieee', 'std-colors', 'no-latex','grid'])
    # plt.rc('font', family='Times New Roman')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    assert args.dataset.lower() == 'ppi'
    train_set, val_set, test_set = dataset_loader(path = args.path, name = args.dataset)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)



    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu':F.rrelu, 'elu': F.elu})[args.activation]
    base_layer = ({'GCNConv': GCNConv, 'SAGEConv': SAGEConv})[args.base_layer]

    encoder = Encoder(train_set.num_features, args.dim_h, activation,
                      base_layer=base_layer, num_layer=args.num_layer).to(device)

    model = Model(encoder, args.dim_h, args.dim_p, args.tau, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    with tqdm(total=args.train_epoch, desc='(T)') as pbar:
        for epoch in range(1, args.train_epoch+1):
            loss = train(model, train_loader, optimizer, args)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test(model, train_loader, val_loader, test_loader, args)
