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
import scienceplots
import matplotlib.ticker as mtick
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj, dropout_edge, add_remaining_self_loops,\
    segregate_self_loops, index_to_mask, to_undirected

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GAE
from model import Encoder, MLP, Model
from eval import node_classification, node_classification_wiki, LREvaluator
from utils import tab_printer, dataset_loader, drop_feature, get_split, save_variable, edgeindex2adj, add_edge
import pickle
from torch.utils.tensorboard import SummaryWriter

def train(model: Model, x, edge_index, optimizer, epoch, args):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_mask_1 = dropout_edge(edge_index, p=args.drop_edge_rate) #pyg_2.2.0
    edge_index_2, edge_mask_2 = dropout_edge(edge_index, p=args.drop_edge_rate)
    x_1 = drop_feature(x, args.drop_feature_rate)
    x_2 = drop_feature(x, args.drop_feature_rate)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    edge_ind = edge_index[:, edge_mask_1 * edge_mask_2]
    adj = edgeindex2adj(edge_ind, x_1.shape[0]).to_dense()

    loss = model.gc_loss(z1, z2, adj, args.batch_size)

    loss.backward()
    optimizer.step()
    return loss.item()

def train_big(model: Model, x, edge_index, optimizer, epoch, args):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_mask_1 = dropout_edge(edge_index, p=args.drop_edge_rate) #pyg_2.2.0
    edge_index_2, edge_mask_2 = dropout_edge(edge_index, p=args.drop_edge_rate)

    x_1 = drop_feature(x, args.drop_feature_rate)
    x_2 = drop_feature(x, args.drop_feature_rate)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    edge_ind = edge_index[:, edge_mask_1 * edge_mask_2]
    loss = model.gc_loss(z1, z2, edge_ind, args.batch_size, epoch)

    loss.backward()
    optimizer.step()
    return loss.item()

def test(model: Model, data, args):
    model.eval()
    z = model(data.x, data.edge_index)
    if args.dataset.lower() == 'wikics':
        LREvaluator(args.le_epoch)(z, data, wikics=True)
    else:
        LREvaluator(args.le_epoch)(z, data)

def test_cv(model: Model, data, args):
    model.eval()
    z = model(data.x, data.edge_index)
    rec = (torch.mm(F.normalize(z), F.normalize(z).t())+ torch.Tensor([1.0]).to(z.device)) / 2
    adj = edgeindex2adj(data.edge_index, z.shape[0]).to_dense()
    loss = -((adj * torch.log(rec + 1e-6)).sum(1) / adj.sum(1) + (
            (1 - adj) * torch.log(1 - rec + 1e-6)).sum(1) / (1 - adj).sum(1))
    print('rec_error: ', loss.mean())

    if args.dataset.lower() == 'wikics':
        node_classification_wiki(z, data)
    else:
        node_classification(z, data.y)

if __name__ == '__main__':
    args = parameter_parser()
    tab_printer(args)
    plt.style.use(['science', 'ieee', 'std-colors', 'no-latex','grid'])
    plt.rc('font', family='Times New Roman')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    dataset = dataset_loader(path = args.path, name = args.dataset)
    data = dataset[0].to(device)

    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu':F.rrelu})[args.activation]
    base_layer = ({'GCNConv': GCNConv, 'SAGEConv': SAGEConv})[args.base_layer]

    encoder = Encoder(dataset.num_features, args.dim_h, activation,
                      base_layer=base_layer, num_layer=args.num_layer).to(device)



    model = Model(encoder, args.dim_h, args.dim_p, args.tau, args, data.y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses = []
    with tqdm(total=args.train_epoch, desc='(T)') as pbar:
        for epoch in range(1, args.train_epoch+1):
            loss = train_big(model, data.x, data.edge_index, optimizer, epoch, args)

            losses.append(loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_cv(model, data, args)
