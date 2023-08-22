import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  SAGEConv, GATConv, ClusterGCNConv, GCNConv, DeepGCNLayer, GENConv, LayerNorm
from torch_geometric.utils import add_remaining_self_loops, get_laplacian
from utils import edgeindex2adj
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import silhouette_score as sh_score
from tqdm import tqdm



class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self): # Override function that finds format to use.
        self.format = "%1.1f" # Give format here


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_layer=GCNConv, num_layer: int = 2):
        super(Encoder, self).__init__()

        self.base_layer = base_layer
        self.activation = activation
        # assert num_layer >= 2
        self.num_layer = num_layer
        self.factor = 2
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if num_layer >=2:
            self.convs.append(base_layer(in_channels, self.factor*out_channels))

            self.bns.append(torch.nn.BatchNorm1d(self.factor * out_channels))
            for _ in range(1, num_layer-1):
                self.convs.append(base_layer(self.factor*out_channels, self.factor*out_channels))
                self.bns.append(torch.nn.BatchNorm1d(self.factor * out_channels))
            self.convs.append(base_layer(self.factor*out_channels, out_channels))
            self.bns.append(torch.nn.BatchNorm1d(out_channels))
        else:
            self.convs.append(base_layer(in_channels, out_channels))
            self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        rep = []
        for i in range(self.num_layer):
            x = self.activation(self.convs[i](x, edge_index))
            rep.append(x)
        return x

    @torch.no_grad()  # for large graph
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Inferring')
        device = x_all.device
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = self.activation(conv(x, batch.edge_index.to(device)))
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all.to(device)


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 num_layer: int = 2):
        super(MLP, self).__init__()
        self.activation = activation
        assert num_layer >= 2
        self.num_layer = num_layer
        self.factor = 2
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, self.factor*out_channels))
        self.bns.append(torch.nn.BatchNorm1d(self.factor*out_channels))
        for _ in range(1, num_layer-1):
            self.lins.append(nn.Linear(self.factor*out_channels, self.factor*out_channels))
            self.bns.append(torch.nn.BatchNorm1d(self.factor * out_channels))
        self.lins.append(nn.Linear(self.factor*out_channels, out_channels))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        for i in range(self.num_layer):
            x = self.activation(self.bns[i](self.lins[i](x)))
        return x

    @torch.no_grad()  # for large graph
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset))
        pbar.set_description('Inferring')
        device = x_all.device
        xs = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)][:batch.batch_size].to(device)
            for i in range(self.num_layer):
                x = self.activation(self.bns[i](self.lins[i](x)))
            xs.append(x.cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class Model(torch.nn.Module):
    def __init__(self, encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, args=None, y=None):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.factor = 1
        self.fc1 = torch.nn.Linear(num_hidden, self.factor * num_proj_hidden)
        self.bn1 = torch.nn.BatchNorm1d(self.factor * num_proj_hidden)
        self.fc2 = torch.nn.Linear(self.factor *num_proj_hidden, self.factor *num_hidden)
        self.bn2 = torch.nn.BatchNorm1d(self.factor * num_proj_hidden)
        self.gcn1= GCNConv(num_hidden, self.factor*num_proj_hidden)
        self.gcn2 = GCNConv(self.factor *num_proj_hidden, self.factor *num_hidden)
        self.args = args
        self.y = y
        self.k = 0
        self.activation = torch.nn.PReLU()#
        self.EPS = torch.Tensor([1e-6]).to(self.device)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(z)))


    def gcn_projection(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.activation(self.gcn1(z, edge_index))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):

        return torch.mm(z1, z2.t())

    def cossim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch: int, args = None, semi = None):
        intra_sim = self.sim(z1,z1)
        inter_sim = self.sim(z1,z2)
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(intra_sim)
        between_sim = f(inter_sim)
        if epoch == self.args.train_epoch:

            torch.save(z1.clone().detach().cpu(), 'results/' + args.dataset + '/grace_z.pt')
            torch.save(between_sim.clone().detach().cpu(), 'results/' + self.args.dataset + '/grace_rec.pt')
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int, epoch: int, args = None, semi = None):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            intra_sim = self.sim(z1[mask], z1)
            inter_sim = self.sim(z1[mask], z2)
            refl_sim = f(intra_sim)  # [B, N]
            between_sim = f(inter_sim)  # [B, N]
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0, epoch: int = 0, args = None):

        h1 = F.normalize(self.projection(z1))
        h2 = F.normalize(self.projection(z2))

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, epoch, args, semi='a')
            l2 = self.semi_loss(h2, h1, epoch, args, semi='b')
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size, epoch, args, semi='a')
            l2 = self.batched_semi_loss(h2, h1, batch_size, epoch, args, semi='b')

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

    def gc_loss_asy(self, z1: torch.Tensor, z2: torch.Tensor, edge_ind1: torch.Tensor, edge_ind2: torch.Tensor, batch_size: int = 0):
        h1 = F.normalize(z1)
        h2 = F.normalize(z2)
        if batch_size == 0:
            adj_1 = edgeindex2adj(edge_ind1, h1.shape[0]).to_dense()
            adj_2 = edgeindex2adj(edge_ind2, h2.shape[0]).to_dense()
            rec_1 = (torch.mm(h1,h2.t())+torch.Tensor([1.0]).to(self.device))/2
            rec_2 = (torch.mm(h2,h1.t())+torch.Tensor([1.0]).to(self.device))/2
            l1 = -((adj_1*torch.log(rec_1+self.EPS)).sum(1)/adj_1.sum(1)+((1-adj_1)*torch.log(1-rec_1+self.EPS)).sum(1)/(1-adj_1).sum(1))
            l2 = -((adj_2*torch.log(rec_2+self.EPS)).sum(1)/adj_2.sum(1)+((1-adj_2)*torch.log(1-rec_2+self.EPS)).sum(1)/(1-adj_2).sum(1))
            return torch.mean((l1+l2)/2)
        else:
            l1 = self.batch_gc_loss_asy(h1, h2, edge_ind1, batch_size)
            return torch.mean(l1)


    def batch_gc_loss_asy(self, h1: torch.Tensor, h2: torch.Tensor, edge_index: torch.Tensor, batch_size: int = 0):
        device = h1.device
        num_nodes = h1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        adj = edgeindex2adj(edge_index, h1.shape[0])  # torch.sparse.tensor
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]  # index for node
            adj_b = adj.index_select(dim=0, index=mask).to_dense()  # row_sampling
            rec_b = (torch.mm(h1[mask], h2.t()) + torch.Tensor([1.0]).to(self.device)) / 2
            losses.append(-((adj_b * torch.log(rec_b + self.EPS)).sum(1) / adj_b.sum(1) + (
                    (1 - adj_b) * torch.log(1 - rec_b + self.EPS)).sum(1) / (1 - adj_b).sum(1)))
        return torch.cat(losses)


    def gc_loss(self, z1: torch.Tensor, z2: torch.Tensor, edge_index: torch.Tensor, batch_size: int = 0, epoch: int =0):
        h1 = F.normalize(z1)
        h2 = F.normalize(z2)

        if batch_size == 0:
            adj = edgeindex2adj(edge_index, h1.shape[0]).to_dense()
            rec = (torch.mm(h1, h2.t()) + torch.Tensor([1.0]).to(self.device)) / 2
            loss = -((adj * torch.log(rec + self.EPS)).sum(1) / adj.sum(1) + (
                        (1 - adj) * torch.log(1 - rec + self.EPS)).sum(1) / (1 - adj).sum(1))
        else:
            loss = self.batch_gc_loss(h1, h2, edge_index, batch_size)
        ret = loss.mean()
        return ret

    def batch_gc_loss(self, h1: torch.Tensor, h2: torch.Tensor, edge_index: torch.Tensor, batch_size: int = 0):
        device = h1.device
        num_nodes = h1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        adj = edgeindex2adj(edge_index, h1.shape[0])  # torch.sparse.tensor
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]  # index for node
            adj_b = adj.index_select(dim=0, index=mask).to_dense()  # row_sampling
            rec_b = (torch.mm(h1[mask], h2.t()) + torch.Tensor([1.0]).to(self.device)) / 2
            losses.append(-((adj_b * torch.log(rec_b + self.EPS)).sum(1) / adj_b.sum(1) + (
                    (1 - adj_b) * torch.log(1 - rec_b + self.EPS)).sum(1) / (1 - adj_b).sum(1)))
        return torch.cat(losses)

















