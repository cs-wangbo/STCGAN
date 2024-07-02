# Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import GCNConv, GATConv, GATv2Conv, SAGEConv, SGConv, TAGConv


class Discriminator(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(Discriminator, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_feature, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        y = self.fc(x1)
        y1 = self.softmax(y)
        y = self.fc(x2)
        y2 = self.softmax(y)
        logits = torch.cat((y1, y2), 0)
        return logits


class GCN(nn.Module):
    def __init__(self, nfeat, out):
        super(GCN, self).__init__()
        #GCNConv, GATConv, GATv2Conv, SAGEConv, SGConv, TAGConv
        from torch_geometric.nn import Sequential, BatchNorm
        self.gc1 = Sequential('x, edge_index', [
            (GATv2Conv(nfeat, out * 2), 'x, edge_index -> x1'),
            BatchNorm(out * 2),
            nn.ReLU(inplace=True),
        ])
        self.gc2 = Sequential('x, edge_index', [
            (GATv2Conv(out * 2, out), 'x, edge_index -> x1'),
        ])
        self.gc3 = Sequential('x, edge_index', [
            (GATv2Conv(out * 2, out), 'x, edge_index -> x1'),
        ])

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        mu = self.gc2(x, adj)
        logvar = self.gc3(x, adj)
        return mu, logvar


class ST(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(ST, self).__init__()
        self.GCN = GCN(nfeat, nhid2)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU(),
            torch.nn.Linear(nhid1, nfeat))
        self.da = InnerProductDecoder(0)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.GCN(x, adj)
        emb = self.reparameterize(mu, logvar)
        readj = self.da(emb)
        re_x = self.decoder(emb)
        return emb, re_x, readj, mu, logvar


class SCC(torch.nn.Module):
    def __init__(self, input_dim=2000):
        super(SCC, self).__init__()
        self.input_dim = input_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.input_dim))

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return emb, out,


class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)

    def forward(self):
        return self.M


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()

    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)
