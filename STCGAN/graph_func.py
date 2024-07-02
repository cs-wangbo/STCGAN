#
import networkx as nx
import numpy as np
import sklearn
import torch
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph


def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def spatial_construct_graph(adj_coo, metric='euclidean', k=15):
    edgeList = []
    l = 200

    for node_idx in range(adj_coo.shape[0]):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat_spatial = distance.cdist(tmp, adj_coo, metric)
        distMat = np.exp(-0.5 * (distMat_spatial ** 2) / (l ** 2))

        res = np.argsort(-distMat)[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1
            else:
                weight = 0
            edgeList.append((node_idx, res[0][j], weight))
    return edgeList


def graph_construction(adj_coo, metric='euclidean', k=15):
    adata_S = spatial_construct_graph(adj_coo, metric=metric, k=k)
    graphdict_s = edgeList2edgeDict(adata_S, adj_coo.shape[0], )

    adj = nx.to_scipy_sparse_array(nx.from_dict_of_lists(graphdict_s)).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict_s))

    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    graph_dict = {
        "edge_index": edge_index,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }
    return graph_dict
