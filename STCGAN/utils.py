# Utils
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import networkx as nx
import random
from scipy.spatial import distance
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-15


def add_contrastive_label(n_spot):
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    return np.concatenate([one_matrix, zero_matrix], axis=0)


def load_data(path):
    st_adata = sc.read_h5ad(path + "st_adata.h5ad")
    sc_adata = sc.read_h5ad(path + "sc_adata.h5ad")
    graph_dict = np.load(path + "graph_dict.npy", allow_pickle=True).tolist()
    return st_adata, sc_adata, graph_dict


def filter_with_overlap_gene(st_adata, sc_adata):
    genes = list(set(st_adata.var.index) & set(sc_adata.var.index))
    print('Number of overlap genes:', len(genes))
    genes.sort()
    st_adata.uns["overlap_genes"] = genes
    sc_adata.uns["overlap_genes"] = genes
    st_adata = st_adata[:, genes]
    sc_adata = sc_adata[:, genes]
    return st_adata, sc_adata


def extract_top_value(map_matrix, retain_percent=0.1):
    top_k = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    return output


def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)

    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    mat = mat.iloc[:, 0:n_type]
    return mat


def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    map_matrix = adata.obsm['map_matrix']
    map_matrix = extract_top_value(map_matrix, retain_percent)
    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_projection = map_matrix.dot(matrix_cell_type.values)
    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=matrix_cell_type.columns)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)
    adata.obs[df_projection.columns] = df_projection


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


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
