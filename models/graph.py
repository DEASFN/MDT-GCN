import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch

# joint_graph = {0: [0, 1, 14, 15],
#                1: [0, 1, 2, 5, 8, 11],
#                2: [1, 2, 3],
#                3: [2, 3, 4],
#                4: [3, 4],
#                5: [1, 5, 6],
#                6: [5, 6, 7],
#                7: [6, 7],
#                8: [1, 8, 9],
#                9: [8, 9, 10],
#                10:[9, 10],
#                11:[1, 11, 12],
#                12:[11, 12, 13],
#                13:[12, 13],
#                14:[0, 14, 16],
#                15:[0, 15, 17],
#                16:[14, 16],
#                17:[15, 17]}

JH_graph = {0: [0, 1, 14, 15, 18],
               1: [0, 1, 2, 5, 8, 11, 18],
               2: [1, 2, 3, 18],
               3: [2, 3, 4, 18],
               4: [3, 4, 18],
               5: [1, 5, 6, 18],
               6: [5, 6, 7, 18],
               7: [6, 7, 18],
               8: [1, 8, 9, 18],
               9: [8, 9, 10, 18],
               10:[9, 10, 18],
               11:[1, 11, 12, 18],
               12:[11, 12, 13, 18],
               13:[12, 13, 18],
               14:[0, 14, 16, 18],
               15:[0, 15, 17, 18],
               16:[14, 16, 18],
               17:[15, 17, 18],
               18:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}

JH_graph_center = {0: [0],
                   1: [1],
                   2: [2],
                   3: [3],
                   4: [4],
                   5: [5],
                   6: [6],
                   7: [7],
                   8: [8],
                   9: [9],
                   10: [10],
                   11: [11],
                   12: [12],
                   13: [13],
                   14: [14],
                   15: [15],
                   16: [16],
                   17: [17],
                   18: [18]}

JH_graph_closer = {0: [1],
                   1: [],
                   2: [1],
                   3: [2],
                   4: [3],
                   5: [1],
                   6: [5],
                   7: [6],
                   8: [1],
                   9: [8],
                   10: [9],
                   11: [1],
                   12: [11],
                   13: [12],
                   14: [0, 16],
                   15: [0, 17],
                   16: [],
                   17: [],
                   18: []}

JH_graph_farther = {0: [14, 15],
                    1: [0, 2, 5, 8, 11],
                    2: [3],
                    3: [4],
                    4: [],
                    5: [6],
                    6: [7],
                    7: [],
                    8: [9],
                    9: [10],
                    10: [],
                    11: [12],
                    12: [13],
                    13: [],
                    14: [],
                    15: [],
                    16: [14],
                    17: [15],
                    18: []}

JH_graph_symmetric = {0: [],
                       1: [],
                       2: [5],
                       3: [6],
                       4: [7],
                       5: [2],
                       6: [3],
                       7: [4],
                       8: [11],
                       9: [12],
                       10: [13],
                       11: [8],
                       12: [9],
                       13: [10],
                       14: [15],
                       15: [14],
                       16: [17],
                       17: [16],
                       18: []}

JH_graph_background = {0: [18],
                       1: [18],
                       2: [18],
                       3: [18],
                       4: [18],
                       5: [18],
                       6: [18],
                       7: [18],
                       8: [18],
                       9: [18],
                       10: [18],
                       11: [18],
                       12: [18],
                       13: [18],
                       14: [18],
                       15: [18],
                       16: [18],
                       17: [18],
                       18: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}

# pascal_graph = {0:[0],
#                 1:[1, 2],
#                 2:[1, 2, 3, 5],
#                 3:[2, 3, 4],
#                 4:[3, 4],
#                 5:[2, 5, 6],
#                 6:[5, 6]}
#
# cihp_graph = {0: [],
#               1: [2, 13],
#               2: [1, 13],
#               3: [14, 15],
#               4: [13],
#               5: [6, 7, 9, 10, 11, 12, 14, 15],
#               6: [5, 7, 10, 11, 14, 15, 16, 17],
#               7: [5, 6, 9, 10, 11, 12, 14, 15],
#               8: [16, 17, 18, 19],
#               9: [5, 7, 10, 16, 17, 18, 19],
#               10:[5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17],
#               11:[5, 6, 7, 10, 13],
#               12:[5, 7, 10, 16, 17],
#               13:[1, 2, 4, 10, 11],
#               14:[3, 5, 6, 7, 10],
#               15:[3, 5, 6, 7, 10],
#               16:[6, 8, 9, 10, 12, 18],
#               17:[6, 8, 9, 10, 12, 19],
#               18:[8, 9, 16],
#               19:[8, 9, 17]}
#
# atr_graph = {0: [],
#               1: [2, 11],
#               2: [1, 11],
#               3: [11],
#               4: [5, 6, 7, 11, 14, 15, 17],
#               5: [4, 6, 7, 8, 12, 13],
#               6: [4,5,7,8,9,10,12,13],
#               7: [4,11,12,13,14,15],
#               8: [5,6],
#               9: [6, 12],
#               10:[6, 13],
#               11:[1,2,3,4,7,14,15,17],
#               12:[5,6,7,9],
#               13:[5,6,7,10],
#               14:[4,7,11,16],
#               15:[4,7,11,16],
#               16:[14,15],
#               17:[4,11],
#               }
#
# cihp2pascal_adj = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#                               [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])




def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj)) # return a adjacency matrix of adj ( type is numpy)
    adj_normalized = normalize_adj(adj) #

    #import pdb
    #pdb.set_trace()
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()

def preprocess_adj_5part(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    x = np.zeros(shape=(19, 19))

    for key, value in adj.items():
        for i in range(len(value)):
            x[key][value[i]] = 1

    D = np.array(np.sum(x, axis=1)).reshape(-1)
    D = np.power(D, -1)
    D[np.isinf(D)] = 0

    for i in range(19):
        x[i] = x[i] * D[i]

    return x

def row_norm(inputs):
    outputs = []
    for x in inputs:
        xsum = x.sum()
        x = x / xsum
        outputs.append(x)
    return outputs


def normalize_adj_torch(adj):
    # print(adj.size())
    if len(adj.size()) == 4:
        new_r = torch.zeros(adj.size()).type_as(adj)
        for i in range(adj.size(1)):
            adj_item = adj[0,i]
            rowsum = adj_item.sum(1)
            d_inv_sqrt = rowsum.pow_(-0.5)
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_item), d_mat_inv_sqrt)
            new_r[0,i,...] = r
        return new_r
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow_(-0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    r = torch.matmul(torch.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    return r

# def row_norm(adj):




if __name__ == '__main__':
    #a= row_norm(cihp2pascal_adj)
    #print(a)
    #print(cihp2pascal_adj)
    # print(a.shape)
    adj = preprocess_adj(pascal_graph)
    #adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    #adj3 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7).cuda()
    import pdb
    pdb.set_trace()

    preprocess_adj(joint_graph)

