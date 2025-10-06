import os
import random
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale

def load_data(args):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    if args.dataset in ["HW", "scene15"]:
        labels = data['truth'].flatten()
        features = data['X']
        if args.dataset == 'scene15':
            for i in range(features.shape[1]):
                features[0][i] = features[0][i].T
    else:
        features = data['X']

        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    train_mask, valid_mask, test_mask = generate_partition(labels, args)
    labels = torch.from_numpy(labels).long()
    adj_list = []
    fea_list = []

    # ######Construct KNN
    for i in range(features.shape[1]):
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix(feature):
            feature = feature.todense()
        direction_judge = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '/' + str(args.knns) + '_adj.npz'
        if os.path.exists(direction_judge):
            print("Loading laplacian matrix from " + str(i) + "th view of " + args.dataset)
            adj_s = torch.from_numpy(ss.load_npz(direction_judge).todense()).float().to(args.device)
        else:
            print("Constructing the adjacency matrix of "+ 'v' + str(i) + args.dataset)
            adj_s = construct_adjacency_matrix(feature, args.knns, args.pr1, args.pr2, args.common_neighbors)
            adj_s = ss.coo_matrix(adj_s)
            adj_s = adj_s + adj_s.T.multiply(adj_s.T > adj_s) - adj_s.multiply(adj_s.T > adj_s)
            adj_s_hat = construct_adj_hat(adj_s)
            save_direction = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '/' + str(args.knns)
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the adjacency matrix to " + save_direction)
            ss.save_npz(save_direction + '_adj.npz', adj_s_hat)
            adj_s = torch.from_numpy(adj_s_hat.todense()).float().to(args.device)
        adj_list.append(adj_s)

        feature = torch.from_numpy(feature).float().to(args.device)
        fea_list.append(feature)


    return adj_list, fea_list, labels, train_mask, valid_mask, test_mask




def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_construct = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>
    adj = ss.coo_matrix(adj_construct)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if prunning_one:
        original_adj = adj.toarray()
        judges_matrix = original_adj == original_adj.T
        adj = original_adj * judges_matrix
        adj = ss.csc_matrix(adj)
    adj = adj - ss.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        adj = adj.toarray()
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = ss.coo_matrix(adj)
        adj.eliminate_zeros()
    adj = ss.coo_matrix(adj)
    return adj


def construct_adj_hat(adj):
    """
        construct the Laplacian matrix
    :param adj: original adj matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    # adj = ss.coo_matrix(adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # lp = ss.eye(adj.shape[0]) - adj_wave
    return adj_wave


def generate_partition(gnd, args):
    '''
    Generate permutation for training, validating and testing data.
    '''
    train_ratio =  args.train_ratio
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio
    N = gnd.shape[0]
    each_class_num = count_each_class_num(gnd)
    training_each_class_num = {} ## number of labeled samples for each class

    for label in each_class_num.keys():
        if args.data_split_mode == "Ratio":
            training_each_class_num[label] = max(round(each_class_num[label] * train_ratio), 1) # min is 1
            valid_num = max(round(N * valid_ratio), 0) # min is 1
            test_num= max(round(N * test_ratio), 1) # min is 1
        else:
            training_each_class_num[label] = args.num_train_per_class
            valid_num = args.num_val
            test_num = args.num_test

    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    # shuffle the data
    data_idx = [i for i in range(len(gnd))]
    # print(index)
    if args.seed >= 0:
        random.seed(args.seed)
        random.shuffle(data_idx)

    # Get training data
    for idx in data_idx:
        label = gnd[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        if (valid_num > 0):
            valid_num -= 1
            valid_mask[idx] = True
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True
    return train_mask, valid_mask, test_mask


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict
