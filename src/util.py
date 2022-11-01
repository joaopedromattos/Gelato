import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, k_hop_subgraph


def set_random_seed(random_seed):
    """
    Set the random seed.
    :param random_seed: Seed to be set.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_dataset(dataset):
    """
    Load the dataset from PyG.

    :param dataset: name of the dataset. Options: 'Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers'
    :return: PyG dataset data.
    """
    data_folder = f'data/{dataset}/'
    if dataset in ('Cora', 'CiteSeer', 'PubMed'):
        pyg_dataset = Planetoid(data_folder, dataset)
    elif dataset in ('Photo', 'Computers'):
        pyg_dataset = Amazon(data_folder, dataset)
    else:
        raise NotImplementedError(f'{dataset} not supported. ')
    data = pyg_dataset.data
    return data


def split_dataset(data, valid_ratio=0.05, test_ratio=0.1, random_seed=0):
    """
    Split the edges/nonedges for biased training, full training, (full) validation and (full) testing.

    :param data: PyG dataset data.
    :param valid_ratio: ratio of validation edges.
    :param test_ratio: ratio of test edges.
    :param random_seed: random seed for the split.
    :return: edge splits
    """

    set_random_seed(random_seed)
    n = data.num_nodes
    split_data = train_test_split_edges(data, valid_ratio, test_ratio)
    split_edge = {'biased_train': {}, 'valid': {}, 'test': {}, 'train': {}}

    # Biased training with negative sampling.
    split_edge['biased_train']['edge'] = split_data.train_pos_edge_index.t()
    edge_index, _ = add_self_loops(split_data.train_pos_edge_index)  # To avoid negative sampling of self loops.
    split_data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=split_data.num_nodes,
        num_neg_samples=split_data.train_pos_edge_index.size(1))
    split_edge['biased_train']['edge_neg'] = split_data.train_neg_edge_index.t()

    # Full training with all negative pairs in the training graph (including validation and testing positive edges).
    split_edge['train']['edge'] = split_data.train_pos_edge_index.t()
    train_edge_neg_mask = torch.ones((n, n), dtype=bool)
    train_edge_neg_mask[tuple(split_edge['train']['edge'].t().tolist())] = False
    train_edge_neg_mask = torch.triu(train_edge_neg_mask, 1)
    split_edge['train']['edge_neg'] = torch.nonzero(train_edge_neg_mask)

    # Full validation with all negative pairs in the training graph (including testing positive edges, excluding validation positive edges).
    split_edge['valid']['edge'] = split_data.val_pos_edge_index.t()
    valid_edge_neg_mask = train_edge_neg_mask.clone()
    valid_edge_neg_mask[tuple(split_edge['valid']['edge'].t().tolist())] = False
    split_edge['valid']['edge_neg'] = torch.nonzero(valid_edge_neg_mask)

    # Full testing with all negative pairs in the training graph (excluding validation and testing positive edges).
    split_edge['test']['edge'] = split_data.test_pos_edge_index.t()
    test_edge_neg_mask = valid_edge_neg_mask.clone()
    test_edge_neg_mask[tuple(split_edge['test']['edge'].t().tolist())] = False
    split_edge['test']['edge_neg'] = torch.nonzero(test_edge_neg_mask)

    return split_edge


def compute_edges(split_edge):
    """
    Compute the train, valid, and test edges based on edge split.

    :param split_edge: edge split.
    :return: train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true
    """

    # Train edges.
    train_edges_pos = split_edge['train']['edge']
    train_edges_pos = train_edges_pos[train_edges_pos[:, 0] < train_edges_pos[:, 1]]  # Only include upper triangle.
    train_edges_neg = split_edge['train']['edge_neg']

    # Valid edges.
    valid_edges = torch.vstack([split_edge['valid']['edge'], split_edge['valid']['edge_neg']])
    valid_true = torch.cat([torch.ones(split_edge['valid']['edge'].shape[0], dtype=int), torch.zeros(split_edge['valid']['edge_neg'].shape[0], dtype=int)])
    index = torch.randperm(valid_edges.shape[0])  # Shuffle edges for expected values of precision@k for ties.
    valid_edges = valid_edges[index]
    valid_true = valid_true[index]

    # Test edges.
    test_edges = torch.vstack([split_edge['test']['edge'], split_edge['test']['edge_neg']])
    test_true = torch.cat([torch.ones(split_edge['test']['edge'].shape[0], dtype=int), torch.zeros(split_edge['test']['edge_neg'].shape[0], dtype=int)])
    index = torch.randperm(test_edges.shape[0])
    test_edges = test_edges[index]
    test_true = test_true[index]

    return train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true


def compute_batches(rows, batch_size, shuffle=True):
    """
    Compute the batches of rows. This implementation is much faster than pytorch's dataloader.

    :param rows: rows to split into batches.
    :param batch_size: size of each batch.
    :param shuffle: whether to shuffle the rows before splitting.
    :return:
    """

    if shuffle:
        return torch.split(rows[torch.randperm(rows.shape[0])], batch_size)
    else:
        return torch.split(rows, batch_size)


def compute_k_hop_neighborhood_edges(hops, edges, edge_index, device="cpu", relabel=False):
    """
    Given an edge (or a set of edges), returns the edges and the nodes
    that constitute the k-hop subgraph around this edge.
    Params:
    hops (int) -> Number of hops to create the neighborhood
    edges ()
    """
    neighbors_a, edges_neighborhood_node_a, _, _ = k_hop_subgraph(
        node_idx=edges[0], num_hops=hops, edge_index=edge_index, relabel_nodes=False)
    neighbors_b, edges_neighborhood_node_b, _, _ = k_hop_subgraph(
        node_idx=edges[1], num_hops=hops, edge_index=edge_index, relabel_nodes=False)

    k_hop_neighborhood_edges = torch.unique(torch.cat((edges_neighborhood_node_a.to(
        device), edges_neighborhood_node_b.to(device), edges), axis=1), dim=1)

    # Compute trained edge weights.
    neighbors = torch.unique(torch.cat((neighbors_a.to(device), neighbors_b.to(
        device), edges[:, 0], edges[:, 1])), sorted=True)

    return neighbors, k_hop_neighborhood_edges
