import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges


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
    
    
 def graph_splits(graph_partitions, train_graph, val_graph, test_graph, full_training=True, save_dir=None, ogbl=False):
    '''
    Given the train, validation and test graphs, along with the graph partitions, this function
    organizes our training data in different splits, one per graph partition and removes possible leaks between
    training, validation and test due to the sampling procedure done per partition.
    
    :param graph_partitions: ClusterData object of the graph partitioned.
    :param train_graph: torch_geometric.data.Data object with the training graph as the edge index.
    :param val_graph: torch_geometric.data.Data object with the validation pairs.
    :param test_graph: torch_geometric.data.Data object with the test pairs.
    :param full_training: generates splits using full training (True) or biased training (False).
    :param save_dir: path in which we will cache the splits.
    :param ogbl: in case we are using full-training in ogbl datasets, it samples validation and test negative pairs only within each cluster.
    
    :return: splits, intercluster_splits, node_to_partition
    '''

    n_partitions = len(graph_partitions)

    # If we have cached data, then we will just read it
    if save_dir and os.path.isfile(save_dir + f'splits_{n_partitions}.pt'):
        splits = torch.load(save_dir + f'splits_{n_partitions}.pt')
        intercluster_splits = torch.load(save_dir + f'intercluster_splits_{n_partitions}.pt')
        node_to_partition = torch.load(save_dir + f'node_to_partition_{n_partitions}.pt')

        return splits, intercluster_splits, node_to_partition


    # Stores partition metadata regarding each node of the network.
    node_to_partition = {}
    partition_idx = 0
    node_partition_idx = 0
    for idx, node in enumerate(graph_partitions.perm):
        if idx >= graph_partitions.partptr[partition_idx + 1]:
            partition_idx += 1
            node_partition_idx = 0

        node_to_partition[int(node)] = dict()
        node_to_partition[int(node)]['node_partition_idx'] = partition_idx # In which partition this node is located
        node_to_partition[int(node)]['node_idx'] = node_partition_idx # What is its index in the partition it is located

        node_partition_idx += 1
    
    splits = dict()


    valid_edge_index = val_graph.edge_index
    test_edge_index = test_graph.edge_index

    valid_edges = val_graph.edge_label_index
    valid_true = val_graph.edge_label

    test_edges = test_graph.edge_label_index
    test_true = test_graph.edge_label

    intercluster_mask_val = torch.zeros((val_graph.edge_label_index.shape[1])).bool()
    intercluster_mask_test = torch.zeros((test_graph.edge_label_index.shape[1])).bool()

    intracluster_pairs_val, intracluster_idx_val = remove_intercluster(val_graph.edge_label_index, node_to_partition=node_to_partition)
    intracluster_pairs_test, intracluster_idx_test = remove_intercluster(test_graph.edge_label_index, node_to_partition=node_to_partition)

    intercluster_mask_val = (intracluster_idx_val == - 1)
    intercluster_mask_test = (intracluster_idx_test == -1) 
    
    _, intracluster_edge_idx_val = remove_intercluster(val_graph.edge_index, node_to_partition=node_to_partition)
    _, intracluster_edge_idx_test = remove_intercluster(test_graph.edge_index, node_to_partition=node_to_partition)

    
    for idx, partition in tqdm(enumerate(graph_partitions), desc="Processing graph partitions"):

        subgraph = partition

        train_edge_index = subgraph.edge_index

        splits[idx] = dict()

        splits[idx]['subgraph'] = subgraph

        splits[idx]['train_edge_index'] = train_edge_index


        del subgraph.edge_label, subgraph.edge_label_index

        
        # Resampling subgraph pairs.
        # The idea here is to take advantage of the fact that we have
        # a subgraph computed using METIS to sample more informative pairs
        # than the ones contained in the original training set.
        transform = RandomLinkSplit(is_undirected=False, num_val=0.0, num_test=0.0)
        subgraph_train, subgraph_val, subgraph_test = transform(subgraph)


        num_pos_edges = subgraph_train.edge_label.bool().sum()
        num_neg_edges = (~subgraph_train.edge_label.bool()).sum()

        train_edges_pos = subgraph_train.edge_label_index[:, subgraph_train.edge_label.bool()]

        if full_training:
            n = subgraph.num_nodes
            # code.interact(local=locals())
            train_edge_neg_mask = torch.ones((n, n), dtype=bool)
            train_edge_neg_mask[tuple(train_edges_pos.tolist())] = False
            train_edge_neg_mask = torch.triu(train_edge_neg_mask, 1)
            train_edges_neg = torch.nonzero(train_edge_neg_mask).t()

            if (ogbl):

                subgraph_val_edges_pos = subgraph_val.edge_label_index[:, subgraph_val.edge_label.bool()]
                subgraph_test_edges_pos = subgraph_test.edge_label_index[:, subgraph_test.edge_label.bool()]
                
                valid_edge_neg_mask = train_edge_neg_mask.clone()
                valid_edge_neg_mask[tuple(subgraph_val_edges_pos.tolist())] = False
                valid_edges_neg = torch.nonzero(valid_edge_neg_mask).t()

                # Full testing with all negative pairs in the training graph (excluding validation and testing positive edges).
                test_edge_neg_mask = valid_edge_neg_mask.clone()
                test_edge_neg_mask[tuple(subgraph_test_edges_pos.tolist())] = False
                test_edges_neg = torch.nonzero(test_edge_neg_mask).t()

        else:
            train_edges_neg = subgraph_train.edge_label_index[:, ~subgraph_train.edge_label.bool()] 


        splits[idx]['train_edges_pos'] = train_edges_pos
        splits[idx]['train_edges_neg'] = train_edges_neg
        

        
        cluster_mask_val = (intracluster_idx_val == idx)
        cluster_mask_test = (intracluster_idx_test == idx)

        if (ogbl):

            valid_edges_pos = convert_to_partition_index(valid_edges[:, cluster_mask_val], node_to_partition)

            test_edges_pos = convert_to_partition_index(test_edges[:, cluster_mask_test], node_to_partition)

            splits[idx]['valid_edges'] = torch.cat([valid_edges_pos, valid_edges_neg], dim=1)
            splits[idx]['valid_true'] = torch.cat([torch.ones(valid_edges_pos.shape[1]), torch.zeros(valid_edges_neg.shape[1])])

            splits[idx]['test_edges'] = torch.cat([test_edges_pos, test_edges_neg], dim=1)
            splits[idx]['test_true'] = torch.cat([torch.ones(test_edges_pos.shape[1]), torch.zeros(test_edges_neg.shape[1])])

            splits[idx]['valid_edge_index'] = subgraph_val.edge_index
            
            splits[idx]['test_edge_index'] = subgraph_test.edge_index

        else:
            # Selecting the validation edges contained in the cluster 'idx'
            splits[idx]['valid_edges'] = convert_to_partition_index(valid_edges[:, cluster_mask_val], node_to_partition)
            splits[idx]['valid_true'] = valid_true[cluster_mask_val]

            # Selecting the validation edges contained in the cluster 'idx'
            splits[idx]['test_edges'] = convert_to_partition_index(test_edges[:, cluster_mask_test], node_to_partition)
            splits[idx]['test_true'] = test_true[cluster_mask_test]


            cluster_edge_idx_mask_val = (intracluster_edge_idx_val == idx)
            cluster_edge_idx_mask_test = (intracluster_edge_idx_test == idx)

            splits[idx]['valid_edge_index'] = convert_to_partition_index(valid_edge_index[:, cluster_edge_idx_mask_val], node_to_partition)

            splits[idx]['test_edge_index'] = convert_to_partition_index(test_edge_index[:, cluster_edge_idx_mask_test], node_to_partition)


    # Accounting for the intercluster pairs left behind
    intercluster_splits = dict()
    
    intercluster_splits['valid_edges'] = valid_edges[:, intercluster_mask_val]
    intercluster_splits['valid_true'] = valid_true[intercluster_mask_val]

    intercluster_splits['test_edges'] = test_edges[:, intercluster_mask_test]
    intercluster_splits['test_true'] = test_true[intercluster_mask_test]


    torch.save(splits, save_dir + f'splits_{n_partitions}.pt')
    torch.save(intercluster_splits, save_dir + f'intercluster_splits_{n_partitions}.pt')
    torch.save(node_to_partition, save_dir + f'node_to_partition_{n_partitions}.pt')

    return splits, intercluster_splits, node_to_partition



def compute_batch_stats(out, running_sum, running_sum_squared, running_n):
    results = out.detach().clone()
    running_sum += results.sum()
    running_sum_squared += torch.dot(results, results)
    running_n += len(results)
    
    running_mean = running_sum / running_n
    
    running_std = torch.sqrt(running_sum_squared / (running_n - 1)) - (running_mean ** 2)

    # mean = min(out_flatten[valid_idx])
    # std = max(out_flatten[valid_idx]) - min(out_flatten[valid_idx])
    return running_mean, running_std, running_sum, running_sum_squared, running_n
