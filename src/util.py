import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid, Amazon, PPI
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, k_hop_subgraph
from tqdm import tqdm

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
    elif dataset in ("PPI"):
        pyg_dataset = PPI(data_folder, dataset)
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



def compute_k_hop_neighborhood_edges(hops, edges, edge_index, device="cpu", relabel=False, max_neighborhood_size=None):
    """
    Given an edge (or a set of edges), returns the edges and the nodes
    that constitute the k-hop subgraph around this edge.

    Params:
    hops (int) -> Number of hops to create the neighborhood
    edges ()
    """
    # print("Max neigh size", max_neighborhood_size)

    neighbors, edges_neighborhood_node, mapping, _ = k_hop_subgraph(node_idx=edges.flatten(), num_hops=hops, edge_index=edge_index, relabel_nodes=relabel)
    
    # Compute trained edge weights.
    if (max_neighborhood_size and (max_neighborhood_size < len(neighbors))):
        indices = random.sample(range(len(neighbors)), int(max_neighborhood_size))
        indices = torch.tensor(indices)
        neighbors = neighbors[indices]
        print("Neighbors", len(neighbors))

        neighbors = torch.unique(torch.cat((neighbors.to(device), edges.flatten())), sorted=True)

        # mask = []
        # for idx, edge in enumerate(edges_neighborhood_node.T):
        #     if (edge[0] in neighbors) and (edge[1] in neighbors):
        #         mask.append(idx)

        # edges_neighborhood_node = edges_neighborhood_node[:, torch.tensor(mask)]

        # print(edges_neighborhood_node.shape)
        # print(mask)
        # edges_neighborhood_node = torch.unique(torch.cat((edges_neighborhood_node.to(device), edges), axis=1), dim=1)


    print("New neighborhood size", neighbors.shape)
    return neighbors, edges_neighborhood_node




def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
