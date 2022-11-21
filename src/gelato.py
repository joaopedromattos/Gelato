from math import floor

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_kernels
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import spspmm
from tqdm import tqdm

import util
import random


class Gelato(torch.nn.Module):

    def __init__(self, A, X, eta, alpha, beta, add_self_loop, trained_edge_weight_batch_size,
                 graph_learning_type, graph_learning_params,
                 topological_heuristic_type, topological_heuristic_params, batch_version,
                 max_neighborhood_size, all_edges
                 ):
        super(Gelato, self).__init__()

        self.register_buffer('A', A.to(torch.float32), persistent=False)
        self.register_buffer('X', X, persistent=False)

        # Hyperparameters.
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.add_self_loop = add_self_loop
        self.trained_edge_weight_batch_size = trained_edge_weight_batch_size
        self.batch_version = batch_version
        self.max_neighborhood_size = max_neighborhood_size

        # Graph learning and topological heuristic components.
        self.graph_learning_type = graph_learning_type
        self.graph_learning_params = graph_learning_params
        self.topological_heuristic_type = topological_heuristic_type
        self.topological_heuristic_params = topological_heuristic_params

        self.graph_learning = {
            'mlp': PairwiseMLP,
        }[graph_learning_type](**graph_learning_params)

        self.topological_heuristic = {
            'ac': Autocovariance,
        }[topological_heuristic_type](**topological_heuristic_params)

        # Compute untrained edge weights and the augmented edges.
        S = pairwise_kernels(self.X, metric='cosine')
        if self.eta != 0.0:
            num_edges = (self.A != 0).sum()
            num_untrained_similarity_edges = floor(num_edges * self.eta)
            np.fill_diagonal(S, 0)
            threshold = np.partition(
                S.flatten(), -num_untrained_similarity_edges)[-num_untrained_similarity_edges]
            self.register_buffer('untrained_similarity_edge_mask', torch.BoolTensor(
                S > threshold), persistent=False)
        else:
            self.register_buffer('untrained_similarity_edge_mask', torch.BoolTensor(
                np.zeros_like(S)), persistent=False)

        augmented_edge_mask = self.A.to(
            bool) + self.untrained_similarity_edge_mask
        self.register_buffer('S', torch.relu(torch.FloatTensor(
            S) * augmented_edge_mask), persistent=False)
        self.augmented_edges = augmented_edge_mask.triu().nonzero(as_tuple=False)

        self.augmented_edge_loader = util.compute_batches(
            self.augmented_edges, batch_size=self.trained_edge_weight_batch_size, shuffle=False)


        # Preprocesses the neighborhoods of our graph based on the AUGMENTED EDGES.
        # The intuition behind this is to incorporate attribute information in the
        # batched version of our model. It would be unfair to compare both models without
        # this information.
        self.pre_computed_neighborhoods = util.preprocess_k_hop_neigborhoods(
            all_edges, A, self.topological_heuristic_params["scaling_parameter"]) if self.batch_version else None

    def _get_pre_computed_neighborhoods(self, edges):
        neighbors = torch.Tensor()
        k_hop_neighborhood_edges = torch.Tensor()

        for node_a, node_b in edges:
            neighborhood_node_a = self.pre_computed_neighborhoods[int(node_a)]["neighbors"]
            neighborhood_node_b = self.pre_computed_neighborhoods[int(node_b)]["neighbors"]
            neighbors = torch.cat((neighbors, neighborhood_node_a, neighborhood_node_b))

            k_hop_neighborhood_edges_node_a = self.pre_computed_neighborhoods[int(node_a)]["k_hop_neighborhood_edges"]
            k_hop_neighborhood_edges_node_b = self.pre_computed_neighborhoods[int(node_b)]["k_hop_neighborhood_edges"]
            k_hop_neighborhood_edges = torch.cat((k_hop_neighborhood_edges, k_hop_neighborhood_edges_node_a, k_hop_neighborhood_edges_node_b), axis=1)

        # We downsample our neighborhood due to optimize memory consupmtion
        # Our neighborhood will be the nodes that compose our edges and
        if (self.max_neighborhood_size and self.max_neighborhood_size <= len(neighbors)):
            print(f"Downsampling neighbors from {len(neighbors)} to ", end="")
            indices = torch.tensor(random.sample(range(0, len(neighbors)), self.max_neighborhood_size))
            indices = torch.tensor(indices)
            neighbors = neighbors[indices]

            # Adding our original edges again in our neighborhood
            valid_edge_indices = []
            for idx, edge in enumerate(k_hop_neighborhood_edges):
                if (edge[0] in neighbors) and (edge[1] in neighbors):
                    valid_edge_indices.append(idx)

            k_hop_neighborhood_edges = k_hop_neighborhood_edges[torch.Tensor(valid_edge_indices).long()]

            print(f"{len(neighbors)}")

        return torch.unique(neighbors).long().to(self.A.device), torch.unique(k_hop_neighborhood_edges, dim=1).long().to(self.A.device)

    def forward_batched(self, edges, edges_pos=None):

        print("Edges shape", edges.shape)
        hops = self.topological_heuristic_params["scaling_parameter"]

        # neighbors, k_hop_neighborhood_edges = self._get_pre_computed_neighborhoods(edges)
        # k_hop_neighborhood = self.pre_computed_neighborhoods[cur_edge]
        neighbors, k_hop_neighborhood_edges = util.compute_k_hop_neighborhood_edges(
            hops, edges, self.augmented_edges.T, device=self.A.device)
    

        print("Neighbors", neighbors.shape)
        print("k_hop_neighborhood_edges", k_hop_neighborhood_edges.shape)

        self.augmented_edge_loader = util.compute_batches(
            k_hop_neighborhood_edges.T, batch_size=self.trained_edge_weight_batch_size, shuffle=False)

        # Positive masking.
        A = self.A.index_put(tuple(edges_pos.t()), torch.zeros(
            edges_pos.shape[0], device=self.A.device)) if self.training else self.A

        # Compute trained edge weights.
        W = torch.zeros(
            (self.A.shape[0], self.A.shape[0]), device=self.A.device)

        print("SHAPE A", A.shape)

        for i, batch in enumerate(tqdm(self.augmented_edge_loader, desc=f'Compute trained weights - Edges: {k_hop_neighborhood_edges.shape}', total=len(self.augmented_edge_loader))):
            print("Batch", batch.shape)
            out = self.graph_learning(self.X, batch.to(self.A.device))
            W[tuple(batch.t())] = out

        W = W + W.t()
        W.fill_diagonal_(1)

        A_enhanced = self.alpha * self.A + (1 - self.alpha) * (
            (self.A + self.untrained_similarity_edge_mask) * ((1 - self.beta) * self.S + self.beta * W))

        if self.add_self_loop:
            A_enhanced.fill_diagonal_(1)  # Add self-loop to all nodes.
        else:
            # Add self-loops to isolated nodes.
            A_enhanced.diagonal().copy_(A_enhanced.sum(axis=1) == 0)

        R = self.topological_heuristic(A_enhanced, neighbors)

        print("Shape R", R.shape)

        neighborhood_idx = {neighbor: idx for idx,
                            neighbor in enumerate(neighbors.tolist())}

        edges_idx_converted = ([neighborhood_idx[edge] for edge in edges[:, 0].tolist()], [
                            neighborhood_idx[edge] for edge in edges[:, 1].tolist()])
        # print("Edges idx converted", edges_idx_converted)
        out = R[edges_idx_converted]
        return out

    def forward_full(self, edges, edges_pos=None):
        # Positive masking.
        A = self.A.index_put(tuple(edges_pos.t()), torch.zeros(
            edges_pos.shape[0], device=self.A.device)) if self.training else self.A

        # Compute trained edge weights.
        W = torch.zeros(
            (self.A.shape[0], self.A.shape[0]), device=self.A.device)

        for i, batch in enumerate(tqdm(self.augmented_edge_loader, desc='Compute trained weights', total=len(self.augmented_edge_loader))):
            out = self.graph_learning(self.X, batch.to(self.A.device))
            W[tuple(batch.t())] = out
        W = W + W.t()
        W.fill_diagonal_(1)

        # Combine topological edge weights, trained edge weights, and untrained edge weights.
        A_enhanced = self.alpha * A + (1 - self.alpha) * (
            (A.to(bool) + self.untrained_similarity_edge_mask) * (self.beta * W + (1 - self.beta) * self.S))
        if self.add_self_loop:
            A_enhanced.fill_diagonal_(1)  # Add self-loop to all nodes.
        else:
            # Add self-loops to isolated nodes.
            A_enhanced.diagonal().copy_(A_enhanced.sum(axis=1) == 0)

        R = self.topological_heuristic(A_enhanced)
        out = R[tuple(edges.t())]
        print("tuple(edges.t())", tuple(edges.t()))

        return out

    def forward(self, edges, edges_pos=None):

        if self.batch_version:
            out = self.forward_batched(edges, edges_pos)
        else:
            out = self.forward_full(edges, edges_pos)

        return out


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, relu_first, batch_norm):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if batch_norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first
        self.batch_norm = batch_norm

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            if self.batch_norm:
                x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)

            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class PairwiseMLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, relu_first, batch_norm, permutation_invariant):

        super(PairwiseMLP, self).__init__()
        self.mlp = MLP(in_channels, hidden_channels, out_channels,
                       num_layers, dropout, relu_first, batch_norm)
        self.permutation_invariant = permutation_invariant

    def forward(self, x, edges):

        if self.permutation_invariant:
            edge_x = torch.cat([x[edges[:, 0]] + x[edges[:, 1]],
                               torch.abs(x[edges[:, 0]] - x[edges[:, 1]])], dim=1)
        else:
            edge_x = torch.cat([x[edges[:, 0]], x[edges[:, 1]]], dim=1)

        return self.mlp(edge_x).exp()[:, 1]


class Autocovariance(torch.nn.Module):

    def __init__(self, scaling_parameter):
        super(Autocovariance, self).__init__()
        self.scaling_parameter = scaling_parameter

    def forward(self, A, batch_idx=None):

        if batch_idx is not None:
            A = A[batch_idx][:, batch_idx]

        # Compute Autocovariance matrix.
        d = A.sum(dim=1)
        pi = F.normalize(d, p=1, dim=0)
        M = A / d[:, None]
        R = torch.diag(pi) @ torch.matrix_power(M,
                                                self.scaling_parameter) - torch.outer(pi, pi)

        # Standardize Autocovariance entries.
        R = (R - R.mean())/R.std()

        return R
