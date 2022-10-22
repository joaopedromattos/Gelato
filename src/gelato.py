import torch
import torch.nn.functional as F
from math import floor
from sklearn.metrics import pairwise_kernels
from tqdm import tqdm
import numpy as np
import util


class Gelato(torch.nn.Module):

    def __init__(self, A, X, eta, alpha, beta, add_self_loop, trained_edge_weight_batch_size,
                 graph_learning_type, graph_learning_params,
                 topological_heuristic_type, topological_heuristic_params,
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

        # Graph learning and topological heuristic components.
        self.graph_learning_type = graph_learning_type
        self.graph_learning_params = graph_learning_params
        self.topological_heuristic_type = topological_heuristic_type
        self.topological_heuristic_params = topological_heuristic_params

        if (self.topological_heuristic_type == 'an'):
            self.X_anchors, self.anchors = util.sample_anchors(self.X, self.topological_heuristic_params["number_of_anchors"])

        self.graph_learning = {
            'mlp': PairwiseMLP,
        }[graph_learning_type](**graph_learning_params)

        self.topological_heuristic = {
            'ac': Autocovariance,
            'an': AnchorAutocovariance,
        }[topological_heuristic_type](**topological_heuristic_params)

        # Compute untrained edge weights and the augmented edges.
        if (self.topological_heuristic_type == 'ac'):
            S = pairwise_kernels(self.X, metric='cosine')
        elif (self.topological_heuristic_type == 'an'):
            print("[DEBUG] self.anchors -----> ", type(self.anchors), self.anchors, self.anchors)
            print("[DEBUG] self.X -----> ", type(self.X), self.X.shape, self.X)
            S = pairwise_kernels(self.X, self.X_anchors, metric='cosine')

        if self.eta != 0.0:
            num_edges = (self.A != 0).sum()
            num_untrained_similarity_edges = floor(num_edges * self.eta)


            if (self.topological_heuristic_params == 'an'):
                # If our algorithm uses anchors, so we have to iterate over the anchors
                # to zero-out the cases in which we are comparing an anchor with itself.
                for column_idx, anchor_node_idx in enumerate(self.anchors):
                    S[anchor_node_idx][column_idx] = 0
            else:
                np.fill_diagonal(S, 0)

            threshold = np.partition(S.flatten(), -num_untrained_similarity_edges)[-num_untrained_similarity_edges]
            self.register_buffer('untrained_similarity_edge_mask', torch.BoolTensor(S > threshold), persistent=False)
        else:
            self.register_buffer('untrained_similarity_edge_mask', torch.BoolTensor(np.zeros_like(S)), persistent=False)
        
        augmented_edge_mask = self.A.to(bool) + self.untrained_similarity_edge_mask

        print("[DEBUG] augmented_edge_mask -----> ", type(augmented_edge_mask), augmented_edge_mask.shape, augmented_edge_mask)
        self.register_buffer('S', torch.relu(torch.FloatTensor(S) * augmented_edge_mask), persistent=False)
        self.augmented_edges = augmented_edge_mask.triu().nonzero(as_tuple=False)
        self.augmented_edge_loader = util.compute_batches(self.augmented_edges, batch_size=self.trained_edge_weight_batch_size, shuffle=False)

    def forward(self, edges, edges_pos=None):

        # Positive masking.
        A = self.   A.index_put(tuple(edges_pos.t()), torch.zeros(edges_pos.shape[0], device=self.A.device)) if self.training else self.A

        # Compute trained edge weights.
        if (self.topological_heuristic_type == 'an'):
            W = torch.zeros((self.A.shape[0], self.topological_heuristic_params.number_of_anchors), device=self.A.device)
        else:
            W = torch.zeros((self.A.shape[0], self.A.shape[0]), device=self.A.device)

        for i, batch in enumerate(tqdm(self.augmented_edge_loader, desc='Compute trained weights', total=len(self.augmented_edge_loader))):
            out = self.graph_learning(self.X, batch.to(self.A.device))
            W[tuple(batch.t())] = out
        W = W + W.t()
        W.fill_diagonal_(1)

        # Combine topological edge weights, trained edge weights, and untrained edge weights.
        A_enhanced = self.alpha * A + (1 - self.alpha) * (
            (A.to(bool) + self.untrained_similarity_edge_mask) * (
                self.beta * W + (1 - self.beta) * self.S
            )
        )
        if self.add_self_loop:
            A_enhanced.fill_diagonal_(1)  # Add self-loop to all nodes.
        else:
            A_enhanced.diagonal().copy_(A_enhanced.sum(axis=1) == 0)  # Add self-loops to isolated nodes.

        R = self.topological_heuristic(A_enhanced)
        out = R[tuple(edges.t())]
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
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers, dropout, relu_first, batch_norm)
        self.permutation_invariant = permutation_invariant

    def forward(self, x, edges):

        if self.permutation_invariant:
            edge_x = torch.cat([x[edges[:, 0]] + x[edges[:, 1]], torch.abs(x[edges[:, 0]] - x[edges[:, 1]])], dim=1)
        else:
            edge_x = torch.cat([x[edges[:, 0]], x[edges[:, 1]]], dim=1)

        return self.mlp(edge_x).exp()[:, 1]


class Autocovariance(torch.nn.Module):

    def __init__(self, scaling_parameter, **kwargs):
        super(Autocovariance, self).__init__()
        self.scaling_parameter = scaling_parameter

    def forward(self, A):

        # Compute Autocovariance matrix.
        d = A.sum(dim=1)
        pi = F.normalize(d, p=1, dim=0)
        M = A / d[:, None]
        R = torch.diag(pi) @ torch.matrix_power(M, self.scaling_parameter) - torch.outer(pi, pi)

        # Standardize Autocovariance entries.
        R = (R - R.mean())/R.std()

        return R

class AnchorAutocovariance(torch.nn.Module):

    def __init__(self, scaling_parameter, number_of_anchors):
        super(AnchorAutocovariance, self).__init__()
        self.scaling_parameter = scaling_parameter
        self.number_of_anchors = number_of_anchors


    def forward(self, A):

        # Compute Autocovariance matrix.
        d = A.sum(dim=1)
        pi = F.normalize(d, p=1, dim=0)
        M = A / d[:, None]
        R = torch.diag(pi) @ torch.matrix_power(M, self.scaling_parameter) - torch.outer(pi, pi)

        # Standardize Autocovariance entries.
        R = (R - R.mean())/R.std()

        return R