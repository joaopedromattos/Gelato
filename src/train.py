import argparse
import torch
from math import ceil
from tqdm import tqdm
from ast import literal_eval
import os
import shutil

import util
from gelato import Gelato
from eval import valid

import wandb


def n_pair_loss(out_pos, out_neg):
    """
    Compute the N-pair loss.

    :param out_pos: similarity scores for positive pairs.
    :param out_neg: similarity scores for negative pairs.
    :return: loss (normalized by the total number of pairs)
    """

    # Number of negative pairs matched to a positive pair.
    agg_size = out_neg.shape[0] // out_pos.shape[0]
    agg_size_p1 = agg_size + 1
    # Number of positive pairs that should be matched to agg_size + 1 instead because of the remainder.
    agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]
    out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
    out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
    out_neg_agg_p1 = out_neg[:agg_size_p1_count *
                             agg_size_p1].reshape(-1, agg_size_p1)
    out_neg_agg = out_neg[agg_size_p1_count *
                          agg_size_p1:].reshape(-1, agg_size)
    # Difference between negative and positive scores.
    out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1
    # Difference between negative and positive scores.
    out_diff_agg = out_neg_agg - out_pos_agg
    out_diff_exp_sum_p1 = torch.exp(torch.clamp(
        out_diff_agg_p1, max=80.0)).sum(axis=1)
    out_diff_exp_sum = torch.exp(torch.clamp(
        out_diff_agg, max=80.0)).sum(axis=1)
    out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
    loss = torch.log(1 + out_diff_exp_cat).sum() / \
        (len(out_pos) + len(out_neg))

    return loss


def train(model, optimizer, train_edges_pos, train_edges_neg, train_batch_ratio):
    """
    Train the model.

    :param model: Gelato model.
    :param train_edges_pos: train positive edges.
    :param train_edges_neg: train negative edges.
    :param train_batch_ratio: ratio of training edges per train batch.
    :return: train loss
    """

    model.train()

    total_loss = 0
    edges_pos_loader = util.compute_batches(train_edges_pos, batch_size=ceil(
        len(train_edges_pos)*train_batch_ratio), shuffle=True)
    edges_neg_loader = util.compute_batches(train_edges_neg, batch_size=ceil(
        len(train_edges_neg)*train_batch_ratio), shuffle=True)

    for batch_edges_pos, batch_edges_neg in tqdm(zip(edges_pos_loader, edges_neg_loader), desc='Train Batch', total=len(edges_pos_loader)):

        optimizer.zero_grad()

        batch_edges = torch.vstack(
            [batch_edges_pos, batch_edges_neg]).to(model.A.device)
        batch_true = torch.cat([torch.ones(batch_edges_pos.shape[0], dtype=int), torch.zeros(
            batch_edges_neg.shape[0], dtype=int)]).to(model.A.device)
        out = model(batch_edges, batch_edges_pos.to(model.A.device))

        # Compute loss.
        out_pos = out[:batch_edges_pos.shape[0]]
        out_neg = out[batch_edges_pos.shape[0]:]

        loss = n_pair_loss(out_pos, out_neg)
        total_loss += loss.item() * len(batch_true)

        loss.backward()

        # wandb.log({"step_loss": loss/len(batch_edges)})

        # Skipping the updating of nan gradients.
        nan_grad = False
        for param in model.parameters():
            if torch.isnan(param.grad).any():
                nan_grad = True
                break
        if not nan_grad:
            optimizer.step()

    return total_loss / len(train_edges_pos)


def parse_args():
    """
    Parse the arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--dataset', default='Photo',
                        help='Dataset. Default is Photo. ')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='Proportion of added edges. Default is 0.0. ')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Topological weight. Default is 0.0. ')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Trained weight. Default is 1.0. ')
    parser.add_argument('--add-self-loop', type=literal_eval, default=False,
                        help='Whether to add self-loops to all nodes. Default is False. ')
    parser.add_argument('--trained-edge-weight-batch-size', type=int, default=20000,
                        help='Batch size for computing the trained edge weights. Default is 20000. ')
    parser.add_argument('--graph-learning-type', default='mlp',
                        help='Type of the graph learning component. Default is mlp. ')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of layers in mlp. Default is 3. ')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Number of hidden channels in mlp. Default is 128. ')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5. ')
    parser.add_argument('--topological-heuristic-type', default='ac',
                        help='Type of the topological heuristic component. Default is ac. ')
    parser.add_argument('--scaling-parameter', type=int, default=3,
                        help='Scaling parameter of ac. Default is 3. ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001. ')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of epochs. Default is 250. ')
    parser.add_argument('--train-batch-ratio', type=float, default=0.1,
                        help='Ratio of training edges per train batch. Default is 0.1. ')
    parser.add_argument('--random-seed', type=int, default=1,
                        help='Random seed for training. Default is 1. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--batch-version', type=bool, default=False,
                        help='Use the batch or full graph version. ')
    parser.add_argument('--max-neighborhood-size', type=int, default=500,
                        help='Only used in the batch version. Defines the maximum amount of neighbors to be sampled in a batch.')

    return parser.parse_args()


def main():
    """
    Pipeline for training.

    """
    args = parse_args()
    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Set up results folder.
    results_folder = f'data/{args.dataset}/model/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    log_file = results_folder + 'log.txt'

    # Load dataset and split edges.
    data = util.load_dataset(args.dataset)
    data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=int)
    split_edge = util.split_dataset(data)
    data.edge_index = split_edge['train']['edge'].t()
    data.edge_weight = data.train_pos_edge_attr
    train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true = util.compute_edges(
        split_edge)

    # Hyperparameters
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_weight.squeeze(), size=(
        data.num_nodes, data.num_nodes)).to_dense()
    X = data.x
    hyperparameters = {
        'gelato': {
            'A': A,
            'X': X,
            'eta': args.eta,
            'alpha': args.alpha,
            'beta': args.beta,
            'add_self_loop': args.add_self_loop,
            'trained_edge_weight_batch_size': args.trained_edge_weight_batch_size,
            'graph_learning_type': args.graph_learning_type,
            'graph_learning_params': {
                'in_channels': X.shape[-1] * 2,
                'hidden_channels': args.hidden_channels,
                'out_channels': 2,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'relu_first': True,
                'batch_norm': True,
                'permutation_invariant': True
            },
            'topological_heuristic_type': args.topological_heuristic_type,
            'topological_heuristic_params': {
                'scaling_parameter': args.scaling_parameter
            },
            'batch_version': args.batch_version,
            'all_edges': torch.cat((train_edges_pos, train_edges_neg, valid_edges)),
            'max_neighborhood_size': args.max_neighborhood_size
        },
        'lr': args.lr,
        'epochs': args.epochs,
        'train_batch_ratio': args.train_batch_ratio,
    }

    wandb.init(project="gelato", entity="joaopedromattos", config=args)

    batched_version_name = f"Batched-{args.train_batch_ratio}-{args.max_neighborhood_size}" if args.batch_version else "Full"
    wandb.run.name = f"{args.dataset}-{batched_version_name}-{args.scaling_parameter}"

    # Training.
    util.set_random_seed(args.random_seed)
    model = Gelato(**hyperparameters['gelato']).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=hyperparameters['lr'])
    wandb.watch(model)

    best_valid_prec = -1
    best_epoch = -1
    epoch_iterator = tqdm(
        range(1, 1 + hyperparameters['epochs']), desc='Epoch')
    for epoch in epoch_iterator:

        loss = train(model, optimizer, train_edges_pos,
                     train_edges_neg, hyperparameters['train_batch_ratio'])

        valid_prec = valid(model, valid_edges, valid_true)
        with open(log_file, 'a') as f:
            print(f"Epoch = {epoch}:", file=f)
            print(f"Loss = {loss:.4e}", file=f)
            print(f"Valid precision@100%: {valid_prec:.2%}", file=f)

        wandb.log({"val_epoch_loss": loss})
        wandb.log({"val_precision_@_100%": valid_prec})

        if valid_prec > best_valid_prec:
            best_epoch = epoch
            best_valid_prec = valid_prec

        torch.save(model.state_dict(), results_folder +
                   f'model_checkpoint{epoch}.pth')
        torch.save(optimizer.state_dict(), results_folder +
                   f'optimizer_checkpoint{epoch}.pth')

        wandb.log({"epoch_loss": loss})

    # Record the best model.
    with open(log_file, 'a') as f:
        print(f"Best epoch = {best_epoch}", file=f)
    shutil.copyfile(
        results_folder + f'model_checkpoint{best_epoch}.pth', results_folder + f'model_best.pth')


if __name__ == "__main__":

    main()
