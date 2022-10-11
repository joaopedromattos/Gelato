import argparse
import torch
import os
from ast import literal_eval
from math import ceil
from sklearn.metrics import average_precision_score

import util
from gelato import Gelato


def precision_at_k(eval_true, eval_pred, k):
    """
    Compute precision@k.

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Precision@k
    """

    eval_top_index = torch.topk(eval_pred, k, sorted=False).indices.cpu()
    eval_tp = eval_true[eval_top_index].sum().item()
    pk = eval_tp / k

    return pk


def hits_at_k(eval_true, eval_pred, k):
    """
    Compute hits@k.

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Hits@k.
    """

    pred_pos = eval_pred[eval_true == 1]
    pred_neg = eval_pred[eval_true == 0]
    kth_score_in_negative_edges = torch.topk(pred_neg, k)[0][-1]
    hitsk = float(torch.sum(pred_pos > kth_score_in_negative_edges).cpu()) / len(pred_pos)
    return hitsk


def average_precision(eval_true, eval_pred):
    """
    Compute Average Precision (AP).

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :return: AP.
    """

    return average_precision_score(eval_true, eval_pred.cpu())


@torch.no_grad()
def valid(model, valid_edges, valid_true):
    """
    Compute precision@100% (of positive validation edges) for model validation.

    :param model: Gelato model.
    :param valid_edges: validation edges.
    :param valid_true: validation edge labels.
    :return: precision@100%
    """

    model.eval()
    num_valid_edges = valid_true.sum().item()
    valid_edges = valid_edges.to(model.A.device)
    valid_pred = model(valid_edges)
    pk = precision_at_k(valid_true, valid_pred, num_valid_edges)

    return pk


@torch.no_grad()
def test(model, test_edges, test_true):
    """
    Compute AP, precision@k (for k = 10%, 20%, ..., 100% of positive test edges), hits@k (for k = 25, 50, ..., 6400).

    :param model: Gelato model.
    :param test_edges: testing edges.
    :param test_true: testing edge labels.
    :return: AP, precision@k, hits@k
    """

    model.eval()
    num_test_edges = test_true.sum().item()
    test_edges = test_edges.to(model.A.device)
    test_pred = model(test_edges)
    ap = average_precision(test_true, test_pred)
    pks = [precision_at_k(test_true, test_pred, k) for k in [ceil(num_test_edges * ratio) for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]]
    hitsks = [hits_at_k(test_true, test_pred, k) for k in (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)]
    return ap, pks, hitsks


def parse_args():
    """
    Parse the arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Evaluation")

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
    parser.add_argument('--topological-heuristic-type', default='ac',
                        help='Type of the topological heuristic component. Default is ac. ')
    parser.add_argument('--scaling-parameter', type=int, default=3,
                        help='Scaling parameter of ac. Default is 3. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')

    return parser.parse_args()


def main():
    """
    Pipeline for evaluation.

    """
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
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
    train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true = util.compute_edges(split_edge)

    # Hyperparameters
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_weight.squeeze(), size=(data.num_nodes, data.num_nodes)).to_dense()
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
                'dropout': 0.5,
                'relu_first': True,
                'batch_norm': True,
                'permutation_invariant': True
            },
            'topological_heuristic_type': args.topological_heuristic_type,
            'topological_heuristic_params': {
                'scaling_parameter': args.scaling_parameter
            },
        }
    }

    # Load the trained model.
    model = Gelato(**hyperparameters['gelato']).to(device)
    model.load_state_dict(torch.load(results_folder + f'model_best.pth', map_location=device))

    # Testing.
    ap, pks, hitsks = test(model, test_edges, test_true)
    with open(log_file, 'a') as f:
        print(f"Test AP: {ap:.2%}")
        print(f"Test AP: {ap:.2%}", file=f)
        for i, k in enumerate(('10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%')):
            print(f"Test precision@{k}: {pks[i]:.2%}")
            print(f"Test precision@{k}: {pks[i]:.2%}", file=f)
        for i, k in enumerate((25, 50, 100, 200, 400, 800, 1600, 3200, 6400)):
            print(f"Test hits@{k}: {hitsks[i]:.2%}")
            print(f"Test hits@{k}: {hitsks[i]:.2%}", file=f)


if __name__ == "__main__":

    main()