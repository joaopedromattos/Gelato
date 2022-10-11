from tqdm import tqdm
import torch
import os
import argparse
from gelato import Gelato
import util
from train import train
from eval import valid, test


def parse_args():
    """
    Parse the arguments.

    """
    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument('--dataset', default='Photo',
                        help='Dataset. Default is Photo. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    return parser.parse_args()


def main():
    """
    Pipeline for reproducing the main results.

    """
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    # Set up results folder.
    results_folder = f'data/{args.dataset}/experiment/'
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

    # Hyperparameters.
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_weight.squeeze(), size=(data.num_nodes, data.num_nodes)).to_dense()
    X = data.x
    hyperparameters = {

        'Cora': {
            'gelato': {
                'A': A,
                'X': X,
                'eta': 0.5,
                'alpha': 0.5,
                'beta': 0.25,
                'add_self_loop': True,
                'trained_edge_weight_batch_size': 50000,
                'graph_learning_type': 'mlp',
                'graph_learning_params': {
                    'in_channels': X.shape[-1] * 2,
                    'hidden_channels': 128,
                    'out_channels': 2,
                    'num_layers': 3,  # One input layer, one hidden layer, and one output layer.
                    'dropout': 0.5,
                    'relu_first': True,
                    'batch_norm': True,
                    'permutation_invariant': True
                },
                'topological_heuristic_type': 'ac',
                'topological_heuristic_params': {
                    'scaling_parameter': 3
                }
            },
            'lr': 0.001,
            'epochs': 100,
            'train_batch_ratio': 0.1,
        },

        'CiteSeer': {
            'gelato': {
                'A': A,
                'X': X,
                'eta': 0.75,
                'alpha': 0.5,
                'beta': 0.5,
                'add_self_loop': True,
                'trained_edge_weight_batch_size': 20000,
                'graph_learning_type': 'mlp',
                'graph_learning_params': {
                    'in_channels': X.shape[-1] * 2,
                    'hidden_channels': 128,
                    'out_channels': 2,
                    'num_layers': 3,  # One input layer, one hidden layer, and one output layer.
                    'dropout': 0.5,
                    'relu_first': True,
                    'batch_norm': True,
                    'permutation_invariant': True
                },
                'topological_heuristic_type': 'ac',
                'topological_heuristic_params': {
                    'scaling_parameter': 3
                }
            },
            'lr': 0.001,
            'epochs': 100,
            'train_batch_ratio': 0.1,
        },

        'PubMed': {
            'gelato': {
                'A': A,
                'X': X,
                'eta': 0.0,
                'alpha': 0.0,
                'beta': 1.0,
                'add_self_loop': False,
                'trained_edge_weight_batch_size': 100000,
                'graph_learning_type': 'mlp',
                'graph_learning_params': {
                    'in_channels': X.shape[-1] * 2,
                    'hidden_channels': 128,
                    'out_channels': 2,
                    'num_layers': 3,  # One input layer, one hidden layer, and one output layer.
                    'dropout': 0.5,
                    'relu_first': True,
                    'batch_norm': True,
                    'permutation_invariant': True
                },
                'topological_heuristic_type': 'ac',
                'topological_heuristic_params': {
                    'scaling_parameter': 3
                },
            },
            'lr': 0.001,
            'epochs': 100,
            'train_batch_ratio': 0.1,
        },

        'Photo': {
            'gelato': {
                'A': A,
                'X': X,
                'eta': 0.0,
                'alpha': 0.0,
                'beta': 1.0,
                'add_self_loop': False,
                'trained_edge_weight_batch_size': 20000,
                'graph_learning_type': 'mlp',
                'graph_learning_params': {
                    'in_channels': X.shape[-1] * 2,
                    'hidden_channels': 128,
                    'out_channels': 2,
                    'num_layers': 3,  # One input layer, one hidden layer, and one output layer.
                    'dropout': 0.5,
                    'relu_first': True,
                    'batch_norm': True,
                    'permutation_invariant': True
                },
                'topological_heuristic_type': 'ac',
                'topological_heuristic_params': {
                    'scaling_parameter': 3
                },
            },
            'lr': 0.001,
            'epochs': 250,
            'train_batch_ratio': 0.1,
        },

        'Computers': {
            'gelato': {
                'A': A,
                'X': X,
                'eta': 0.0,
                'alpha': 0.0,
                'beta': 1.0,
                'add_self_loop': False,
                'trained_edge_weight_batch_size': 75000,
                'graph_learning_type': 'mlp',
                'graph_learning_params': {
                    'in_channels': X.shape[-1] * 2,
                    'hidden_channels': 128,
                    'out_channels': 2,
                    'num_layers': 3,  # One input layer, one hidden layer, and one output layer.
                    'dropout': 0.5,
                    'relu_first': True,
                    'batch_norm': True,
                    'permutation_invariant': True
                },
                'topological_heuristic_type': 'ac',
                'topological_heuristic_params': {
                    'scaling_parameter': 3
                },
            },
            'lr': 0.001,
            'epochs': 250,
            'train_batch_ratio': 0.1,
        },

    }[args.dataset]

    # Run experiments.
    runs = 10
    best_scores = [None] * runs
    for run in tqdm(range(1, runs+1), desc='Run'):

        with open(log_file, 'a') as f:
            print(f"Run = {run}", file=f)
        util.set_random_seed(run)

        model = Gelato(**hyperparameters['gelato']).to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=hyperparameters['lr'])

        scores = []
        best_valid_prec = -1
        best_epoch = -1
        epoch_iterator = tqdm(range(1, 1 + hyperparameters['epochs']), desc='Epoch')

        # Training.
        for epoch in epoch_iterator:

            loss = train(model, optimizer, train_edges_pos, train_edges_neg, hyperparameters['train_batch_ratio'])

            valid_prec = valid(model, valid_edges, valid_true)
            with open(log_file, 'a') as f:
                print(f"Epoch = {epoch}:", file=f)
                print(f"Loss = {loss:.4e}", file=f)
                print(f"Valid precision@100%: {valid_prec:.2%}", file=f)
            scores.append({
                'epoch': epoch,
                'loss': loss,
                'valid_prec': valid_prec,
            })
            if valid_prec > best_valid_prec:
                best_epoch = epoch
                best_valid_prec = valid_prec

            torch.save(model.state_dict(), results_folder + f'run{run}_model_checkpoint{epoch}.pth')
            torch.save(optimizer.state_dict(), results_folder + f'run{run}_optimizer_checkpoint{epoch}.pth')

        # Testing.
        model.load_state_dict(torch.load(results_folder + f'run{run}_model_checkpoint{best_epoch}.pth', map_location=device))
        ap, pks, hitsks = test(model, test_edges, test_true)
        with open(log_file, 'a') as f:
            print(f"Best epoch = {best_epoch}:", file=f)
            print(f"Loss = {scores[best_epoch-1]['loss']:.4e}", file=f)
            print(f"Valid precision@100%: {scores[best_epoch-1]['valid_prec']:.2%}", file=f)
            print(f"Test AP: {ap:.2%}", file=f)
            for i, k in enumerate(('10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%')):
                print(f"Test precision@{k}: {pks[i]:.2%}", file=f)
            for i, k in enumerate((25, 50, 100, 200, 400, 800, 1600, 3200, 6400)):
                print(f"Test hits@{k}: {hitsks[i]:.2%}", file=f)
            print(file=f)
        best_scores[run - 1] = {
            'run': run,
            'best_epoch': best_epoch,
            'valid_prec': scores[best_epoch-1]['valid_prec'],
            'ap': ap,
            'pks': pks,
            'hitsks': hitsks
        }

    # Summary across runs.
    valid_precs = torch.tensor([best_score['valid_prec'] for best_score in best_scores])
    aps = torch.tensor([best_score['ap'] for best_score in best_scores])
    pkss = torch.tensor([best_score['pks'] for best_score in best_scores])
    hitskss = torch.tensor([best_score['hitsks'] for best_score in best_scores])
    pkss_mean = pkss.mean(axis=0)
    pkss_std = pkss.std(axis=0)
    hitskss_mean = hitskss.mean(axis=0)
    hitskss_std = hitskss.std(axis=0)
    with open(log_file, 'a') as f:
        print("\nAverage best scores across runs:")
        print("\nAverage best scores across runs:", file=f)
        print(f"Valid precision@100%: {valid_precs.mean():.2%} ± {valid_precs.std():.2%}")
        print(f"Valid precision@100%: {valid_precs.mean():.2%} ± {valid_precs.std():.2%}", file=f)
        print(f"Test AP: {aps.mean():.2%} ± {aps.std():.2%}")
        print(f"Test AP: {aps.mean():.2%} ± {aps.std():.2%}", file=f)
        for i, k in enumerate(('10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%')):
            print(f"Test precision@{k}: {pkss_mean[i]:.2%} ± {pkss_std[i]:.2%}")
            print(f"Test precision@{k}: {pkss_mean[i]:.2%} ± {pkss_std[i]:.2%}", file=f)
        for i, k in enumerate((25, 50, 100, 200, 400, 800, 1600, 3200, 6400)):
            print(f"Test hits@{k}: {hitskss_mean[i]:.2%} ± {hitskss_std[i]:.2%}")
            print(f"Test hits@{k}: {hitskss_mean[i]:.2%} ± {hitskss_std[i]:.2%}", file=f)


if __name__ == "__main__":

    main()
