import torch_geometric
import torch
from get_dfs import get_dfs, get_dfs_current_pair, get_dfs_current_batch
from util import *
from gelato import Gelato
import networkx as nx 
from torch.utils.data import DataLoader, Dataset
import code
import random
from math import ceil, inf
from tqdm import tqdm
# from scalene import scalene_profiler
from multiprocessing import Pool
from itertools import repeat
import os
import shutil
import wandb
import pickle
from eval import precision_at_k, average_precision, hits_at_k, mean_reciprocal_rank


def n_pair_loss(out_pos, out_neg):
    """
    Compute the N-pair loss.
    :param out_pos: similarity scores for positive pairs.
    :param out_neg: similarity scores for negative pairs.
    :return: loss (normalized by the total number of pairs)
    """

    agg_size = out_neg.shape[0] // out_pos.shape[0]  # Number of negative pairs matched to a positive pair.
    agg_size_p1 = agg_size + 1
    agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]  # Number of positive pairs that should be matched to agg_size + 1 instead because of the remainder.
    out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
    out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
    out_neg_agg_p1 = out_neg[:agg_size_p1_count * agg_size_p1].reshape(-1, agg_size_p1)
    # print(out_pos, out_neg)

    out_neg_agg = out_neg[agg_size_p1_count * agg_size_p1:].reshape(-1, agg_size)
    out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1  # Difference between negative and positive scores.
    out_diff_agg = out_neg_agg - out_pos_agg  # Difference between negative and positive scores.
    out_diff_exp_sum_p1 = torch.exp(torch.clamp(out_diff_agg_p1, max=80.0)).sum(axis=1)
    out_diff_exp_sum = torch.exp(torch.clamp(out_diff_agg, max=80.0)).sum(axis=1)
    out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
    loss = torch.log(1 + out_diff_exp_cat).sum() / (len(out_pos) + len(out_neg))

    return loss


def train(model, A, X, S, optimizer, train_edges_pos, train_edges_neg, train_batch_ratio, training_device, running_n, running_sum, running_sum_squared) -> float:

    edges_pos_loader = compute_batches(train_edges_pos, batch_size=ceil(len(train_edges_pos)*train_batch_ratio), shuffle=True)
    edges_neg_loader = compute_batches(train_edges_neg, batch_size=ceil(len(train_edges_neg)*train_batch_ratio), shuffle=True)

    total_loss = 0
    loss = 0
    total_pairs_predicted = 0

    tau = model.topological_heuristic_params['scaling_parameter']
    
    memory_allocated = []


    # edges_pos_loader = torch.unsqueeze(edges_pos_loader[0], 0)
    # edges_neg_loader = torch.unsqueeze(edges_neg_loader[0], 0)

    A = A.to(training_device)
    S = S.to(training_device)
    X = X.to(training_device)

    # code.interact(local=locals())
    model.train()
    for batch_edges_pos, batch_edges_neg in tqdm(zip(edges_pos_loader, edges_neg_loader), desc='Train Batch', total=len(edges_pos_loader)):

        optimizer.zero_grad()

        batch_edges = torch.vstack([batch_edges_pos, batch_edges_neg])
        batch_true = torch.cat([torch.ones(batch_edges_pos.shape[0], dtype=int), torch.zeros(batch_edges_neg.shape[0], dtype=int)])


        print(f"\n[TRAIN] -- Pos Edges - {len(batch_edges_pos)} --- Neg Edges - {len(batch_edges_neg)}")

        # Masking positive pairs 
        mask = tuple(batch_edges_pos.t())
        A[mask] = 0.0 # This is almost 10x faster than torch.index_put()


        out = model(A, X, S)

        A[mask] = 1.0


        # Adding the "channel" index
        mask = tuple(batch_edges.t())

        out = out[mask]

        running_mean, running_std, running_sum, running_sum_squared, running_n = compute_batch_stats(out, running_sum=running_sum, running_sum_squared=running_sum_squared, running_n=running_n)

        out = (out - running_mean) / running_std

    
        # Compute loss.
        out_pos = out[:batch_edges_pos.shape[0]]
        out_neg = out[batch_edges_pos.shape[0]:]
        loss = n_pair_loss(out_pos, out_neg)

        loss.backward()


        total_loss += loss.detach() * len(batch_true)
        total_pairs_predicted += batch_edges_pos.shape[0]
        

        print('Loss', loss)
        
        nan_grad = False
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad).any():
                nan_grad = True
                break 
        if not nan_grad:
            optimizer.step()

        # optimizer.step()

        if training_device != 'cpu':
            # Saving the memory allocated in GB 
            # We're using this tracking instead of nvidia-smi
            # Because it measures the memory *really* allocated.
            # nvidia-smi considers caching and other huge imprecisions.
            memory_allocated.append(torch.cuda.memory_allocated())
        
        pk = precision_at_k(batch_true, out, batch_true.sum().item())
        print("Prec@K --------", pk)

        del loss, out

    return total_loss / total_pairs_predicted, memory_allocated, running_mean, running_std, running_sum, running_sum_squared, running_n


@torch.no_grad()
def valid(model, A, X, S, valid_edges, valid_true, validation_device, validation_batch_ratio, mean, std):
    """
    Predict the autocovariance given a graph and a set of pairs.
    :param model: Gelato model.
    :param valid_edges: validation edges.
    :param valid_true: validation edge labels.
    :return: precision@100%
    """

    A = A.to(validation_device)
    S = S.to(validation_device)
    X = X.to(validation_device)

    memory_allocated = []

    model.eval()
    
    valid_pred = model(A, X, S)

    print(f"\n[VAL] -- Pos Edges - {valid_true.sum()} --- Neg Edges - {valid_edges.shape[0] - valid_true.sum()}")

    mask = tuple(valid_edges.t())

    valid_pred = valid_pred[mask]

    valid_pred = (valid_pred - mean) / std
    
    # code.interact(local=locals())
    # print('[VAL]', valid_pred.shape, valid_pred.shape)
    valid_pred = valid_pred.to('cpu').tolist()

    if validation_device != 'cpu':
        allocated_mem = torch.cuda.memory_allocated()
        print("Allocated mem on GPU", allocated_mem)
        memory_allocated.append(allocated_mem)

    return valid_pred, memory_allocated


if __name__ == '__main__':

    args = parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    data, pyg_dataset = load_dataset(args.dataset)
    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)

    # Set up data folder.
    # dataset_folder = f'/scratch/jrm28/data/{args.dataset}_{args.num_partitions}_CLEAN_{"full" if args.full_training else "biased"}/'
    dataset_folder = f'/home/jrm28/gelato/original_gelato/dataset_splits/{args.dataset}/'

    # Getting full training splits.
    data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=int)

    if args.dataset.startswith('ogbl'):


        set_random_seed(0)
        transform = RandomLinkSplit(add_negative_train_samples=False)
        train_graph, val_graph, test_graph = transform(data)

        data.edge_weight = torch.ones(size=(data.edge_index.shape[1],))

    else:
        split_edge = split_dataset(data)

        data.edge_index = split_edge['train']['edge'].t()

        data.edge_weight = data.train_pos_edge_attr

        train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true = compute_edges(split_edge)

        train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true = train_edges_pos.t(), train_edges_neg.t(), valid_edges.t(), valid_true.t(), test_edges.t(), test_true.t()

        # Processing
        train_graph = torch_geometric.data.Data(edge_index=data.edge_index, edge_weight=data.edge_weight, x=data.x, train_edges_pos=train_edges_pos, train_edges_neg=train_edges_neg)

        val_graph = torch_geometric.data.Data(edge_index=data.edge_index, edge_label_index=valid_edges, edge_label=valid_true)

        test_graph = torch_geometric.data.Data(edge_index=data.edge_index, edge_label_index=test_edges, edge_label=test_true)


    graph_partitions = torch_geometric.loader.cluster.ClusterData(train_graph, num_parts=args.num_partitions, log=True, save_dir=dataset_folder)

    splits, intercluster_splits, node_to_partition = graph_splits(graph_partitions, train_graph, val_graph, test_graph, full_training=args.full_training, save_dir=dataset_folder, ogbl=args.dataset.startswith('ogbl'))

    torch_geometric.seed_everything(args.random_seed)

    hyperparameters = {
        'gelato': {
            'eta': args.eta,
            'alpha': args.alpha,
            'beta': args.beta,
            'add_self_loop': args.add_self_loop,
            'trained_edge_weight_batch_size': args.trained_edge_weight_batch_size,
            'graph_learning_type': args.graph_learning_type,
            'graph_learning_params': {
                'in_channels': data.x.shape[-1] * 2,
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
                'scaling_parameter': args.scaling_parameter,
                # 'graph_volume': graph_volume
            },
            'device': device
        },
        'lr': args.lr,
        'epochs': args.epochs,
        'train_batch_ratio': args.train_batch_ratio,
    }


    best_valid_prec = -1
    best_epoch = -1

    running_n = 0
    running_sum = 0
    running_sum_squared = 0

    model = Gelato(**hyperparameters['gelato']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])

    version_name = f"CLUSTER-{args.train_batch_ratio}-seed-{args.random_seed}[HP-TUNNED]_{args.num_partitions}"
    
    run_name = f"{args.dataset}-{version_name}-{args.scaling_parameter}"


    # Set up results folder.
    results_folder = f'/scratch/jrm28/data/{run_name}/model/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    log_file = results_folder + 'log.txt'


    mode = 'disabled' if args.no_wandb else 'online'

    wandb.init(project="gelato", entity="joaopedromattos", config=args, name=run_name, mode=mode, settings=wandb.Settings(start_method='fork'))

    wandb.watch(model)

    # scalene_profiler.start()
    for epoch in tqdm(range(args.epochs), desc="Epochs"):

        valid_preds = []
        valid_labels = []
        total_loss = 0
        mem_allocated_values = []

        # code.interact(local=locals())

        for partition in splits.keys():
            # torch.cuda.empty_cache()

            print(f"PARTITION - {partition} ----- ")
            
            train_edge_index = splits[partition]['train_edge_index']

            train_edges_pos = splits[partition]['train_edges_pos']
            train_edges_neg = splits[partition]['train_edges_neg']

            num_nodes = splits[partition]['subgraph'].num_nodes

            # masking
            A = torch.sparse_coo_tensor(train_edge_index, torch.ones(size=(train_edge_index.shape[1], )), size=(num_nodes, num_nodes)).to_dense()            
            X = splits[partition]['subgraph'].x
            S = X @ X.t()
            print("--------- A.size()", A.shape)
            # code.interact(local=locals())

            training_params = {
                'model': model, 
                'A': A,
                'X': X.float(),
                'S': S.float(),
                'optimizer': optimizer,
                'train_edges_pos': train_edges_pos.t(),
                'train_edges_neg': train_edges_neg.t(),
                'train_batch_ratio': args.train_batch_ratio,
                'training_device': device,
                'running_n': running_n,
                'running_sum': running_sum,
                'running_sum_squared': running_sum_squared
            }

            loss, memory_allocated_train, running_mean, running_std, running_sum, running_sum_squared, running_n = train(**training_params)

            total_loss += loss
            mem_allocated_values.append(memory_allocated_train)

            code.interact(local=locals())
            

            valid_edges = splits[partition]['valid_edges']
            valid_true = splits[partition]['valid_true']

            
            A = torch.sparse_coo_tensor(train_edge_index, torch.ones(size=(train_edge_index.shape[1], )), size=(num_nodes, num_nodes)).to_dense()            
            X = splits[partition]['subgraph'].x
            S = X @ X.t()
            
        
            validation_params = {
                'model': model,
                'A': A,
                'X': X.float(),
                'S': S.float(),
                'valid_edges': valid_edges.t(),
                'valid_true': valid_true.t(),
                'validation_batch_ratio': args.validation_batch_ratio,
                'validation_device': device,
                'mean': running_mean, 
                'std': running_std
            }

            valid_pred, memory_allocated_val = valid(**validation_params)

            valid_preds.extend(valid_pred)
            valid_labels.extend(valid_true.tolist())


            del A, S

        
        print('Valid preds --->', valid_preds)
        print('Wrong  --->', valid_preds[valid_labels.bool().sum() == False])
        print(torch.tensor(valid_labels).int().bool().sum())
        wrong_pairs = torch.sort(torch.tensor(valid_preds)).indices[torch.tensor(valid_labels).int().bool().sum().item():]
        print("Wrong pairs", torch.tensor(valid_labels)[wrong_pairs], wrong_pairs)
        code.interact(local=locals())

        if not(args.intracluster_only):
            intercluster_preds = torch.zeros(size=(intercluster_splits['valid_edges'].shape[1], )).fill_(-1000).tolist()

            valid_preds.extend(intercluster_preds)
            valid_labels.extend(intercluster_splits['valid_true'].tolist())
            

        valid_preds = torch.tensor(valid_preds)
        valid_labels = torch.tensor(valid_labels)
        

        print("Preds --- ", valid_preds)
        valid_prec_at_100 = precision_at_k(valid_labels, valid_preds, valid_labels.int().sum().item())
        valid_ap = average_precision(eval_true=valid_labels, eval_pred=valid_preds)
        valid_hits_at_50 = hits_at_k(valid_labels, valid_preds, k=50)
        valid_hits_at_100 = hits_at_k(valid_labels, valid_preds, k=100)
        valid_hits_at_1000 = hits_at_k(valid_labels, valid_preds, k=1000)
        
                
        wandb.log({"epoch_loss":total_loss, 
                "val_precision_@_100%":valid_prec_at_100,
                'val_ap': valid_ap, 
                'val_hits_@_50': valid_hits_at_50,
                'val_hits_@_100': valid_hits_at_100,
                'val_hits_@_1000': valid_hits_at_1000,
                'mem_alloc_train_mean':np.mean(mem_allocated_values),
                'epoch': epoch})


        if valid_prec_at_100 > best_valid_prec:
            best_epoch = epoch
            best_valid_prec = valid_prec_at_100

        
        torch.save(model.state_dict(), results_folder + f'model_checkpoint{epoch}.pth')
        torch.save(optimizer.state_dict(), results_folder + f'optimizer_checkpoint{epoch}.pth')
        torch.save(running_mean, results_folder + f'running_mean.pth')
        torch.save(running_std, results_folder + f'running_mean.pth')



    if valid_prec_at_100 > best_valid_prec:
        best_epoch = epoch
        best_valid_prec = valid_prec_at_100


    # Record the best model.
    with open(log_file, 'a') as f:
        print(f"Best epoch = {best_epoch}", file=f)

    if (best_epoch != -1):
        shutil.copyfile(results_folder + f'model_checkpoint{best_epoch}.pth', results_folder + f'model_best.pth')
    else:
        shutil.copyfile(results_folder + f'model_checkpoint{epoch}.pth', results_folder + f'model_best.pth')
