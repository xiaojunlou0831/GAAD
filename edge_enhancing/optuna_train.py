import copy

import dgl
import pickle
import logging
import argparse
import numpy as np
import scipy.sparse as sp
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from edge_enhancing.GNN import GAT
from edge_enhancing.f1_scores import f1_score

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred


def test_gaugm(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset', default='DBLP')
    parser.add_argument('--model', type=str, default='gat')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layer')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--slope', type=float, default=0.05)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    adaptive_lr = args.adaptive_lr
    dropout = args.dropout
    n_heads = args.n_heads
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    with open('data/' + args.dataset + '/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open('data/' + args.dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open('data/' + args.dataset + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    num_nodes = edges[0].shape[0]

    remove_pct = trial.suggest_int('remove_pct', 0, 15)
    add_pct = trial.suggest_int('add_pct', 0, 15)

    A_pred = pickle.load(open(f'data/dblp/dblp_pred.pkl', 'rb'))
    adj = sample_graph_det(sum(edges), A_pred, remove_pct, add_pct)
    g1 = sum(edges)
    print(edges[1])
    # assert sp.issparse(adj)
    # if not isinstance(adj, sp.coo_matrix):
    #     adj = sp.coo_matrix(adj)
    # adj.setdiag(1)
    g = dgl.DGLGraph(adj)
    g1 = dgl.DGLGraph(sum(edges))
    # g.from_scipy(sum(edges))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to('cpu')

    # print(g)
    """for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)"""

    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor).tolist()
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor).tolist()
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor).tolist()
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)

    node_list = train_node + valid_node + test_node

    num_classes = torch.max(train_target).item() + 1
    # print(node_features.size())
    final_f1 = 0
    heads = ([n_heads] * num_layers) + [1]
    a = node_features.size()[1]
    for l in range(1):
        if args.model == 'gat':
            model = GAT(g, num_layers, node_features.size()[1], node_dim, num_classes, heads, F.elu, dropout, dropout,
                        args.slope, False)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': model.weight},
                                          {'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {"params": model.layers.parameters(), "lr": 0.5}
                                          ], lr=0.005, weight_decay=0.001)
        loss_fn = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0

        for i in range(epochs):
            # for param_group in optimizer.param_groups:
            #    if param_group['lr'] > 0.005:
            #        param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ', i + 1)
            model.zero_grad()
            model.train()
            y = model(node_features)
            train_loss = loss_fn(y[train_node], train_target)
            train_f1 = torch.mean(f1_score(torch.argmax(y[train_node].detach(), dim=1), train_target,
                                           num_classes=num_classes)).cpu().numpy()
            print('Train - Loss: {}, Macro_F1: {}'.format(train_loss.detach().cpu().numpy(), train_f1))
            train_loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                y = model.forward(node_features)
                val_loss = loss_fn(y[valid_node], valid_target)
                val_f1 = torch.mean(
                    f1_score(torch.argmax(y[valid_node], dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                y = model.forward(node_features)
                test_loss = loss_fn(y[test_node], test_target)
                test_f1 = torch.mean(
                    f1_score(torch.argmax(y[test_node], dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = train_loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        final_f1 += best_test_f1

    return final_f1


if __name__ == "__main__":
    logging.info('start')
    study = optuna.create_study(direction='maximize')
    study.optimize(test_gaugm, n_trials=200)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
