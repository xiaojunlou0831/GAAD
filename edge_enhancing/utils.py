import copy
import pickle
import time
import warnings
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def get_scores(edges_pos, edges_neg, A_pred, adj_label):
    # get logists and labels
    preds = A_pred[edges_pos.T]
    preds_neg = A_pred[edges_neg.T]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # logists = A_pred.view(-1)
    # labels = adj_label.to_dense().view(-1)
    # calc scores
    roc_auc = roc_auc_score(labels, logists)
    ap_score = average_precision_score(labels, logists)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    warnings.simplefilter('ignore', RuntimeWarning)
    f1s = np.nan_to_num(2 * precisions * recalls / (precisions + recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]
    # calc reconstracted adj_mat and accuracy with the threshold for best f1
    adj_rec = copy.deepcopy(A_pred)
    adj_rec[adj_rec < thresh] = 0
    adj_rec[adj_rec >= thresh] = 1
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = adj_rec.view(-1).long()
    recon_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    results = {'roc': roc_auc,
               'pr': pr_auc,
               'ap': ap_score,
               'pre': pre,
               'rec': rec,
               'f1': f1,
               'acc': recon_acc,
               'adj_recon': adj_rec}
    return results


def train_model(args, dl, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # weights for log_lik loss
    adj_t = dl.adj_train
    norm_w = adj_t.shape[0] ** 2 / float((adj_t.shape[0] ** 2 - adj_t.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_t.shape[0] ** 2 - adj_t.sum()) / adj_t.sum()]).to(args.device)
    # move input data and label to gpu if needed
    features = dl.features
    adj_label = dl.adj_label.to_dense()

    best_vali_criterion = 0.0
    best_state_dict = None
    model.train()
    for epoch in range(args.epochs):
        t = time.time()
        A_pred = model(features)
        optimizer.zero_grad()
        loss = log_lik = norm_w * F.binary_cross_entropy_with_logits(A_pred, adj_label, pos_weight=pos_weight)
        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        print(
            'Epoch{:3}: train_loss: {:.4f} recon_acc: {:.4f} val_roc: {:.4f} val_ap: {:.4f} f1: {:.4f} time: {:.4f}'.format(
                epoch + 1, loss.item(), r['acc'], r['roc'], r['ap'], r['f1'], time.time() - t))
        if r[args.criterion] > best_vali_criterion:
            best_vali_criterion = r[args.criterion]
            best_state_dict = copy.deepcopy(model.state_dict())
            # r_test = get_scores(dl.test_edges, dl.test_edges_false, A_pred, dl.adj_label)
            r_test = r
            print("          test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
                r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))
        loss.backward()
        optimizer.step()

    print("Done! final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
        r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

    model.load_state_dict(best_state_dict)
    return model


def gen_graphs(args, dl, model):
    adj_orig = dl.adj_orig
    assert adj_orig.diagonal().sum() == 0
    # sp.csr_matrix
    if args.gae:
        pickle.dump(adj_orig, open(f'graphs/{args.dataset}_graph_0_gae.pkl', 'wb'))
    else:
        pickle.dump(adj_orig, open(f'graphs/{args.dataset}_graph_0.pkl', 'wb'))
    # sp.lil_matrix
    pickle.dump(dl.features_orig, open(f'graphs/{args.dataset}_features.pkl', 'wb'))
    features = dl.features.to(args.device)
    for i in range(args.gen_graphs):
        with torch.no_grad():
            A_pred = model(features)
        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        adj_recon = A_pred.numpy()
        np.fill_diagonal(adj_recon, 0)
        # np.ndarray
        if args.gae:
            filename = f'graphs/{args.dataset}_graph_{i + 1}_logits_gae.pkl'
        else:
            filename = f'graphs/{args.dataset}_graph_{i + 1}_logits.pkl'
        pickle.dump(adj_recon, open(filename, 'wb'))
