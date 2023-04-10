import argparse
from edge_enhancing.dataloader import DataLoader
from utils import *
import numpy as np
from edge_enhancing.gae import VGAE

def get_args():
    parser = argparse.ArgumentParser(description='VGAE')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--emb_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gen_graphs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--val_frac', type=float, default=0.05)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--criterion', type=str, default='roc')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--gae', action='store_true')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_layers', type=int, default=3)
    # # tmp args for debuging
    parser.add_argument("--w_r", type=float, default=1)
    parser.add_argument("--w_kl", type=float, default=1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    return args

def main(args):
    args.device = torch.device(f'cuda:{args.cuda}' if args.cuda>=0 else 'cpu')

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dl = DataLoader(args)

    g = dl.g
    heads = ([8] * 2) + [1]

    model = VGAE(dl.adj_norm, 334, 32, 16, args.gae)
    model.to(args.device)
    model = train_model(args, dl, model)

    if args.gen_graphs > 0:
        gen_graphs(args, dl, model)

if __name__ == "__main__":
    args = get_args()
    main(args)