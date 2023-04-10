import pickle

import torch
import scipy.sparse as sp
from sklearn.preprocessing import normalize

features = pickle.load(open(f'../data/dblp/cora_features.pkl', 'rb'))
if isinstance(features, torch.Tensor):
    features = features.numpy()
features = sp.csr_matrix(features)
features_orig = normalize(features, norm='l1', axis=1)
print(features_orig)