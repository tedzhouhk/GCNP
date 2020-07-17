import torch
import scipy.sparse as sp
import numpy as np
import os
import time
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='2'

def _coo_scipy2torch(adj, coalesce=True):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    ans = torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))
    if coalesce:
        ans = ans.cuda().coalesce()
    return ans

adj_index = 51200 * 32
adj_row = np.zeros(adj_index)
adj_col = np.zeros(adj_index)
adj_data = np.zeros(adj_index, dtype=np.float32)
for i in tqdm(range(51200)):
    for j in range(32):
        adj_row[i * 32 + j] = i
        adj_col[i * 32 + j] = i * 32 + j
        adj_data[i] = 1 / 32
adj=sp.coo_matrix((adj_data,(adj_row,adj_col)),shape=(51200,1638400))
feat_all = torch.zeros(2000000, 300, device='cuda')
index = np.arange(1638400)
mask = np.random.choice(a=[False, True], size=(1638400), p=[0.15, 0.85])
feat = torch.zeros(1638400, 300, device='cuda')
index_masked=index[mask]
del feat

time_s = time.time()
feat = torch.zeros(1638400, 300, device='cuda')
feat[mask]=feat_all[index_masked]
torch.cuda.synchronize()
print('create time:', time.time() - time_s)

time_s = time.time()
adj_tensor = _coo_scipy2torch(adj)
ans1 = torch.sparse.mm(adj_tensor, feat)
torch.cuda.synchronize()
print('sparse time:', time.time() - time_s)

time_s = time.time()
feat_dense = feat.view(51200, 32, 300)
ans2 = feat_dense.mean(dim=1)
torch.cuda.synchronize()
print('dense time:', time.time() - time_s)
