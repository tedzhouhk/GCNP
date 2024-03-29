from GNN.globals import *
import math
from GNN.utils import *
from GNN.graph_samplers import *
from GNN.norm_aggr import *
import torch
import scipy.sparse as sp
import scipy
import numpy as np
import time
import hashlib


def _coo_scipy2torch(adj, coalesce=True, use_cuda=False):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    ans = torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))
    if use_cuda:
        ans = ans.cuda()
    if coalesce:
        ans = ans.coalesce()
    return ans


class Minibatch:
    """
    This minibatch iterator iterates over nodes for supervised learning.
    """
    def __init__(self,
                 adj_full_norm,
                 adj_train,
                 adj_val_norm,
                 role,
                 train_params,
                 cpu_eval=False):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        """
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda = False

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])
        self.node_trainval = np.concatenate((self.node_train,self.node_val),axis=None)

        self.adj_full_norm_sp = adj_full_norm.tocsr()
        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())
        self.adj_val_norm = _coo_scipy2torch(adj_val_norm.tocoo())
        self.adj_train = adj_train
        if self.use_cuda:  # now i put everything on GPU. Ideally, full graph adj/feat should be optionally placed on CPU
            self.adj_full_norm = self.adj_full_norm.cuda()
            self.adj_val_norm = self.adj_val_norm.cuda()
        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []


        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        # norm_loss_test is used in full batch evaluation (without sampling). so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        _denom = len(self.node_train) + len(self.node_val) + len(
            self.node_test)
        self.norm_loss_test[self.node_train] = 1. / _denom
        self.norm_loss_test[self.node_val] = 1. / _denom
        self.norm_loss_test[self.node_test] = 1. / _denom
        self.norm_loss_test = torch.from_numpy(
            self.norm_loss_test.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()
        self.norm_aggr_train = np.zeros(self.adj_train.size)

        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()

    def set_sampler(self, train_phases):
        self.subgraphs_remaining_indptr = list()
        self.subgraphs_remaining_indices = list()
        self.subgraphs_remaining_data = list()
        self.subgraphs_remaining_nodes = list()
        self.subgraphs_remaining_edge_index = list()
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'mrw':
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000  # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = mrw_sampling(self.adj_train,self.node_train,\
                self.size_subg_budget,train_phases['size_frontier'],_deg_clip)
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root'] * train_phases[
                'depth']
            self.graph_sampler = rw_sampling(self.adj_train,self.node_train,\
                self.size_subg_budget,int(train_phases['num_root']),int(train_phases['depth']))
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(self.adj_train, self.node_train,
                                               train_phases['size_subg_edge'])
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(self.adj_train, self.node_train,
                                               self.size_subg_budget)
        elif self.method_sample == 'full_batch':
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(self.adj_train,
                                                     self.node_train,
                                                     self.size_subg_budget)
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)

        # caching norm_aggr_train and norm_loss_train
        text = args_global.data_prefix
        for k, v in train_phases.items():
            text += str(k) + str(v)
        path = 'pytorch_models/sample' + hashlib.md5(text.encode('utf-8')).hexdigest() + '.npz'
        if os.path.isfile(path):
            print('Found existing sampling normalization coeefficients, loading from', path)
            samplef = np.load(path)
            self.norm_loss_train = samplef['norm_loss_train']
            self.norm_aggr_train = samplef['norm_aggr_train']
        else:
            print('Saving sampling normalization coeefficients to', path)
            # For edge sampler, no need to estimate norm factors, we can calculate directly.
            # However, for integrity of the framework, we decide to follow the same procedure for all samplers:
            # 1. sample enough number of subgraphs
            # 2. estimate norm factor alpha and lambda
            tot_sampled_nodes = 0
            while True:
                self.par_graph_sample('train')
                tot_sampled_nodes = sum(
                    [len(n) for n in self.subgraphs_remaining_nodes])
                if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                    break
            print()
            num_subg = len(self.subgraphs_remaining_nodes)
            for i in range(num_subg):
                self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
                self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
            assert self.norm_loss_train[self.node_val].sum(
            ) + self.norm_loss_train[self.node_test].sum() == 0
            for v in range(self.adj_train.shape[0]):
                i_s = self.adj_train.indptr[v]
                i_e = self.adj_train.indptr[v + 1]
                val = np.clip(
                    self.norm_loss_train[v] / self.norm_aggr_train[i_s:i_e], 0,
                    1e4)
                val[np.isnan(val)] = 0.1
                self.norm_aggr_train[i_s:i_e] = val
            self.norm_loss_train[np.where(self.norm_loss_train == 0)[0]] = 0.1
            self.norm_loss_train[self.node_val] = 0
            self.norm_loss_train[self.node_test] = 0
            self.norm_loss_train[
                self.node_train] = num_subg / self.norm_loss_train[
                    self.node_train] / self.node_train.size
            np.savez(path, norm_loss_train=self.norm_loss_train, norm_aggr_train=self.norm_aggr_train)
        
        self.norm_loss_train = torch.from_numpy(
                self.norm_loss_train.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self, phase):
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(
            phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0),
              end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def one_batch(self, mode='train'):
        if mode in ['val', 'test']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])
            if mode == 'val':
                adj = self.adj_val_norm
            elif mode == 'test':
                adj = self.adj_full_norm
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(),\
                                 self.subgraphs_remaining_indices.pop(),\
                                 self.subgraphs_remaining_indptr.pop()),\
                                 shape=(self.size_subgraph,self.size_subgraph))
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            norm_aggr(adj.data,
                      adj_edge_index,
                      self.norm_aggr_train,
                      num_proc=args_global.num_cpu_core)
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])
            adj = _coo_scipy2torch(adj.tocoo(),use_cuda=self.use_cuda)
            self.batch_num += 1
        norm_loss = self.norm_loss_test if mode in ['val', 'test'
                                                    ] else self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph]
        return self.node_subgraph, adj, norm_loss

    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] /
                         float(self.size_subg_budget))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return (self.batch_num +
                1) * self.size_subg_budget >= self.node_train.shape[0]
