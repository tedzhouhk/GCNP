import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from graphsaint.utils import *
import graphsaint.pytorch_version.layers as layers
from tqdm import tqdm
from graphsaint.pytorch_version.minibatch_sampler import *
from graphsaint.pytorch_version.minibatch import _coo_scipy2torch

class GraphSAINT(nn.Module):
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Inputs:
            arch_gcn            parsed arch of GCN
            train_params        parameters for training
        """
        super(GraphSAINT,self).__init__()
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False
        if "attention" in arch_gcn:
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls=layers.GatedAttentionAggregator
                    self.mulhead=int(arch_gcn['attention'])
            else:
                self.aggregator_cls=layers.AttentionAggregator
                self.mulhead=int(arch_gcn['attention'])
        else:
            self.aggregator_cls=layers.HighOrderAggregator
            self.mulhead=1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.train_params=train_params
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = (arch_gcn['loss']=='sigmoid')
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()
        self.num_classes = num_classes
        _dims,self.order_layer,self.act_layer,self.bias_layer,self.aggr_layer \
                        = parse_layer_yml(arch_gcn,self.feat_full.shape[1])
        self._dims=_dims
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)

        self.loss = 0
        self.opt_op = None

        # build the model below        
        self.aggregators = self.get_aggregators()
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes,\
                            act='I', order=0, dropout=self.dropout, bias='bias')
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)

    def set_dims(self,dims):
        self.dims_feat = [dims[0]] + [((self.aggr_layer[l]=='concat')*self.order_layer[l]+1)*dims[l+1] for l in range(len(dims)-1)]
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        idx_conv = np.where(np.array(self.order_layer)>=1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer)-1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer)==1)[0])


    def forward(self, node_subgraph, adj_subgraph):
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg))
        # import pdb; pdb.set_trace()
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        pred_subg = self.classifier((None, emb_subg_norm))[1]
        return pred_subg, label_subg, label_subg_converted


    def _loss(self, preds, labels, norm_loss):
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss,reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss*_ls).sum()


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        aggregators = []
        for l in range(self.num_layers):
            aggrr = self.aggregator_cls(*self.dims_weight[l],dropout=self.dropout,\
                    act=self.act_layer[l], order=self.order_layer[l], \
                    aggr=self.aggr_layer[l], bias=self.bias_layer[l], mulhead=self.mulhead)
            aggregators.append(aggrr)
        return aggregators

    def predict(self, preds):
        # return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)
        if self.sigmoid_loss:
            preds.sigmoid_()
        else:
            preds.exp_()
            preds/=torch.sum(preds,dim=1,keepdim=True)
        return preds
        
        
    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward and backward propagation
        """
        self.train()
        self.optimizer.zero_grad()
        preds,labels,labels_converted = self(node_subgraph, adj_subgraph)    # will call the forward function
        loss = self._loss(preds,labels_converted,norm_loss_subgraph) # labels.squeeze()?
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return loss,self.predict(preds),labels

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward propagation only
        """
        self.eval()
        with torch.no_grad():
            # preds,labels,labels_converted = self(node_subgraph, adj_subgraph)
            # loss = self._loss(preds,labels_converted,norm_loss_subgraph)
            assert node_subgraph.shape[0]==self.feat_full.shape[0]
            _feat=self.feat_full
            for layer in self.aggregators:
                feat=layer.inplace_forward(_feat,adj_subgraph)
                del _feat
                _feat=feat
            F.normalize(_feat,p=2,dim=1,out=_feat)
            preds=self.classifier.inplace_forward(_feat)
            del _feat
            labels_converted=self.label_full if self.sigmoid_loss else self.label_full_cat
            loss=self._loss(preds,labels_converted,norm_loss_subgraph)
        return loss,self.predict(preds),self.label_full

    def get_input_activation(self,node_subgraph,adj_subgraph,norm_loss_subgraph,layer_step):
        self.eval()
        with torch.no_grad():
            assert node_subgraph.shape[0]==self.feat_full.shape[0]
            if layer_step==0:
                return self.feat_full
            else:
                _feat=self.feat_full
                for layer in self.aggregators[:layer_step]:
                    feat=layer.inplace_forward(_feat,adj_subgraph)
                    del _feat
                    _feat=feat
                return _feat
        return

    def minibatched_eval(self,node_test,adj,inf_params):
        print('start minibatch inference ...')
        self.eval()
        t_forward=0
        t_sampling=0
        minibatch_sampler=MinibatchSampler(adj.indptr,adj.indices,inf_params['neighbors'])
        with torch.no_grad():
            minibatches=np.array_split(node_test.astype(np.int32),int(node_test.shape[0]/inf_params['batch_size']))
            preds=list()
            labels=list()
            for nodes in tqdm(minibatches):
                t_sampling_s=time.time()
                supports=[nodes]
                last_layer=True
                for layer in reversed(self.aggregators):
                    if layer.order>0:
                        assert layer.order==1
                        if last_layer:
                            support,subg_adj=minibatch_sampler.sparse_sampling(supports[0])
                            subg_adj=_coo_scipy2torch(subg_adj)
                            if self.use_cuda:
                                subg_adj=subg_adj.cuda()
                            supports.insert(0,support)
                            last_layer=False
                        else:
                            support=minibatch_sampler.dense_sampling(supports[0])
                            supports.insert(0,support)
                t_sampling+=time.time()-t_sampling_s
                t_forward_s=time.time()
                support_idx=0
                _feat=self.feat_full[supports[support_idx]]
                for layer in self.aggregators:
                    if support_idx==len(supports)-2:
                        last_layer=True
                    else:
                        last_layer=False
                    if layer.order>0:
                        if last_layer:
                            _feat_self=_feat[:supports[support_idx+1].shape[0]]
                            _feat_neigh=_feat
                            _feat=layer.sparse_forward(_feat_self,_feat_neigh,subg_adj)
                        else:
                            _feat_self=_feat[:supports[support_idx+1].shape[0]]
                            _feat_neigh=_feat[supports[support_idx+1].shape[0]:].view(supports[support_idx+1].shape[0],inf_params['neighbors'],_feat.shape[1])
                            _feat=layer.dense_forward(_feat_self,_feat_neigh)
                        support_idx+=1
                    else:
                        _feat=layer.inplace_forward(_feat)
                F.normalize(_feat,p=2,dim=1,out=_feat)
                pred=self.classifier.inplace_forward(_feat)
                label=self.label_full[nodes]
                t_forward+=time.time()-t_forward_s
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
        return preds,labels,t_forward,t_sampling


class PrunedGraphSAINT(nn.Module):
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, dims_in, dims_out, masks, cpu_eval=False):
        super(PrunedGraphSAINT,self).__init__()
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False
        self.aggregator_cls=layers.PrunedHighOrderAggregator
        self.mulhead=1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = (arch_gcn['loss']=='sigmoid')
        self.feat_full = feat_full
        self.label_full = label_full
        self.dims_in=dims_in
        self.dims_out=dims_out
        self.masks=masks
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()
        else:
            self.feat_full=self.feat_full.cpu()
            self.label_full=self.label_full.cpu()
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.cpu().numpy().argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()
        self.num_classes = num_classes
        _dims,self.order_layer,self.act_layer,self.bias_layer,self.aggr_layer \
                        = parse_layer_yml(arch_gcn,self.feat_full.shape[1])
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()

        self.loss = 0
        self.opt_op = None

        # build the model below        
        self.aggregators = self.get_aggregators()
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.classifier = layers.PrunedHighOrderAggregator(self.dims_in[-1], self.dims_out[-1],\
                            act='I', order=0, dropout=self.dropout, bias='bias')
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)

    def set_idx_conv(self):
        idx_conv = np.where(np.array(self.order_layer)>=1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer)-1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer)==1)[0])


    def forward(self, node_subgraph, adj_subgraph):
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        _,emb_subg,_,_ = self.conv_layers((adj_subgraph, feat_subg, True, self.masks))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        pred_subg = self.classifier((None, emb_subg_norm, False, self.masks))[1]
        return pred_subg, label_subg, label_subg_converted


    def _loss(self, preds, labels, norm_loss):
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss,reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss*_ls).sum()


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        aggregators = []
        for l in range(self.num_layers):
            aggrr = self.aggregator_cls(self.dims_in[l],self.dims_out[l],dropout=self.dropout,\
                    act=self.act_layer[l], order=self.order_layer[l], \
                    aggr=self.aggr_layer[l], bias=self.bias_layer[l], mulhead=self.mulhead)
            aggregators.append(aggrr)
        return aggregators

    def predict(self, preds):
        # return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)
        if self.sigmoid_loss:
            preds.sigmoid_()
        else:
            preds.exp_()
            preds/=torch.sum(preds,dim=1,keepdim=True)
        return preds
        
    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward and backward propagation
        """
        self.train()
        self.optimizer.zero_grad()
        preds,labels,labels_converted = self(node_subgraph, adj_subgraph)    # will call the forward function
        loss = self._loss(preds,labels_converted,norm_loss_subgraph) # labels.squeeze()?
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return loss,self.predict(preds),labels

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward propagation only
        """
        self.eval()
        with torch.no_grad():
            # preds,labels,labels_converted = self(node_subgraph, adj_subgraph)
            # loss = self._loss(preds,labels_converted,norm_loss_subgraph)
            assert node_subgraph.shape[0]==self.feat_full.shape[0]
            _feat=self.feat_full
            first_layer=True
            # import pdb; pdb.set_trace()
            for layer in self.aggregators:
                feat=layer.inplace_forward(_feat,adj_subgraph,first_layer,self.masks)
                first_layer=False
                del _feat
                _feat=feat
            F.normalize(_feat,p=2,dim=1,out=_feat)
            preds=self.classifier.inplace_forward(_feat)
            del _feat
            labels_converted=self.label_full if self.sigmoid_loss else self.label_full_cat
            loss=self._loss(preds,labels_converted,norm_loss_subgraph)
        return loss,self.predict(preds),self.label_full

    def minibatched_eval(self,node_test,adj,inf_params):
        print('start minibatch inference ...')
        self.eval()
        minibatch_sampler=MinibatchSampler(adj.indptr,adj.indices,inf_params['neighbors'])
        t_forward=0
        t_sampling=0
        with torch.no_grad():
            _feat_self_full=self.feat_full[:,self.masks[0]]
            _feat_neigh_full=self.feat_full[:,self.masks[1]]
            minibatches=np.array_split(node_test.astype(np.int32),int(node_test.shape[0]/inf_params['batch_size']))
            preds=list()
            labels=list()
            for nodes in tqdm(minibatches):
                t_sampling_s=time.time()
                supports=[nodes]
                last_layer=True
                for layer in reversed(self.aggregators):
                    if layer.order>0:
                        assert layer.order==1
                        if last_layer:
                            support,subg_adj=minibatch_sampler.sparse_sampling(supports[0])
                            subg_adj=_coo_scipy2torch(subg_adj)
                            if self.use_cuda:
                                subg_adj=subg_adj.cuda()
                            supports.insert(0,support)
                            last_layer=False
                        else:
                            support=minibatch_sampler.dense_sampling(supports[0])
                            supports.insert(0,support)
                t_sampling+=time.time()-t_sampling_s
                t_forward_s=time.time()
                support_idx=0
                _feat_self=_feat_self_full[supports[0][:supports[1].shape[0]]]
                _feat_neigh=_feat_neigh_full[supports[0][supports[1].shape[0]:]].view(supports[1].shape[0],inf_params['neighbors'],_feat_self.shape[1])
                first_layer=True
                for layer in self.aggregators:
                    if support_idx==len(supports)-2:
                        last_layer=True
                    else:
                        last_layer=False
                    if layer.order>0:
                        if last_layer:
                            _feat_self=_feat[:supports[support_idx+1].shape[0]]
                            _feat_neigh=_feat
                            _feat=layer.sparse_forward(_feat_self,_feat_neigh,subg_adj)
                        else:
                            if first_layer:
                                _feat=layer.dense_forward(_feat_self,_feat_neigh)
                            else:
                                _feat_self=_feat[:supports[support_idx+1].shape[0]]
                                _feat_neigh=_feat[supports[support_idx+1].shape[0]:].view(supports[support_idx+1].shape[0],inf_params['neighbors'],_feat.shape[1])
                                _feat=layer.dense_forward(_feat_self,_feat_neigh)
                        support_idx+=1
                        first_layer=False
                    else:
                        _feat=layer.inplace_forward(_feat)
                F.normalize(_feat,p=2,dim=1,out=_feat)
                pred=self.classifier.inplace_forward(_feat)
                label=self.label_full[nodes]
                t_forward+=time.time()-t_forward_s
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
        return preds,labels,t_forward,t_sampling