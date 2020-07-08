import torch
from torch import nn
import scipy.sparse as sp


F_ACT = {'relu': nn.ReLU(),
         'I': lambda x:x}

F_ACT_I = {'relu':nn.ReLU(inplace=True),'I':lambda x:x}


class HighOrderAggregator(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0., act='relu', \
            order=1, aggr='mean', bias='norm', **kwargs):
        super(HighOrderAggregator,self).__init__()
        self.order,self.aggr = order,aggr
        self.act, self.bias = F_ACT[act], bias
        self.inplace_act=F_ACT_I[act]
        self.dropout = dropout
        self.f_lin = list()
        self.f_bias = list()
        self.offset=list()
        self.scale=list()
        for o in range(self.order+1):
            self.f_lin.append(nn.Linear(dim_in,dim_out,bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out)))
            self.offset.append(nn.Parameter(torch.zeros(dim_out)))
            self.scale.append(nn.Parameter(torch.ones(dim_out)))
        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params=nn.ParameterList(self.f_bias+self.offset+self.scale)
        self.f_bias=self.params[:self.order+1]
        self.offset=self.params[self.order+1:2*self.order+2]
        self.scale=self.params[2*self.order+2:]
        self.feat_partial=[]

    def _spmm(self, adj_norm, _feat):
        # alternative ways: use geometric.propagate or torch.mm
        return torch.sparse.mm(adj_norm,_feat)

    def _f_feat_trans(self, _feat, _id):
        feat=self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias=='norm':
            mean=feat.mean(dim=1).view(feat.shape[0],1)
            var=feat.var(dim=1,unbiased=False).view(feat.shape[0],1)+1e-9
            feat_out=(feat-mean)*self.scale[_id]*torch.rsqrt(var)+self.offset[_id]
        else:
            feat_out=feat
        return feat_out

    def forward(self, inputs):
        """
        Inputs:.
            adj_norm        edge-list represented adj matrix
        """
        adj_norm, feat_in = inputs
        feat_in = self.f_dropout(feat_in)
        feat_hop = [feat_in]
        for o in range(self.order):
            feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))
        feat_partial=[self._f_feat_trans(ft,idf) for idf,ft in enumerate(feat_hop)]
        if self.aggr == 'mean':
            feat_out = feat_partial[0]
            for o in range(len(feat_partial)-1):
                feat_out += feat_partial[o+1]
        elif self.aggr == 'concat':
            feat_out = torch.cat(feat_partial,1)
        else:
            raise NotImplementedError
        return adj_norm, feat_out       # return adj_norm to support Sequential
        
    def inplace_forward(self, feat, adj=None):
        """
        first compute X' = X x W, then A x X'
        """
        feat_partial = list()
        for o in range(self.order+1):
            __feat = torch.mm(feat, self.f_lin[o].weight.T)
            for _ in range(o):
                __feat=self._spmm(adj,__feat)
            __feat.add_(self.f_bias[o])
            self.inplace_act(__feat)
            if self.bias=='norm':
                mean=__feat.mean(dim=1).view(__feat.shape[0],1)
                var=__feat.var(dim=1,unbiased=False).view(__feat.shape[0],1)+1e-9
                var.rsqrt_()
                __feat.add_(mean,alpha=-1)
                __feat.mul_(self.scale[o])
                __feat.mul_(var)
                __feat.add_(self.offset[o])
                del mean
                del var
            feat_partial.append(__feat)
            del __feat
        if self.aggr=='concat':
            out=torch.cat(feat_partial,1)
            del feat_partial
        elif self.aggr=='mean':
            out=feat_partial[0]
            for o in range(len(feat_partial)-1):
                out.add_(feat_partial[o+1])
            del feat_partial
        else:
            raise NotImplementedError
        return out

    def sparse_forward(self,feat_self,feat_neigh,adj):
        assert self.order==1
        from_self=feat_self
        from_neigh=self._spmm(adj,feat_neigh)
        from_self=self._f_feat_trans(from_self,0)
        from_neigh=self._f_feat_trans(from_neigh,1)
        if self.aggr=='concat':
            return torch.cat([from_self,from_neigh],1)
        else:
            raise NotImplementedError

    def dense_forward(self,feat_self,feat_neigh):
        assert self.order==1
        from_self=feat_self
        from_neigh=torch.mean(feat_neigh,dim=1)
        from_self=self._f_feat_trans(from_self,0)
        from_neigh=self._f_feat_trans(from_neigh,1)
        if self.aggr=='concat':
            return torch.cat([from_self,from_neigh],1)
        else:
            raise NotImplementedError

class PrunedHighOrderAggregator(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0., act='relu', \
            order=1, aggr='concat', bias='norm', **kwargs):
        super(PrunedHighOrderAggregator,self).__init__()
        self.order,self.aggr = order,aggr
        self.act, self.bias = F_ACT[act], bias
        self.inplace_act=F_ACT_I[act]
        self.dropout = dropout
        self.f_lin = list()
        self.f_bias = list()
        self.offset=list()
        self.scale=list()
        for o in range(self.order+1):
            self.f_lin.append(nn.Linear(dim_in[o],dim_out[o],bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out[o])))
            self.offset.append(nn.Parameter(torch.zeros(dim_out[o])))
            self.scale.append(nn.Parameter(torch.ones(dim_out[o])))
        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params=nn.ParameterList(self.f_bias+self.offset+self.scale)
        self.f_bias=self.params[:self.order+1]
        self.offset=self.params[self.order+1:2*self.order+2]
        self.scale=self.params[2*self.order+2:]
        self.feat_partial=[]

    def _spmm(self, adj_norm, _feat):
        # alternative ways: use geometric.propagate or torch.mm
        return torch.sparse.mm(adj_norm, _feat)

    def _f_feat_trans(self, _feat, _id):
        feat=self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias=='norm':
            mean=feat.mean(dim=1).view(feat.shape[0],1)
            var=feat.var(dim=1,unbiased=False).view(feat.shape[0],1)+1e-9
            feat_out=(feat-mean)*self.scale[_id]*torch.rsqrt(var)+self.offset[_id]
        else:
            feat_out=feat
        return feat_out

    def forward(self, inputs):
        """
        Inputs:.
            adj_norm        edge-list represented adj matrix
        """
        adj_norm, feat_in, first_layer, masks = inputs
        feat_in = self.f_dropout(feat_in)
        if first_layer:
            first_layer=False
            feat_hop=list()
            for o in range(self.order+1):
                feat_hop.append(feat_in[:,masks[o]])
                for _ in range(o):
                    feat_hop[o]=self._spmm(adj_norm,feat_hop[o])
        else:
            feat_hop = [feat_in]
            for o in range(self.order):
                feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))
        feat_partial=[self._f_feat_trans(ft,idf) for idf,ft in enumerate(feat_hop)]
        if self.aggr == 'concat':
            feat_out = torch.cat(feat_partial,1)
        else:
            raise NotImplementedError
        # print(feat_out)
        return adj_norm, feat_out, first_layer, masks      # return adj_norm to support Sequential

    def load_pruned_weight(self,mask_in,mask_out,ref_layer,ref_weight,first_layer=False):
        with torch.no_grad():
            if first_layer:
                for o in range(self.order+1):
                    self.f_lin[o].weight.data.copy_(torch.transpose(ref_weight[o][mask_in[o],:][:,mask_out[o]],0,1).data)
            else:
                for o in range(self.order+1):
                    mask_in_cat=torch.cat(mask_in,0)
                    self.f_lin[o].weight.data.copy_(torch.transpose(ref_weight[o][mask_in_cat,:][:,mask_out[o]],0,1).data)
            for o in range(self.order+1):
                self.f_bias[o].data.copy_(ref_layer.f_bias[o][mask_out[o]].data)
                self.scale[o].data.copy_(ref_layer.scale[o][mask_out[o]].data)
                self.offset[o].data.copy_(ref_layer.offset[o][mask_out[o]].data)

    def inplace_forward(self, feat, adj=None, first_layer=False, masks=None):
        """
        first compute X' = X x W, then A x X'
        """
        feat_partial=list()
        for o in range(self.order+1):
            if first_layer:
                _feat=feat[:,masks[o]]
            else:
                _feat=feat
            __feat = torch.mm(_feat, self.f_lin[o].weight.T)
            del _feat
            for i in range(o):
                __feat=self._spmm(adj,__feat)
            __feat.add_(self.f_bias[o])
            self.inplace_act(__feat)
            if self.bias=='norm':
                mean=__feat.mean(dim=1).view(__feat.shape[0],1)
                var=__feat.var(dim=1,unbiased=False).view(__feat.shape[0],1)+1e-9
                var.rsqrt_()
                __feat.add_(mean,alpha=-1)
                __feat.mul_(self.scale[o])
                __feat.mul_(var)
                __feat.add_(self.offset[o])
                del mean
                del var
            feat_partial.append(__feat)
            del __feat
        if self.aggr=='concat':
            out=torch.cat(feat_partial,1)
            del feat_partial
        elif self.aggr=='mean':
            out=feat_partial[0]
            for o in range(len(feat_partial)-1):
                out.add_(feat_partial[o+1])
            del feat_partial
        else:
            raise NotImplementedError
        return out

    def sparse_forward(self,feat_self,feat_neigh,adj):
        assert self.order==1
        from_self=feat_self
        from_neigh=self._spmm(adj,feat_neigh)
        from_self=self._f_feat_trans(from_self,0)
        from_neigh=self._f_feat_trans(from_neigh,1)
        if self.aggr=='concat':
            return torch.cat([from_self,from_neigh],1)
        else:
            raise NotImplementedError

    def dense_forward(self,feat_self,feat_neigh):
        assert self.order==1
        from_self=feat_self
        from_neigh=torch.mean(feat_neigh,dim=1)
        from_self=self._f_feat_trans(from_self,0)
        from_neigh=self._f_feat_trans(from_neigh,1)
        if self.aggr=='concat':
            return torch.cat([from_self,from_neigh],1)
        else:
            raise NotImplementedError