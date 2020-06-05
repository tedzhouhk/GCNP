import torch
from torch import nn

class Lasso(nn.Module):
    def __init__(self,dim_in,weight,lmbd,beta_lr,weight_lr):
        super(Lasso,self).__init__()
        self.beta=nn.Parameter(torch.ones(dim_in))
        self.weight=nn.Parameter(torch.zeros(weight.shape))
        self.weight.data.copy_(weight.data)
        self.params=nn.ParameterList([self.beta,self.weight])
        self.beta=self.params[0]
        self.weight=self.params[1]
        self.lmbd=lmbd
        self.beta_optimizer=torch.optim.Adam([self.beta],lr=beta_lr)
        self.weight_optimizer=torch.optim.Adam([self.weight],lr=weight_lr)

    def forward(self,inputs,mask_in):
        return torch.mm(inputs,self.weight*self.beta.unsqueeze(1))[:,mask_in]

    def optimize_beta(self,inputs,ref,mask_in):
        self.beta.requires_grad=True
        self.weight.requires_grad=False
        out=self(inputs,mask_in)
        loss_beta=nn.MSELoss()(out,ref[:,mask_in])
        loss_beta+=self.lmbd*torch.norm(self.beta,p=1)
        loss_beta.backward()
        self.beta_optimizer.step()
        self.beta_optimizer.zero_grad()
        return loss_beta
    
    def clip_beta(self,budget):
        with torch.no_grad():
            _,indices=torch.sort(torch.abs(self.beta),descending=False)
            self.beta[indices[:int(self.beta.shape[0]*budget)]=0
        return

    def optimize_weight(self,inputs,ref,mask_in):
        self.beta.requires_grad=False
        self.weight.requires_grad=True
        out=self(inputs,mask_in)
        loss_weight=nn.MSELoss()(out,ref[:,mask_in])
        loss_weight.backward()
        self.weight_optimizer.step()
        self.weight_optimizer.zero_grad()
        return loss_weight

    def norm(self):
        with torch.no_grad():
            normp=torch.norm(self.weight,p='fro')
            self.weight/=normp
            self.beta*=normp
        return