import torch
from torch import nn


class Lasso(nn.Module):
    def __init__(self, dim_in, weight, lmbd_1, lmbd_2, lmbd_1_step,
                 lmbd_2_step, beta_lr, weight_lr):
        super(Lasso, self).__init__()
        self.beta = nn.Parameter(torch.ones(dim_in))
        self.weight = nn.Parameter(torch.zeros(weight.shape))
        self.weight.data.copy_(weight.data)
        self.weight_norm_factor = torch.norm(self.weight, p='fro')
        self.params = nn.ParameterList([self.beta, self.weight])
        self.beta = self.params[0]
        self.weight = self.params[1]
        self.lmbd_1 = lmbd_1
        self.lmbd_2 = lmbd_2
        self.lmbd_1_step = lmbd_1_step
        self.lmbd_2_step = lmbd_2_step
        self.beta_optimizer = torch.optim.Adam([self.beta], lr=beta_lr)
        self.weight_optimizer = torch.optim.Adam([self.weight], lr=weight_lr)

    def forward(self, inputs, mask_in):
        return torch.mm(inputs, self.weight * self.beta.unsqueeze(1))[:,
                                                                      mask_in]

    def optimize_beta(self, inputs, ref, mask_in):
        self.beta_optimizer.zero_grad()
        self.beta.requires_grad = True
        self.weight.requires_grad = False
        out = self(inputs, mask_in)
        loss_beta = nn.MSELoss()(out, ref[:, mask_in])
        loss_beta += self.lmbd_1 * torch.norm(self.beta, p=1)
        loss_beta += self.lmbd_2 * torch.norm(self.beta, p=2)
        loss_beta.backward()
        self.beta_optimizer.step()
        return loss_beta

    def lmbd_step(self):
        self.lmbd_1 += self.lmbd_1_step
        self.lmbd_2 += self.lmbd_2_step

    def clip_beta(self, budget, beta_clip=False):
        with torch.no_grad():
            _, indices = torch.sort(torch.abs(self.beta), descending=False)
            self.beta[indices[:int(self.beta.shape[0] * budget)]] = 0
            mask_out = torch.ones(self.beta.shape[0], dtype=bool)
            mask_out[indices[:int(self.beta.shape[0] * budget)]] = 0
            self.mask_out = mask_out
            if beta_clip:
                # clip to mean of remaining beta
                torch.clamp(self.beta, max=torch.mean(self.beta[mask_out]))
        return mask_out

    def seperated_clip_beta(self, self_budget, neigh_budget, beta_clip=False):
        with torch.no_grad():
            beta_self = self.beta[:int(self.beta.shape[0] / 2)]
            beta_neigh = self.beta[int(self.beta.shape[0] / 2):]
            mask_out = torch.ones(self.beta.shape[0], dtype=bool)
            _, indices = torch.sort(torch.abs(beta_self), descending=False)
            mask_out[indices[:int(beta_self.shape[0] * self_budget)]] = 0
            _, indices = torch.sort(torch.abs(beta_neigh), descending=False)
            indices += beta_self.shape[0]
            mask_out[indices[:int(beta_self.shape[0] * neigh_budget)]] = 0
            self.mask_out = mask_out
            if beta_clip:
                torch.clamp(self.beta, max=torch.mean(self.beta[mask_out]))
        return mask_out
                

    def optimize_weight(self, inputs, ref, mask_in):
        self.weight_optimizer.zero_grad()
        self.beta.requires_grad = False
        self.weight.requires_grad = True
        out = self(inputs, mask_in)
        loss_weight = nn.MSELoss()(out, ref[:, mask_in])
        loss_weight.backward()
        self.weight_optimizer.step()
        return loss_weight

    def norm(self):
        with torch.no_grad():
            normp = torch.norm(self.weight, p='fro') / self.weight_norm_factor
            self.weight /= normp
            self.beta *= normp
        return

    def apply_beta(self):
        with torch.no_grad():
            self.weight *= self.beta.unsqueeze(1)
        return