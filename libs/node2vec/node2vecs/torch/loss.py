# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-09 15:55:30
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Loss function
#
class Node2VecTripletLoss(nn.Module):
    def __init__(self, n_neg):
        super(Node2VecTripletLoss, self).__init__()
        self.n_neg = n_neg
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, model, iwords, owords, nwords):
        ivectors = model.forward_i(iwords).unsqueeze(2)
        ovectors = model.forward_o(owords)
        nvectors = model.forward_o(nwords).neg()
        oloss = self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(dim=1)
        nloss = (
            self.logsigmoid(torch.bmm(nvectors, ivectors).squeeze())
            .view(-1, owords.size()[1], self.n_neg)
            .sum(dim=2)
            .mean(dim=1)
        )
        return -(oloss + nloss).mean()


class ModularityTripletLoss(nn.Module):
    def __init__(self, n_neg):
        super(ModularityTripletLoss, self).__init__()
        self.n_neg = n_neg
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, model, iwords, owords, nwords, base_iwords, base_owords):
        ivectors = model.forward_i(iwords).unsqueeze(2)
        ovectors = model.forward_o(owords)
        nvectors = model.forward_o(nwords)

        base_ivectors = model.forward_i(base_iwords).unsqueeze(2)
        base_ovectors = model.forward_o(base_owords)

        oloss = torch.bmm(ovectors, ivectors).squeeze().mean(dim=1)
        nloss = (
            torch.bmm(nvectors, ivectors)
            .squeeze()
            .view(-1, owords.size()[1], self.n_neg)
            .sum(dim=2)
            .mean(dim=1)
        )

        n_nodes = model.vocab_size 
        base_loss = torch.bmm(base_ovectors, base_ivectors).squeeze()
        base_loss = torch.pow(base_loss, 2).mean(dim=1)
        
        loss = -oloss.mean() + nloss.mean() + 0.5 * n_nodes * base_loss.mean()
        #loss = -(oloss + nloss - 0.5 * torch.pow(base_loss, 2)).mean()

        return loss


class LaplacianEigenMapTripletLoss(nn.Module):
    def __init__(self, n_neg):
        super(LaplacianEigenMapTripletLoss, self).__init__()
        self.n_neg = n_neg

    def forward(self, model, iwords, owords, nwords):
        ivectors = model.forward_i(iwords).unsqueeze(2)
        ovectors = model.forward_o(owords)
        nvectors = model.forward_o(nwords).neg()

        oloss = torch.bmm(ovectors, ivectors).squeeze().mean(dim=1)
        nloss = (
            torch.bmm(nvectors, ivectors)
            .squeeze()
            .view(-1, owords.size()[1], self.n_neg)
            .sum(dim=2)
            .mean(dim=1)
        )

        loss = -(oloss - 0.5 * torch.pow(nloss, 2)).mean()

        return loss


# class ModularityTripletLoss(TripletLoss):
#    def __init__(self, **params):
#        super(ModularityTripletLoss, self).__init__(**params)
#
#    def forward(self, iword, oword, y, pcoef):
#        ivectors = self.model.forward_i(iword)
#        ovectors = self.model.forward_o(oword)
#
#        loss = -self.dist_func(ivectors, ovectors).squeeze()
#        # if self.with_logsigmoid:
#        #    loss = self.logsigmoid(loss)
#        loss = torch.pow(loss, pcoef) * y.squeeze()
#        return -(loss).mean()
#
