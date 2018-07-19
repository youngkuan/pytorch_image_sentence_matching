# -*- coding: utf-8 -*-
import torch
from torch.nn.functional import mse_loss 

class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0, lambda1=1.0, lambda2=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # right_discriminator_scores size (batch_size*1)
    def forward(self, right_discriminator_scores, wrong_discriminator_scores, synthetic_discriminator_scores):
        # loss of gennerator 
        # sum(max(0,margin-synthetic_discriminator_scores+wrong_discriminator_scores))+
        # lambda2*(synthetic_discriminator_scores-right_discriminator_scores)^2
        synthetic_wrong_ranking_scores = torch.max(torch.zeros(scores.size()[0]).cuda()
        ,torch.ones(scores.size()[0]).cuda()*self.margin-synthetic_discriminator_scores+wrong_discriminator_scores)
        
        synthetic_right_similarity = mse_loss(right_discriminator_scores,synthetic_discriminator_scores)

        gennerator_loss =  torch.sum(synthetic_discriminator_scores)+self.lambda2*synthetic_right_similarity

        # loss of discriminator
        # sum(max(0,margin-right_discriminator_scores+wrong_discriminator_scores))+
        # lambda1*sum(max(0,margin-right_discriminator_scores+synthetic_discriminator_scores))
        right_wrong_ranking_scores = torch.max(torch.zeros(scores.size()[0]).cuda()
        ,torch.ones(scores.size()[0]).cuda()*self.margin-right_discriminator_scores+wrong_discriminator_scores)

        right_synthetic_ranking_scores = torch.max(torch.zeros(scores.size()[0]).cuda()
        ,torch.ones(scores.size()[0]).cuda()*self.margin-right_discriminator_scores+synthetic_discriminator_scores)

        discriminator_loss = torch.sum(right_wrong_ranking_scores+self.lambda1*right_synthetic_ranking_scores)




        return generator_loss,discriminator_loss