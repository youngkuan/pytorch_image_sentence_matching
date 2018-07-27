# -*- coding: utf-8 -*-
import torch
from torch.nn.functional import mse_loss
from utils import Utils


class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=0.2, lambda1=1.0, lambda2=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # right_discriminator_scores size (batch_size*1)
    def forward(self, right_discriminator_features, wrong_discriminator_features, synthetic_discriminator_features):
        batch_size = right_discriminator_features[0].size()[0]
        # loss of gennerator 
        # sum(max(0,margin-synthetic_discriminator_scores+wrong_discriminator_scores))+
        # lambda2*(synthetic_discriminator_scores-right_discriminator_scores)^2

        right_image_features = right_discriminator_features[0]
        right_sentence_features = right_discriminator_features[1]

        wrong_image_features = wrong_discriminator_features[0]
        wrong_sentence_features = wrong_discriminator_features[1]

        synthetic_image_features = synthetic_discriminator_features[0]
        synthetic_sentence_features = synthetic_discriminator_features[1]

        right_discriminator_scores = Utils.cosine_similarity(right_image_features, right_sentence_features)
        wrong_discriminator_scores = Utils.cosine_similarity(wrong_image_features, wrong_sentence_features)
        synthetic_discriminator_scores = Utils.cosine_similarity(synthetic_image_features, synthetic_sentence_features)

        synthetic_wrong_ranking_scores = torch.max(torch.zeros(batch_size).cuda()
                                                   , torch.ones(batch_size).cuda()
                                                   * self.margin-synthetic_discriminator_scores
                                                   + wrong_discriminator_scores)

        synthetic_right_distance = Utils.distance(right_image_features, synthetic_image_features)

        generator_loss = torch.sum(synthetic_wrong_ranking_scores+self.lambda2 * synthetic_right_distance)

        # loss of discriminator
        # sum(max(0,margin-right_discriminator_scores+wrong_discriminator_scores))+
        # lambda1*sum(max(0,margin-right_discriminator_scores+synthetic_discriminator_scores))
        right_wrong_ranking_scores = torch.max(torch.zeros(batch_size).cuda()
                                               , torch.ones(batch_size).cuda()
                                               * self.margin - right_discriminator_scores
                                               + wrong_discriminator_scores)

        right_synthetic_ranking_scores = torch.max(torch.zeros(batch_size).cuda()
                                                   , torch.ones(batch_size).cuda()
                                                   * self.margin - right_discriminator_scores
                                                   + synthetic_discriminator_scores)

        discriminator_loss = torch.sum(right_wrong_ranking_scores + self.lambda1 * right_synthetic_ranking_scores)

        return generator_loss, discriminator_loss

        # right_discriminator_scores size (batch_size*1)

    def forward_gan(self, right_discriminator_scores, wrong_discriminator_scores, synthetic_discriminator_scores):
        batch_size = synthetic_discriminator_scores.size()[0]
        # loss of gennerator
        # sum(max(0,margin-synthetic_discriminator_scores+wrong_discriminator_scores))+
        # lambda2*(synthetic_discriminator_scores-right_discriminator_scores)^2
        synthetic_wrong_ranking_scores = torch.max(torch.zeros(batch_size).cuda()
                                                   , torch.ones(batch_size).cuda()
                                                   * self.margin - synthetic_discriminator_scores
                                                   + wrong_discriminator_scores)

        synthetic_right_similarity = mse_loss(right_discriminator_scores, synthetic_discriminator_scores)

        generator_loss = torch.sum(synthetic_wrong_ranking_scores) + self.lambda2 * synthetic_right_similarity

        # loss of discriminator
        # sum(max(0,margin-right_discriminator_scores+wrong_discriminator_scores))+
        # lambda1*sum(max(0,margin-right_discriminator_scores+synthetic_discriminator_scores))
        right_wrong_ranking_scores = torch.max(torch.zeros(batch_size).cuda()
                                               , torch.ones(batch_size).cuda()
                                               * self.margin - right_discriminator_scores
                                               + wrong_discriminator_scores)

        right_synthetic_ranking_scores = torch.max(torch.zeros(batch_size).cuda()
                                                   , torch.ones(batch_size).cuda()
                                                   * self.margin - right_discriminator_scores
                                                   + synthetic_discriminator_scores)

        discriminator_loss = torch.sum(right_wrong_ranking_scores + self.lambda1 * right_synthetic_ranking_scores)

        return generator_loss, discriminator_loss

