# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
from cgan import generator,discriminator
from dataset import load_dataset
from loss import PairwiseRankingLoss
from sample import sample_data

class Trainer(object):
    
    def __init__(self,epochs=15,batch_size=100,margin=0.2,lr=0.0002, save_path):
        self.generator = generator
        self.discriminator = discriminator
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.dataset = load_dataset()

        self.noise_dim = 100
        self.lr = lr
        self.beta1 = 0.5

        # hyper-parameter
        self.lambda1 = 1
        self.lambda2 = 1

        # loss
        self.ranking_loss = PairwiseRankingLoss(margin=margin,self.lambda1,self.lambda2)
        self.ranking_loss = self.ranking_loss.cuda()

        # optmizer
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.save_path = save_path

    
    def train(self,params):
        self.train_cgan(params)
    

    def train_cgan(self,params):
        # train the generator and discriminator

        for epoch in range(self.num_epochs):
            print('epoch:',epoch)


            for samples in self.dataset:

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                
                sentence_embeddings = []
                right_images = []
                wrong_images = []
                for sample in samples:
                    # get the input
                    sentence_embedding = sample["sentence_embedding"]
                    right_image = sample["right_image"]
                    wrong_image = sample["wrong_image"]

                    sentence_embeddings.append(sentence_embedding)
                    right_images.append(right_image)
                    wrong_images.append(wrong_image)


                # get the input tensor
                sentence_embedding_tensors = torch.tensor(sentence_embeddings,requires_grad=False).cuda()
                right_image_tensors = torch.tensor(right_images,requires_grad=False).cuda()
                wrong_image_tensors = torch.tensor(wrong_images,requires_grad=False).cuda()
                noises = torch.randn(sentence_embedding_tensors.size(0), self.noise_dim).cuda()
                #
                synthetic_image_tensors = self.generator(noises,sentence_embedding_tensors)

                right_discriminator_scores = self.discriminator(right_image_tensors,sentence_embedding_tensors)
                wrong_discriminator_scores = self.discriminator(wrong_image_tensors,sentence_embedding_tensors)
                synthetic_discriminator_scores = self.discriminator(synthetic_image_tensors,sentence_embedding_tensors)

                # compute the loss for generator
                # compute the loss for discriminator
                generator_loss,discriminator_loss = self.ranking_loss(right_discriminator_scores,wrong_discriminator_scores,synthetic_discriminator_scores)

                # optmize  discriminator
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # optmize  generator
                generator_loss.backward()
                self.generator_optimizer.step()










if __name__=='__main__':
