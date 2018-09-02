# -*- coding: utf-8 -*-
import torch
from text_image_dataset import Text2ImageDataset
from cgan import Generator, Discriminator
from torch.utils.data import DataLoader
from loss import PairwiseRankingLoss
from utils import Utils
from evaluate import evaluate_model


class Trainer(object):
    
    def __init__(self, sentence_embedding_file, image_ids_file, image_dir, dataset_type="flickr8k"
                 , epochs=15, batch_size=128, margin=0.2, lr=0.01, lambda1=1.0, lambda2=1.0
                 , model_save_path="./model", num_workers=0):

        generator = Generator()
        discriminator = Discriminator()
        self.generator = torch.nn.DataParallel(generator.cuda())
        self.discriminator = torch.nn.DataParallel(discriminator.cuda())

        self.num_epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        self.dataset = Text2ImageDataset(sentence_embedding_file, image_ids_file, image_dir, dataset_type)

        self.noise_dim = 500
        self.lr = lr
        self.beta1 = 0.5

        # hyper-parameter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.margin = margin

        # data loader
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)

        # loss
        self.ranking_loss = PairwiseRankingLoss(margin=margin, lambda1=self.lambda1, lambda2=self.lambda2)
        self.ranking_loss = self.ranking_loss.cuda()

        # initial parameters
        # self.discriminator.apply(Utils.weights_init)
        # self.generator.apply(Utils.weights_init)

        # optmizer
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.model_save_path = model_save_path

    def train(self):
        return self.train_image_text_gan()

    def train_cgan(self):
        # train the generator and discriminator

        for epoch in range(self.num_epochs):
            iteration = 0
            for sample in self.data_loader:
                iteration += 1
                # batch_size x sentence_embedding_size
                sentence_embeddings = sample["sentence_embedding"]
                # batch_size x num_channels x (image_size x image_size)
                right_images = sample["right_image"]
                # batch_size x num_channels x (image_size x image_size)
                wrong_images = sample["wrong_image"]

                # get the input tensor
                sentence_embedding_tensors = torch.tensor(sentence_embeddings, requires_grad=False).cuda()
                right_image_tensors = torch.tensor(right_images, requires_grad=False).cuda()
                wrong_image_tensors = torch.tensor(wrong_images, requires_grad=False).cuda()
                noises = torch.randn(sentence_embedding_tensors.size(0), self.noise_dim).cuda()

                # optmize  generator
                self.generator_optimizer.zero_grad()

                synthetic_image_tensors = self.generator(sentence_embedding_tensors, noises)

                right_discriminator_scores = self.discriminator(right_image_tensors, sentence_embedding_tensors)
                wrong_discriminator_scores = self.discriminator(wrong_image_tensors, sentence_embedding_tensors)
                synthetic_discriminator_scores = self.discriminator(synthetic_image_tensors, sentence_embedding_tensors)

                # compute the loss for generator
                # compute the loss for discriminator
                generator_loss, _ = self.ranking_loss(right_discriminator_scores
                                                                       , wrong_discriminator_scores
                                                                       , synthetic_discriminator_scores)

                generator_loss.backward()
                self.generator_optimizer.step()

                # optmize  discriminator
                self.discriminator_optimizer.zero_grad()
                synthetic_image_tensors = self.generator(sentence_embedding_tensors, noises)
                right_discriminator_scores = self.discriminator(right_image_tensors, sentence_embedding_tensors)
                wrong_discriminator_scores = self.discriminator(wrong_image_tensors, sentence_embedding_tensors)
                synthetic_discriminator_scores = self.discriminator(synthetic_image_tensors, sentence_embedding_tensors)

                _, discriminator_loss = self.ranking_loss(right_discriminator_scores
                                                      , wrong_discriminator_scores
                                                      , synthetic_discriminator_scores)

                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                print("Epoch: %d, iteration: %d, generator_loss= %f, discriminator_loss= %f" %
                      (epoch, iteration, generator_loss.data, discriminator_loss.data))

    def train_image_text_gan(self):
        # train the generator and discriminator
        for epoch in range(self.num_epochs):
            iteration = 0
            for sample in self.data_loader:
                iteration += 1
                # batch_size x sentence_embedding_size
                sentence_embeddings = sample["sentence_embedding"]
                # batch_size x num_channels x (image_size x image_size)
                right_images = sample["right_image"]
                # batch_size x num_channels x (image_size x image_size)
                wrong_images = sample["wrong_image"]

                # get the input tensor
                sentence_embedding_tensors = sentence_embeddings.cuda()
                right_image_tensors = right_images.cuda()
                wrong_image_tensors = wrong_images.cuda()
                noises = torch.randn(sentence_embedding_tensors.size(0), self.noise_dim).cuda()

                # optmize  generator
                self.generator_optimizer.zero_grad()

                synthetic_image_tensors = self.generator(sentence_embedding_tensors, noises)

                right_discriminator_scores = self.discriminator(right_image_tensors, sentence_embedding_tensors)
                wrong_discriminator_scores = self.discriminator(wrong_image_tensors, sentence_embedding_tensors)
                synthetic_discriminator_scores = self.discriminator(synthetic_image_tensors,
                                                                    sentence_embedding_tensors)

                # compute the loss for generator
                # compute the loss for discriminator
                generator_loss, _ = self.ranking_loss(right_discriminator_scores
                                                      , wrong_discriminator_scores
                                                      , synthetic_discriminator_scores)

                generator_loss.backward()
                self.generator_optimizer.step()

                # optmize  discriminator
                self.discriminator_optimizer.zero_grad()
                synthetic_image_tensors = self.generator(sentence_embedding_tensors, noises)
                right_discriminator_scores = self.discriminator(right_image_tensors, sentence_embedding_tensors)
                wrong_discriminator_scores = self.discriminator(wrong_image_tensors, sentence_embedding_tensors)
                synthetic_discriminator_scores = self.discriminator(synthetic_image_tensors,
                                                                    sentence_embedding_tensors)

                _, discriminator_loss = self.ranking_loss(right_discriminator_scores
                                                          , wrong_discriminator_scores
                                                          , synthetic_discriminator_scores)

                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                print("Epoch: %d, iteration: %d, generator_loss= %f, discriminator_loss= %f" %
                      (epoch, iteration, generator_loss.data, discriminator_loss.data))
            (r1, r5, r10, medr) = evaluate_model(self.discriminator)
            print "Epoch: %d, lr:%.4f, margin:%.2f, lambda1:%.2f, lambda2:%.2f ; Image to Text: %.2f, %.2f, %.2f, %.2f" \
                  % (epoch, self.lr, self.margin, self.lambda1, self.lambda2, r1, r5, r10, medr)

            # Utils.save_checkpoint(self.discriminator, self.generator
            #                       , self.model_save_path, self.dataset_type, postfix)
        return self.generator, self.discriminator


