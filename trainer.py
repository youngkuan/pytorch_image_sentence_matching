# -*- coding: utf-8 -*-
import torch
import os
from text_image_dataset import Text2ImageDataset
from gan import Generator, Discriminator
from torch.utils.data import DataLoader
from loss import PairwiseRankingLoss
from utils import Utils
from evaluate import i2t, t2i
from torchvision import transforms


class Trainer(object):

    def __init__(self, sentence_embedding_file, image_ids_file, image_dir, dataset_type="flickr8k"
                 , epochs=15, batch_size=8, margin=0.2, lr=0.01, lambda1=1.0, lambda2=1.0
                 , model_save_path="./model", num_workers=0):

        generator = Generator()
        discriminator = Discriminator()
        self.generator = torch.nn.DataParallel(generator.cuda())
        self.discriminator = torch.nn.DataParallel(discriminator.cuda())

        self.num_epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = 224
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

        self.ToPILImage = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224))
        ])

        self.ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])

        # set requires grad
        for m in self.discriminator.modules():
            if isinstance(m, Discriminator):
                for param in m.pretrained_vgg16_model.parameters():
                    param.requires_grad = False

        # initial parameters
        self.discriminator.apply(Utils.discriminator_weights_init)
        self.generator.apply(Utils.generator_weights_init)

        # define optimizer
        self.generator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=self.lr, betas=(self.beta1, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=self.lr, betas=(self.beta1, 0.999))

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
                print "sentence_embedding_tensors.size()", sentence_embedding_tensors.size()
                right_image_tensors = torch.tensor(right_images, requires_grad=False).cuda()
                print "right_image_tensors.size()", right_image_tensors.size()
                wrong_image_tensors = torch.tensor(wrong_images, requires_grad=False).cuda()
                print "wrong_image_tensors.size()", wrong_image_tensors.size()
                noises = torch.randn(sentence_embedding_tensors.size(0), self.noise_dim).cuda()

                # optmize  generator
                self.generator_optimizer.zero_grad()

                synthetic_image_tensors = self.generator(sentence_embedding_tensors, noises)
                print "synthetic_image_tensors.size()", synthetic_image_tensors.size()
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
        results = []
        for epoch in range(self.num_epochs):
            iteration = 0
            for sample in self.data_loader:
                iteration += 1
                # batch_size*10 x sentence_embedding_size
                sentence_embeddings = sample["sentence_embedding"]
                sentence_embeddings = sentence_embeddings.view(-1, 6000)
                # batch_size*10 x num_channels x (image_size x image_size)
                right_images = sample["right_image"]
                right_images = right_images.view(-1, 3, self.image_size, self.image_size)
                # batch_size*10 x num_channels x (image_size x image_size)
                wrong_images = sample["wrong_image"]
                wrong_images = wrong_images.view(-1, 3, self.image_size, self.image_size)

                # get the input tensor
                sentence_embedding_tensors = sentence_embeddings.cuda()
                right_image_tensors = right_images.cuda()
                wrong_image_tensors = wrong_images.cuda()
                noises = torch.randn(sentence_embedding_tensors.size(0), self.noise_dim).cuda()

                # optimize  generator
                for param in self.generator.parameters():
                    param.requires_grad = True
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                # redefine optimizer
                # self.generator_optimizer = torch.optim.Adam(
                #     filter(lambda p: p.requires_grad, self.generator.parameters()),
                #     lr=self.lr * (0.95 ** epoch),
                #     betas=(self.beta1, 0.999))

                self.generator_optimizer.zero_grad()

                # print "sentence_embedding_tensors.size():", sentence_embedding_tensors.size()
                # print "right_image_tensors.size():", right_image_tensors.size()
                # print "wrong_image_tensors.size():", wrong_image_tensors.size()
                # print "synthetic_image_tensors.size():",synthetic_image_tensors.size()

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

                # optimize  discriminator
                for param in self.generator.parameters():
                    param.requires_grad = False
                for param in self.discriminator.parameters():
                    param.requires_grad = True
                # set requires grad
                for m in self.discriminator.modules():
                    if isinstance(m, Discriminator):
                        for param in m.pretrained_vgg16_model.parameters():
                            param.requires_grad = False
                # redefine optimizer
                # self.discriminator_optimizer = torch.optim.Adam(
                #     filter(lambda p: p.requires_grad, self.discriminator.parameters()),
                #     lr=self.lr * (0.95 ** epoch),
                #     betas=(self.beta1, 0.999))

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

            # compute recall score every 4 epochs
            if ((epoch + 1) % (self.num_epochs + 1)) == 0:
                data_path = "../data/flickr8k"
                image_tensors, sentence_embedding_tensors = Utils.load_validate_data(data_path)

                i2t_r1, i2t_r5, i2t_r10, i2t_medr = i2t(self.discriminator, image_tensors, sentence_embedding_tensors)
                t2i_r1, t2i_r5, t2i_r10, t2i_medr = t2i(self.discriminator, image_tensors, sentence_embedding_tensors)
                result = [epoch, self.lr, self.margin, self.lambda1, self.lambda2, i2t_r1, i2t_r5, i2t_r10, i2t_medr]
                results.append(result)
                # save result
                np_path = './model/flickr8k/result.npy'
                txt_path = './model/flickr8k/result.txt'
                Utils.save_results(results, np_path, txt_path)

                # print recall
                print "Epoch: %d, lr:%.4f, margin:%.2f, lambda1:%.2f, lambda2:%.2f ; Image to Text: %.2f, %.2f, %.2f, %.2f" \
                      % (epoch, self.lr, self.margin, self.lambda1, self.lambda2, i2t_r1, i2t_r5, i2t_r10, i2t_medr)
                print "Epoch: %d, lr:%.4f, margin:%.2f, lambda1:%.2f, lambda2:%.2f ; Text to Image: %.2f, %.2f, %.2f, %.2f" \
                      % (epoch, self.lr, self.margin, self.lambda1, self.lambda2, t2i_r1, t2i_r5, t2i_r10, t2i_medr)

        return self.generator, self.discriminator
