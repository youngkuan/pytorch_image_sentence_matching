import torch
import torch.nn as nn
from torchvision import models

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.image_size = 128
        self.num_channels = 3
        self.noise_dim = 500
        self.sentence_embedding_size = 6000
        self.projected_sentence_embedding_size = 512
        self.latent_dim = self.noise_dim + self.projected_sentence_embedding_size
        self.ngf = 64


        self.projector = nn.Sequential(
            nn.Linear(in_features=self.sentence_embedding_size, out_features=self.projected_sentence_embedding_size),
            nn.BatchNorm1d(num_features=self.projected_sentence_embedding_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.generator_net = nn.Sequential(
            # input (self.noise_dim + self.projected_sentence_embedding_size)*(1*1)
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.ngf, self.ngf / 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf / 2),
            nn.Tanh(),

            # state size. (ngf/2) x 128 x 128
            nn.ConvTranspose2d(self.ngf/2, self.num_channels, kernel_size=2, stride=2, padding=16, bias=False),
            nn.Tanh()
            # state size. num_channels x 224 x 224

        )

    def forward(self, sentence_embedding, noise):
        # size: batch_size x projected_sentence_embed_dim
        projected_sentence_embedding = self.projector(sentence_embedding).unsqueeze(2).unsqueeze(3)
        # size: batch_size x projected_sentence_embed_dim x 1 x 1
        noise = noise.unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_sentence_embedding, noise], 1)
        # size: batch_size x (projected_sentence_embed_dim+noise_dim) x 1 x 1
        output = self.generator_net(latent_vector)
        # size: batch_size x (num_channels) x 128 x 128

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.sentence_embedding_size = 6000
        self.vgg16_feature_size = 4096
        self.common_size = 512
        self.pretrained_vgg16_model = models.vgg16(pretrained=True).features

        self.sentence_projector = nn.Sequential(
            nn.Linear(in_features=self.sentence_embedding_size, out_features=self.common_size),
            nn.BatchNorm1d(num_features=self.common_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.image_projector = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.vgg16_feature_size),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(in_features=self.vgg16_feature_size, out_features=self.common_size),
            nn.BatchNorm1d(num_features=self.common_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image, sentence_embedding):
        # batch_size x (num_channels) x image_size x image_size -> batch_size x vgg16_feature_size
        # batch_size x (num_channels) x 224 x 224 -> batch_size x 4096
        image_feature = self.pretrained_vgg16_model(image)
        image_feature = image_feature.view(image_feature.size(0), 512 * 7 * 7)
        image_projected_feature = self.image_projector(image_feature)

        # batch_size x embed_sentence_dim -> batch_size x projected_embed_sentence_dim
        sentence_projected_embedding = self.sentence_projector(sentence_embedding)

        return [image_projected_feature, sentence_projected_embedding]
