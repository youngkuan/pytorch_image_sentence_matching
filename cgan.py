import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.image_size = 128
		self.num_channels = 3
		self.noise_dim = 500
		self.sentence_embed_dim = 6000
		self.projected_sentence_embed_dim = 512
		self.latent_dim = self.noise_dim + self.projected_sentence_embed_dim
		self.ngf = 64

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.sentence_embed_dim, out_features=self.projected_sentence_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_sentence_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)

		self.netG = nn.Sequential(
			# input (self.noise_dim + self.projected_sentence_embed_dim)*(1*1)
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
			nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (num_channels) x 128 x 128
		)

	def forward(self, embed_vector, z):
		# size: batch_size x projected_sentence_embed_dim
		projected_sentence_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		# size: batch_size x projected_sentence_embed_dim x 1 x 1
		z = z.unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_sentence_embed, z], 1)
		# size: batch_size x (projected_sentence_embed_dim+noise_dim) x 1 x 1
		output = self.netG(latent_vector)
		# size: batch_size x (num_channels) x 128 x 128

		return output


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.image_size = 128
		self.num_channels = 3
		self.sentence_embed_dim = 6000
		self.projected_sentence_embed_dim = 512
		self.ndf = 64

		self.netD_1 = nn.Sequential(
			# input is (nc) x 128 x 128
			nn.Conv2d(self.num_channels, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 64 x 64
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 32 x 32
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 16 x 16
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 8 x 8
			nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 16),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*16) x 4 x 4
		)

		self.sentence_projector = nn.Sequential(
			nn.Linear(in_features=self.sentence_embed_dim, out_features=self.projected_sentence_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_sentence_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)

		self.image_projector = nn.Sequential(
			nn.Linear(in_features=(self.ndf*16)*4*4, out_features=self.projected_sentence_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_sentence_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)

		self.netD_2 = nn.Sequential(
			# state size. (self.ndf*16+ self.projected_sentence_embed_dim) x 4 x 4
			nn.Conv2d(self.ndf * 16 + self.projected_sentence_embed_dim, 1, kernel_size=4, stride=1, padding=0, bias=False),
			# (1) x 1 x 1
			nn.Sigmoid()
			)
	'''
		def forward_gan(self, image, sentence_embed):
		# batch_size x (num_channels) x image_size x image_size -> batch_size x (ndf*16) x 4 x 4
		image_feature = self.netD_1(image)

		# batch_size x embed_sentence_dim -> batch_size x projected_embed_sentence_dim
		sentence_projected_embed = self.projector(sentence_embed)
		sentence_projected_embed = sentence_projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)

		# batch_size x projected_embed_sentence_dim x (4 x 4) ; batch_size x (ndf*16) x 4 x 4
		hidden_concat = torch.cat([image_feature, sentence_projected_embed], 1)
		# batch_size x ((ndf*16)+projected_embed_sentence_dim) x (4 x 4)
		x = self.netD_2(hidden_concat)
		# batch_size x 1 x 1 x 1 -> batch_size x 1

		return x.view(-1, 1).squeeze(1)
	'''

	def forward(self, image, sentence_embed):
		# batch_size x (num_channels) x image_size x image_size -> batch_size x (ndf*16) x 4 x 4
		image_feature = self.netD_1(image)
		image_feature = image_feature.view(-1, (self.ndf * 16) * 4 * 4)
		image_projected_feature = self.image_projector(image_feature)

		# batch_size x embed_sentence_dim -> batch_size x projected_embed_sentence_dim
		sentence_projected_embed = self.sentence_projector(sentence_embed)

		# batch_size x sentence_projected_embed
		# similarities = Utils.cosine_similarity(image_projected_feature, sentence_projected_embed)

		# return similarities
		return [image_projected_feature, sentence_projected_embed]
