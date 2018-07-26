from torch import  autograd
import torch
import os


class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod

    # based on:  https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def compute_GP(netD, real_data, real_embed, fake_data, LAMBDA):
        BATCH_SIZE = real_data.size(0)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates, real_embed)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

