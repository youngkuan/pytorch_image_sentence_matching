from torch import  autograd
import torch
import os
import torch.nn.functional as F


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

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        r"""Returns cosine similarity between x1 and x2, computed along dim.

        .. math ::
            \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

        Args:
            x1 (Tensor): First input.
            x2 (Tensor): Second input (of size matching x1).
            dim (int, optional): Dimension of vectors. Default: 1
            eps (float, optional): Small value to avoid division by zero.
                Default: 1e-8

        Shape:
            - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
            - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

        Example::
            >>> input1 = torch.randn(100, 128)
            >>> input2 = torch.randn(100, 128)
            >>> output = F.cosine_similarity(input1, input2)
            >>> print(output)
        """
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return w12 / (w1 * w2).clamp(min=eps)

    @staticmethod
    def distance(w1, w2):
        p_w1 = F.pairwise_distance(w1, torch.zeros(w1.size()).cuda())
        p_w2 = F.pairwise_distance(w2, torch.zeros(w2.size()).cuda())
        p_w1_w2 = F.pairwise_distance(w1, w2)
        return p_w1_w2/(p_w1*p_w2)



