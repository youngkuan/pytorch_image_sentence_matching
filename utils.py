import torch
import os
import torch.nn.functional as F
import h5py
from PIL import Image
import numpy as np

class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset


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

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, postfix):
        path = os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, postfix))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, postfix))

    @staticmethod
    def load_data(image_ids_file, sentence_embedding_file, image_dir):
        # load sentence
        sentence_embeddings = h5py.File(sentence_embedding_file, 'r')
        #
        sentence_embeddings = np.asarray(sentence_embeddings['val_vectors_'])

        # load image ids
        # image ids (n * 1)
        image_ids = []
        image_ids_h5py = h5py.File(image_ids_file, 'r')
        hdf5_objects = image_ids_h5py['val_image_ids']
        length = hdf5_objects.shape[1]
        for i in range(length):
            image_ids.append(''.join([chr(v[0]) for v in image_ids_h5py[hdf5_objects[0][i]].value]))

        images = []
        for i in range(length):
            image_path = os.path.join(image_dir, image_ids[i])
            image = Image.open(image_path).resize((128, 128))
            image = np.array(image, dtype=float)
            image = image.transpose(2, 0, 1)
            images.append(image)
        images = np.array(images, dtype=float)

        return torch.FloatTensor(images), torch.FloatTensor(sentence_embeddings)










