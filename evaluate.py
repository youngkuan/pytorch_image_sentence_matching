from cgan import Generator, Discriminator
import torch
import os
from utils import Utils
import numpy as np


def load_generator(generator_model_path):
    # load generator model
    generator = Generator()
    generator = torch.nn.DataParallel(generator.cuda())
    generator.load_state_dict(torch.load(generator_model_path))

    return generator


def load_discriminator(discriminator_model_path):
    # load discriminator model
    discriminator = Discriminator()
    discriminator = torch.nn.DataParallel(discriminator.cuda())
    discriminator.load_state_dict(torch.load(discriminator_model_path))
    return discriminator


def i2t(discriminator):
    """
       Images->Text (Image Annotation)
       Images: (5N, K=512) matrix of images
       Captions: (5N, K=512) matrix of captions
    """
    data_path = "../data/flickr8k"
    val_data_path = os.path.join(data_path, "val")
    val_sentence_embedding_file = os.path.join(val_data_path, "val_vectors_.mat")
    val_image_ids_file = os.path.join(val_data_path, "val_image_ids.mat")
    image_dir = os.path.join(data_path, "images")

    image_tensors, sentence_embedding_tensors = Utils.load_data(val_image_ids_file, val_sentence_embedding_file,
                                                                image_dir)
    # image_number * 512, image_number * 512
    [image_projected_feature, sentence_projected_embed] = discriminator(image_tensors[0:1000]
                                                                        , sentence_embedding_tensors[0:1000])

    size = image_projected_feature.size()[0]
    ranks = np.zeros(size/5)
    for index in range(size/5):
        images = image_projected_feature[index*5].expand(size, 512)
        s = Utils.cosine_similarity(images, sentence_projected_embed)

        sort_s, indices = torch.sort(s, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(indices == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i(discriminator):
    """
        Text->Images (Image Search)
        Images: (5N, K=512) matrix of images
        Captions: (5N, K=512) matrix of captions
    """
    data_path = "../data/flickr8k"
    val_data_path = os.path.join(data_path, "val")
    val_sentence_embedding_file = os.path.join(val_data_path, "val_vectors_.mat")
    val_image_ids_file = os.path.join(val_data_path, "val_image_ids.mat")
    image_dir = os.path.join(data_path, "images")

    image_tensors, sentence_embedding_tensors = Utils.load_data(val_image_ids_file, val_sentence_embedding_file,
                                                                image_dir)
    # image_number * 512, image_number * 512
    [image_projected_feature, sentence_projected_embed] = discriminator(image_tensors[0:1000]
                                                                        , sentence_embedding_tensors[0:1000])

    size = image_projected_feature.size()[0]
    ims = torch.cat([image_projected_feature[i].unsqueeze(0) for i in range(0, size, 5)])

    ranks = np.zeros(size)
    for index in range(size):

        # get query sentence and repeat sentence
        query_sentence = sentence_projected_embed[index].expand(size/5, 512)

        s = Utils.cosine_similarity(ims, query_sentence)

        sort_s, indices = torch.sort(s, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        ranks[index] = np.where(indices == (index/5))[0][0]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)








