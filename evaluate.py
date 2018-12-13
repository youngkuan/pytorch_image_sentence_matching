import torch
import numpy as np
from utils import Utils
import os



def i2t(discriminator, image_tensors, sentence_embedding_tensors):
    """
       Images->Text (Image Annotation)
       Images: (5N, K=512) matrix of images
       Captions: (5N, K=512) matrix of captions
    """
    # image_number * 512, image_number * 512
    image_projected_feature, sentence_projected_embed \
        = map_image_and_sentence(discriminator, image_tensors, sentence_embedding_tensors)

    size = image_projected_feature.size()[0]
    ranks = np.zeros(size / 5)
    for index in range(size / 5):
        images = image_projected_feature[index * 5].expand(size, 512)
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
    return r1, r5, r10, medr


def t2i(discriminator, image_tensors, sentence_embedding_tensors):
    """
        Text->Images (Image Search)
        Images: (5N, K=512) matrix of images
        Captions: (5N, K=512) matrix of captions
    """

    # image_number * 512, image_number * 512
    image_projected_feature, sentence_projected_embed \
        = map_image_and_sentence(discriminator, image_tensors, sentence_embedding_tensors)

    size = image_projected_feature.size()[0]
    print "size: %d" % size
    ims = torch.cat([image_projected_feature[i].unsqueeze(0) for i in range(0, size, 5)])

    ranks = np.zeros(size)
    for index in range(size):
        # get query sentence and repeat sentence
        query_sentence = sentence_projected_embed[index].expand(size / 5, 512)

        s = Utils.cosine_similarity(ims, query_sentence)

        sort_s, indices = torch.sort(s, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        ranks[index] = np.where(indices == (index / 5))[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return r1, r5, r10, medr


def map_image_and_sentence(discriminator, image_tensors, sentence_embedding_tensors):
    image_projected_features = []
    sentence_projected_embeddings = []
    print "image_tensors.size()[0]:",image_tensors.size()[0]
    step = 5
    with torch.no_grad():
        for i in range(0, image_tensors.size()[0], step):
            [image_projected_feature, sentence_projected_embedding] = discriminator(image_tensors[i:i + step],
                                                                                    sentence_embedding_tensors[i:i + step])
            image_projected_features.append(image_projected_feature)
            sentence_projected_embeddings.append(sentence_projected_embedding)

        image_projected_features = torch.stack(image_projected_features)
        image_projected_features = image_projected_features.view(-1, 512)
        print "image_projected_features.size():",image_projected_features.size()

        sentence_projected_embeddings = torch.stack(sentence_projected_embeddings)
        sentence_projected_embeddings = sentence_projected_embeddings.view(-1, 512)
        print "sentence_projected_embeddings.size():", sentence_projected_embeddings.size()

    # image_number * 512, image_number * 512
    return image_projected_features, sentence_projected_embeddings

def validate_model():
    data_path = "../data/flickr8k"
    image_tensors, sentence_embedding_tensors = Utils.load_validate_data(data_path)

    model_path = "./model/flickr8k/"
    discriminator_model_path = os.path.join(model_path, "disc_4_0.1000_0.30_1.00_1.00.pth")
    discriminator = Utils.load_discriminator(discriminator_model_path)

    i2t_r1, i2t_r5, i2t_r10, i2t_medr = i2t(discriminator, image_tensors, sentence_embedding_tensors)
    t2i_r1, t2i_r5, t2i_r10, t2i_medr = t2i(discriminator, image_tensors, sentence_embedding_tensors)
    print "Image to Text: %.2f, %.2f, %.2f, %.2f" % (i2t_r1, i2t_r5, i2t_r10, i2t_medr)
    print "Text to Image: %.2f, %.2f, %.2f, %.2f" % (t2i_r1, t2i_r5, t2i_r10, t2i_medr)
