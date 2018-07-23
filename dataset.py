import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import os
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import numpy as np


# load_type: train/val
def load_dataset(name='mscoco2014', load_type='train'):
    root = os.path.join('../data', name, load_type, 'images')
    annFile = os.path.join('../data', name, load_type, 'annotations', 'captions.json')
    dataset = dset.CocoCaptions(root=root,
                        annFile=annFile,
                        transform=transforms.ToTensor())
    dts = transform_dataset(dataset)
    return dts


def transform_dataset(dataset):
    imgs = []
    for img, captions in dataset:
        imgs.append(img)

    datas = []
    for img, captions in dataset:
            for caption in captions:
                sample = {
                'sentence_embedding': caption,
                'right_image': img,
                #sample a wrong image from images
                'wrong_image': random.sample(imgs,1)[0]
                 }
                datas.append(sample)
    return datas


if __name__ == '__main__':
    # cap = load_dataset()
    # print('Number of samples: ', len(cap))
    # img, target = cap[3] # load 4th sample

    # print("Image Size: ", img.size())
    # print(target)
    image_ids = []
    data_root = "../data"
    flickr8k_image_ids_path = os.path.join(data_root, "flickr8k/flickr8k_hglmm_30_ica_sent_vecs_pca_6000.mat")
    sentence_embeddings = h5py.File(flickr8k_image_ids_path, 'r')
    # load the sentence embeddings ;size: n * 6000
    print(sentence_embeddings['vectors_'])
    sentence_embeddings = np.asarray(sentence_embeddings['vectors_'])
    # print(sentence_embeddings['vectors_'])
    print(type(sentence_embeddings))
    print(sentence_embeddings[0])


