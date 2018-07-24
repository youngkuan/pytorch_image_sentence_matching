import os
import text_image_dataset
from PIL import Image
import io
from torch.utils.data import DataLoader
import h5py
import torch
import numpy as np

flickr8k_root_path = "../data/flickr8k"
flickr30k_root_dir = "../data/flickr30k"
mscoco_root_dir = "../data/mscoco"

if __name__ == '__main__':
    sentence_embedding_file = os.path.join(flickr8k_root_path, "flickr8k_hglmm_30_ica_sent_vecs_pca_6000.mat")
    image_ids_file = os.path.join(flickr8k_root_path, "flickr8k_image_ids.mat")
    image_dir = os.path.join(flickr8k_root_path, "images")
    # dataset = text_image_dataset.Text2ImageDataset(sentence_embedding_file, image_ids_file, image_dir, "flickr8k")
    # data_loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    # cnt = 0
    # for sample in data_loader:
    #     print('************', cnt, '**************')
    #     # ('sentence_embedding:', (5, 6000))
    #     sentence_embedding = sample['sentence_embedding']
    #     print('sentence_embedding:', sentence_embedding.size())
    #     # ('right_images:', (5, 64, 64, 3))
    #     right_images = sample['right_image']
    #     print('right_images:', right_images.size())
    #     cnt += 1
    img = Image.open(os.path.join(image_dir, '667626_18933d713e.jpg'))
    img = img.resize((128, 128))
    print(img.size)
    img.save('tmp.jpg', 'jpeg')









