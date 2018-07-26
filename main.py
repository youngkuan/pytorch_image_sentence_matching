import torch
from trainer import Trainer
from text_image_dataset import Text2ImageDataset
import os
from torch.utils.data import DataLoader


if __name__ == '__main__':

    data_path = "../data/flickr8k/"
    sentence_embedding_file = os.path.join(data_path, "flickr8k_hglmm_30_ica_sent_vecs_pca_6000.mat")
    image_ids_file = os.path.join(data_path, "flickr8k_image_ids.mat")
    image_dir = os.path.join(data_path,"images")

    trainer = Trainer(sentence_embedding_file, image_ids_file, image_dir)
    trainer.train()

    # dataset = Text2ImageDataset(sentence_embedding_file, image_ids_file, image_dir)
    # data_loader = DataLoader(dataset, batch_size=5, shuffle=True,
    #                          num_workers=0)
    # for samples in data_loader:
    #     sentence_embedding = samples["sentence_embedding"]
    #     print(sentence_embedding.size())
    #     right_image = samples["right_image"]
    #     print(right_image.size())
    #     wrong_image = samples["wrong_image"]
    #     print(wrong_image.size())








