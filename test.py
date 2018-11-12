import os
from utils import Utils
from evaluate import load_discriminator,t2i
from text_image_dataset import Text2ImageDataset
from torch.utils.data import DataLoader
import torch


def main():


    # load model
    model_path = "./model/flickr8k/"
    generator_model_path = os.path.join(model_path, "gen_14.pth")
    discriminator_model_path = os.path.join(model_path, "disc_0.0020_0.50_3.00_1.00.pth")
    discriminator = load_discriminator(discriminator_model_path)

    # load validate data
    data_path = "../data/flickr8k"
    val_data_path = os.path.join(data_path, "val")
    val_sentence_embedding_file = os.path.join(val_data_path, "val_vectors_.mat")
    val_image_ids_file = os.path.join(val_data_path, "val_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    image_tensors, sentence_embedding_tensors = Utils.load_data(val_image_ids_file, val_sentence_embedding_file,
                                                                image_dir)

    r1, r5, r10, medr = t2i(discriminator,image_tensors,sentence_embedding_tensors)
    print "Text to Image: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)


def data_set_test():
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    sentence_embedding_file = os.path.join(train_data_path, "train_vectors_.mat")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    dataset_type = "flickr8k"
    dataset = Text2ImageDataset(sentence_embedding_file, image_ids_file, image_dir, dataset_type)
    data_loader = DataLoader(dataset, 32, shuffle=True,
                             num_workers=0)
    for sample in data_loader:
        # samples size 10

        right_image = sample["right_image"]
        right_image = right_image.view(-1,3,128,128)
        print right_image.size()
        wrong_image = sample["wrong_image"]
        wrong_image = wrong_image.view(-1, 3, 128, 128)
        print wrong_image.size()
        sentence_embedding = sample["sentence_embedding"]
        sentence_embedding = sentence_embedding.view(-1, 6000)
        print sentence_embedding.size()


if __name__ == '__main__':
    # main()
    # data_set_test()
    print 0.95**2