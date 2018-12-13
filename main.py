# -*- coding:utf-8 -*-
from trainer import Trainer
import os
from utils import Utils
from evaluate import i2t, t2i
import numpy as np


def train_loop():
    '''
    train the networks
    use loop to select parameters
    :return:
    '''
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    sentence_embedding_file = os.path.join(train_data_path, "train_vectors_.mat")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    model_save_path = "./model"
    dataset_type = "flickr8k"

    # 3
    lrs = [0.01, 0.002, 0.001]
    # 7
    margins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # 5
    lambda1s = [1]
    # 5
    lambda2s = [1]
    results = []
    for lr in lrs:
        for margin in margins:
            for lambda1 in lambda1s:
                for lambda2 in lambda2s:
                    print "hyper parameter %.4f, %.2f, %.2f, %.2f" % (lr, margin, lambda1, lambda2)
                    epochs = 8
                    batch_size = 32
                    trainer = Trainer(sentence_embedding_file, image_ids_file, image_dir, epochs=epochs,
                                      batch_size=batch_size, margin=margin, lr=lr, lambda1=lambda1, lambda2=lambda2)
                    generator, discriminator = trainer.train()

                    data_path = "../data/flickr8k"
                    image_tensors, sentence_embedding_tensors = Utils.load_validate_data(data_path)
                    i2t_r1, i2t_r5, i2t_r10, i2t_medr = i2t(discriminator, image_tensors, sentence_embedding_tensors)
                    t2i_r1, t2i_r5, t2i_r10, t2i_medr = t2i(discriminator, image_tensors, sentence_embedding_tensors)
                    print "lr:%.4f, margin:%.2f, lambda1:%.2f, lambda2:%.2f ; Image to Text: %.2f, %.2f, %.2f, %.2f" \
                          % (lr, margin, lambda1, lambda2, i2t_r1, i2t_r5, i2t_r10, i2t_medr)
                    print "lr:%.4f, margin:%.2f, lambda1:%.2f, lambda2:%.2f ; Text to Image: %.2f, %.2f, %.2f, %.2f" \
                          % (lr, margin, lambda1, lambda2, t2i_r1, t2i_r5, t2i_r10, t2i_medr)
                    result = [lr, margin, lambda1, lambda2, i2t_r1, i2t_r5, i2t_r10, i2t_medr]
                    results.append(result)
                    # save result
                    np_path = './model/flickr8k/result.npy'
                    txt_path = './model/flickr8k/result.txt'

                    if os.path.exists(np_path):
                        os.remove(np_path)
                    if os.path.exists(txt_path):
                        os.remove(txt_path)

                    np.save(np_path, results)
                    np.savetxt(txt_path, results)


def train(lr, margin, lambda1, lambda2):
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    sentence_embedding_file = os.path.join(train_data_path, "train_vectors_.mat")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    model_save_path = "./model"
    dataset_type = "flickr8k"
    epochs = 32
    batch_size = 8

    print "hyper parameter %.4f, %.2f, %.2f, %.2f" % (lr, margin, lambda1, lambda2)
    trainer = Trainer(sentence_embedding_file, image_ids_file, image_dir, epochs=epochs, batch_size=batch_size
                      , margin=margin, lr=lr, lambda1=lambda1, lambda2=lambda2)
    generator, discriminator = trainer.train()

    # save model
    postfix = "%d_%.4f_%.2f_%.2f_%.2f" % (epochs, lr, margin, lambda1, lambda2)
    Utils.save_checkpoint(discriminator, generator, model_save_path, dataset_type, postfix)

    # # load validate data
    # data_path = "../data/flickr8k"
    # val_data_path = os.path.join(data_path, "val")
    # val_sentence_embedding_file = os.path.join(val_data_path, "val_vectors_.mat")
    # val_image_ids_file = os.path.join(val_data_path, "val_image_ids.mat")
    # image_dir = os.path.join(data_path, "images")
    # image_tensors, sentence_embedding_tensors = Utils.load_data(val_image_ids_file, val_sentence_embedding_file,
    #                                                             image_dir)
    #
    # i2t_r1, i2t_r5, i2t_r10, i2t_medr = i2t(discriminator, image_tensors, sentence_embedding_tensors)
    # t2i_r1, t2i_r5, t2i_r10, t2i_medr = t2i(discriminator, image_tensors, sentence_embedding_tensors)
    # print "Image to Text: %.2f, %.2f, %.2f, %.2f" % (i2t_r1, i2t_r5, i2t_r10, i2t_medr)
    # print "Text to Image: %.2f, %.2f, %.2f, %.2f" % (t2i_r1, t2i_r5, t2i_r10, t2i_medr)
    # results = []
    # result = [lr, margin, lambda1, lambda2, i2t_r1, i2t_r5, i2t_r10, i2t_medr]
    # results.append(result)

    # save result
    # np_path = './model/flickr8k/result.npy'
    # txt_path = './model/flickr8k/result.txt'
    # Utils.save_results(results,np_path,txt_path)


if __name__ == '__main__':
    # lr = 0.001
    # margin = 0.3
    # lambda1 = 3.2
    # lambda2 = 1.4
    # train(lr, margin, lambda1, lambda2)
    train_loop()
