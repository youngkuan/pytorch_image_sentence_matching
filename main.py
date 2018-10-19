# -*- coding:utf-8 -*-
from trainer import Trainer
import os
from utils import Utils
from evaluate import i2t
import numpy as np

def train2select_parameters():
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
    epochs = 15

    # 3
    lrs = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
    # 7
    margins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # 5
    lambda1s = [2.5, 2.8, 3, 3.2, 3.5]
    # 5
    lambda2s = [1, 1.2, 1.3, 1.4, 1.5]
    best_generator = None
    best_discriminator = None
    best_r1, best_r5, best_r10, best_medr = 0, 0, 0, 0
    best_lr, best_lambda1, best_lambda2, best_margin = 0, 0, 0, 0
    results = []
    flag = True
    for lr in lrs:
        for margin in margins:
            for lambda1 in lambda1s:
                for lambda2 in lambda2s:
                    print "hyper parameter %.4f, %.2f, %.2f, %.2f" % (lr, margin, lambda1, lambda2)
                    # if (lr != 0.1 or margin != 0.2 or lambda1 != 1 or lambda2 != 2) and flag is True:
                    #     continue
                    # elif lr == 0.1 and margin == 0.2 and lambda1 == 1 and lambda2 == 2:
                    #     flag = False
                    #     continue
                    trainer = Trainer(sentence_embedding_file, image_ids_file, image_dir
                                      , margin=margin, lr=lr, lambda1=lambda1, lambda2=lambda2)
                    generator, discriminator = trainer.train()
                    (r1, r5, r10, medr) = i2t(discriminator)
                    print "Image to Text: %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr)
                    if (r1 >= best_r1) and (r5 >= best_r5) and (r10 >= best_r10):
                        best_r1, best_r5,  best_r10, best_medr = r1, r5, r10, medr
                        best_lr, best_lambda1, best_lambda2, best_margin = lr, lambda1, lambda2, margin
                        best_generator = generator
                        best_discriminator = discriminator
                        postfix = "%1.4f_%1.2f_%1.2f_%1.2f" \
                                  % (best_lr, best_margin, best_lambda1, best_lambda2)
                        Utils.save_checkpoint(best_discriminator, best_generator
                                              , model_save_path, dataset_type, postfix)
                    result = [lr, margin, lambda1, lambda2, r1, r5, r10, medr]
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
    postfix = "best_result_%1.4f_%1.2f_%1.2f_%1.2f" % (best_lr, best_margin, best_lambda1, best_lambda2)
    print "best result-- lr:%.4f, margin:%.2f, lambda1:%.2f, lambda2:%.2f ; Image to Text: %.2f, %.2f, %.2f, %.2f" \
                  % (best_lr, best_margin, best_lambda1, best_lambda2, best_r1, best_r5,  best_r10, best_medr)
    if best_discriminator is not None:
        Utils.save_checkpoint(best_discriminator, best_generator
                              , model_save_path, dataset_type, postfix)


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


def train(lr, margin, lambda1, lambda2):
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    sentence_embedding_file = os.path.join(train_data_path, "train_vectors_.mat")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    model_save_path = "./model"
    dataset_type = "flickr8k"
    epochs = 64

    print "hyper parameter %.4f, %.2f, %.2f, %.2f" % (lr, margin, lambda1, lambda2)
    trainer = Trainer(sentence_embedding_file, image_ids_file, image_dir,epochs=epochs
                      , margin=margin, lr=lr, lambda1=lambda1, lambda2=lambda2)
    generator, discriminator = trainer.train()
    (r1, r5, r10, medr) = i2t(discriminator)
    print "Image to Text: %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr)
    results = []
    result = [lr, margin, lambda1, lambda2, r1, r5, r10, medr]
    results.append(result)
    # save result
    np_path = './model/flickr8k/result.npy'
    txt_path = './model/flickr8k/result.txt'
    # Utils.save_results(results,np_path,txt_path)


if __name__ == '__main__':
    lr = 0.001
    margin = 0.3
    lambda1 = 3
    lambda2 = 1.4
    train(lr, margin, lambda1, lambda2)




