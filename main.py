from trainer import Trainer
import os
from utils import Utils
from evaluate import evaluate_model
import numpy as np


if __name__ == '__main__':

    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    sentence_embedding_file = os.path.join(train_data_path, "train_vectors_.mat")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    model_save_path = "./model"
    dataset_type = "flickr8k"
    epochs = 15

    # 7
    lrs = [0.1, 0.01, 0.02, 0.05, 0.001, 0.002, 0.005]
    # 6
    lambda1s = [5, 2, 1, 0.5, 0.2, 0.1]
    # 6
    lambda2s = [5, 2, 1, 0.5, 0.2, 0.1]
    # 9
    margins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_generator = None
    best_discriminator = None
    best_r1, best_r5, best_r10, best_medr = 0, 0, 0, 0
    best_lr, best_lambda1, best_lambda2, best_margin = 0, 0, 0, 0
    results = []
    for margin in margins:
        for lr in lrs:
            for lambda1 in lambda1s:
                for lambda2 in lambda2s:
                    trainer = Trainer(sentence_embedding_file, image_ids_file, image_dir
                                      , margin=margin, lr=lr, lambda1=lambda1, lambda2=lambda2)
                    generator, discriminator = trainer.train()
                    (r1, r5, r10, medr) = evaluate_model(discriminator)
                    print "Image to Text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
                    if (r1 >= best_r1) and (r5 >= best_r5) and (r10 >= best_r10):
                        best_r1, best_r5,  best_r10, best_medr = r1, r5, r10, medr
                        best_lr, best_lambda1, best_lambda2, best_margin = lr, lambda1, lambda2, margin
                        best_generator = generator
                        best_discriminator = discriminator
                    result = [lr, lambda1, lambda2, margin, r1, r5, r10, medr]
                    results.append(result)
    postfix = "best_result_%1.1f_%1.1f_%1.1f_%1.1f" % (best_lr, best_lambda1, best_lambda2, best_margin)
    print "best result-- lr:%.1f, lambda1:%.1f, lambda2:%.1f, margin:%.1f ; Image to Text: %.1f, %.1f, %.1f, %.1f" \
                  % (lr, lambda1, lambda2, margin, r1, r5, r10, medr)
    Utils.save_checkpoint(best_discriminator, best_generator
                          , model_save_path, dataset_type, postfix)
    np.save('./model/flickr8k/result.npy', results)


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








