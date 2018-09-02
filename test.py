from utils import Utils
import os
import torch
from utils import Utils
from evaluate import evaluate_model,load_discriminator
import numpy as np


def main():
    model_path = "./model/flickr8k/"
    # generator_model_path = os.path.join(model_path, "gen_14.pth")
    # discriminator_model_path = os.path.join(model_path, "disc_14.pth")
    # discriminator = load_discriminator(discriminator_model_path)
    # (r1, r5, r10, medr) = evaluate_model(discriminator)
    # print "Image to Text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)

    # result = np.load(model_path+'result.npy')
    print(torch.randn(10, 1))



if __name__ == '__main__':
    main()

