import torch
from torch.nn.functional import mse_loss,pairwise_distance
import torch.nn as nn


def main():
    # cap = dset.CocoCaptions(root='F:\\COCO\\data\\mscoco2014\\train\\images',
    #                         annFile='F:/COCO/data/mscoco2014/train/annotations/captions_train2014.json',
    #                         transform=transforms.ToTensor())
    bn2D = nn.BatchNorm2d(6)

    x1 = torch.ones(5, 6)
    x2 = torch.zeros(5, 6)
    x1 = pairwise_distance(x1,x2)
    print(x1)


if __name__ == '__main__':
    main()

