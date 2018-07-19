import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import os
import random
import numpy as np

# load_type: train/val
def load_dataset(name='mscoco2014',load_type='train'):
    root = os.path.join('F:/COCO/data/data',name,load_type,'images')
    annFile = os.path.join('./data',name,load_type,'annotations','captions.json')
    dataset = dset.CocoCaptions(root = root,
                        annFile = annFile,
                        transform=transforms.ToTensor())
    dts = transform_dataset(dataset)
    return dts

def transform_dataset(dataset):
    imgs = []
    for img,captions in dataset:
        imgs.append(img)

    datas = []
    for img,captions in dataset:
            for caption in captions:
                sample = {
                'sentence_embedding': caption,
                'right_image': img,
                #sample a wrong image from images
                'wrong_image': random.sample(imgs,1)[0]
                 }
                datas.append(sample)
    return datas

    



if __name__=='__main__':
    # cap = load_dataset()
    # print('Number of samples: ', len(cap))
    # img, target = cap[3] # load 4th sample

    # print("Image Size: ", img.size())
    # print(target)

    imgs = [1,2,3,4,5]
    n = np.array(imgs)
    print(n)
    tensor = torch.tensor(imgs,requires_grad=False)
    print(tensor)