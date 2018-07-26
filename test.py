from main import load_dataset


def main():
    # cap = dset.CocoCaptions(root='F:\\COCO\\data\\mscoco2014\\train\\images',
    #                         annFile='F:/COCO/data/mscoco2014/train/annotations/captions_train2014.json',
    #                         transform=transforms.ToTensor())

    dset = load_dataset()
    print(dset.size())
    print(dset[0])


if __name__ == '__main__':
    main()

