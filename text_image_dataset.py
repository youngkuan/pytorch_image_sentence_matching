from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import h5py
import random
import os


class Text2ImageDataset(Dataset):

    def __init__(self, sentence_embedding_file, image_ids_file, image_dir, dataset_type="flickr8k"):
        self.sentence_embedding_file = sentence_embedding_file
        self.image_ids_file = image_ids_file
        self.image_dir = image_dir
        self.sentence_embeddings = None
        self.image_ids = None
        self.images = None
        self.dataset_type = dataset_type

    def __len__(self):
        sentence_embeddings = h5py.File(self.sentence_embedding_file, 'r')
        # load the sentence embeddings ;size: n * 6000
        self.sentence_embeddings = sentence_embeddings['train_vectors_']
        length = self.sentence_embeddings.shape[0]
        print(length)
        return length

    def __getitem__(self, idx):
        if self.sentence_embeddings is None:
            sentence_embeddings = h5py.File(self.sentence_embedding_file, 'r')
            # load the sentence embeddings ;size: n * 6000
            self.sentence_embeddings = np.asarray(sentence_embeddings['train_vectors_'])

        if self.image_ids is None:
            image_ids = []
            image_ids_h5py = h5py.File(self.image_ids_file, 'r')
            hdf5_objects = image_ids_h5py['train_image_ids']
            length = hdf5_objects.shape[1]
            for i in range(length):
                image_ids.append(''.join([chr(v[0]) for v in image_ids_h5py[hdf5_objects[0][i]].value]))
            # image ids (n * 1)
            self.image_ids = image_ids

        # find all images for train
        # self.find_all_images()

        sentence_embedding = self.sentence_embeddings[idx]
        right_images = self.find_right_image(idx)
        wrong_images = self.find_wrong_image(idx)
        # print "right_images: %d" % len(right_images)
        # print "wrong_images: %d" % len(wrong_images)

        right_images_all = []
        wrong_images_all = []
        sentence_embeddings_all = []
        for index in range(len(right_images)):
            # right_image = np.array(right_images[index], dtype=float)
            right_image = np.array(right_images[index])
            # print "right_image.shape:",right_image.shape
            # print "np.size(right_image):",np.size(right_image)
            right_image = right_image.transpose(2, 0, 1)

            # wrong_image = np.array(wrong_images[index], dtype=float)
            wrong_image = np.array(wrong_images[index])
            # print "wrong_image.shape:",wrong_image.shape
            # print "np.size(wrong_image):",np.size(wrong_image)
            wrong_image = wrong_image.transpose(2, 0, 1)

            right_image = self.validate_image(right_image)
            wrong_image = self.validate_image(wrong_image)

            right_image = torch.FloatTensor(right_image).sub_(127.5).div_(127.5)
            wrong_image = torch.FloatTensor(wrong_image).sub_(127.5).div_(127.5)


            right_images_all.append(right_image)
            wrong_images_all.append(wrong_image)
            sentence_embeddings_all.append(torch.FloatTensor(sentence_embedding))


        sample = {
            'sentence_embedding': torch.stack(sentence_embeddings_all,0),
            'right_image': torch.stack(right_images_all,0),
            # sample a wrong image from images
            'wrong_image': torch.stack(wrong_images_all,0),
        }
        return sample

    def find_wrong_image(self, idx):
        new_idx = random.randint(0, self.sentence_embeddings.shape[0]-1)
        while abs(new_idx - idx) < 10:
            new_idx = random.randint(0, self.sentence_embeddings.shape[0]-1)

        return self.find_image(self.image_ids[new_idx])

    def find_right_image(self, idx):
        return self.find_image(self.image_ids[idx])

    def find_image(self, image_id):
        if self.dataset_type == 'flickr8k':
            return self.find_flickr8k_image(image_id)
        elif self.dataset_type == 'flickr30k':
            return self.find_all_flickr30k_images(image_id)
        elif self.dataset_type == 'mscoco':
            return self.find_all_mscoco_images(image_id)
        else:
            raise Exception("the dataset %s does no been provided! please make sure arguments", self.dataset_type)

    def find_flickr8k_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).resize((224, 224))
        images = [image]
        return images

    def find_crop_flickr8k_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path)
        # print "image_id",image_id
        # print image
        w, h = image.size
        image_lt = image.crop((0,0,128,128))
        image_lt2 = image_lt.transpose(Image.FLIP_TOP_BOTTOM)
        image_rt = image.crop((w-128,0,w,128))
        image_rt2 = image_rt.transpose(Image.FLIP_TOP_BOTTOM)
        image_lb = image.crop((0,h-128,128,h))
        image_lb2 = image_lb.transpose(Image.FLIP_TOP_BOTTOM)
        image_rb = image.crop((w-128,h-128,w,h))
        image_rb2 = image_rb.transpose(Image.FLIP_TOP_BOTTOM)
        image_center = image.crop((w/2-64,h/2-64,w/2+64,h/2+64))
        image_center2 = image_center.transpose(Image.FLIP_TOP_BOTTOM)
        images = [image_lt,image_lt2,image_rt,image_rt2
            ,image_lb,image_lb2,image_rb,image_rb2,image_center,image_center2]
        return images



    def validate_image(self, img):
        # img = np.array(img, dtype=float)
        # if len(img.shape) < 3:
        #     rgb = np.empty((64, 64, 3), dtype=np.float32)
        #     rgb[:, :, 0] = img
        #     rgb[:, :, 1] = img
        #     rgb[:, :, 2] = img
        #     img = rgb
        #
        # return img.transpose(2, 0, 1)
        return img


