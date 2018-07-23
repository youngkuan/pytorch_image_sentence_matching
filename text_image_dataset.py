import io
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch
import h5py


class Text2ImageDataset(Dataset):

    def __init__(self, sentence_embedding_file, image_ids_file, image_dir):
        self.sentence_embedding_file = sentence_embedding_file
        self.image_ids_file = image_ids_file
        self.image_dir = image_dir
        self.sentence_embeddings = None
        self.image_ids = None
        self.images = None

    def __len__(self):
        sentence_embeddings = h5py.File(self.sentence_embedding_file, 'r')
        # load the sentence embeddings ;size: n * 6000
        self.sentence_embeddings = sentence_embeddings['vectors_']
        length = len(self.sentence_embeddings.shape[0])
        return length

    def __getitem__(self, idx):
        if self.sentence_embeddings is None:
            sentence_embeddings = h5py.File(self.sentence_embedding_file, 'r')
            # load the sentence embeddings ;size: n * 6000
            self.sentence_embeddings = np.asarray(sentence_embeddings['vectors_'])

        if self.image_ids is None:
            image_ids = []
            image_ids_h5py = h5py.File(self.image_ids_file, 'r')
            hdf5_objects = image_ids_h5py['image_ids']
            length = hdf5_objects.shape[1]
            for i in range(length):
                image_ids.append(''.join([chr(v[0]) for v in image_ids_h5py[hdf5_objects[0][i]].value]))
            # image ids (n * 1)
            self.image_ids = image_ids


        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        txt = np.array(example['txt']).astype(str)

        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'inter_embed': torch.FloatTensor(inter_embed),
                'txt': str(txt)
                 }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

