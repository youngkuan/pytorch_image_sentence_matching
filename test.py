import os
from utils import Utils
from text_image_dataset import Text2ImageDataset
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import torch
from torchvision import transforms
from gan import Generator,Discriminator
from evaluate import i2t,t2i
from torchvision import models
import torch.nn as nn
from evaluate import map_image_and_sentence
from evaluate import validate_model


def main():
    # load model
    model_path = "./model/flickr8k/"
    generator_model_path = os.path.join(model_path, "gen_14.pth")
    discriminator_model_path = os.path.join(model_path, "disc_0.0020_0.50_3.00_1.00.pth")
    discriminator = Utils.load_discriminator(discriminator_model_path)

    # load validate data
    data_path = "../data/flickr8k"
    val_data_path = os.path.join(data_path, "val")
    val_sentence_embedding_file = os.path.join(val_data_path, "val_vectors_.mat")
    val_image_ids_file = os.path.join(val_data_path, "val_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    image_tensors, sentence_embedding_tensors = Utils.load_data(val_image_ids_file, val_sentence_embedding_file,
                                                                image_dir)

    r1, r5, r10, medr = t2i(discriminator, image_tensors, sentence_embedding_tensors)
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
        right_image = right_image.view(-1, 3, 128, 128)
        print right_image.size()
        wrong_image = sample["wrong_image"]
        wrong_image = wrong_image.view(-1, 3, 128, 128)
        print wrong_image.size()
        sentence_embedding = sample["sentence_embedding"]
        sentence_embedding = sentence_embedding.view(-1, 6000)
        print sentence_embedding.size()


def image_test():
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    image_ids = []
    image_ids_h5py = h5py.File(image_ids_file, 'r')
    hdf5_objects = image_ids_h5py['train_image_ids']
    length = hdf5_objects.shape[1]
    print ("loading image id")
    for i in range(length):
        image_ids.append(''.join([chr(v[0]) for v in image_ids_h5py[hdf5_objects[0][i]].value]))
    print ("loading image")
    cnt = 0
    for image_id in image_ids:
        image_path = os.path.join(image_dir, image_id)
        image = Image.open(image_path)
        (height, width) = image.size

        if height < 256 or width < 256:
            print ("image size: ", image.size)
            cnt += 1
    print ("the number of small image: %d" % cnt)

def image_crop():
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    image_ids = []
    image_ids_h5py = h5py.File(image_ids_file, 'r')
    hdf5_objects = image_ids_h5py['train_image_ids']
    length = hdf5_objects.shape[1]
    print ("loading image id")
    for i in range(length):
        image_ids.append(''.join([chr(v[0]) for v in image_ids_h5py[hdf5_objects[0][i]].value]))
    print ("loading image")
    images = []
    for image_id in image_ids[0:2]:
        image_path = os.path.join(image_dir, image_id)
        image = Image.open(image_path).resize([128,128])
        images.append(image)
    print images
    images_tensor = torch.stack([transforms.ToTensor()(i) for i in images])
    print images_tensor.size()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224))
    ])
    images_recreate = transform(images[0])
    print images_recreate

def vgg16_test():
    pretrained_model = models.vgg16(pretrained=True).features
    data_path = "../data/flickr8k"
    train_data_path = os.path.join(data_path, "train")
    image_ids_file = os.path.join(train_data_path, "train_image_ids.mat")
    image_dir = os.path.join(data_path, "images")
    image_ids = []
    image_ids_h5py = h5py.File(image_ids_file, 'r')
    hdf5_objects = image_ids_h5py['train_image_ids']
    length = hdf5_objects.shape[1]
    print ("loading image id")
    for i in range(length):
        image_ids.append(''.join([chr(v[0]) for v in image_ids_h5py[hdf5_objects[0][i]].value]))
    print ("loading image")
    images = []
    for image_id in image_ids[0:2]:
        image_path = os.path.join(image_dir, image_id)
        image = Image.open(image_path).resize([224, 224])
        images.append(image)
    images_tensor = torch.stack([transforms.ToTensor()(i) for i in images])
    print "images_tensor.size():",images_tensor.size()
    output = pretrained_model(images_tensor)
    print "output.size():",output.size()

def init_weight_test(m):
    classname = m.__class__.__name__
    print "classname:",classname
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    # for m in o.modules():
    #     print "m:",m
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.01)
    #         nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    validate_model()
