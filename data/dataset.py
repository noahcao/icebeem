"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .smallnorb import SmallNORB
import h5py


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        assert 0, "should not use this class here"
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        '''
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2
        '''
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)
        if img1.shape[0] == 1:
            img1 = img1.repeat(3,1,1)
        return img1

    def __len__(self):
        return self.data_tensor.size(0)


class Custom3DshapesDataset(Dataset):
    '''
    dataset used for 3Dshapes: https://github.com/deepmind/3d-shapes
    images are of shape [64, 64, 3]
    Latent factor values
        floor hue: 10 values linearly spaced in [0, 1]
        wall hue: 10 values linearly spaced in [0, 1]
        object hue: 10 values linearly spaced in [0, 1]
        scale: 8 values linearly spaced in [0, 1]
        shape: 4 values in [0, 1, 2, 3]
        orientation: 15 values linearly spaced in [-30, 30]
    '''
    def __init__(self, data, transform=None):
        self.imgs, self.targets = data 
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        img = self.imgs[index1]
        label = self.targets[index1]
        if self.transform is not None:
            img = self.transform(img)
        return (img, label)

    def __len__(self):
        return self.imgs.size(0)


class CustomLabeledTensorDataset(Dataset):
    # this is for dsprites or other datasets having labels 
    # for classification or regression
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        '''
        self.imgs = torch.from_numpy(self.data_tensor['imgs']).unsqueeze(1).float()
        self.values = torch.from_numpy(self.data_tensor['latents_values'])
        self.classes = torch.from_numpy(self.data_tensor['latents_classes'])
        '''
        self.imgs, self.targets, self.classes = data_tensor
        self.indices = range(len(self))

       
    def __getitem__(self, index1):
        img1 = self.imgs[index1]
        latent_cls = self.classes[index1]
        latent_value = self.targets[index1]
        if self.transform is not None:
            img1 = self.transform(img1)
        if img1.shape[0] == 1:
            img1 = img1.repeat(3,1,1)
        return (img1, (latent_cls, latent_value))

    def __len__(self):
        return self.imgs.shape[0]


class COCODataset(Dataset):
    '''
        For Factor disentanglement study on the classic COCO dataset
        for each category of object contained in the image, it's 
        accounted for a single 'factor', we maintain in total 100 factors
        in total, though the total category number on COCO is not that many
    '''
    def __init__(self, dirs, transform=None):
        self.img_dir = dirs[0]
        self.img_json = dirs[1]
        self.factor_anno = dirs[2]
        self.w, self.h = 256, 256
        self.transform = transform

        self.image_info = json.load(open(self.img_json))["images"]
        factors = np.load(self.factor_anno)

        self.images = []
        self.ids = dict()
        self.factors = dict()

        for image in self.image_info:
            self.images.append((image["id"], image["file_name"]))
            self.factors[image["id"]] = np.zeros(100)

        for img_id in range(factors.shape[0]):
            image_id = int(factors[img_id][0])
            factor = factors[img_id][1:].astype(np.uint8)
            self.factors[image_id] = factor 
    
    def __getitem__(self, index):
        image_id, image_name = self.images[index]
        factor = self.factors[image_id]
        image_path = os.path.join(self.img_dir, image_name)
        im = cv2.imread(image_path)
        im = cv2.resize(im, (self.w, self.h))
        im = np.transpose(im, (2,0,1))
        if self.transform is not None:
            im = self.transform(im)
        factor[factor>1] = 1
        im = torch.Tensor(im)
        factor = torch.Tensor(factor)
        return (im, factor)

    def __len__(self):
        return len(self.images)



def dset_generate(name, dset_dir, image_size, mode="full-train"):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])

    # for 3-dim images
    transforms3D = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]
    )

    if name.lower() == 'celeba':
        imageroot = os.path.join(dset_dir, 'CelebA/data.npy')
        split = os.path.join(dset_dir, "CelebA/split.npz")
        attributes = os.path.join(dset_dir, 'CelebA/attr.npy')

        attributes = torch.from_numpy(np.load(attributes))
        images = np.load(imageroot)

        if mode == "train" or mode == "full-train":
            indices = np.load(split)["train"]
        elif mode == "valid":
            indices = np.load(split)["valid"]
        
        images = torch.from_numpy(images)
        images = torch.transpose(images, 1, 3)
        images = torch.transpose(images, 2, 3)
        images = images / 255.0
        images = images[indices]
        attributes = attributes[indices]
        data = (images, attributes)
        train_kwargs = {'data': data, 'transform':transforms3D}
        dset = Custom3DshapesDataset
    elif name.lower() == "3dshapes":
        root = os.path.join(dset_dir, "3dshapes/3dshapes.h5")
        data = h5py.File(root)

        splited = np.load(os.path.join(dset_dir, "3dshapes/split.npz"))

        images = torch.from_numpy(np.array(data["images"][:]))
        labels = torch.from_numpy(np.array(data["labels"][:]))

        if mode == "train":
            train_indices = splited["train"]
            images = images[train_indices]
            labels = labels[train_indices]
        elif mode == "valid":
            valid_indices = splited["valid"]
            images = images[valid_indices]
            labels = labels[valid_indices]

        images = torch.transpose(images, 1, 3)
        images = torch.transpose(images, 2, 3)
        images = images / 255.0

        # the last factor - orientation: 15 values linearly spaced in [-30, 30]
        # we make them into int: 0-14
        labels[:, -1] = (labels[:, -1] + 30) / (60.0/14.0)
        data = (images, labels)

        train_kwargs = {'data': data, 'transform': transforms3D}
        dset = Custom3DshapesDataset
    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        imgs = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        classes = torch.from_numpy(data['latents_classes'])
        values = torch.from_numpy(data['latents_values'])

        if mode == "train":
            train_indices = torch.load(os.path.join(dset_dir, "dsprites/dsprites_train.pth"))
            imgs = imgs[train_indices]
            values = values[train_indices]
            classes = classes[train_indices]
        elif mode == "valid":
            valid_indices = torch.load(os.path.join(dset_dir, "dsprites/dsprites_valid.pth"))
            imgs = imgs[valid_indices]
            values = values[valid_indices]
            classes = classes[valid_indices]

        data = (imgs, values, classes)
        train_kwargs = {'data_tensor':data}
        dset = CustomLabeledTensorDataset
    elif name.lower() == "coco":
        root = os.path.join(dset_dir)
        img_dir = os.path.join(root, "val2017")
        image_json = os.path.join(root, "annotations/instance_val2017.json")
        factor = os.path.join(root, "coco_val2017_factor.npy")
        data = (image_dir, image_json, factor_json)
        train_kwargs = {'dirs': data}
        dset = COCODataset
    elif name.lower() == "smallnorb":
        root = os.path.join(dset_dir, "smallnorb")
        train_kwargs = {"root": root}
        dset = SmallNORB
    elif name.lower() == "cars3d":
        root = os.path.join(dset_dir, "cars")
        train_kwargs = {"root": root}
        dset = Cars3D
    else:
        raise NotImplementedError

    data = dset(**train_kwargs)

    return data


def return_data(args, mode):
    name = args["DATALOADER"]["dataset"]
    dset_dir = args["DATALOADER"]["dset_dir"]
    batch_size = args["DATALOADER"]["batch_size"]
    num_workers = args["DATALOADER"]["num_workers"]
    image_size = args["MODEL"]["image_size"]
    assert image_size == 64, 'currently only image size of 64 is supported'
    assert mode in ["train", "valid", "full-train"]

    print("Loading {}-{} dataset...".format(name, mode))
    train_data = dset_generate(name, dset_dir, image_size, mode=mode)

    print("Completed loading dataset")

    shuffle = False if mode == "valid" else True
    data_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)
    # data_loader = train_loader
    return data_loader


def return_dataset(name, dset_dir, img_size):
    print("Loading {} dataset".format(name))

    if name.lower() == "dsprites":
        root = os.path.join(dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        imgs = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        classes = torch.from_numpy(data['latents_classes'])
        values = torch.from_numpy(data['latents_values'])

        data = (imgs, values, classes)
        train_kwargs = {'data_tensor':data}
        dset = CustomLabeledTensorDataset
    elif name.lower() == "smallnorb":
        root = os.path.join(dset_dir, "smallnorb")
        train_kwargs = {"root": root}
        dset = SmallNORB
    elif name.lower() == "cars3d":
        from .cars3d import Cars3D
        root = os.path.join(dset_dir, "cars")
        train_kwargs = {"root": root}
        dset = Cars3D
    elif name.lower() == "3dshapes":
        root = os.path.join(dset_dir, "3dshapes/3dshapes.h5")
        data = h5py.File(root)

        splited = np.load(os.path.join(dset_dir, "3dshapes/split.npz"))

        images = torch.from_numpy(np.array(data["images"][:]))
        labels = torch.from_numpy(np.array(data["labels"][:]))
        
        mode = "train"

        if mode == "train":
            train_indices = splited["train"]
            images = images[train_indices]
            labels = labels[train_indices]
        elif mode == "valid":
            valid_indices = splited["valid"]
            images = images[valid_indices]
            labels = labels[valid_indices]

        images = torch.transpose(images, 1, 3)
        images = torch.transpose(images, 2, 3)
        images = images / 255.0

        # the last factor - orientation: 15 values linearly spaced in [-30, 30]
        # we make them into int: 0-14
        labels[:, -1] = (labels[:, -1] + 30) / (60.0/14.0)
        data = (images, labels)

        train_kwargs = {'data': data}
        dset = Custom3DshapesDataset
    else:
        NotImplementedError("dataset {} not implemented yet".format(name))
    
    

    dataset = dset(**train_kwargs)
    return dataset
