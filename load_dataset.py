import glob
import os
import random
import numpy as np
import pandas as pd
import glob
from PIL import Image, ImageOps, ImageFilter, ImageFile
from torch.utils.data import Dataset
#from config import *
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN
from torchvision import datasets
import torch
import torchvision
import pickle
# from timm.data.transforms import _pil_interp
#import pyvips
ImageFile.LOAD_TRUNCATED_IMAGES = True
torchvision.set_image_backend('accimage')
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        
        ret = []
        if self.transform is not None:
            # for t in self.transform:
            #     ret.append(t(image))
            ret.append(self.transform(image))
        else:
            ret.append(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        ret.append(target)

        return ret

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class faceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        k_class = 0
        self.f_names = []
        self.labels = []
        self.ids = []
        no_dir = os.listdir(self.img_dir)
        for dir_name in no_dir:
            # print(self.img_dir + dir_name)
            self.f_names += glob.glob(self.img_dir + dir_name + "/*.jpg")
            # print(self.f_names)
            self.labels  += [k_class for x in range(len(glob.glob(self.img_dir + dir_name  + "/*.jpg")))]
            k_class += 1
            # print(self.labels)
        self.ids = list(range(0, len(self.f_names)))

    def __len__(self):
        return len(self.f_names)

    def __getitem__(self, index):
        idx = self.ids[index]
        transform = T.Compose([T.Resize(128),
                               T.CenterCrop(128),
                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform(Image.open(self.f_names[idx]))
        target = self.labels[idx]
        return img, target, idx

class intelDataset(Dataset):
    def __init__(self, img_dir, transf):
        self.transform = transf
        self.img_dir = img_dir
        k_class = 0
        self.f_names = []
        self.labels = []
        self.ids = []
        no_dir = os.listdir(self.img_dir)
        for dir_name in no_dir:
            # print(self.img_dir + dir_name)
            self.f_names += glob.glob(self.img_dir + dir_name + "/*.jpg")
            # print(self.f_names)
            self.labels  += [k_class for x in range(len(glob.glob(self.img_dir + dir_name  + "/*.jpg")))]
            k_class += 1
            # print(self.labels)
        self.ids = list(range(0, len(self.f_names)))

    def __len__(self):
        return len(self.f_names)

    def __getitem__(self, index):
        idx = self.ids[index]
        # transform = T.Compose([T.Resize(128),
        #                        T.CenterCrop(128),
        #                        T.ToTensor(),
        #                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = Image.open(self.f_names[idx])
        img = self.transform(img)
        target = self.labels[idx]
        return img, target, idx

class BoundingBoxImageLoader(Dataset):
    """Animal Bounding Box Crop."""

    def __init__(self, pickle_file, root_dir, transform=None, is_test=False,pseudo_labels={}):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_pickle(pickle_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        # self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataframe)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     img_name = os.path.join(self.root_dir,
    #                             self.dataframe.iloc[idx, 0]+'.JPG')

    #     image = Image.open(img_name)
    #     #image = Image.new('RGB',(128, 128))
    #     #image = pyvips.Image.new_from_file(img_name, access='sequential')
    #     bbox_im = self.dataframe.iloc[idx, 1]
    #     image_croped = T.functional.crop(
    #         image, int(bbox_im[1]), int(bbox_im[0]), int(bbox_im[3]), int(bbox_im[2])) # top, left, height, width
        
    #     sample = image_croped

    #     # get pseudo label if available
    #     # if idx in self.pseudo_labels:
    #     #     target = self.pseudo_labels[idx]
    #     # else:
    #     target = self.dataframe.iloc[idx, 2]
        
    #     if self.transform:
    #         sample = self.transform(sample)
    #     # if self.is_test:
    #     #     return sample, target
    #     return sample, target, idx
        
 # testing croped resized images
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 3]+'.JPG')

        image = Image.open(img_name)
        #image = Image.new('RGB',(128, 128))
        #image = pyvips.Image.new_from_file(img_name, access='sequential')
        # bbox_im = self.dataframe.iloc[idx, 1]
        # image_croped = T.functional.crop(
        #     image, int(bbox_im[1]), int(bbox_im[0]), int(bbox_im[3]), int(bbox_im[2])) # top, left, height, width
        
        sample = image
        target = self.dataframe.iloc[idx, 2]
        
        if self.transform:
            sample = self.transform(sample)
        if self.is_test:
            return sample, target
        return sample, target, idx




class BoundingBoxImageLoaderWithPseudoLabels(BoundingBoxImageLoader):
    def __init__(self, pickle_file, root_dir, transform=None, pseudo_labels_pickle_file=None):
        super().__init__(pickle_file,root_dir,transform=transform,is_test=False)
        pseudo_labels = {}
        if os.path.exists(pseudo_labels_pickle_file): # first cycle will not have a file.
            with open(pseudo_labels_pickle_file, 'rb') as f:
                pseudo_labels = pickle.load(f)

        self.pseudo_labels = pseudo_labels # dict
    
    def __len__(self):
        return super().__len__()

    def __getitem__(self,idx):
        sample, target, idx = super().__getitem__(idx)

        if idx in self.pseudo_labels:
            target = self.pseudo_labels[idx]

        return sample, target, idx

class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf):
        self.dataset_name = dataset_name
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('./cifar10', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == 'cifar10_test':
            self.cifar10 = CIFAR10('./cifar10', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('../cifar100', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../fashionMNIST', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('../svhn', split="train", 
                                    download=True, transform=transf)


    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == 'cifar10_test':
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        if self.dataset_name == 'cifar10_test':
            return len(self.cifar10)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)
##

# Data
def load_dataset(dataset, args, add_ssl=False, image_size=32):
    if dataset == 'cifar10' or dataset == 'cifar10im':
        normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        IMG_SIZE = image_size
    elif dataset == 'cifar100':
        normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        IMG_SIZE = image_size
    elif dataset == 'SnapshotSerengeti10':
        normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        IMG_SIZE = image_size
    elif dataset in ['SnapshotSerengeti','SnapshotSerengetiSmall']:
        normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        IMG_SIZE = image_size
    else:
        normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        IMG_SIZE = image_size
    # Weak augmentations
    cifar100_train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomCrop(size=IMG_SIZE, padding=4),
        T.ToTensor(),
       normalize
    ])

    cifar10_train_transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.RandomHorizontalFlip(0.5),
        T.RandomCrop(size=IMG_SIZE, padding=4),
        T.ToTensor(),
        normalize
    ])
    svhn_transform = T.Compose([
                            T.RandomHorizontalFlip(0.5),
                            T.RandomCrop(size=IMG_SIZE, padding=4),
                            T.ToTensor(),
                        ])

    snapshotserengeti10_train_transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.RandomHorizontalFlip(0.5),
        T.RandomCrop(size=IMG_SIZE, padding=4),
        T.ToTensor(),
        normalize
    ])
    snapshotserengeti_train_transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.RandomHorizontalFlip(0.5),
        T.RandomCrop(size=IMG_SIZE, padding=4),
        T.ToTensor(),
        normalize
    ])

    if add_ssl:
        # Strong augmentations
        transform_2 = T.Compose([
            # T.Resize(IMG_SIZE, interpolation=_pil_interp('bicubic')),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur()], p=0.2),
            T.RandomApply([ImageOps.solarize], p=0.2),
            T.ToTensor(),
            normalize,
        ])


    
        svhn_transform2 = T.Compose([
                            T.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.)),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply([GaussianBlur()], p=0.2),
                            T.RandomApply([ImageOps.solarize], p=0.2),
                            T.ToTensor(),
                            ])
        # cifar10_train_transform1 = transform
        cifar10_train_transform2 = transform_2 
        # cifar100_train_transform1= transform
        cifar100_train_transform2  = transform_2 

        snapshotserengeti10_train_transform2 = transform_2
        snapshotserengeti_train_transform2 = transform_2
    
    # Test augmentations
    cifar100_test_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    cifar10_test_transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.ToTensor(),
        normalize
    ])
    snapshotserengeti10_test_transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.ToTensor(),
        normalize
    ])
    snapshotserengeti_test_transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.ToTensor(),
        normalize
    ])
    intel_test_transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        normalize
    ])
    data_train2, data_unlabeled2 = [], []
        # cifar100_train_transform
    if dataset == 'cifar10': 
        data_train = CIFAR10('./cifar10', train=True, download=True, transform=cifar10_train_transform)
        if add_ssl:
            data_train2 = CIFAR10('./cifar10', train=True, download=True, transform=cifar10_train_transform2)
            data_unlabeled2 = MyDataset(dataset, True, cifar10_train_transform2)
        data_unlabeled = MyDataset(dataset, True, cifar10_test_transform)
        # data_test  = CIFAR10('./cifar10', train=False, download=True, transform=cifar10_test_transform)
        data_test = MyDataset('cifar10_test', False, cifar10_test_transform)
        NO_CLASSES = 10
        #adden = ADDENDUM
        no_train = 50000
    
    elif dataset == 'cifar10im': 
        data_train = CIFAR10('./cifar10', train=True, download=True, transform=cifar10_train_transform)

        #data_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        targets = np.array(data_train.targets)
        #NUM_TRAIN = targets.shape[0]
        classes, _ = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [500, 5000] * 5
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        # print(NUM_TRAIN)
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        data_unlabeled = MyDataset(dataset[:-2], True, cifar10_test_transform)
        data_unlabeled.cifar10.targets = targets[imb_class_idx]
        data_unlabeled.cifar10.data = data_unlabeled.cifar10.data[imb_class_idx]

        if add_ssl:
            data_train2 = CIFAR10('./cifar10', train=True, download=True, transform=cifar10_train_transform2)
            data_train2.targets = targets[imb_class_idx]
            data_unlabeled2 = MyDataset(dataset[:-2], True, cifar10_train_transform2)
            data_unlabeled2.cifar10.targets = targets[imb_class_idx]
            data_unlabeled2.cifar10.data = data_unlabeled2.cifar10.data[imb_class_idx]
        data_test  = CIFAR10('./cifar10', train=False, download=True, transform=cifar10_test_transform)
        NO_CLASSES = 10
        #adden = ADDENDUM

    elif dataset == 'cifar100':
        data_train = CIFAR100('../cifar100', train=True, download=True, transform=cifar100_train_transform)
        if add_ssl:
            data_train2 = CIFAR100('../cifar100', train=True, download=True, transform=cifar100_train_transform2)
            data_unlabeled2 = MyDataset(dataset, True, cifar100_train_transform2)
        data_unlabeled = MyDataset(dataset, True, cifar100_train_transform)
        data_test  = CIFAR100('../cifar100', train=False, download=True, transform=cifar100_test_transform)
        NO_CLASSES = 100
        #adden = 2000
        no_train = 50000

    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = FashionMNIST('../fashionMNIST', train=False, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        #adden = ADDENDUM
        no_train = 60000

    elif dataset == 'svhn':
        data_train = SVHN('../svhn', split='train', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = SVHN('../svhn', split='test', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        #adden = ADDENDUM
        no_train = 73257
    elif dataset == 'svhn5':
        data_train = SVHN('../svhn', split='train', download=True, 
                                    transform=svhn_transform)
        nb_classes = 5
        targets = np.array(data_train.labels)

        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        class_idxs = np.hstack(class_idxs)
        no_train = len(class_idxs)
        data_train.labels = targets[class_idxs]
        data_train.data = data_train.data[class_idxs]
        if add_ssl:
            data_train2 = SVHN('../svhn', split='train', download=True, 
                                    transform=svhn_transform2)
            data_unlabeled2 = SVHN('../svhn', split='train', download=True, 
                                    transform=svhn_transform2)
            data_unlabeled2.labels = targets[class_idxs]
            data_unlabeled2.data = data_unlabeled2.data[class_idxs]
        data_unlabeled = SVHN('../svhn', split='train', download=True, 
                                    transform=svhn_transform2)
        data_unlabeled.labels = targets[class_idxs]
        data_unlabeled.data = data_unlabeled.data[class_idxs]
        data_test  = SVHN('../svhn', split='test', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        targets = np.array(data_test.labels)
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        class_idxs = np.hstack(class_idxs)
        data_test.labels = targets[class_idxs]
        data_test.data = data_test.data[class_idxs]
        NO_CLASSES = 5
        #adden = 100
        
    elif dataset == 'SnapshotSerengeti10':
        if args.sampling_strategy == 'corelbpseudo': # create subclass that additionally has pseudolabels
            with open(f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl', 'wb') as file:
                pickle.dump({}, file, protocol=pickle.HIGHEST_PROTOCOL)
            data_train = BoundingBoxImageLoaderWithPseudoLabels(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti10_train_transform,
                pseudo_labels_pickle_file=f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl'
                )
        else:
            data_train = BoundingBoxImageLoader(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti10_train_transform)
            
        data_unlabeled = BoundingBoxImageLoader(
            pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_train.df',
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=snapshotserengeti10_train_transform)
        data_test = BoundingBoxImageLoader(
            pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_test.df',
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=snapshotserengeti10_test_transform,
            is_test=True)

        if add_ssl:
            if args.sampling_strategy == 'corelbpseudo':
                data_train2 = BoundingBoxImageLoaderWithPseudoLabels(
                    pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_train.df',
                    root_dir=os.environ['DATA_DIR_PATH'],
                    transform=snapshotserengeti10_train_transform2,
                    pseudo_labels_pickle_file=f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl'
                    )                
            else: 
                data_train2 = BoundingBoxImageLoader(
                    pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_train.df',
                    root_dir=os.environ['DATA_DIR_PATH'],
                    transform=snapshotserengeti10_train_transform2)
            
            data_unlabeled2 = BoundingBoxImageLoader(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_balanced_top_10_metadata_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti10_train_transform2)
        NO_CLASSES = 10
        no_train = 175000
    
    elif dataset == 'SnapshotSerengeti':
        if args.sampling_strategy == 'corelbpseudo':
            with open(f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl', 'wb') as file:
                pickle.dump({}, file, protocol=pickle.HIGHEST_PROTOCOL)
            data_train = BoundingBoxImageLoaderWithPseudoLabels(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti_train_transform,
                pseudo_labels_pickle_file=f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl'
            )
        else:
            data_train = BoundingBoxImageLoader(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti_train_transform)
        data_unlabeled = BoundingBoxImageLoader(
            pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=snapshotserengeti_train_transform)
        data_test = BoundingBoxImageLoader(
            pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_test.df',
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=snapshotserengeti_test_transform,
            is_test=True)
        if add_ssl:
            if args.sampling_strategy == 'corelbpseudo':
                data_train2 = BoundingBoxImageLoaderWithPseudoLabels(
                    pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
                    root_dir=os.environ['DATA_DIR_PATH'],
                    transform=snapshotserengeti_train_transform2,
                    pseudo_labels_pickle_file = f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl'
                )
            else:
                data_train2 = BoundingBoxImageLoader(
                    pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
                    root_dir=os.environ['DATA_DIR_PATH'],
                    transform=snapshotserengeti_train_transform2)
            data_unlabeled2 = BoundingBoxImageLoader(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti_train_transform2)
        NO_CLASSES = 47
        no_train = 1094430
    
    elif dataset == 'SnapshotSerengetiSmall':
        if args.sampling_strategy == 'corelbpseudo':
            with open(f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl', 'wb') as file:
                pickle.dump({}, file, protocol=pickle.HIGHEST_PROTOCOL)
            data_train = BoundingBoxImageLoaderWithPseudoLabels(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_unique_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti_train_transform,
                pseudo_labels_pickle_file=f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl'
            )
        else:
            data_train = BoundingBoxImageLoader(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_unique_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti_train_transform)
        data_unlabeled = BoundingBoxImageLoader(
            pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_train.df',
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=snapshotserengeti_train_transform)
        data_test = BoundingBoxImageLoader(
            pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_test.df',
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=snapshotserengeti_test_transform,
            is_test=True)
        if add_ssl:
            if args.sampling_strategy == 'corelbpseudo':
                data_train2 = BoundingBoxImageLoaderWithPseudoLabels(
                    pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_unique_train.df',
                    root_dir=os.environ['DATA_DIR_PATH'],
                    transform=snapshotserengeti_train_transform2,
                    pseudo_labels_pickle_file = f'./MoBYv2AL/models/pseudo_labels_{args.dataset}_{args.method_type}_{args.run_id}.pkl'
                )
            else:
                data_train2 = BoundingBoxImageLoader(
                    pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_unique_train.df',
                    root_dir=os.environ['DATA_DIR_PATH'],
                    transform=snapshotserengeti_train_transform2)
            data_unlabeled2 = BoundingBoxImageLoader(
                pickle_file=os.environ['DATA_DIR_PATH']+ '/' + 'df_metadata_unique_train.df',
                root_dir=os.environ['DATA_DIR_PATH'],
                transform=snapshotserengeti_train_transform2)
        NO_CLASSES = 46
        no_train = 512522#512857

    return data_train, data_unlabeled, data_test, NO_CLASSES, no_train, data_train2, data_unlabeled2

class DataAugmentationDINO(object):
    def __init__(self, img_sz, global_crops_scale, local_crops_scale, local_crops_number, normalize):
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        # first global crop
        self.global_transfo1 = T.Compose([
            T.RandomResizedCrop(img_sz, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            T.ToTensor(),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = T.Compose([
            T.RandomResizedCrop(img_sz, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            T.ToTensor(),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = T.Compose([
            T.RandomResizedCrop(int(img_sz/2.23), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.5),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

