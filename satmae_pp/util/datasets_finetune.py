import os
import pandas as pd
import numpy as np
import warnings
import random
import json
import cv2
from glob import glob
from typing import Any, Optional, List
import rasterio
from rasterio import logging

import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    
"""CATEGORIES=[
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]"""

CATEGORIES=['parkinglot',
 'harbor',
 'buildings',
 'river',
 'denseresidential',
 'overpass',
 'intersection',
 'sparseresidential',
 'forest',
 'runway',
 'chaparral',
 'storagetanks',
 'airplane',
 'golfcourse',
 'agricultural',
 'mediumresidential',
 'freeway',
 'baseballdiamond',
 'tenniscourt',
 'beach',
 'mobilehomepark']

"""CATEGORIES=['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 
'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 
'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 
'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 
'intersection', 'island', 'lake', 'meadow', 'medium_residential', 
'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 
'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 
'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 
'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']"""



class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, json_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param json_path: json_path (string): path to json file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transform
        self.categories = CATEGORIES

        # extract base folder path from json file path
        self.base_path = ''
        tokens = json_path.strip().split('/')
        for tok in tokens:
            if '.json' in tok:
                continue
            self.base_path = self.base_path + '/' + tok.strip()
        if 'train' in json_path:
            self.base_path = os.path.join(self.base_path, 'train')
        else:
            self.base_path = os.path.join(self.base_path, 'val')

        # read json data
        fid = open(json_path)
        data = json.load(fid)
        fid.close()

        # get list of image paths
        self.image_arr = []
        for key in data.keys():
            self.image_arr.append(data[key])

        self.data_len = len(self.image_arr)

        assert self.data_len > 0

    def __getitem__(self, index):
        # get image path at index
        item = self.image_arr[index]

        # read image
        abs_img_path = os.path.join(self.base_path, item['img_path'])
        #img = cv2.imread(abs_img_path)
        img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)
        
        # crop the image using bounding box coordinates
        box = item['box']
        cropped = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]

        # convert to PIL image
        img_as_img = Image.fromarray(cropped)
        
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)

        single_image_label = self.categories.index(item['label'])

        return {'img':img_as_tensor, 'label':single_image_label}

    def __len__(self):
        return self.data_len



#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = None,
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands = [0, 9, 10],
                 subset_indices: Optional[np.ndarray] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.csv_path = csv_path
        
        
        position = self.csv_path.find('sateuro')
        if position != -1:
          path_before_sateuro = self.csv_path[:position]
          self.base_path = path_before_sateuro.rstrip(os.sep)

        self.df = pd.read_csv(csv_path) \
            .sort_values(['ClassName'])


        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]
            
        # Apply subset if subset_indices is provided
        if subset_indices is not None:
            self.df = self.df.iloc[subset_indices].reset_index(drop=True)    
    
        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a dict.
        """
        selection = self.df.iloc[idx]

        folder = r'sateuro/EuroSATallBands'
        

        image_path = selection['Filename']

        abs_img_path = os.path.join(self.base_path, folder, image_path)

        images = self.open_image(abs_img_path)  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['ClassName'])
        
        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return {'img':img_as_tensor, 'label':labels}

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


#########################################################
# EuroSAT (RGB)
#########################################################

class EuroSat_RGB(SatelliteDataset):
    mean = [0.485, 0.456, 0.406] # ImageNet
    #mean = [0.368, 0.381, 0.3436] # actual (from scale mae)
    std = [0.229, 0.224, 0.225] # ImageNet
    #std = [0.2035, 0.1854, 0.1849] # actual (from scale mae)


    def __init__(self, base_path, file_path, transform):
        """
        Creates dataset for RGB single image classification for EuroSAT.
        :param base_path: path to the dataset root folder.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        """
        super().__init__(3)
        self.base_path = base_path
        self.file_path=file_path

        position = self.file_path.find('sateuro')
        if position != -1:
          path_before_sateuro = self.file_path[:position]
          self.base_path = path_before_sateuro.rstrip(os.sep)

        self.df = pd.read_csv(self.file_path) \
            .sort_values(['ClassName'])

        # transforms
        self.transform = transform
        self.categories = CATEGORIES


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        selection = self.df.iloc[idx]
        folder = r'sateuro/EuroSAT'
        image_path = selection['Filename']

        abs_img_path = os.path.join(self.base_path, folder, image_path)
        img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)

        # get integer label of the image
        label = self.categories.index(selection['ClassName'])

        # apply transforms
        img_as_tensor = self.transform(img)  # (c, h, w)

        return {'img':img_as_tensor, 'label':label}

#########################################################
# UC-Merced
#########################################################

class UCMerced(SatelliteDataset):
    #mean = [0.4842, 0.4901, 0.4505] # from scale mae
    #std = [0.2180, 0.2021, 0.1958] # from scale mae
    mean = [0.485, 0.456, 0.406] # ImageNet
    std = [0.229, 0.224, 0.225] # ImageNet


    def __init__(self, base_path, file_path, transform,subset_indices: Optional[np.ndarray] = None):
        """
        Creates dataset for RGB single image classification for UCMerced.
        :param base_path: path to the dataset root folder.
        :param file_path: path to txt file containing paths to image data for UCMerced.
        :param transform: pytorch Transform for transforms and tensor conversion
        """
        super().__init__(3)
        self.base_path = "/home/maloulou/UCMerced_LandUse/Images"
        self.file_path=file_path

        # read data split file
        self.df = pd.read_csv(self.file_path) \
            .sort_values(['ClassName'])   
        #reduce by 10 percent    
        if subset_indices is not None:
            self.df = self.df.iloc[subset_indices].reset_index(drop=True)  

        # get list of images and corresponding labels
        self.img_paths = self.df['ClassName'].tolist()
        self.labels = self.df['Label'].tolist()
 

        # transforms
        self.transform = transform
        self.categories = CATEGORIES


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get image path at idx
        selection = self.df.iloc[idx]
        folder=selection['Label']
        image_path = selection['ClassName']

        #read image
        abs_img_path = os.path.join(self.base_path, folder, image_path)
        img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)

        # get integer label
        label = self.categories.index(selection['Label'])

        # apply transforms
        img_as_tensor = self.transform(img)  # (c, h, w)

        return {'img':img_as_tensor, 'label':label}

#########################################################
# RESISC-45
#########################################################

class Resisc_45(SatelliteDataset):
    mean = [0.368, 0.381, 0.3436] # from scale mae
    std = [0.2035, 0.1854, 0.1849] # from scale mae
    #mean = [0.485, 0.456, 0.406] # ImageNet
    #std = [0.229, 0.224, 0.225] # ImageNet


    def __init__(self, is_train,base_path, file_path, transform,subset_indices: Optional[np.ndarray] = None):
        """
        Creates dataset for RGB single image classification for RESISC-45.
        :param base_path: path to the dataset root folder.
        :param file_path: path to txt file containing paths to image data for RESISC-45.
        :param transform: pytorch Transform for transforms and tensor conversion
        """
        super().__init__(3)
        self.base_path = "/mnt/isilon/maloulou/"

        self.is_train=is_train
        self.file_path=file_path

        # read data split file
        self.df = pd.read_csv(self.file_path) \
            .sort_values(['ClassName'])   
        #reduce by 10 percent    
        if subset_indices is not None:
            self.df = self.df.iloc[subset_indices].reset_index(drop=True)     

        # get list of images and corresponding labels
        self.img_paths = self.df['ClassName'].tolist()
        self.labels = [label[:-1] for label in self.df['Label'].tolist()]
        # transforms
        self.transform = transform
        self.categories = CATEGORIES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get image path at idx
        img_path = self.img_paths[idx]
        img_path = Path(img_path)
        img_path_str = str(img_path).replace("\\", "/")

        #read image
        folder='resisc_45/Dataset'
        folder=os.path.join(folder,'train','train') 
        abs_img_path = os.path.join(self.base_path, folder,str(self.labels[idx]),img_path_str)
        img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2RGB)

        # get integer label of the image
        label = self.categories.index(self.labels[idx])

        # apply transforms
        img_as_tensor = self.transform(img)  # (c, h, w)

        return {'img':img_as_tensor, 'label':label}

###################################################################################################################

def build_fmow_dataset(is_train: bool, args,subset_indices: np.ndarray=None) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    file_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(file_path, transform)
    
    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(file_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands,subset_indices=subset_indices)
    
    elif args.dataset_type == 'euro_sat':
        mean, std = EuroSat_RGB.mean, EuroSat_RGB.std
        transform = EuroSat_RGB.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat_RGB(args.base_path, file_path, transform)

    elif args.dataset_type == 'ucmerced':
        mean, std = UCMerced.mean, UCMerced.std
        transform = UCMerced.build_transform(is_train, args.input_size, mean, std)
        dataset = UCMerced(args.base_path, file_path, transform,subset_indices=subset_indices)

    elif args.dataset_type == 'resisc':
        mean, std = Resisc_45.mean, Resisc_45.std
        transform = Resisc_45.build_transform(is_train, args.input_size, mean, std)
        dataset = Resisc_45(is_train,args.base_path, file_path, transform, subset_indices=subset_indices)
    
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
