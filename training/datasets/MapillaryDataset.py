from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import namedtuple

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
DATASET_ROOT = '/root/data/MSLS/'

def input_transform():
    return transforms.Compose([
        transforms.Resize((480, 640), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')

dbStruct = namedtuple('dbStruct', ['numDb', 'numQ'])

class MSLS(Dataset):
    def __init__(self, input_transform = input_transform()):
        
        self.input_transform = input_transform
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load('./datasets/msls_val/msls_val_dbImages.npy')
        
        # hard coded query image names.
        self.qImages = np.load('./datasets/msls_val/msls_val_qImages.npy')
        
        self.dbStruct = dbStruct(len(self.dbImages), len(self.qImages))
        
        # hard coded index of query images
        self.qIdx = np.load('./datasets/msls_val/msls_val_qIdx.npy')
        
        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('./datasets/msls_val/msls_val_pIdx.npy', allow_pickle=True)
        
        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
    
    def getPositives(self):
        return self.pIdx
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)