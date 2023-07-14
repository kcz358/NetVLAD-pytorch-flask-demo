import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

struct_dir = 'datasets'
database_dir = '.'
queries_dir = 'queries_real'

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_training_set(root_dir, onlyDB=False):
    return WholeDatasetFromStruct(root_dir,
                                  structFile='pitts30k_train.mat',
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

def get_whole_val_set(root_dir):
    return WholeDatasetFromStruct(root_dir,
                                  structFile='pitts30k_val.mat',
                                  input_transform=input_transform())

def get_250k_val_set(root_dir):
    return WholeDatasetFromStruct(root_dir,
                                  structFile='pitts250k_val.mat',
                                  input_transform=input_transform())
def get_whole_test_set(root_dir):
    return WholeDatasetFromStruct(root_dir,
                                  structFile='pitts30k_test.mat',
                                  input_transform=input_transform())

def get_250k_test_set(root_dir):
    return WholeDatasetFromStruct(root_dir,
                                  structFile='pitts250k_test.mat',
                                  input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, root_dir, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.root_dir = root_dir
        self.struct_dir = join(self.root_dir, struct_dir)
        self.database_dir = join(self.root_dir, database_dir)
        self.queries_dir = join(self.root_dir, queries_dir)

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(join(self.struct_dir, structFile))
        self.images = [join(self.database_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(self.queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def get_path(self, index):
        path = self.images[index]
        return path

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
