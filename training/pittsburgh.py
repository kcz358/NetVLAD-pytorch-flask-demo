import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors

root_dir = "raw_data/"
query_dir = "raw_data/queries_real/"
def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

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
    

class WholeDataset(data.Dataset):
    def __init__(self, structFile, input_transform) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.db = parse_dbStruct(structFile)
        self.whichSet = self.db.whichSet
        self.dataset = self.db.dataset
        self.image = [join(root_dir, dbImage) for dbImage in self.db.dbImage]
        self.image += [join(query_dir, qImage) for qImage in self.db.qImage]
        
        self.positive = None
        self.distance = None
    
    def __getitem__(self, index):
        image = Image.open(self.image[index])
        image = self.input_transform(image)
        return image, index
    
    def __len__(self):
        return len(self.image)
    
    def get_positive(self):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db.utmDb)
        
        self.distance, self.positive = knn.radius_neighbors(self.db.utmQ, radius = self.db.posDistThr, sort_results=True)
        return self.positive
        

class QueryDataset(data.Dataset):
    def __init__(self, structFile, nNeg=10, input_transform=None):
        super().__init__()
        self.db = parse_dbStruct(structFile)
        self.whichSet = self.db.whichSet
        self.datasest = self.db.dataset
        self.nNeg = nNeg
        self.input_transform = input_transform
        
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db.utmDb)
        
        #sort nontrivialpositives by distance
        self.nontrivialpositives = list(knn.radius_neighbors(self.db.utmQ, radius = self.db.nonTrivPosDistSqThr**0.5, sort_results=True)[1])
        
        #Filter out elements with no trivial positives
        #np.where produce a tuple, we want array inside, thus adding[0] after
        self.queries = np.where(np.array([len(x) for x in self.nontrivialpositives]))[0]
        
        potential_positive = knn.radius_neighbors(self.db.utmQ, radius = self.db.posDistThr, return_distance=False)
        
        self.potential_negative = []
        for pos in potential_positive:
            self.potential_negative.append(np.setdiff1d(np.arange(self.db.numDb), pos, assume_unique=True))
            
    def __getitem__(self, index):
        index = self.queries[index]
        positive_index = self.nontrivialpositives[index][0]
        query = Image.open(join(query_dir, self.db.qImage[index]))
        positive = Image.open(join(root_dir, self.db.dbImage[positive_index]))
        
        if self.input_transform != None:
            query = self.input_transform(query)
            positive = self.input_transform(positive)
        else:
            query = torch.from_numpy(query)
            positive = torch.from_numpy(positive)
        
        negatives = []
        neg_sample = np.random.choice(self.potential_negative[index], size = self.nNeg, replace=False)
        for neg in neg_sample:
            negative = Image.open(join(root_dir, self.db.dbImage[neg]))
            if self.input_transform != None:
                negative = self.input_transform(negative)
            else:
                negative = torch.from_numpy(negative)
            negatives.append(negative)
        
        negatives = torch.stack(negatives, 0)
        return query, positive, negatives
    
    def __len__(self):
        return len(self.queries)
        
        