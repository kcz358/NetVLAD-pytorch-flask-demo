import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
import netvlad
import faiss
import numpy as np
import pittsburgh

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def test(epoch):
    checkpoints = torch.load("checkpoints/epoch_{}".format(epoch))
    model.load_state_dict(checkpoints["model_state_dict"])
    dataset = pittsburgh.WholeDataset("raw_data/pitts30k_test.mat", input_transform=pittsburgh.input_transform())
    test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dbfeatures = np.empty((len(dataset), 512*64))
    model.eval()
    with torch.no_grad():
        print("Extracting features")
        for iteration, (image, indice) in enumerate(test_data_loader, 1):
            image = image.to(device)
            image_encoding = model.encode(image)
            vlad_encoding = model.netvlad_pool(image_encoding)
            
            dbfeatures[indice, :] = vlad_encoding.detach().cpu().numpy()
            
            if iteration % 50 == 0:
                print("====> Batch [{} / {}]".format(iteration, len(test_data_loader)))
                file_object = open('encode.txt', 'a')
                file_object.write("====> Batch [{} / {}]\n".format(iteration, len(test_data_loader)))
                file_object.close()     
            del image, image_encoding, vlad_encoding
            
    del test_data_loader
    qFeat = dbfeatures[dataset.db.numDb:].astype("float32")
    dbFeat = dbfeatures[:dataset.db.numDb].astype("float32")
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(512*64)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 

    # for each query get those within threshold distance
    gt = dataset.get_positive() 

    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / dataset.db.numQ
    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))        
            
    
    return recalls
        

encoder = models.vgg16(pretrained=True)
layers = list(encoder.features.children())[:-2]

for l in layers:
    for p in l.parameters():
        p.requires_grad = False
        
layers.append(L2Norm())
encoder = nn.Sequential(*layers)
model = nn.Module()
model.add_module("encode", encoder)

netvlad_layer = netvlad.NetVLAD(num_clusters=64, dim=512,vladv2=True)
model.add_module("netvlad_pool", netvlad_layer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.encode = nn.DataParallel(model.encode)
model.netvlad_pool = nn.DataParallel(model.netvlad_pool)
model.to(device)

test(6)