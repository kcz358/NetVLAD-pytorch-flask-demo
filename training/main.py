from genericpath import exists
from tracemalloc import start
import pittsburgh
import netvlad
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.io import loadmat


train_loss = 0
def train(epoch):
    epoch_loss = 0
    nBatches = len(train_set) // batch_size + 1
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    model.train()
    for iteration, (query, positives, negatives) in enumerate(train_dataloader):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
        if query is None: continue # in case we get an empty batch
        B, C, H, W = query.shape
        
        negatives = negatives.reshape(-1, C, H, W)
        input = torch.cat([query, positives, negatives])
        input = input.to(device)
        image_encoding = model.encode(input)
        vlad_encoding = model.netvlad_pool(image_encoding)
        vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, B*nNeg])
        optimizer.zero_grad()
        
        loss = 0
        for b in range(B):
            for n in range(nNeg):
                loss += criterion(vladQ[b:b+1], vladP[b:b+1], vladN[b*nNeg+n: b*nNeg+n+1])
        
        loss /= nNeg
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
        if (iteration + 1) % 50 == 0:
            print("=====> Epoch[{}] ({}/{}) Loss: {}".format(epoch+1, iteration+1, nBatches, batch_loss))
            file_object = open('loss.txt', 'a')
            file_object.write("=====> Epoch[{}] ({}/{}) Loss: {}\n".format(epoch+1, iteration+1, nBatches, batch_loss))
            file_object.close()
    del train_dataloader, loss
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    avg_loss = epoch_loss/nBatches
    print("====> Epoch[{}] Completed. Avg Loss {}".format(epoch+1, avg_loss))
    file_object = open('loss.txt', 'a')
    file_object.write("====> Epoch[{}] Completed. Avg Loss {}\n".format(epoch+1, avg_loss))
    file_object.close()       

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

train_set = pittsburgh.QueryDataset("raw_data/datasets/pitts30k_train.mat", nNeg = 10, input_transform=pittsburgh.input_transform())
batch_size = 4
nNeg = 10
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
traindescs = loadmat("initdata/pitts30k_train_vd16_conv5_3_preL2_traindescs.mat")['trainDescs'].reshape(-1,512)
clst = loadmat("initdata/pitts250k_train_vd16_conv5_3_preL2_k064_clst.mat")['clsts'].reshape(-1,512)
netvlad_layer.init_params(clst, traindescs)
model.add_module("netvlad_pool", netvlad_layer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.encode = nn.DataParallel(model.encode)
model.netvlad_pool = nn.DataParallel(model.netvlad_pool)
model.to(device)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=0.001,
                momentum=0.9,
                weight_decay=0.001)

criterion = nn.TripletMarginLoss(margin=10**0.5, 
                p=2, reduction='sum').to(device)

start_epoch = 0
epochs = 10
checkpoint = None
for i in range(1,21):
    if exists("checkpoints/epoch_{}".format(i)):
        checkpoint = torch.load("checkpoints/epoch_{}".format(i))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = i

file_object = open('loss.txt', 'a')
file_object.write("Start Training\n")
file_object.close()
for epoch in range(start_epoch, epochs):
    train(epoch)
    torch.save({
        'epoch' : epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "checkpoints/epoch_{}".format(epoch))
