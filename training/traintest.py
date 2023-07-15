from math import ceil
import shutil
from os.path import join, exists, realpath, split
from os import remove
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.models as models
import h5py
import faiss
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from utils.pca import PCA

from tensorboardX import SummaryWriter
import numpy as np
from datasets import tokyo247, pittsburgh, GSVDatasets, nordlands

from model import netvlad_ghost_norm, netvlad_ghost_context
import argparse
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 32
WARM_UP = 650
LR = 1e-3
DESCRIPTOR_DIM = 32768
EPOCHS = 20
LR_STEPS = [5,10,15]
LR_GAMMA = 0.3

#Hard-coded data path
TEST_PATH = ".../tokyo247"


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpu", type=int, required=False, help="number of gpu used in training")
    parser.add_argument("--off_the_shelf", type=bool, required=False, default=False, help="Whether to test the off the shelf model or not")
    return parser.parse_args()


def train(train_set, epoch, model, device):
    global global_step
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    training_data_loader = DataLoader(dataset=train_set, 
                batch_size=BATCH_SIZE, shuffle=False)
    
    nBatches = len(training_data_loader)
    for w in range (args.num_gpu):
        print('GPU: ', w, ' Allocated:', round(torch.cuda.memory_allocated(w)/1024**3,1), 'GB')
        print('GPU: ', w, ' Cached:', round(torch.cuda.memory_reserved(w)/1024**3,1), 'GB')

    model.train()
    for iteration, (images, labels) in enumerate(training_data_loader, startIter):
        BS, K, C, H, W = images.shape
        images = images.view(-1, C, H, W)
        labels = labels.view(-1).to(device)
        images = images.to(device)
        image_encoding = model.features(images)
        vlad_encoding, _ = model.pool(image_encoding)

        optimizer.zero_grad()
        
        hard_pairs = miner(vlad_encoding, labels)
        loss = criterion(vlad_encoding, labels, hard_pairs)

        loss.backward()
        
        del images, vlad_encoding, image_encoding, labels, hard_pairs,_
        #del query, positives, negatives

        batch_loss = loss.item()
        epoch_loss += batch_loss
        
        if global_step < WARM_UP:
            lr_scale = min(1., float(global_step + 1) / WARM_UP)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * LR
        
        optimizer.step()        
        global_step += 1

        del loss
        if iteration % 200 == 0:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                nBatches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * nBatches) + iteration)
            '''writer.add_scalar('Train/nNeg', nNeg, 
                    ((epoch-1) * nBatches) + iteration)'''
            for w in range (args.num_gpu):
                print('GPU: ', w, ' Allocated:', round(torch.cuda.memory_allocated(w)/1024**3,1), 'GB')
                print('GPU: ', w, ' Cached:', round(torch.cuda.memory_reserved(w)/1024**3,1), 'GB')

    startIter += len(training_data_loader)
    del training_data_loader
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    

def encode_database(dataset, model, device, whiten=False):
    #dataset = pittsburgh.WholeDataset("/export/home/iceicehyhy/dataset/Pitts30k/raw_data/datasets/pitts30k_test.mat", 
    #                                  input_transform=pittsburgh.input_transform())
    test_dataset_q = Subset(dataset, np.arange(dataset.dbStruct.numDb, len(dataset)))
    test_dataset_db = Subset(dataset, np.arange(dataset.dbStruct.numDb))
    
    data_loader_q = DataLoader(test_dataset_q, batch_size=32, shuffle=False)
    data_loader_db = DataLoader(test_dataset_db, batch_size=32, shuffle=False)
    
    #dbfeatures = np.empty((len(test_dataset_db), 512*64), dtype="float32")
    #qfeatures = np.empty((len(test_dataset_q), 512*64), dtype="float32")
    dbfeatures = torch.empty((len(test_dataset_db), 512*64), dtype=torch.float32)
    qfeatures = torch.empty((len(test_dataset_q), 512*64), dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        print("Extracting features database")
        for iteration, (images, indice) in enumerate(data_loader_db, 1):
            images = images.to(device)
            image_encoding = model.features(images)
            #(B, K, C, H, W)
            vlad_encoding, _ = model.pool(image_encoding)
            
                
            #dbfeatures[indice, :] = vlad_encoding.detach().cpu().numpy()
            dbfeatures[indice, :] = vlad_encoding.detach().cpu()
                
            
            if iteration % 100 == 0:
                print("====> Batch [{} / {}]".format(iteration, len(data_loader_db)))  
 
                
            del images, image_encoding, vlad_encoding, _
        
        print("Extracting features query")
        for iteration, (images, indice) in enumerate(data_loader_q, 1):
            images = images.to(device)
            image_encoding = model.features(images)
            #(B, K, C, H, W)
            vlad_encoding, _ = model.pool(image_encoding)
            
                
            #qfeatures[indice - dataset.dbStruct.numDb, :] = vlad_encoding.detach().cpu().numpy()
            qfeatures[indice - dataset.dbStruct.numDb, :] = vlad_encoding.detach().cpu()
                
                
            if iteration % 100 == 0:
                print("====> Batch [{} / {}]".format(iteration, len(data_loader_q)))
                
            del images, image_encoding, vlad_encoding, _
    print("Finishing encoding")
    torch.cuda.empty_cache()
    
    if whiten == True:
        pca = PCA(pca_n_components=4096, 
              pca_whitening=True,
              pca_parameters_path='./train_log/pca_params.h5',
              save = False)
        if len(dataset) >= 20000:
            print("Features number larger than 20000, randomly sampled 20000 features")
            index = np.random.choice(np.arange(0, len(dataset)), size = 20000, replace = False)
            pca.train(torch.cat([dbfeatures, qfeatures], dim=0)[index, :], device)
        else:
            pca.train(torch.cat([dbfeatures, qfeatures], dim=0), device)
        dbfeatures = pca.infer(dbfeatures.to(device))
        qfeatures = pca.infer(qfeatures.to(device))
    return np.ascontiguousarray(dbfeatures.detach().cpu().numpy()), np.ascontiguousarray(qfeatures.detach().cpu().numpy())

def test(epoch, dataset, model, device, write_board = False, whiten = False):
    #dataset = pittsburgh.get_whole_test_set("/root/hz_ws/pytorch-NetVlad/datasets/pitts30k/raw_data")
    dbFeat, qFeat = encode_database(dataset, model, device, whiten)

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(DESCRIPTOR_DIM)
    #faiss_index = faiss.IndexFlatIP(cfg.dsc_dim)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 

    # for each query get those within threshold distance
    gt = dataset.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / dataset.dbStruct.numQ
    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))  
        if write_board:      
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)    
    del dbFeat, qFeat
    return recalls

def is_best_recall(recalls, best_score):
    if recalls[1] > best_score[1]:
        return True
    elif recalls[1] < best_score[1]:
        return False
    else:
        if recalls[5] > best_score[5]:
            return True
        elif recalls[5] < best_score[5]:
            return False
        else:
            return recalls[10] > best_score[10]


def save_checkpoint(state, is_best, epoch_num, savePath, filename='checkpoint.pth.tar'):
    model_out_path = join(savePath, str(epoch_num)+filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))

if __name__ == "__main__":
    args = parse_argument()
    seed = 123
    device = torch.device("cuda")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading dataset(s)')
    # getting training set for clustering
    
    train_set = GSVDatasets.GSVCitiesDataset()
    print('====> Training query set:', len(train_set))   # each query, 1+, 10-

    test_set = tokyo247.get_whole_val_set(TEST_PATH)
    print(f'===> Evaluating on val set, {len(test_set)}')

    model = nn.Module()

    print('===> Building model: Backbone')
    encoder = models.vgg16(pretrained=True)
    layers = list(encoder.features.children())[:-2]

    for l in layers[:-5]:
        for p in l.parameters():
            p.requires_grad = False

    encoder = nn.Sequential(*layers)

    model.add_module("features", encoder)
    clsts = h5py.File("centroids.h5", "r")
    clsts_r = clsts['representative'][:].astype('float32')
    clsts_g = clsts['ghost'][:].astype('float32')
    sampled_feat = h5py.File("sampled_feat.h5", "r")['representative'][:10000].astype('float32')
    print("===> Building model: Pooling")


    net_vlad = netvlad_ghost_norm.NetVlad_Ghost(
        num_clusters=64, 
        dim=512, 
        normalize_input=True, 
        vladv2=False, 
        num_ghost_clusters = 6)
    net_vlad.init_params(clsts_r, clsts_g, sampled_feat)
        
    del clsts, clsts_r, clsts_g, sampled_feat

    model.add_module("pool", net_vlad)

    print('Training using {} GPUs!'.format(args.num_gpu))

    model.features = nn.DataParallel(model.features)
    model.pool = nn.DataParallel(model.pool)
    #model = nn.DataParallel(model)
    isParallel = True
    
    
    params = list(model.named_parameters())
    params_group = []
    params_group_name = {}
    params_group_name['train'] = []
    params_group_name['freeze'] = []
    
    freeze = {
        'features' : False,
        'pool.module.conv' : False,
        'parametric_norm' : True
    }
    
    for key in freeze.keys():
        if freeze.get(key):
            print("Freezing parameters {}".format(key))
            params_group.append({
                'params' : [par for name, par in params if key in name],
                'lr' : 0
            })
            params_group_name['freeze'].append([name for name, par in params if key in name])
        else:
            params_group.append({
                'params' : [par for name, par in params if key in name and par.requires_grad == True],
                'lr' : LR
            })
            
            params_group_name['train'].append([name for name, par in params if key in name and par.requires_grad == True])
            
    print("Training parameters ==>")
    print(params_group_name['train'])
    print("Freezing parameters ===>")
    print(params_group_name['freeze'])
        
    optimizer = optim.SGD(params_group, lr=LR,
                momentum=0.9,
                weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=LR_STEPS, 
                                               gamma=LR_GAMMA)
    
    criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=DotProductSimilarity())

    model = model.float()
    model = model.to(device)

    #Your pretrained weight and tensorboard weight
    resume = False
    ckpt_path = "..."
    tf_path = "..."
    print('===> Training model')
    if resume == True:
        writer = SummaryWriter(log_dir = tf_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
    else:
        writer = SummaryWriter(log_dir=join(tf_path, 
                                            datetime.now().strftime('%b%d_%H-%M-%S')+'_'+'Patch'+'_'+'GhostVladNorm'))
        start_epoch = 1
    logdir = writer.file_writer.get_logdir()
    print (logdir)
    not_improved = 0
    best_score = {'epoch':0, 1:0, 5:0, 10:0}
    global_step = 0
    
    whiten = False
        
    is_best = False
    recalls = best_score
    
    if args.off_the_shelf:
        # testing for off the shelf
        recalls = test(0, test_set, model, device, write_board=True, whiten = whiten)
        is_best = is_best_recall(recalls, best_score)
        if is_best:
            not_improved = 0
            best_score = {'epoch':0, 1:recalls[1], 5:recalls[5], 10:recalls[10]}
        else:
            not_improved += 1

        save_checkpoint({
                'epoch': 0,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'optimizer' : optimizer.state_dict(),
                'parallel' : isParallel,
        }, is_best, 0, "checkpoints/")
    
    for epoch in range(start_epoch, EPOCHS):

        train(train_set, epoch, model, device)
        scheduler.step()
        if (epoch % 1) == 0:
            # testing
            recalls = test(epoch, test_set, model, device, write_board=True, whiten = whiten)
            is_best = recalls[1] > best_score[1]
            if is_best:
                not_improved = 0
                best_score = {'epoch':epoch, 1:recalls[1], 5:recalls[5], 10:recalls[10]}
            else:
                not_improved += 1
                
            if not_improved > (10 / 1):
                print('Performance did not improve for', 10, 'epochs. Stopping.')
                break
                        
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'optimizer' : optimizer.state_dict(),
                'parallel' : isParallel,
        }, is_best, epoch, "checkpoints/")

            

    print("=> Best Epoch: {}".format(best_score['epoch']), flush=True)
    print("=> Best Recall@1: {:.4f}".format(best_score[1]), flush=True)
    print("=> Best Recall@5: {:.4f}".format(best_score[5]), flush=True)
    print("=> Best Recall@10: {:.4f}".format(best_score[10]), flush=True)
    writer.close()