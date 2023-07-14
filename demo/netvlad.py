import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from math import sqrt, floor

class ParametricNorm(nn.Module):
    def __init__(self, num_clusters) -> None:
        super().__init__()
        self.parametric_norm = nn.Parameter(torch.ones(num_clusters,1)/sqrt(num_clusters))
    
    def forward(self, vlad):
        N, K, C = vlad.shape
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad * self.parametric_norm
        vlad = vlad.view(N, -1)
        return vlad

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False, num_ghost_clusters = 4):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters  # 64
        self.num_ghost_clusters = num_ghost_clusters  # 8
        self.dim = 512#dim   # VGG 512
        self.alpha = 0
        self.vladv2 = vladv2
        # normalize input
        self.normalize_input = normalize_input
        # conv 2d layer, with (1,1) kernel size(1,1), input 128, output 64, bias = False
        self.conv = nn.Conv2d(dim, num_clusters + num_ghost_clusters , kernel_size=(1,1), bias=vladv2, stride=1)
        self.centroids = nn.Parameter(torch.rand(num_clusters + num_ghost_clusters, dim))
        #self.centroids_r = nn.Parameter(torch.rand(num_clusters, dim))
        #self.centroids_s = nn.Parameter(torch.rand(num_ghost_clusters, dim))
        print ("Ghost NetVLAD init done...")

    def init_params(self, clsts_r, clsts_s, traindescs_r):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            # normalization along feature_dim axis
            clsts_combined = np.concatenate((clsts_r, clsts_s), axis=0)
            clstsAssign = clsts_combined / np.linalg.norm(clsts_combined, axis=1, keepdims=True)
            print(clstsAssign.shape, clstsAssign[:self.num_clusters].shape)
            # dot product of two matrix
            dots = np.dot(clstsAssign[:self.num_clusters], traindescs_r.T)
            dots.sort(0)
            # in descending order
            dots = dots[::-1, :] # sort, descending
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item() 
            self.centroids = nn.Parameter(torch.from_numpy(clsts_combined))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
            
        else:
            clsts_combined = np.concatenate((clsts_r, clsts_s), axis=0)
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs_r)
            del traindescs_r
            # find 2 nearest neighbors of a point, return dist and index
            # dssq of each query and two nearest descriptors
            dsSq = np.square(knn.kneighbors(clsts_r, 2)[0])
            del knn
            # get the mean of dist of all the further point and the nearer point
            # alpha = 1538.135277, big enough
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            print("alpha is : ", self.alpha)
            
            # conv weight now 72x5 , 512, 1, 1
            self.centroids = nn.Parameter(torch.from_numpy(clsts_combined))
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            # 72x5
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        # input batch x channel, N X C X H X W, 48x512x30x40
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # 48 x 72 x 1200
        soft_assign = self.conv(x)

        # 1st soft-assignment,  48 x (64+8) x 1200
        soft_assign = F.softmax(soft_assign, dim=1)

        store_residual = torch.zeros([N, self.num_clusters, C, H, W], dtype=x.dtype, layout=x.layout, device=x.device)
        for j in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x.unsqueeze(0).permute(1, 0, 2, 3, 4) - \
                self.centroids[j:j + 1, :].expand(x.size(2), x.size(3), -1, -1).permute(2, 3, 0, 1).unsqueeze(0)

            residual *= soft_assign[:, j:j + 1, :].unsqueeze(2)  # residual should be size [N K C H W]
            store_residual[:, j:j + 1, :, :, :] = residual
        return store_residual, soft_assign