import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
#import faiss
from math import ceil, sqrt,log
import numpy as np
import netvlad


class NetVlad_Ghost(netvlad.NetVLAD):
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
        super(NetVlad_Ghost, self).__init__(num_clusters=num_clusters, dim=dim, 
                 normalize_input=normalize_input, vladv2=vladv2, num_ghost_clusters = num_ghost_clusters)
        self.parametric_norm = netvlad.ParametricNorm(num_clusters)
    def init_params(self, clsts_r, clsts_g, traindescs_r):
        #TODO replace numpy ops with pytorch ops
        super(NetVlad_Ghost, self).init_params(clsts_r, clsts_g, traindescs_r)

    def forward(self, x):
        # input batch x channel, N X C X H X W, 48x512x30x40
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # 48 x 72 x 1200
        soft_assign = self.conv(x)

        # 1st soft-assignment,  48 x (64+8) x 1200
        soft_assign = F.softmax(soft_assign, dim=1)

        # 48 x 512 x 1200
        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            # 48 x 1 x 512 x 1200 - 1 x 512 x 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1).permute(1,0).unsqueeze(0).unsqueeze(0)
            # 1st soft-assignment
            residual *= soft_assign.view(N, self.num_clusters + self.num_ghost_clusters,  -1)[:,C:C+1,:].unsqueeze(2)

            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = self.parametric_norm(vlad)
        
        return vlad, soft_assign
        