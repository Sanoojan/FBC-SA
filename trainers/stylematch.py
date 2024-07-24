import contextlib
import random
import os
import time
import datetime
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param
from dassl.modeling.ops import deactivate_mixstyle, run_with_mixstyle

from .adain.adain import AdaIN

from dassl.metrics import compute_accuracy
from dassl.modeling.ops.utils import sigmoid_rampup, ema_model_update
import copy
from timm.models.vision_transformer import Block

import matplotlib.pyplot as plt
import seaborn as sns

from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchprofile import profile_macs

@contextlib.contextmanager
def freeze_models_params(models):
    try:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(False)
        yield
    finally:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(True)


class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.temp = temp

    def forward(self, x, stochastic=True):
        mu = self.mu
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4)  # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu

        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.linear(x, weight)
        score = score / self.temp

        return score
    
class NormalClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x, stochastic=True):
        return self.linear(x)
# class NormalClassifier(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.linear = nn.Linear(num_features, num_classes)
#         self.count = 0

#     def forward(self, x, stochastic=True):
#         # Plot histograms for each set of weights
#         weights = self.linear.weight.data.cpu().numpy()
#         fig, axs = plt.subplots(nrows=1, ncols=weights.shape[0], figsize=(15, 5), sharey=True)
#         # fig, axs = plt.subplots(nrows=weights.shape[0], figsize=(8, 6), sharex=True)

#         for i in range(weights.shape[0]):
#             axs[i].hist(weights[i, :], bins=50, alpha=0.75, density=True)
#             axs[i].set_title(f'Weight Histogram - Neuron {i+1}')

#         plt.tight_layout()

#         # Save the plot as an image file (e.g., PNG)
#         plt.savefig(f'vis/weight_histograms_{self.count}.png')
#         print("Saved weight histograms to weight_histograms.png")
#         self.count += 1
#         return self.linear(x)

# class NormalClassifier(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.linear = nn.Linear(num_features, num_classes)
#         self.count = 0

#     def forward(self, x, stochastic=True):
#         # Plot histograms for each set of weights
#         weights = self.linear.weight.data.cpu().numpy()
#         fig, axs = plt.subplots(nrows=1, ncols=weights.shape[0], figsize=(21, 5), sharey=True)
#         # fig, axs = plt.subplots(nrows=weights.shape[0], figsize=(8, 6), sharex=True)

#         for i in range(weights.shape[0]):
#             sns.histplot(weights[i, :], stat='count', bins=50, kde=True, color='blue', ax=axs[i], element='poly', linewidth=0.5, alpha=0.25, line_kws={'linewidth': 2})
#             # axs[i].hist(weights[i, :], bins=50, alpha=0.75, density=True)
#             axs[i].set_title(f'Neuron {i+1}')

#         # plt.tight_layout()
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         # plt.title("Weight Histograms")
#         plt.suptitle('Weight Distributions for Each Class - FixMatch', fontsize=16)
#         # Save the plot as an image file (e.g., PNG)
#         plt.savefig(f'vis/weight_histograms_FixMatch.png')
        
#         print("Saved weight histograms to weight_histograms.png")
#         # self.count += 1
#         return self.linear(x)

# class NormalClassifier(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.linear = nn.Linear(num_features, num_classes)
#         self.count = 0

#     def forward(self, x, stochastic=True):
#         # Plot histograms for each set of weights
#         weights = self.linear.weight.data.cpu().numpy()
#         weights = self.w.data.cpu().numpy()
#         plt.figure(figsize=(10, 6))
#         sns.heatmap(weights, annot=False, cmap="coolwarm", xticklabels=False, yticklabels=False, cbar=False)
#         plt.title('Weight Matrix Heatmap for the Classifier')
#         plt.xlabel('Input Features')
#         plt.ylabel('Output Classes')

#         plt.savefig('weight_matrix_heatmap_base.png')
        
#         print("Saved weight histograms to weight_histograms.png")
#         # self.count += 1
#         plt.close()
#         return self.linear(x)

class HyperClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifier_PL(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False, mask=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator

            if mask:
                return torch.matmul(x, self.w_new.t()), w_modulator
            
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifier_PL2(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False, PL=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean) # 1x512
            # print("x1: ", x1.shape) 

            vf = x1.detach().clone()
            jp = x1.detach().clone()
            jn = x1.detach().clone()

            # zero out all the negative values in vf 

            jp[vf < 0] = 0

            # zero out all the positive values in jn
            jn[vf > 0] = 0


            x2 = self.h2(x_mean) # 1xC

            vc = x2.detach().clone()
            vc_p = x2.detach().clone()
            vc_n = x2.detach().clone()

            # zero out all the negative values in vc

            vc_p[vc < 0] = 0

            # zero out all the positive values in vc

            vc_n[vc > 0] = 0

            # print("x2: ", x2.shape) 
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator


            w_modulator_p = torch.sigmoid(torch.matmul(vc_p.t(), jp))

            w_modulator_n = torch.sigmoid(torch.matmul(vc_n.t(), jn))

            self.w_new_p = self.w * w_modulator_p

            self.w_new_n = self.w * w_modulator_n

            # for par in x1.parameters():
            #     print(par.grad)

            # print the gradient values for x1 


            if PL:
           

                return torch.matmul(x, self.w_new.t()), torch.matmul(x, self.w_new_p.t()), torch.matmul(x, self.w_new_n.t())


            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

# class HyperClassifier(nn.Module):
#     def __init__(self, num_features, num_classes, hypernet=False, noise=False):
#         super().__init__()
#         self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
#         # self.b = nn.Parameter(torch.Tensor(num_classes))

#         if hypernet:
#             self.h1 = nn.Linear(num_features, num_features)
#             self.h2 = nn.Linear(num_features, num_classes)

#         self.p1 = nn.Linear(num_features, num_features//2)
#         self.p2 = nn.Linear(num_features//2, num_features//4)
#         self.p3 = nn.Linear(num_features//4, num_features//8)

#         self.p4 = nn.Linear(num_features//4, num_features//2)
#         self.p5 = nn.Linear(num_features//2, num_features)

#         stdv = 1./math.sqrt(self.w.size(1))
#         self.w.data.uniform_(-stdv, stdv)
#         # self.b.data.uniform_(-stdv, stdv)


#     def forward(self, x, stochastic=True, hypernet=False, noise=False):
#         if hypernet:
#             x_mean = x.mean(0).unsqueeze(0)
#             # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

#             x_mean = torch.relu(self.p1(x_mean)) # 256
#             x_mean = torch.relu(self.p2(x_mean)) # 128
#             x_mean = torch.sigmoid(self.p3(x_mean)) # 64

#             if noise:
#                 noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
#                 # x_mean = self.mean_projection(x_mean)
#                 x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
#             else:
#                 x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

#             x_mean = torch.relu(self.p4(x_mean)) # 256
#             x_mean = torch.relu(self.p5(x_mean)) # 512

#             # print("x_mean: ", x_mean.shape)
#             x1 = self.h1(x_mean)
#             # print("x1: ", x1.shape)

#             x2 = self.h2(x_mean)
#             # print("x2: ", x2.shape)
#             w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
#             # print("w_modulator: ", w_modulator.shape)
#             self.w_new = self.w * w_modulator



#             # Plot histograms for each set of weights
#             weights = self.w_new.data.cpu().numpy()
#             fig, axs = plt.subplots(nrows=1, ncols=weights.shape[0], figsize=(21, 5), sharey=True)
#             # fig, axs = plt.subplots(nrows=weights.shape[0], figsize=(8, 6), sharex=True)

#             for i in range(weights.shape[0]):
#                 sns.histplot(weights[i, :], stat='count', bins=50, kde=True, color='blue', ax=axs[i], element='poly', linewidth=0.5, alpha=0.25, line_kws={'linewidth': 2})
#                 # axs[i].hist(weights[i, :], bins=50, alpha=0.75, density=True)
#                 axs[i].set_title(f'Neuron {i+1}')

#             # plt.tight_layout()
#             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#             # plt.title("Weight Histograms")
#             plt.suptitle('Weight Distributions for Each Class - Modulated (Photo)', fontsize=16)
#             # Save the plot as an image file (e.g., PNG)
#             plt.savefig(f'vis/weight_histograms_modulated_photo.png')
            
#             print("Saved weight histograms to weight_histograms.png")
#             # self.count += 1
#             plt.close()

#             return torch.matmul(x, self.w_new.t()) 
                                    
#         else:
#             # Plot histograms for each set of weights
#             weights = self.w.data.cpu().numpy()
#             fig, axs = plt.subplots(nrows=1, ncols=weights.shape[0], figsize=(21, 5), sharey=True)
#             # fig, axs = plt.subplots(nrows=weights.shape[0], figsize=(8, 6), sharex=True)

#             for i in range(weights.shape[0]):
#                 sns.histplot(weights[i, :], stat='count', bins=50, kde=True, color='blue', ax=axs[i], element='poly', linewidth=0.5, alpha=0.25, line_kws={'linewidth': 2})
#                 # axs[i].hist(weights[i, :], bins=50, alpha=0.75, density=True)
#                 axs[i].set_title(f'Neuron {i+1}')

#             # plt.tight_layout()
#             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#             # plt.title("Weight Histograms")
#             plt.suptitle('Weight Distributions for Each Class - FixMatch+Ours (Shared Classifier)', fontsize=16)
#             # Save the plot as an image file (e.g., PNG)
#             plt.savefig(f'vis/weight_histograms_Ours_common.png')
            
#             print("Saved weight histograms to weight_histograms.png")
#             # self.count += 1
#             plt.close()
#             return torch.matmul(x, self.w.t()) 
#             # return F.linear(x, self.w, self.b)

# class HyperClassifier(nn.Module):
#     def __init__(self, num_features, num_classes, hypernet=False, noise=False):
#         super().__init__()
#         self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
#         # self.b = nn.Parameter(torch.Tensor(num_classes))

#         if hypernet:
#             self.h1 = nn.Linear(num_features, num_features)
#             self.h2 = nn.Linear(num_features, num_classes)

#         self.p1 = nn.Linear(num_features, num_features//2)
#         self.p2 = nn.Linear(num_features//2, num_features//4)
#         self.p3 = nn.Linear(num_features//4, num_features//8)

#         self.p4 = nn.Linear(num_features//4, num_features//2)
#         self.p5 = nn.Linear(num_features//2, num_features)

#         stdv = 1./math.sqrt(self.w.size(1))
#         self.w.data.uniform_(-stdv, stdv)
#         # self.b.data.uniform_(-stdv, stdv)


#     def forward(self, x, stochastic=True, hypernet=False, noise=False):
#         if hypernet:
#             x_mean = x.mean(0).unsqueeze(0)
#             # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

#             x_mean = torch.relu(self.p1(x_mean)) # 256
#             x_mean = torch.relu(self.p2(x_mean)) # 128
#             x_mean = torch.sigmoid(self.p3(x_mean)) # 64

#             if noise:
#                 noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
#                 # x_mean = self.mean_projection(x_mean)
#                 x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
#             else:
#                 x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

#             x_mean = torch.relu(self.p4(x_mean)) # 256
#             x_mean = torch.relu(self.p5(x_mean)) # 512

#             # print("x_mean: ", x_mean.shape)
#             x1 = self.h1(x_mean)
#             # print("x1: ", x1.shape)

#             x2 = self.h2(x_mean)
#             # print("x2: ", x2.shape)
#             w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
#             # print("w_modulator: ", w_modulator.shape)
#             self.w_new = self.w * w_modulator



#             # Plot histograms for each set of weights
#             weights = self.w_new.data.cpu().numpy()
#             fig, axs = plt.subplots(nrows=1, ncols=weights.shape[0], figsize=(21, 5), sharey=True)
#             # fig, axs = plt.subplots(nrows=weights.shape[0], figsize=(8, 6), sharex=True)

#             for i in range(weights.shape[0]):
#                 sns.histplot(weights[i, :], stat='count', bins=50, kde=True, color='blue', ax=axs[i], element='poly', linewidth=0.5, alpha=0.25, line_kws={'linewidth': 2})
#                 # axs[i].hist(weights[i, :], bins=50, alpha=0.75, density=True)
#                 axs[i].set_title(f'Neuron {i+1}')

#             # plt.tight_layout()
#             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#             # plt.title("Weight Histograms")
#             plt.suptitle('Weight Distributions for Each Class - Modulated (Photo)', fontsize=16)
#             # Save the plot as an image file (e.g., PNG)
#             plt.savefig(f'vis/weight_histograms_modulated_photo.png')
            
#             print("Saved weight histograms to weight_histograms.png")
#             # self.count += 1
#             plt.close()

#             return torch.matmul(x, self.w_new.t()) 
                                    
#         else:
#             # Plot histograms for each set of weights
#             weights = self.w.data.cpu().numpy()
#             plt.figure(figsize=(10, 6))
#             sns.heatmap(weights, annot=False, cmap="coolwarm", xticklabels=False, yticklabels=False, cbar=False)
#             plt.title('Weight Matrix Heatmap for the Classifier')
#             plt.xlabel('Input Features')
#             plt.ylabel('Output Classes')

#             plt.savefig('weight_matrix_heatmap.png')
            
#             print("Saved weight histograms to weight_histograms.png")
#             # self.count += 1
#             plt.close()
#             return torch.matmul(x, self.w.t()) 
#             # return F.linear(x, self.w, self.b)

class D_Con_weight_mod(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*num_classes)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            x1 = self.h1(x_mean)
            w_modulator = torch.sigmoid(x1.reshape(self.w.shape[0], self.w.shape[1]))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 

class D_Con_weight_mod_dropout(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))
        self.dropout = nn.Dropout(0.2)

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*num_classes)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            x1 = self.h1(self.dropout(x_mean))
            w_modulator = torch.sigmoid(x1.reshape(self.w.shape[0], self.w.shape[1]))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
        
class D_Con_weight_mod_low_rank(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            x1 = self.h1(x_mean)
            x2 = self.h2(x_mean)

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 

class Seperate_HyperClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        # self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        # stdv = 1./math.sqrt(self.w.size(1))
        # self.w.data.uniform_(-stdv, stdv)
        


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))

            # print("w_modulator: ", w_modulator.shape)
            # print("w_modulator: ", w_modulator.shape)
            # self.w_new = self.w * w_modulator
            return w_modulator 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class simple_classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)

    def forward(self, x, w_modulator):
        w_new = self.w * w_modulator
        return torch.matmul(x, w_new.t())

class HyperClassifier_with_single_mlp(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            # x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(x1.reshape(self.w.shape[0], self.w.shape[1]))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifier_vit(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False, batch_size=16):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        self.embed_dim = 768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # print("cls_token_size: ", self.cls_token.shape)
        self.pos_embed = nn.Parameter(torch.randn(1, batch_size+1, self.embed_dim) * .02)
        self.norm = nn.LayerNorm(self.embed_dim)

        self.linear_proj = nn.Linear(num_features, self.embed_dim)

        if hypernet:
            self.h1 = nn.Linear(self.embed_dim, num_features)
            self.h2 = nn.Linear(self.embed_dim, num_classes)

        self.p1 = nn.Linear(self.embed_dim, self.embed_dim//2)
        self.p2 = nn.Linear(self.embed_dim//2, self.embed_dim//4)
        self.p3 = nn.Linear(self.embed_dim//4, self.embed_dim//8)

        self.p4 = nn.Linear(self.embed_dim//4, self.embed_dim//2)
        self.p5 = nn.Linear(self.embed_dim//2, self.embed_dim)

        self.domain_extractor = Block(dim=self.embed_dim, num_heads=12)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = self.linear_proj(x)
        
            x_mean = x_mean.unsqueeze(0)
            # print("x_mean: ", x_mean.shape)

            # cls_token = self.cls_token.expand(x.unsqueeze(0).shape[0], -1, -1)
            # print("cls_token: ", cls_token.shape)
            # print("x: ", x.unsqueeze(0).shape)

            x_mean = torch.cat((self.cls_token, x_mean), dim=1)
            # print("x_mean: ", x_mean.shape)
            # print("pos_embed: ", self.pos_embed.shape)

            x_mean = x_mean + self.pos_embed

            x_mean = self.domain_extractor(x_mean)
            x_mean = self.norm(x_mean)

            x_mean = x_mean[:, 0]

            # print("x_mean: ", x_mean.shape)

            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, self.embed_dim//8).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, self.embed_dim//8).to(x.device)), dim=1) # 128

            # print("x_mean: ", x_mean.shape)

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifier_Cov(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = x.mean(0).unsqueeze(0)
            covariance_mat = torch.matmul(x.t(), x)
            # print("covariance_mat: ", covariance_mat.shape)
            eigen_values, eigen_vectors = torch.linalg.eig(covariance_mat)
            # print("eigen_values: ", eigen_values)
            # print("eigen_values: ", eigen_vectors.mean(0).shape)
            # print(eigen_vectors[0].shape)
            x_mean = eigen_vectors[0].unsqueeze(0).real
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifier_Cov_mean(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = x.mean(0).unsqueeze(0)
            covariance_mat = torch.matmul(x.t(), x)
            # print("covariance_mat: ", covariance_mat.shape)
            eigen_values, eigen_vectors = torch.linalg.eig(covariance_mat)
            # print("eigen_values: ", eigen_values)
            # print("eigen_values: ", eigen_vectors.mean(0).shape)
            # print(eigen_vectors[0].shape)
            x_mean = eigen_vectors.mean(0).unsqueeze(0).real
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)


@TRAINER_REGISTRY.register()
class StyleMatch_Motiv(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.C(self.G(xu_k), stochastic=False)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            y_u_true = y_u_true.chunk(K)
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            # y_u_pred = torch.cat(y_u_pred, 0)
            # mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats_1 = self.assess_y_pred_quality(y_u_pred[0], y_u_true[0], mask_u[0])
            # y_u_pred_stats_2 = self.assess_y_pred_quality(y_u_pred[1], y_u_true[1], mask_u[1])
            # y_u_pred_stats_3 = self.assess_y_pred_quality(y_u_pred[2], y_u_true[2], mask_u[2])

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre_1"] = y_u_pred_stats_1["acc_thre"]
        loss_summary["y_u_pred_acc_raw_1"] = y_u_pred_stats_1["acc_raw"]
        loss_summary["y_u_pred_keep_rate_1"] = y_u_pred_stats_1["keep_rate"]

        # loss_summary["y_u_pred_acc_thre_2"] = y_u_pred_stats_2["acc_thre"]
        # loss_summary["y_u_pred_acc_raw_2"] = y_u_pred_stats_2["acc_raw"]
        # loss_summary["y_u_pred_keep_rate_2"] = y_u_pred_stats_2["keep_rate"]

        # loss_summary["y_u_pred_acc_thre_3"] = y_u_pred_stats_3["acc_thre"]
        # loss_summary["y_u_pred_acc_raw_3"] = y_u_pred_stats_3["acc_raw"]
        # loss_summary["y_u_pred_keep_rate_3"] = y_u_pred_stats_3["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    # def model_inference(self, input):
    #     features = self.G(input)

    #     if self.inference_mode == "deterministic":
    #         prediction = self.C(features, stochastic=False)

    #     elif self.inference_mode == "ensemble":
    #         prediction = 0
    #         for _ in range(self.n_ensemble):
    #             prediction += self.C(features, stochastic=True)
    #         prediction = prediction / self.n_ensemble

    #     else:
    #         raise NotImplementedError

    #     return prediction

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction


    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


@TRAINER_REGISTRY.register()
class StyleMatchCopy(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.C(self.G(xu_k), stochastic=False)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    # def model_inference(self, input):
    #     features = self.G(input)

    #     if self.inference_mode == "deterministic":
    #         prediction = self.C(features, stochastic=False)

    #     elif self.inference_mode == "ensemble":
    #         prediction = 0
    #         for _ in range(self.n_ensemble):
    #             prediction += self.C(features, stochastic=True)
    #         prediction = prediction / self.n_ensemble

    #     else:
    #         raise NotImplementedError

    #     return prediction

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction


    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        self.flops = 0

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)


        dummy_in = torch.randn(1, 512).to(self.device)
        macs = profile_macs(self.C, dummy_in)
        print(f"FLOPs: {macs*2}")
        # flops = FlopCountAnalysis(self.C, dummy_in)

        # # print(flop_count_table(flops))
        # total_flops = flops.total()
        # print(f"Total FLOPs: {total_flops}")

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.C(self.G(xu_k), stochastic=False)
                macs = profile_macs(self.G, xu_k)
                self.flops += 2*macs
                macs = profile_macs(self.C, self.G(xu_k))
                self.flops += 2*macs
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # print(xu_k.shape)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                # flops = FlopCountAnalysis(self.adain, (xu_k, xu_k2))
                # self.flops += flops.total()
                macs = profile_macs(self.adain, (xu_k, xu_k2))
                self.flops += 2*macs
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True)
            # flops = FlopCountAnalysis(self.G, x_k)
            # self.flops += flops.total()
            # flops = FlopCountAnalysis(self.C, self.G(x_k))
            # self.flops += flops.total()
            macs = profile_macs(self.G, x_k)
            self.flops += 2*macs
            macs = profile_macs(self.C, self.G(x_k))
            self.flops += 2*macs
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)
                # flops = FlopCountAnalysis(self.G, xu_k_aug)
                # self.flops += flops.total()
                # flops = FlopCountAnalysis(self.C, f_xu_k_aug)
                # self.flops += flops.total()
                macs = profile_macs(self.G, xu_k_aug)
                self.flops += 2*macs
                macs = profile_macs(self.C, f_xu_k_aug)
                self.flops += 2*macs
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                # flops = FlopCountAnalysis(self.G, xu_k_sty)
                # self.flops += flops.total()

                macs = profile_macs(self.G, xu_k_sty)
                self.flops += 2*macs

                macs = profile_macs(self.C, f_xu_k_sty)
                self.flops += 2*macs

                # flops = FlopCountAnalysis(self.C, f_xu_k_sty)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        print(f"FLOPs: {self.flops}")
        self.flops = 0
        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction


    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_PL(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_PL(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            masks = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                # z_xu_k, mask_k = self.C(f_xu_k, stochastic=False, hypernet=True, noise=True, mask=True)
                # masks.append(mask_k)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
                
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def before_epoch(self):
        train_loader_x_iter = iter(self.train_loader_u)

        masks = [[], [], []]


        # for self.batch_idx in range(len(self.train_loader_u)):
        for self.batch_idx in range(50):
            batch_u = next(train_loader_x_iter)
            u = batch_u["img"]
            u = u.to(self.device)

            K = self.num_source_domains

            u = u.chunk(K)

            with torch.no_grad():
                for k in range(K):
                    u_k = u[k]
                    f_u_k = self.G(u_k)
                    # f_u_k = f_u_k.detach().clone()
                    z_u_k, mask_k = self.C(f_u_k, stochastic=False, hypernet=True, noise=True, mask=True)
                    masks[k].append(mask_k-0.5)

                    # print(mask_k.shape)

        # print(len(masks[0]))
        # print(len(masks))

        # get a random integer 0-25
        r = random.randint(0, 24)

        m_selected = masks[0][r].flatten().unsqueeze(0)

        # print(m_selected.shape)

        same_dot = 0
        # same_dot=0

        for mask_i in masks[0]:
            # same_dot += torch.mm(m_selected, mask_i.flatten().t())
            same_dot += torch.mm(m_selected, mask_i.flatten().unsqueeze(0).t()).item()
        same_dot = same_dot / len(masks[0])


        print("Averaged dot product between masks of the same domain")
        print("----------------------------------------------------------")
        print(same_dot)

        # dot_12 = torch.zeros((7,7)).to(self.device)
        dot_12=0

        for mask_i in masks[1]:
            dot_12 += torch.mm(m_selected, mask_i.flatten().unsqueeze(0).t()).item()

        dot_12 = dot_12 / len(masks[1])

        print("Averaged dot product between masks of different domains (1-2)")
        print("----------------------------------------------------------")

        print(dot_12)

        # dot_13 = torch.zeros((7,7)).to(self.device)
        dot_13=0

        for mask_i in masks[2]:
            dot_13 += torch.mm(m_selected, mask_i.flatten().unsqueeze(0).t()).item()

        dot_13 = dot_13 / len(masks[2])

        print("Averaged dot product between masks of different domains (1-3)")
        print("----------------------------------------------------------")

        print(dot_13)




        # masks = [torch.stack(masks[k], 0) for k in range(K)]

        # print(masks[0].shape)
        # print(torch.mean(masks[0], 0).shape)


            


    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_PL2(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_PL2(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            p_xu_p = []
            p_xu_n = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)

                # p_xu_k_p = F.softmax(z_xu_k_p, 1)
                # p_xu_p.append(p_xu_k_p)

                # p_xu_k_n = F.softmax(z_xu_k_n, 1)
                # p_xu_n.append(p_xu_k_n)

            p_xu = torch.cat(p_xu, 0)
            # p_xu_p = torch.cat(p_xu_p, 0)
            # p_xu_n = torch.cat(p_xu_n, 0)



            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            # p_xu_maxval_p, y_xu_pred_p = p_xu_p.max(1)
            # mask_xu_p = (p_xu_maxval_p >= self.conf_thre).float()

            # p_xu_maxval_n, y_xu_pred_n = p_xu_n.max(1)
            # mask_xu_n = (p_xu_maxval_n >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # y_xu_pred_p = y_xu_pred_p.chunk(K)
            # mask_xu_p = mask_xu_p.chunk(K)

            # y_xu_pred_n = y_xu_pred_n.chunk(K)
            # mask_xu_n = mask_xu_n.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

            # Calculate pseudo-label's accuracy for p
            y_u_pred_p = []
            mask_u_p = []
            
        #     for y_xu_k_pred_p, mask_xu_k in zip(y_xu_pred_p, mask_xu):
        #         y_u_pred_p.append(
        #             y_xu_k_pred_p.chunk(2)[1]
        #         )  # only take the 2nd half (unlabeled data)
        #         mask_u_p.append(mask_xu_k.chunk(2)[1])

        #     y_u_pred_p = torch.cat(y_u_pred_p, 0)
        #     mask_u_p = torch.cat(mask_u_p, 0)
        #     y_u_pred_stats_p = self.assess_y_pred_quality(y_u_pred_p, y_u_true, mask_u_p)

        #     # Calculate pseudo-label's accuracy for n
        #     y_u_pred_n = []
        #     mask_u_n = []

        #     for y_xu_k_pred_n, mask_xu_k in zip(y_xu_pred_n, mask_xu):
        #         y_u_pred_n.append(
        #             y_xu_k_pred_n.chunk(2)[1]
        #         )  # only take the 2nd half (unlabeled data)
        #         mask_u_n.append(mask_xu_k.chunk(2)[1])
        #     y_u_pred_n = torch.cat(y_u_pred_n, 0)
        #     mask_u_n = torch.cat(mask_u_n, 0)
        #     y_u_pred_stats_n = self.assess_y_pred_quality(y_u_pred_n, y_u_true, mask_u_n)

        # ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug , z_xu_k_aug_p, z_xu_k_aug_n = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True, PL=True)

                z_xu_k_aug.requires_grad = True

                print(z_xu_k_aug.grad)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        # loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # loss_summary["y_u_pred_acc_thre_p"] = y_u_pred_stats_p["acc_thre"]
        # loss_summary["y_u_pred_acc_raw_p"] = y_u_pred_stats_p["acc_raw"]
        # # loss_summary["y_u_pred_keep_rate_p"] = y_u_pred_stats_p["keep_rate"]

        # loss_summary["y_u_pred_acc_thre_n"] = y_u_pred_stats_n["acc_thre"]
        # loss_summary["y_u_pred_acc_raw_n"] = y_u_pred_stats_n["acc_raw"]
        # # loss_summary["y_u_pred_keep_rate_n"] = y_u_pred_stats_n["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Motiv(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            y_u_true = y_u_true.chunk(K)
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            # y_u_pred = torch.cat(y_u_pred, 0)
            # mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats_1 = self.assess_y_pred_quality(y_u_pred[0], y_u_true[0], mask_u[0])
            # y_u_pred_stats_2 = self.assess_y_pred_quality(y_u_pred[1], y_u_true[1], mask_u[1])
            # y_u_pred_stats_3 = self.assess_y_pred_quality(y_u_pred[2], y_u_true[2], mask_u[2])

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre_1"] = y_u_pred_stats_1["acc_thre"]
        loss_summary["y_u_pred_acc_raw_1"] = y_u_pred_stats_1["acc_raw"]
        loss_summary["y_u_pred_keep_rate_1"] = y_u_pred_stats_1["keep_rate"]

        # loss_summary["y_u_pred_acc_thre_2"] = y_u_pred_stats_2["acc_thre"]
        # loss_summary["y_u_pred_acc_raw_2"] = y_u_pred_stats_2["acc_raw"]
        # loss_summary["y_u_pred_keep_rate_2"] = y_u_pred_stats_2["keep_rate"]

        # loss_summary["y_u_pred_acc_thre_3"] = y_u_pred_stats_3["acc_thre"]
        # loss_summary["y_u_pred_acc_raw_3"] = y_u_pred_stats_3["acc_raw"]
        # loss_summary["y_u_pred_keep_rate_3"] = y_u_pred_stats_3["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class EntMin_Ours(TrainerXU):
    """Entropy Minimization.

    http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.ENTMIN.LMDA

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier(self.G.fdim, self.num_classes, hypernet=True, noise=True)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)

        output_x = self.C(self.G(input_x))
        loss_x = F.cross_entropy(output_x, label_x)

        output_u = F.softmax(self.C(self.G(input_u)), 1)
        loss_u = (-output_u * torch.log(output_u + 1e-5)).sum(1).mean()

        loss = loss_x + loss_u * self.lmda

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(output_x, label_x)[0].item(),
            "loss_u": loss_u.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.C(self.G(input), stochastic=False, hypernet=True, noise=False)
    
@TRAINER_REGISTRY.register()
class MeanTeacher_Ours(TrainerXU):
    """Mean teacher.

    https://arxiv.org/abs/1703.01780.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.MEANTEACHER.WEIGHT_U
        self.ema_alpha = cfg.TRAINER.MEANTEACHER.EMA_ALPHA
        self.rampup = cfg.TRAINER.MEANTEACHER.RAMPUP

        self.teacher_G = copy.deepcopy(self.G)
        self.teacher_C = copy.deepcopy(self.C)
        self.teacher_G.train()
        self.teacher_C.train()
        for param in self.teacher_G.parameters():
            param.requires_grad_(False)
        for param in self.teacher_C.parameters():
            param.requires_grad_(False)

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier(self.G.fdim, self.num_classes, hypernet=True, noise=True)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)


    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)

        logit_x = self.C(self.G(input_x))
        loss_x = F.cross_entropy(logit_x, label_x)

        target_u = F.softmax(self.teacher_C(self.teacher_G(input_u)), 1)
        prob_u = F.softmax(self.C(self.G(input_u)) , 1)
        loss_u = ((prob_u - target_u)**2).sum(1).mean()

        weight_u = self.weight_u * sigmoid_rampup(self.epoch, self.rampup)
        loss = loss_x + loss_u*weight_u
        self.model_backward_and_update(loss)

        global_step = self.batch_idx + self.epoch * self.num_batches
        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        ema_model_update(self.G, self.teacher_G, ema_alpha)
        ema_model_update(self.C, self.teacher_C, ema_alpha)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(logit_x, label_x)[0].item(),
            "loss_u": loss_u.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.C(self.G(input), stochastic=False, hypernet=True, noise=False)
    
@TRAINER_REGISTRY.register()
class ablation_DClass(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        # for param in self.G.parameters():
        #     param.requires_grad_(False)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Domian Cs")
        self.D = []
        self.optim_D = []
        self.sched_D = []
        
        for i in range(self.num_source_domains):
            if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
                self.D.append(StochasticClassifier(self.G.fdim, self.num_classes))
            else:
                self.D.append(NormalClassifier(self.G.fdim, self.num_classes))
            self.D[i].to(self.device)
            print("# params: {:,}".format(count_num_param(self.D[i])))
            self.optim_D.append(build_optimizer(self.D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.sched_D.append(build_lr_scheduler(self.optim_D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.register_model("D"+str(i), self.D[i], self.optim_D[i], self.sched_D[i])

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.D[k](self.G(xu_k), stochastic=False)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_D = [0 for i in range(K)]
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            # z_x_k = self.C(self.G(x_k), stochastic=True)
            f_x_k = self.G(x_k)
            z_x_k =  self.C(f_x_k, stochastic=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

            z_x_k_D = self.D[k](f_x_k, stochastic=True)
            loss_x_D[k] += F.cross_entropy(z_x_k_D, y_x_k_true)

            

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)

                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                z_xu_k_aug_D = self.D[k](f_xu_k_aug, stochastic=True)
                loss_D = F.cross_entropy(z_xu_k_aug_D, y_xu_k_pred, reduction="none")
                loss_D = (loss_D * mask_xu_k).mean()
                loss_x_D[k] += loss_D

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

                z_xu_k_sty_D = self.D[k](f_xu_k_sty, stochastic=True)
                loss_D = F.cross_entropy(z_xu_k_sty_D, y_xu_k_pred, reduction="none")
                loss_D = (loss_D * mask_xu_k).mean()
                loss_x_D[k] += loss_D

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        # self.model_backward_and_update(loss_all)


        for k in range(K):  
            self.optim_D[k].zero_grad()
            loss_x_D[k].backward(retain_graph=True)
            self.optim_D[k].step()
        
        self.optim_G.zero_grad()
        self.optim_C.zero_grad()

        loss_all.backward()

        self.optim_G.step()
        self.optim_C.step()

        for k in range(K):
            loss_summary["loss_x_D"+str(k)] = loss_x_D[k].item()



        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

    def before_epoch(self):
        train_loader_x_iter = iter(self.train_loader_x)
        total_x = []
        total_y = []
        total_d = []
        for self.batch_idx in range(len(self.train_loader_x)):
            batch_x = next(train_loader_x_iter)

            input_x = batch_x["img0"]
            label_x = batch_x["label"]
            domain_x = batch_x["domain"]

            total_x.append(input_x)
            total_y.append(label_x)
            total_d.append(domain_x)

        x = torch.cat(total_x, dim=0)
        y = torch.cat(total_y, dim=0)
        d = torch.cat(total_d, dim=0)
        
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        global_feat = []

        for i in range(K):
            idx = d == i
            imgs = x[idx]
            labels = y[idx]
            with torch.no_grad():
                z_imgs = F.normalize(self.G(imgs.to(self.device)), p=2., dim=1)

            f = []
            for j in range(self.num_classes):
                idx = labels == j
                z = z_imgs[idx]
                f.append(z.mean(dim=0))
            feat = torch.stack(f)
            global_feat.append(feat)

        self.feat = torch.cat(global_feat, dim=0).chunk(K)

@TRAINER_REGISTRY.register()
class ablation_D_Con_weight_mod(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = D_Con_weight_mod(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class ablation_D_Con_weight_mod_low_rank(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = D_Con_weight_mod_low_rank(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_without_noise(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=False)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Seperate_HyperClassifier(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"


    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.w = simple_classifier(self.G.fdim, self.num_classes)
        print("# params: {:,}".format(count_num_param(self.w)))
 
        self.w.to(self.device)

        self.optim_C = build_optimizer(self.w, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.w, self.optim_C, self.sched_C)

        print("Building HyperClassifier")
        self.D = []
        self.optim_D = []
        self.sched_D = []

        for i in range(self.num_source_domains):
            self.D.append(Seperate_HyperClassifier(self.G.fdim, self.num_classes, hypernet=True, noise=False))
            self.D[i].to(self.device)
            print("# params: {:,}".format(count_num_param(self.D[i])))
            self.optim_D.append(build_optimizer(self.D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.sched_D.append(build_lr_scheduler(self.optim_D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.register_model("D"+str(i), self.D[i], self.optim_D[i], self.sched_D[i])


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                w_modulator = self.D[k](f_xu_k, stochastic=False, hypernet=True)
                # print(w_modulator.shape)
                # w_new = self.w * w_modulator
                # z_xu_k = torch.matmul(f_xu_k, w_new.t())
                z_xu_k = self.w(f_xu_k, w_modulator)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            x_w_modulator = self.D[k](self.G(x_k), stochastic=True, hypernet=True, noise=True)
            # x_w_new = self.w * x_w_modulator
            # z_x_k = torch.matmul(self.G(x_k), x_w_new.t())
            z_x_k = self.w(self.G(x_k), x_w_modulator)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                u_w_modulator = self.D[k](f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                # u_w_new = self.w * u_w_modulator
                # z_xu_k_aug = torch.matmul(f_xu_k_aug, u_w_new.t())
                z_xu_k_aug = self.w(f_xu_k_aug, u_w_modulator)

                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                sty_w_modulator = self.D[k](f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                # sty_w_new = self.w * sty_w_modulator
                # z_xu_k_sty = torch.matmul(f_xu_k_sty, sty_w_new.t())
                z_xu_k_sty = self.w(f_xu_k_sty, sty_w_modulator)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

     

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        w_modulator = torch.ones(self.num_classes, self.G.fdim).to(self.device)

        if self.inference_mode == "deterministic":
            # prediction = self.C(features, stochastic=False)
            prediction = self.w(features, w_modulator)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_with_single_mlp(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_with_single_mlp(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=False)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class  ablation_D_Con_weight_mod_dropout(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = D_Con_weight_mod_dropout(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=False)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_with_single_mlp_noise(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_with_single_mlp(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_ViT(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE
        self.batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE // self.num_source_domains

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_vit(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                # xu_k = torch.cat([x_k, u_k], 0)
                f_x_k = self.G(x_k)
                f_u_k = self.G(u_k)
                z_x_k = self.C(f_x_k, stochastic=False, hypernet=True)
                z_u_k = self.C(f_u_k, stochastic=False, hypernet=True)
                z_xu_k = torch.cat([z_x_k, z_u_k], 0)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                # xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                # f_xu_k_aug = self.G(xu_k_aug)
                # z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                f_x_k_aug = self.G(x_k_aug)
                f_u_k_aug = self.G(u_k_aug)
                z_x_k_aug = self.C(f_x_k_aug, stochastic=True, hypernet=True, noise=True)
                z_u_k_aug = self.C(f_u_k_aug, stochastic=True, hypernet=True, noise=True)
                z_xu_k_aug = torch.cat([z_x_k_aug, z_u_k_aug], 0)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k].chunk(2)
                x_k_sty = xu_k_sty[0]
                u_k_sty = xu_k_sty[1]

                f_x_k_sty = self.G(x_k_sty)
                f_u_k_sty = self.G(u_k_sty)
                z_x_k_sty = self.C(f_x_k_sty, stochastic=True, hypernet=True, noise=False)
                z_u_k_sty = self.C(f_u_k_sty, stochastic=True, hypernet=True, noise=False)
                z_xu_k_sty = torch.cat([z_x_k_sty, z_u_k_sty], 0)

                # f_xu_k_sty = self.G(xu_k_sty)
                # z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Cov_principle(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Cov(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Cov_mean(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Cov_mean(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_with_Domain(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x_d.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_with_Domain(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building Domain_G")
        self.Domain_G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.Domain_G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_G)))
        self.optim_Domain_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_Domain_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("Domain_G", self.Domain_G, self.optim_Domain_G, self.sched_Domain_G)

        print("Building C")
        self.C = HyperClassifier_with_Domain(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.Domain_G.fdim, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Domain_G(xu_k)
                z_xu_k = self.C(f_xu_k,d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            with torch.no_grad():
                d_x_k = self.Domain_G(x_k)
            # print(y_x_k_true.shape)
            z_x_k = self.C(self.G(x_k), d_x_k,  stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        all_images = []
        d_labels = []
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                all_images.append(xu_k_aug)
                d_labels.append(torch.tensor([k]*xu_k_aug.shape[0]).to(self.device))
                f_xu_k_aug = self.G(xu_k_aug)
                with torch.no_grad():
                    d_xu_k_aug = self.Domain_G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                with torch.no_grad():
                    d_xu_k_sty = self.Domain_G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, d_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        all_images = torch.cat(all_images, 0)
        d_labels = torch.cat(d_labels, 0)

        # shuffle both images and labels
        idx = torch.randperm(all_images.shape[0])
        all_images = all_images[idx]
        d_labels = d_labels[idx]

        # train domain classifier
        d_logits = self.Domain_C(self.Domain_G(all_images))
        loss_d = F.cross_entropy(d_logits, d_labels)


        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_d
        loss_summary["loss_d"] = loss_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)
        d_features = self.Domain_G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, d_features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Domain_MLP(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_with_Domain_MLP(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_with_Domain(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.Domain_G.fdim, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            # print(y_x_k_true.shape)
            z_x_k = self.C(self.G(x_k),  stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        all_images = []
        d_labels = []
        means = []
        means_d = [i for i in range(K)]
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)

                all_images.append(xu_k_aug)
                d_labels.append(torch.tensor([k]*xu_k_aug.shape[0]).to(self.device))
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        all_images = torch.cat(all_images, 0)
        d_labels = torch.cat(d_labels, 0)

        # shuffle both images and labels
        idx = torch.randperm(all_images.shape[0])
        all_images = all_images[idx]
        d_labels = d_labels[idx]

        # train domain classifier
        d_logits = self.Domain_C(self.G(all_images))
        loss_d = F.cross_entropy(d_logits, d_labels)


        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_d
        loss_summary["loss_d"] = loss_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)


        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))


        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)


        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = torch.sigmoid(x_d) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifier_Proj3(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)
        self.p3 = nn.Linear(num_features//8, num_features//4)
        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            # x_mean = torch.relu(self.p1(x_mean)) # 256
            # x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(x_d) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                # x_mean = torch.cat((x_mean, noise), dim=1) # 128
                x_mean = x_mean + noise
            
            # else:
            #     x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p3(x_mean)) # 256
            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device))

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_aug_d
        loss_summary["loss_u_aug_d"] = loss_u_aug_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_add_3l(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//8, num_features//4)
        self.p5 = nn.Linear(num_features//4, num_features//2)
        self.p6 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.relu(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                # x_mean = torch.cat((x_mean, noise), dim=1) # 128
                x_mean = x_mean + noise

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512
            x_mean = torch.relu(self.p6(x_mean)) # 1024

            # print("x_mean: ", x_mean.shape)

            x1 = self.h1(x_mean)
            x2 = self.h2(x_mean)

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))

            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_add_3l(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_add_3l(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_add_2l(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//8, num_features//4)
        self.p5 = nn.Linear(num_features//4, num_features//2)
        self.p6 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            # x_mean = torch.relu(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/4)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                # x_mean = torch.cat((x_mean, noise), dim=1) # 128
                x_mean = x_mean + noise

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512
            x_mean = torch.relu(self.p6(x_mean)) # 1024

            # print("x_mean: ", x_mean.shape)

            x1 = self.h1(x_mean)
            x2 = self.h2(x_mean)

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))

            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_add_2l(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_add_2l(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_diff(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            x_mean0 = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
                
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128



            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512


            with torch.no_grad():
                x_mean0 = torch.relu(self.p4(x_mean0)) # 256
                x_mean0 = torch.relu(self.p5(x_mean0)) # 512
                x10 = self.h1(x_mean0)
                x20 = self.h2(x_mean0)
                w_modulator0 = torch.sigmoid(torch.matmul(x20.t(), x10))
                self.w_new0 = self.w * w_modulator0

                w_mse = F.mse_loss(self.w_new0, self.w)
            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator

            with torch.no_grad():
                mse = F.mse_loss(x_mean, x.mean(0).unsqueeze(0))
                mse0 = F.mse_loss(x_mean0, x.mean(0).unsqueeze(0))

            return torch.matmul(x, self.w_new.t()) , mse, mse0, w_mse
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_diff(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_diff(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            loss_mse = 0
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k, mse,_,_ = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                loss_mse += mse
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_mse = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k, mse_x_k, _,_ = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            loss_x_mse += mse_x_k

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_mse = 0
        loss_u_mse0 = 0
        w_mse = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug, loss_u_mse_k , loss_u_mse_k0 , wmse = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss
                loss_u_mse += loss_u_mse_k
                loss_u_mse0 += loss_u_mse_k0
                w_mse += wmse

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty, _ , _, _= self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # loss_summary["loss_x_mse"] = loss_x_mse.item()  
        loss_summary["loss_u_mse_strong"] = loss_u_mse.item()
        # loss_summary["loss_u_mse_weak"] = loss_mse.item()
        loss_summary["loss_u_mse_strong0"] = loss_u_mse0.item()
        loss_summary["loss_w_mse"] = w_mse.item()


        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_capacity(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_features))
            self.h2 = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_classes))

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//4, num_features//2)
        # self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            # x_mean = torch.relu(self.p1(x_mean)) # 256
            # x_mean = torch.relu(self.p2(x_mean)) # 128
            # x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            # if noise:
            #     noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
            #     # x_mean = self.mean_projection(x_mean)
            #     x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            # else:
            #     x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            # x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Capacity(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_capacity(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.fc_mu = nn.Linear(input_dim//4, input_dim//8)
        self.fc_logvar = nn.Linear(input_dim//4, input_dim//8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(output_dim//8, output_dim//4)
        self.fc2 = nn.Linear(output_dim//4, output_dim//2)
        self.fc3 = nn.Linear(output_dim//2, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        x_recon = F.relu(self.fc3(z))
        return x_recon

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        # recon, kl = vae_loss(x_recon, x, mu, logvar)
        return x_recon, mu, logvar
    
class HyperClassifier_VAE(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.vae = VAE(num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)

    def vae_loss(self, x_recon, x, mu, logvar):
        # recon_loss = F.mse(x_recon, x)
        mse = F.mse_loss(x_recon, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (mse , kl_divergence)

    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean, mu, logvar = self.vae(x_mean)

            recon, kl = self.vae_loss(x_mean, x.mean(0).unsqueeze(0), mu, logvar)

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()), recon, kl
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_VAE(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_VAE(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k, _ ,_ = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_recon = 0
        loss_x_kl = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            # z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            z_x_k, recon, kl = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            loss_x_recon += recon
            loss_x_kl += kl

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_recon = 0
        loss_u_aug_kl = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                # z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                z_xu_k_aug, recon, kl = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss
                loss_u_aug_recon += recon
                loss_u_aug_kl += kl

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        loss_all += loss_x_recon
        loss_summary["loss_x_recon"] = loss_x_recon.item()

        loss_all += loss_x_kl
        loss_summary["loss_x_kl"] = loss_x_kl.item()

        if self.apply_aug:
            loss_all += loss_u_aug_recon
            loss_summary["loss_u_aug_recon"] = loss_u_aug_recon.item()

        if self.apply_aug:
            loss_all += loss_u_aug_kl
            loss_summary["loss_u_aug_kl"] = loss_u_aug_kl.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj2(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        xu_d = torch.cat([x_d, u_d], 0)
        y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        idx = torch.randperm(xu_d.shape[0])
        xu_d_shuf = xu_d[idx]
        y_xu_d_shuf = y_xu_d[idx]

        with torch.no_grad():
            f_xu_d = self.G(xu_d_shuf)
        d_xu_d = self.Proj(f_xu_d)
        d_z_xu_d = self.Domain_C(d_xu_d)
        loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj3(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj3(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        xu_d = torch.cat([x_d, u_d], 0)
        y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        idx = torch.randperm(xu_d.shape[0])
        xu_d_shuf = xu_d[idx]
        y_xu_d_shuf = y_xu_d[idx]

        with torch.no_grad():
            f_xu_d = self.G(xu_d_shuf)
        d_xu_d = self.Proj(f_xu_d)
        d_z_xu_d = self.Domain_C(d_xu_d)
        loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj4(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj5(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


class HyperClassifier_Proj20(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h = nn.Linear(num_features, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = torch.sigmoid(x_d) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                x_mean = torch.cat((x_mean, noise), dim=1) 
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # x1 = self.h1(x_mean)
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            w_modulator = torch.sigmoid(self.h(x_mean).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj10(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj20(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj21(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h = nn.Linear(num_features//4, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//4, num_features//2)
        # self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = torch.sigmoid(x_d) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                x_mean = torch.cat((x_mean, noise), dim=1) 
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            # x_mean = torch.relu(self.p5(x_mean)) # 512

            # x1 = self.h1(x_mean)
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            w_modulator = torch.sigmoid(self.h(x_mean).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj11(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj21(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj22(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h = nn.Linear(num_features//8, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//4, num_features//2)
        # self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = tor(x_d) # 64

            # if noise:
            #     noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
            #     x_mean = torch.cat((x_mean, noise), dim=1) 
            
            # else:
            #     x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            # x_mean = torch.relu(self.p5(x_mean)) # 512

            # x1 = self.h1(x_mean)
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            w_modulator = torch.sigmoid(self.h(x_d).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj12(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj22(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj23(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            # self.h = nn.Linear(num_features//8, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)
            self.h1 = nn.Sequential(nn.Linear(num_features//8, num_features//4), nn.ReLU(), nn.Linear(num_features//4, num_features//2), nn.ReLU(), nn.Linear(num_features//2, num_features), nn.ReLU())
            self.h2 = nn.Sequential(nn.Linear(num_features//8, num_features//4), nn.ReLU(), nn.Linear(num_features//4, num_features//2), nn.ReLU(), nn.Linear(num_features//2, num_classes), nn.ReLU())

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//4, num_features//2)
        # self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = tor(x_d) # 64

            # if noise:
            #     noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
            #     x_mean = torch.cat((x_mean, noise), dim=1) 
            
            # else:
            #     x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            # x_mean = torch.relu(self.p5(x_mean)) # 512

            x1 = self.h1(torch.sigmoid(x_d)) # 256

            x2 = self.h2(torch.sigmoid(x_d)) # 512

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # w_modulator = torch.sigmoid(self.h(x_d).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj13(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj23(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj24(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            # self.h = nn.Linear(num_features//8, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)
            self.h1 = nn.Sequential(nn.Linear(num_features//8, num_features//4), nn.ReLU(), nn.Linear(num_features//4, num_features), nn.ReLU())
            self.h2 = nn.Sequential(nn.Linear(num_features//8, num_features//4), nn.ReLU(), nn.Linear(num_features//4, num_classes), nn.ReLU())

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//4, num_features//2)
        # self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = tor(x_d) # 64

            # if noise:
            #     noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
            #     x_mean = torch.cat((x_mean, noise), dim=1) 
            
            # else:
            #     x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            # x_mean = torch.relu(self.p5(x_mean)) # 512


            x1 = self.h1(torch.sigmoid(x_d)) # 256

            x2 = self.h2(torch.sigmoid(x_d)) # 512

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # w_modulator = torch.sigmoid(self.h(x_d).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj14(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj24(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


class HyperClassifier_Proj25(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            # self.h = nn.Linear(num_features//8, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)
            self.h1 = nn.Sequential(nn.Linear(num_features//8, num_features), nn.ReLU())
            self.h2 = nn.Sequential(nn.Linear(num_features//8, num_classes), nn.ReLU())

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        # self.p4 = nn.Linear(num_features//4, num_features//2)
        # self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            # x_mean = tor(x_d) # 64

            # if noise:
            #     noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
            #     x_mean = torch.cat((x_mean, noise), dim=1) 
            
            # else:
            #     x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            # x_mean = torch.relu(self.p4(x_mean)) # 256
            # x_mean = torch.relu(self.p5(x_mean)) # 512


            x1 = self.h1(torch.sigmoid(x_d)) # 256

            x2 = self.h2(torch.sigmoid(x_d)) # 512

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # w_modulator = torch.sigmoid(self.h(x_d).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj15(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj25(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj26(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            # self.h = nn.Linear(num_features//8, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)
            self.h1 = nn.Sequential(nn.Linear(num_features//2, num_features), nn.ReLU())
            self.h2 = nn.Sequential(nn.Linear(num_features//2, num_classes), nn.ReLU())

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//8, num_features//4)
        self.p5 = nn.Linear(num_features//4, num_features//2)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_d = torch.relu(self.p4(torch.relu(x_d))) # 256
            x_d = torch.relu(self.p5(x_d)) # 512


            x1 = self.h1(x_d) # 256

            x2 = self.h2(x_d) # 512

            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # w_modulator = torch.sigmoid(self.h(x_d).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj16(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj26(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifier_Proj27(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h = nn.Linear(num_features//2, num_features*num_classes)
            # self.h2 = nn.Linear(num_features, num_classes)
            # self.h1 = nn.Sequential(nn.Linear(num_features//2, num_features), nn.ReLU())
            # self.h2 = nn.Sequential(nn.Linear(num_features//2, num_classes), nn.ReLU())

        # self.p1 = nn.Linear(num_features, num_features//2)
        # self.p2 = nn.Linear(num_features//2, num_features//4)
        # self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//8, num_features//4)
        self.p5 = nn.Linear(num_features//4, num_features//2)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_d, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_d = torch.relu(self.p4(torch.relu(x_d))) # 256
            x_d = torch.relu(self.p5(x_d)) # 512


            # x1 = self.h1(x_d) # 256

            # x2 = self.h2(x_d) # 512

            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # x2 = self.h2(x_mean)
            # w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            w_modulator = torch.sigmoid(self.h(x_d).view(self.w.shape))
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_Ours_Proj17(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifier_Proj27(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Proj")
        self.Proj = nn.Sequential(nn.Linear(self.G.fdim, self.G.fdim//4), nn.ReLU(), nn.Linear(self.G.fdim//4, self.G.fdim//8))
        self.Proj.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Proj)))
        self.optim_Proj = build_optimizer(self.Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Proj = build_lr_scheduler(self.optim_Proj, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Proj", self.Proj, self.optim_Proj, self.sched_Proj)

        print("Building Domain_C")
        self.Domain_C = NormalClassifier(self.G.fdim//8, self.num_source_domains)
        self.Domain_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.Domain_C)))
        self.optim_Domain_C = build_optimizer(self.Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_Domain_C = build_lr_scheduler(self.optim_Domain_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("Domain_C", self.Domain_C, self.optim_Domain_C, self.sched_Domain_C)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]
        y_x_d = parsed_batch["y_x_d"]
        # print("y_x_d: ", y_x_d)

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor
        y_u_d = parsed_batch["y_u_d"]

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                d_xu_k = self.Proj(f_xu_k).mean(0).unsqueeze(0)
                z_xu_k = self.C(f_xu_k, d_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_d = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            d_x_k = self.Proj(f_x_k.detach().clone())
            z_x_k = self.C(f_x_k, d_x_k.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
            d_z_x_k = self.Domain_C(d_x_k)
            loss_x_d += F.cross_entropy(d_z_x_k, torch.tensor([k]*d_z_x_k.shape[0]).to(self.device)) / K

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_aug_d = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                d_xu_k_aug = self.Proj(f_xu_k_aug.detach().clone())
                z_xu_k_aug = self.C(f_xu_k_aug, d_xu_k_aug.mean(0).unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # d_z_xu_k_aug = self.Domain_C(d_xu_k_aug)
                # loss_u_aug_d += F.cross_entropy(d_z_xu_k_aug, torch.tensor([k]*d_z_xu_k_aug.shape[0]).to(self.device))

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                d_xu_k_sty = self.Proj(f_xu_k_sty.detach().clone()).mean(0).unsqueeze(0)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, d_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

            
            # x_d_k = x[k]
            # u_d_k = u[k]
            # xu_d_k = torch.cat([x_d_k, u_d_k], 0)
            # with torch.no_grad():
            #     f_xu_d_k = self.G(xu_d_k)
            # d_xu_d_k = self.Proj(f_xu_d_k).mean(0).unsqueeze(0)
            # d_z_xu_d_k = self.Domain_C(d_xu_d_k)
            # loss_u_aug_d += F.cross_entropy(d_z_xu_d_k, torch.tensor([k]*d_z_xu_d_k.shape[0]).to(self.device))

        # x_d = torch.cat([i for i in x_aug], 0)
        u_d = torch.cat([i for i in u_aug], 0)
        # y_x_d = torch.cat([i for i in y_x_d], 0)
        y_u_d = torch.cat([i for i in y_u_d], 0)

        # xu_d = torch.cat([x_d, u_d], 0)
        # y_xu_d = torch.cat([y_x_d, y_u_d], 0)

        # idx = torch.randperm(xu_d.shape[0])
        # xu_d_shuf = xu_d[idx]
        # y_xu_d_shuf = y_xu_d[idx]

        idx = torch.randperm(u_d.shape[0])
        u_d_shuf = u_d[idx]
        y_u_d_shuf = y_u_d[idx]

        # with torch.no_grad():
        #     f_xu_d = self.G(xu_d_shuf)
        # d_xu_d = self.Proj(f_xu_d)
        # d_z_xu_d = self.Domain_C(d_xu_d)
        # loss_u_d = F.cross_entropy(d_z_xu_d, y_xu_d_shuf)

        with torch.no_grad():
            f_u_d = self.G(u_d_shuf)
        d_u_d = self.Proj(f_u_d)
        d_z_u_d = self.Domain_C(d_u_d)
        loss_u_d = F.cross_entropy(d_z_u_d, y_u_d_shuf)



        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_x_d
        loss_summary["loss_x_d"] = loss_x_d.item()

        loss_all += loss_u_d
        loss_summary["loss_u_d"] = loss_u_d.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]
        y_x_d = batch_x["domain"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)
        y_x_d = y_x_d.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only
        y_u_d = batch_u["domain"]

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)
        y_u_d = y_u_d.to(self.device)


        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        y_x_d = y_x_d.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        y_u_d = y_u_d.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            "y_x_d": y_x_d,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
            "y_u_d": y_u_d,

        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        d_featiues = self.Proj(features).mean(0).unsqueeze(0)

        if self.inference_mode == "deterministic":
            prediction = self.C(features,d_featiues, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


@TRAINER_REGISTRY.register()
class DomainC(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Domian Cs")
        self.D = []
        self.optim_D = []
        self.sched_D = []
        
        for i in range(self.num_source_domains):
            if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
                self.D.append(StochasticClassifier(self.G.fdim, self.num_classes))
            else:
                self.D.append(NormalClassifier(self.G.fdim, self.num_classes))
            self.D[i].to(self.device)
            print("# params: {:,}".format(count_num_param(self.D[i])))
            self.optim_D.append(build_optimizer(self.D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.sched_D.append(build_lr_scheduler(self.optim_D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.register_model("D"+str(i), self.D[i], self.optim_D[i], self.sched_D[i])

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.D[k](self.G(xu_k), stochastic=False)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_D = [0 for i in range(K)]
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            # z_x_k = self.C(self.G(x_k), stochastic=True)
            f_x_k = self.G(x_k)
            z_x_k =  self.C(f_x_k, stochastic=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

            z_x_k_D = self.D[k](f_x_k.detach().clone(), stochastic=True)
            loss_x_D[k] += F.cross_entropy(z_x_k_D, y_x_k_true)

            

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_d = [0 for i in range(K)]
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)

                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                z_xu_k_aug_D = self.D[k](f_xu_k_aug.detach().clone(), stochastic=True)
                loss_D = F.cross_entropy(z_xu_k_aug_D, y_xu_k_pred, reduction="none")
                loss_D = (loss_D * mask_xu_k).mean()
                loss_u_d[k] += loss_D

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

                z_xu_k_sty_D = self.D[k](f_xu_k_sty.detach().clone(), stochastic=True)
                loss_D = F.cross_entropy(z_xu_k_sty_D, y_xu_k_pred, reduction="none")
                loss_D = (loss_D * mask_xu_k).mean()
                loss_u_d[k] += loss_D

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        loss_all += sum(loss_x_D)
        loss_summary["loss_x_D"] = sum(loss_x_D).item()

        loss_all += sum(loss_u_d)
        loss_summary["loss_u_d"] = sum(loss_u_d).item()

        self.model_backward_and_update(loss_all)


        for k in range(K):
            loss_summary["loss_x_D"+str(k)] = loss_x_D[k].item()



        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # # Save sigma
        # if self.save_sigma:
        #     sigma_raw = np.stack(self.sigma_log["raw"])
        #     np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

        #     sigma_std = np.stack(self.sigma_log["std"])
        #     np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class DomainC2(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building Domian Cs")
        self.D = []
        self.optim_D = []
        self.sched_D = []
        
        for i in range(self.num_source_domains):
            if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
                self.D.append(StochasticClassifier(self.G.fdim, self.num_classes))
            else:
                self.D.append(NormalClassifier(self.G.fdim, self.num_classes))
            self.D[i].to(self.device)
            print("# params: {:,}".format(count_num_param(self.D[i])))
            self.optim_D.append(build_optimizer(self.D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.sched_D.append(build_lr_scheduler(self.optim_D[i], cfg.TRAINER.STYLEMATCH.C_OPTIM))
            self.register_model("D"+str(i), self.D[i], self.optim_D[i], self.sched_D[i])

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.D[k](self.G(xu_k), stochastic=False)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        loss_x_D = [0 for i in range(K)]
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            # z_x_k = self.C(self.G(x_k), stochastic=True)
            f_x_k = self.G(x_k)
            z_x_k =  self.C(f_x_k, stochastic=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

            z_x_k_D = self.D[k](f_x_k.detach().clone(), stochastic=True)
            loss_x_D[k] += F.cross_entropy(z_x_k_D, y_x_k_true)

            

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        loss_u_d = [0 for i in range(K)]
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)

                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

                # z_xu_k_aug_D = self.D[k](f_xu_k_aug.detach().clone(), stochastic=True)
                # loss_D = F.cross_entropy(z_xu_k_aug_D, y_xu_k_pred, reduction="none")
                # loss_D = (loss_D * mask_xu_k).mean()
                # loss_u_d[k] += loss_D

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

                # z_xu_k_sty_D = self.D[k](f_xu_k_sty.detach().clone(), stochastic=True)
                # loss_D = F.cross_entropy(z_xu_k_sty_D, y_xu_k_pred, reduction="none")
                # loss_D = (loss_D * mask_xu_k).mean()
                # loss_u_d[k] += loss_D

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        loss_all += sum(loss_x_D)
        loss_summary["loss_x_D"] = sum(loss_x_D).item()

        # loss_all += sum(loss_u_d)
        # loss_summary["loss_u_d"] = sum(loss_u_d).item()

        self.model_backward_and_update(loss_all)


        for k in range(K):
            loss_summary["loss_x_D"+str(k)] = loss_x_D[k].item()



        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # # Save sigma
        # if self.save_sigma:
        #     sigma_raw = np.stack(self.sigma_log["raw"])
        #     np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

        #     sigma_std = np.stack(self.sigma_log["std"])
        #     np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifierR3(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))
        self.num_features = num_features
        self.num_classes = num_classes

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*3)
            self.h2 = nn.Linear(num_features, num_classes*3)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            x1 = x1.view(self.num_features,-1)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            x2 = x2.view(self.num_classes,-1)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2, x1.t()))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_R3(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierR3(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifierR5(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))
        self.num_features = num_features
        self.num_classes = num_classes

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*5)
            self.h2 = nn.Linear(num_features, num_classes*5)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            x1 = x1.view(self.num_features,-1)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            x2 = x2.view(self.num_classes,-1)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2, x1.t()))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_R5(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierR5(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifierR7(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))
        self.num_features = num_features
        self.num_classes = num_classes

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*7)
            self.h2 = nn.Linear(num_features, num_classes*7)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            x1 = x1.view(self.num_features,-1)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            x2 = x2.view(self.num_classes,-1)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2, x1.t()))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_R7(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierR7(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifierR9(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))
        self.num_features = num_features
        self.num_classes = num_classes

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features*9)
            self.h2 = nn.Linear(num_features, num_classes*9)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            x1 = x1.view(self.num_features,-1)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            x2 = x2.view(self.num_classes,-1)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2, x1.t()))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_R9(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierR9(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

class HyperClassifierN01(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise*0.1), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifierN05(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise*0.5), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

class HyperClassifierN20(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise*2.0), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)

@TRAINER_REGISTRY.register()
class StyleMatch_N01(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierN01(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_N05(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierN05(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

@TRAINER_REGISTRY.register()
class StyleMatch_N20(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierN20(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


class HyperClassifierECCV(nn.Module):
    def __init__(self, num_features, num_classes, hypernet=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        # self.b = nn.Parameter(torch.Tensor(num_classes))

        if hypernet:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)


    def forward(self, x, x_mean=None, stochastic=True, hypernet=False, noise=False):
        if hypernet:
            if x_mean is None:
                x_mean = x.mean(0).unsqueeze(0)
            # sample some random noise from a normal distribution to form a 1 x num_features tensor/2

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                # x_mean = self.mean_projection(x_mean)
                x_mean = torch.cat((x_mean, noise), dim=1) # 128
            
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            # print("x_mean: ", x_mean.shape)
            x1 = self.h1(x_mean)
            # print("x1: ", x1.shape)

            x2 = self.h2(x_mean)
            # print("x2: ", x2.shape)
            w_modulator = torch.sigmoid(torch.matmul(x2.t(), x1))
            # print("w_modulator: ", w_modulator.shape)
            self.w_new = self.w * w_modulator
            return torch.matmul(x, self.w_new.t()) 
                                    
        else:
            return torch.matmul(x, self.w.t()) 
            # return F.linear(x, self.w, self.b)


@TRAINER_REGISTRY.register()
class StyleMatch_ECCV(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        self.DIV = {"0":None, "1":None, "2":None}

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    # def after_epoch(self):
    #     torch.save(self.DIV, "DIV.pt")
    #     self.DIV = [[],[],[]]
    #     print("DIV saved")

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

 
    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = HyperClassifierECCV(self.G.fdim, self.num_classes, hypernet=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                x_mean = f_xu_k.mean(0)
                if self.DIV[str(k)] is None:
                    self.DIV[str(k)] = x_mean
                else:
                    self.DIV[str(k)] = 0.1*x_mean + 0.9*self.DIV[str(k)]

                z_xu_k = self.C(f_xu_k, x_mean=self.DIV[str(k)].unsqueeze(0),  stochastic=False, hypernet=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), x_mean=self.DIV[str(k)].unsqueeze(0), stochastic=True, hypernet=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, x_mean=self.DIV[str(k)].unsqueeze(0), stochastic=True, hypernet=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, x_mean=self.DIV[str(k)].unsqueeze(0),  stochastic=True, hypernet=True, noise=False)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        # if self.save_sigma:
        #     sigma_raw = self.C.sigma.data  # (num_classes, num_features)
        #     sigma_std = F.softplus(sigma_raw - 4)
        #     sigma_std = sigma_std.mean(1).cpu().numpy()
        #     self.sigma_log["std"].append(sigma_std)
        #     sigma_raw = sigma_raw.mean(1).cpu().numpy()
        #     self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input, f=False):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False, hypernet=False, noise=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True, hypernet=False, noise=False)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        if f:
            return prediction, features

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)

        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)


        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u['label']

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2, label_u