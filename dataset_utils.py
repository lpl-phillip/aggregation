import torch, os, sys, shutil, json, random
import torch.distributions as D
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import torchvision
from utils import *
from shape_data import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# =================================================================================== #
#                                        Meta                                         #
# =================================================================================== #
# the ones applied in the papers are gaussian_gaussian_centered, gaussian_gaussian_mixture, MNIST, crowd_motion, crowd_motion_two
def prepare_data(args):

    if  args.dataset   =='gaussianmix_single':
        train_generator = gaussianmix_single_generator( args.B_sample, args.dim, args)
        test_generator  = gaussianmix_single_generator( args.B_sample_test, args.dim, args)
    elif  args.dataset   =='gaussianmix_single_wl':
        train_generator = gaussianmix_single_WL_generator( args.B_sample, args.dim, args)
        test_generator  = gaussianmix_single_WL_generator( args.B_sample_test, args.dim, args)
    elif  args.dataset   =='gaussianmix_single_wlinv':
        train_generator = gaussianmix_single_WLinverse_generator( args.B_sample, args.dim, args)
        test_generator  = gaussianmix_single_WLinverse_generator( args.B_sample_test, args.dim, args)
    elif  args.dataset   =='gaussianmix_split':
        train_generator = gaussianmix_split_generator(args.B_dist, args.B_sample, args.dim)
        test_generator  = gaussianmix_split_generator(1, args.B_sample_test, args.dim)
    elif  args.dataset   =='gaussianmix_merge':
        train_generator = gaussianmix_merge_generator(args.B_dist, args.B_sample, args.dim)
        test_generator  = gaussianmix_merge_generator(1, args.B_sample_test, args.dim)
    elif  args.dataset   =='gaussianmix_1to2_bidirection':
        train_generator = gaussianmix_1to2_bidirection_generator(args.B_dist, args.B_sample, args.dim)
        test_generator  = gaussianmix_1to2_bidirection_generator(1, args.B_sample_test, args.dim)   
    elif  args.dataset == 'gaussian_gaussian_SameVariance':
        train_generator = gaussian_gaussian_SameVariance_generator(args.B_dist, args.B_sample, args.dim)
        test_generator  = gaussian_gaussian_SameVariance_generator(1, args.B_sample_test, args.dim)
    elif  args.dataset == 'gaussian_gaussian_DiffVariance':
        train_generator = gaussian_gaussian_DiffVariance_generator(args.B_dist, args.B_sample, args.dim)
        test_generator  = gaussian_gaussian_DiffVariance_generator(1, args.B_sample_test, args.dim)
    #porous_generator
    elif args.dataset == 'porous':
        #assert args.B_sample % 2 == 0, "B_sample must be even"
        #train_generator = porous_sym_generator(args.B_dist, args.B_sample // 2, args.dim, args)
        train_generator = porous_generator(args.B_dist, args.B_sample, args.dim, args)
        test_generator  = porous_generator(1, args.B_sample_test, args.dim, args)
    #nonlocal
    elif args.dataset == 'aggregation':
    #elif args.dataset == 'nonlocal' or 'aggregation':
        train_generator = aggregation_generator(args.B_dist, args.B_sample, args.dim, args)
        test_generator  = aggregation_generator(1, args.B_sample_test, args.dim, args)
    #kalman
    elif args.dataset == 'kalman':
        train_generator = kalman_generator(args.B_dist, args.B_sample, args.dim, args)
        test_generator  = kalman_generator(args.B_dist_test, args.B_sample_test, args.dim, args)
    elif args.dataset == 'MNIST_full':
        train_loader = DataLoader(dataset=MNIST_full(train=True, n_sample=args.B_sample, w=args.MNIST_w), \
                                  batch_size=args.B_dist, num_workers = args.num_dataloader_workers, \
                                  drop_last=True, shuffle=True)
        test_loader = DataLoader(dataset=MNIST_full(train=False, n_sample=args.B_sample_test, w=args.MNIST_w), \
                                 batch_size=args.B_dist_test, num_workers = args.num_dataloader_workers, \
                                 drop_last=True, shuffle=True)
        train_generator = batch_generator(train_loader)
        test_generator  = batch_generator(test_loader)
        args.dim = 2 
    else:  # args.dataset == 'MNIST_SAMPLE:
        train_loader = DataLoader(dataset=MNIST(train=True, n_sample=args.B_sample, w=args.MNIST_w), \
                                  batch_size=args.B_dist, num_workers = args.num_dataloader_workers, \
                                  drop_last=True, shuffle=True)
        test_loader = DataLoader(dataset=MNIST(train=False, n_sample=args.B_sample_test, w=args.MNIST_w), \
                                 batch_size=args.B_dist_test, num_workers = args.num_dataloader_workers, \
                                 drop_last=True, shuffle=True)
        train_generator = batch_generator(train_loader)
        test_generator  = batch_generator(test_loader)
        args.dim = 2    

    
    return train_generator, test_generator, args.dim


def batch_generator(loader, num_batches=int(1e12)):
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return



#initial is fixed, the target is a mxiture of 2 gaussians, with different vaeriance on different dirction(0.1-0.6)
def gaussianmix_split_generator(B_dist, B_sample, d):
    while True:
        

        #inital distribution(P0) 
        mu_0 =  torch.zeros(B_dist, d)
        sigma0 =  torch.eye(d).unsqueeze(0).repeat( B_dist, 1, 1 )
        P_0 = D.MultivariateNormal(mu_0, sigma0)
        X_0 = P_0.sample((B_sample,)) # B x n x d
        probs_X_0 = P_0.log_prob(X_0).exp()
        X_0 = X_0.permute(1,0,2) # B x n x d
        probs_X_0 = probs_X_0.permute(1,0)

        #torch.manual_seed(2)
        X_1_list = []
        probs_X_1_list = []
        P1_means_list = []
        Std = []

        for _ in range(B_dist):
            num_components = 3
            min_separation = 0.3


            #P1_means  =  torch.rand(num_components, d) * 4 - 2
            # Generate means with separation
            while True:
                
                P1_means = torch.rand(num_components, d) * 4 - 2  # Initial random means
                
                # Compute pairwise distances
                diffs = P1_means.unsqueeze(1) - P1_means.unsqueeze(0)  # shape [3,3,d]
                distances = torch.norm(diffs, dim=2)  # Euclidean distance matrix
                assert distances.ndim==2 and distances.shape[1]==num_components, "distance wrong shape"
                print("looking for means", distances)
                
                # Check if all pairwise distances > min_separation
                if (distances + 2*torch.eye(num_components)*min_separation > min_separation).all():                    
                    break  # Exit loop if sufficiently separated


            if torch.rand(1).item() > 0.5:
                mix = D.Categorical(torch.tensor([1, 1, 1]))
            else: 
                mix = D.Categorical(torch.tensor([0.5, 0.5, 0.]))
            stdi = (torch.rand(1,1) * (0.5 - 0.1) + 0.1).repeat( num_components, d   )
            #torch.distributions.normal.Normal(loc, scale) scale means the std
            comp = D.Independent( D.Normal(
                    P1_means,  # means
                    stdi  ),  reinterpreted_batch_ndims=1 )
            gmm = D.MixtureSameFamily(mix, comp)
            P1_samples = gmm.sample((B_sample,))  # shape (N, d)       
            P1_prob = gmm.log_prob(P1_samples).exp()

            P1_means_list.append( P1_means.unsqueeze(0)   )
            X_1_list.append(P1_samples.unsqueeze(0))  # Shape: (1, B_sample, d)
            probs_X_1_list.append(P1_prob.unsqueeze(0))  # Shape: (1, B_sample)
            Std.append( stdi.unsqueeze(0)  )

        # Stack along batch dimension (B_dist, B_sample, d)
        X_1 = torch.cat(X_1_list, dim=0)
        probs_X_1 = torch.cat(probs_X_1_list, dim=0)
        P1_means_all = torch.cat( P1_means_list    ,dim=0)
        Std = torch.cat( Std, dim=0   ) #(B_dist, num_components, d)

        yield X_0, X_1, probs_X_0, probs_X_1, P1_means_all,torch.ones(B_dist,num_components)/num_components, Std


def gaussianmix_merge_generator(B_dist, B_sample, d):
    while True:        

        #inital distribution(P0) 
        P0_means =  torch.zeros(B_dist, d)
        sigma0 = (0.1)* torch.eye(d).unsqueeze(0).repeat( B_dist, 1, 1 )
        P_0 = D.MultivariateNormal(P0_means, sigma0)
        X_0 = P_0.sample((B_sample,)) # B x n x d
        probs_X_0 = P_0.log_prob(X_0).exp()
        X_0 = X_0.permute(1,0,2) # B x n x d
        probs_X_0 = probs_X_0.permute(1,0)

        #torch.manual_seed(2)
        X_1_list = []
        probs_X_1_list = []
        P1_means_list = []
        Std = []

        for _ in range(B_dist):
            num_components = 3
            P1_means  =  torch.rand(num_components, d) * 4 - 2
            if torch.rand(1).item() > 0.5:
                mix = D.Categorical(torch.tensor([1, 1, 1]))
            else: 
                mix = D.Categorical(torch.tensor([0.5, 0.5, 0.]))
            stdi = (torch.rand(1,1) * (0.9 - 0.2) + 0.2).repeat( num_components, d   )
            #torch.distributions.normal.Normal(loc, scale) scale means the std
            comp = D.Independent( D.Normal(
                    P1_means,  # means
                    stdi  ),  reinterpreted_batch_ndims=1 )
            gmm = D.MixtureSameFamily(mix, comp)
            P1_samples = gmm.sample((B_sample,))  # shape (N, d)       
            P1_prob = gmm.log_prob(P1_samples).exp()

            P1_means_list.append( P1_means.unsqueeze(0)   )
            X_1_list.append(P1_samples.unsqueeze(0))  # Shape: (1, B_sample, d)
            probs_X_1_list.append(P1_prob.unsqueeze(0))  # Shape: (1, B_sample)
            Std.append( stdi.unsqueeze(0)  )

        # Stack along batch dimension (B_dist, B_sample, d)
        X_1 = torch.cat(X_1_list, dim=0)
        probs_X_1 = torch.cat(probs_X_1_list, dim=0)
        P1_means_all = torch.cat( P1_means_list    ,dim=0)
        Std = torch.cat( Std, dim=0   ) #(B_dist, num_components, d)

        yield X_1, X_0, probs_X_1, probs_X_0, P0_means.unsqueeze(1),torch.ones(B_dist,1)/1,  0.1*torch.ones(B_dist,1,d)




def gaussianmix_single_WL_generator(B_sample, d, args):
    args.B_dist = 1
    print("B_dist has been reset to be 1")
    while True:
        # Initial distribution (P0) - Two Gaussians at (±1.2, 0) with σ=0.5
        means_p0 = torch.tensor([[-1.2, 0.0], [1.2, 0.0]])  # shape (2, 2)
        mix_p0 = D.Categorical(torch.ones(2,)/2)  # equal weights
        comp_p0 = D.Independent(D.Normal(
            means_p0,  # means
            0.5*torch.ones(2, d)),  # std of 0.5
            reinterpreted_batch_ndims=1)
        gmm_p0 = D.MixtureSameFamily(mix_p0, comp_p0)
        X_0 = gmm_p0.sample((B_sample,)).unsqueeze(0)  # shape (1, B_sample, 2)
        probs_X_0 = gmm_p0.log_prob(X_0.squeeze(0)).exp().unsqueeze(0)
        
        # Target distribution (P1) - Four Gaussians at (±2, ±2) with diagonal cov 0.25
        means_p1 = torch.tensor([[2.0, 2.0], [2.0, -2.0], 
                               [-2.0, 2.0], [-2.0, -2.0]])  # shape (4, 2)
        mix_p1 = D.Categorical(torch.ones(4,)/4)  # equal weights
        comp_p1 = D.Independent(D.Normal(
            means_p1,  # means
            0.5*torch.ones(4, d)),  # std of 0.5 (since variance is 0.25)
            reinterpreted_batch_ndims=1)
        gmm_p1 = D.MixtureSameFamily(mix_p1, comp_p1)
        X_1 = gmm_p1.sample((B_sample,)).unsqueeze(0)  # shape (1, B_sample, 2)
        probs_X_1 = gmm_p1.log_prob(X_1.squeeze(0)).exp().unsqueeze(0)
        
        yield X_0, X_1, probs_X_0, probs_X_1, means_p1.unsqueeze(0), torch.ones(1,4)/4,  0.5*torch.ones(1,4,d)

        
def gaussianmix_single_WLinverse_generator( B_sample, d, args):
    args.B_dist = 1
    print("B_dist has been reset to be 1")

    while True:
        torch_state = torch.get_rng_state()
        random_state = random.getstate()
        np_state = np.random.get_state()

        np.random.seed(42)
        torch.manual_seed(42)
        # Initial distribution (P0) - Two Gaussians at (±1.2, 0) with σ=0.5
        means_p0 = torch.tensor([[-1.2, 0.0], [1.2, 0.0]])  # shape (2, 2)
        std = 0.5*torch.ones(2, d)
        mix_p0 = D.Categorical(torch.ones(2,)/2)  # equal weights
        comp_p0 = D.Independent(D.Normal(
            means_p0,  # means
            std),  # std of 0.5
            reinterpreted_batch_ndims=1)
        gmm_p0 = D.MixtureSameFamily(mix_p0, comp_p0)
        X_0 = gmm_p0.sample((B_sample,)).unsqueeze(0)  # shape (1, B_sample, 2)
        probs_X_0 = gmm_p0.log_prob(X_0.squeeze(0)).exp().unsqueeze(0)        
        std = std.unsqueeze(0)
        
        # Target distribution (P1) - Four Gaussians at (±2, ±2) with diagonal cov 0.25
        means_p1 = torch.tensor([[2.0, 2.0], [2.0, -2.0], 
                               [-2.0, 2.0], [-2.0, -2.0]])  # shape (4, 2)
        mix_p1 = D.Categorical(torch.ones(4,)/4)  # equal weights
        comp_p1 = D.Independent(D.Normal(
            means_p1,  # means
            0.5*torch.ones(4, d)),  # std of 0.5 (since variance is 0.25)
            reinterpreted_batch_ndims=1)
        gmm_p1 = D.MixtureSameFamily(mix_p1, comp_p1)
        X_1 = gmm_p1.sample((B_sample,)).unsqueeze(0)  # shape (1, B_sample, 2)
        probs_X_1 = gmm_p1.log_prob(X_1.squeeze(0)).exp().unsqueeze(0)

        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        np.random.set_state(np_state)
        
        yield X_1, X_0, probs_X_1, probs_X_0, means_p0.unsqueeze(0), torch.ones(1,2)/2, std

def gaussianmix_single_generator( B_sample, d, args):
    args.B_dist = 1
    print("B_dist has been reset to be 1")

    while True:
        torch_state = torch.get_rng_state()
        random_state = random.getstate()
        np_state = np.random.get_state()

        #inital distribution(P0) 
        P0_means =  torch.zeros(1, d) 
        sigma0 = torch.eye(d).repeat(1, 1, 1)
        P_0 = D.MultivariateNormal(P0_means, sigma0)
        X_0 = P_0.sample((B_sample,)) # B x n x d
        probs_X_0 = P_0.log_prob(X_0).exp()
        X_0 = X_0.permute(1,0,2) # B x n x d
        probs_X_0 = probs_X_0.permute(1,0)

        torch.manual_seed(0)
        num_components = 2
        P1_means  =  torch.randn(num_components, d)
        mix = D.Categorical(torch.ones(num_components,))
        stdi = 0.8*torch.ones(num_components,d)
        #torch.distributions.normal.Normal(loc, scale) scale means the std
        comp = D.Independent( D.Normal(
                P1_means,  # means
                stdi  ),  reinterpreted_batch_ndims=1 )
        gmm = D.MixtureSameFamily(mix, comp)
        P1_samples = gmm.sample((B_sample,))  # shape (N, d)       
        P1_prob = gmm.log_prob(P1_samples).exp()
        X_1 = P1_samples.unsqueeze(0)
        probs_X_1 = P1_prob.unsqueeze(0)

        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        np.random.set_state(np_state)

        #yield X_1, X_1, probs_X_1, probs_X_1, P1_means.unsqueeze(0),torch.ones(1,2)/2,  stdi.unsqueeze(0), None, None
        yield X_1, X_0, probs_X_1, probs_X_0, P0_means.unsqueeze(1),torch.ones(1,1)/1,  torch.ones(1,1,d)


def gaussianmix_1to2_bidirection_generator(B_dist, B_sample, d):
    while True:

        #inital distribution(P0) 
        mu_0 =  torch.zeros(B_dist, d)
        s0 = torch.ones((B_dist, 1, 1))
        sigma0 = s0 * torch.eye(d).unsqueeze(0)
        P_0 = D.MultivariateNormal(mu_0, sigma0)
        X_0 = P_0.sample((B_sample,)) # B x n x d
        probs_X_0 = P_0.log_prob(X_0).exp()
        X_0 = X_0.permute(1,0,2) # B x n x d
        probs_X_0 = probs_X_0.permute(1,0)

        #torch.manual_seed(2)
        X_1_list = []
        probs_X_1_list = []
        P1_means_list = []
        Std = []

        
        num_components = 2
        weights = torch.ones(B_dist,num_components)/num_components
        for i in range(B_dist):
            #P1_means  =  torch.randn(num_components, d)
            P1_means  =  torch.rand(num_components, d) * 4 - 2
            mix = D.Categorical(torch.ones(num_components,))
            stdi = torch.rand(num_components, d) * (0.9 - 0.2) + 0.2
            #torch.distributions.normal.Normal(loc, scale) scale means the std
            comp = D.Independent( D.Normal(
                    P1_means,  # means
                    stdi  ),  reinterpreted_batch_ndims=1 )
            gmm = D.MixtureSameFamily(mix, comp)
            P1_samples = gmm.sample((B_sample,))  # shape (N, d)       
            P1_prob = gmm.log_prob(P1_samples).exp()

            if torch.rand(1).item() > 0.5:
                weights[i] = torch.tensor([0.5, 0.5])
                P1_means = torch.zeros_like(P1_means)  # Reset all means to 0
                stdi = torch.ones_like(stdi) 
                X_0[i], P1_samples = P1_samples.clone(), X_0[i].clone()
                probs_X_0[i], P1_prob = P1_prob.clone(), probs_X_0[i].clone()

                
            P1_means_list.append( P1_means.unsqueeze(0)   )
            X_1_list.append(P1_samples.unsqueeze(0))  # Shape: (1, B_sample, d)
            probs_X_1_list.append(P1_prob.unsqueeze(0))  # Shape: (1, B_sample)
            Std.append( stdi.unsqueeze(0)  )


        # Stack along batch dimension (B_dist, B_sample, d)
        X_1 = torch.cat(X_1_list, dim=0)
        probs_X_1 = torch.cat(probs_X_1_list, dim=0)
        P1_means_all = torch.cat( P1_means_list    ,dim=0)
        Std = torch.cat( Std, dim=0   ) #(B_dist, num_components, d)

        yield X_0, X_1, probs_X_0, probs_X_1, P1_means_all,weights, Std








def gaussian_gaussian_SameVariance_generator(B_dist, B_sample, d):
    while True:
        #inital distribution(P0) has changing mean and variance
        mu_0 =  10*torch.rand(B_dist, d).to(device) -5 # Unif[-5,5]^d   
        #s0 = 0.1*torch.ones((B_dist, 1, 1))
        s0 =0.05+ torch.rand(B_dist)[:, None, None]**2
        sigma0 = s0 * torch.eye(d).unsqueeze(0)
        P_0 = D.MultivariateNormal(mu_0, sigma0.to(device))
        #print('dataset initializated: p0:mean Unif[-5,5]^d, variance Unif[0,1]^d')

        #target distribution(P1) is fixded 0 as the mean and 0.1 as the variance
        mu_1 =  10*torch.rand(B_dist, d).to(device) -5 # Unif[-5,5]^d
        #s1 = torch.rand(B_dist)[:, None, None]**2
        #s1 = 0.1*torch.ones((B_dist, 1, 1))
        s1 = s0
        sigma1 = s1 * torch.eye(d).unsqueeze(0)
        P_1 = D.MultivariateNormal(mu_1, sigma1.to(device))

        
        # sample from ground truth density, which is N((1-t)*mu_0 + t*mu_1, sigma^2*I)
        # NOTE: every sample from P_i is one sample from B_dist gaussians
        X_0 = P_0.sample((B_sample,)) # B x n x d
        probs_X_0 = P_0.log_prob(X_0).exp()
        X_0 = X_0.permute(1,0,2) # B x n x d
        probs_X_0 = probs_X_0.permute(1,0)

        X_1 = P_1.sample((B_sample,)) # B x n x d
        probs_X_1 = P_1.log_prob(X_1).exp()
        X_1 = X_1.permute(1,0,2) # B x n x d
        probs_X_1 = probs_X_1.permute(1,0)

        
        yield X_0, X_1, probs_X_0, probs_X_1,  mu_1.unsqueeze(1), torch.ones(B_dist,1)/1, s1
        


def gaussian_gaussian_DiffVariance_generator(B_dist, B_sample, d):
    while True:
        #inital distribution(P0) has changing mean and variance
        mu_0 =  10*torch.rand(B_dist, d).to(device) -5 # Unif[-5,5]^d  
        s0 =0.05+ torch.rand(B_dist)[:, None, None]**2
        sigma0 = s0 * torch.eye(d).unsqueeze(0)
        P_0 = D.MultivariateNormal(mu_0, sigma0.to(device))
        #print('dataset initializated: p0:mean Unif[-5,5]^d, variance Unif[0,1]^d')

        #target distribution(P1) is fixded 0 as the mean and 0.1 as the variance
        mu_1 =  10*torch.rand(B_dist, d).to(device) -5 # Unif[-5,5]^d
        s1 = 0.05+ torch.rand(B_dist)[:, None, None]**2
        sigma1 = s1 * torch.eye(d).unsqueeze(0)
        P_1 = D.MultivariateNormal(mu_1, sigma1.to(device))

        
        # sample from ground truth density, which is N((1-t)*mu_0 + t*mu_1, sigma^2*I)
        # NOTE: every sample from P_i is one sample from B_dist gaussians
        X_0 = P_0.sample((B_sample,)) # B x n x d
        probs_X_0 = P_0.log_prob(X_0).exp()
        X_0 = X_0.permute(1,0,2) # B x n x d
        probs_X_0 = probs_X_0.permute(1,0)

        X_1 = P_1.sample((B_sample,)) # B x n x d
        probs_X_1 = P_1.log_prob(X_1).exp()
        X_1 = X_1.permute(1,0,2) # B x n x d
        probs_X_1 = probs_X_1.permute(1,0)

        
        yield X_0, X_1, probs_X_0, probs_X_1,  mu_1.unsqueeze(1), torch.ones(B_dist,1)/1, s1


#special porous generator
def porous_1d_generator(B_dist, B_sample, dim, args):
    
    while True:
        torch_state = torch.get_rng_state()
        random_state = random.getstate()
        np_state = np.random.get_state()

        np.random.seed(42)
        torch.manual_seed(42)
        samples,densities = generate_1d_barenblatt_samples(B_sample)

        # Convert to tensors (assuming PyTorch)
        dataset_tensor = samples.reshape(1,B_sample,1)
        V_tensor = densities.reshape(1,B_sample)

        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        np.random.set_state(np_state)

        yield dataset_tensor, V_tensor


def initial_barenblatt_radius_density(norms, d, m=2, C=0.8):

    t = 0.001 

    if torch.is_tensor(norms):
        norms = norms.detach().cpu().numpy()
    
    alpha = d/(d*(m-1)+2)
    k = (m-1)*alpha/(2*m*d)
    
    beta = alpha/d

    u = C - (k * norms ** 2)/(t ** (2*beta))
    u[u<0] = 0
    u = (t**(-alpha)) * u**(1/(m-1))
    u = np.reshape(u, (len(u),1))

    return torch.from_numpy(u).flatten()


def porous_generator(B, n_samples, dim, args):
    
    while True:

        # Initialize output tensors
        samples = torch.zeros(B, n_samples, dim)
        densities = torch.zeros(B, n_samples)
        C_list = []
        
        # Process each batch element separately
        for b in range(B):

            c = torch.rand(1).mul(0.9).add(0.1).item()
            C_list.append(c)

            _, r_max = barenblatt_solution_porous(torch.zeros(1,dim), C=c, m=args.porous_m)
            
            
            # Discretize radial profile with Jacobian correction (r^{d-1})
            r_pts = torch.linspace(0, r_max, 1000)            
            rho_r = initial_barenblatt_radius_density(r_pts, dim, C=c, m=args.porous_m)
            rho_d = rho_r.squeeze() * (r_pts ** (dim - 1))  # Jacobian correction
            
            # Compute CDF numerically
            cdf = torch.cumsum(rho_d, dim=0)
            cdf = cdf / cdf[-1]  # Normalize
            
            # Inverse transform sampling
            u = torch.rand(n_samples)
            cdf_np = cdf.numpy()
            r_pts_np = r_pts.numpy()
            radii = np.interp(u.numpy(), cdf_np, r_pts_np)
            radii = torch.from_numpy(radii)

            #set landmark
            smallest_two = torch.topk(radii, k=2, largest=False).values
            gap_ = max( torch.mean(smallest_two).item()/(args.dim ), args.Deltax)
            for i in range(0,1+args.dim):
                radii[-1*(i+1)] = i*gap_

            
            # Sample uniform directions on unit sphere
            directions = torch.randn(n_samples, dim)
            directions = directions / torch.norm(directions, p=2, dim=1, keepdim=True)
            
            # Combine radii and directions
            samples[b] = directions * radii.reshape(-1, 1)  
            mean_to_subtract = samples[b].mean(axis=0)
            count = 0
            while (mean_to_subtract.abs().mean()>1e-3) and count <10:
                samples[b, :-1 - args.dim] -= mean_to_subtract
                mean_to_subtract = samples[b].mean(axis=0)
                count +=1
              
            
            # Compute densities
            densities[b], _ = barenblatt_solution_porous(samples[b], C=c, m=args.porous_m)
        
        yield samples, densities,C_list

  

def porous_generator_(B_dist, n_samples, dim, args):
    
    while True:
        #torch_state = torch.get_rng_state()
        #random_state = random.getstate()
        #np_state = np.random.get_state()

        #if dim==1:                
        #    np.random.seed(42)
        #    torch.manual_seed(42)

        Data_list = []
        V_list = []
        C_list = []

        for _ in range(B_dist):    
            samples = []
            densities = []
        
            # Get support radius first
            c = torch.rand(1).mul(0.9).add(0.1).item()
            #print("------C",c)
            #c = 0.8
            rho0, r = barenblatt_solution_porous(torch.zeros(1,dim), C=c, m=args.porous_m)
            
            while len(samples) < n_samples :
                # Generate random points in 2D
                x = torch.rand(n_samples, dim) * 2 * r - r  # Uniform in [-r, r]^2
                
                # Get densities
                rho, _ = barenblatt_solution_porous(x, C=c, m=args.porous_m)
                rho = rho.numpy()
                
                # Rejection sampling
                accept = np.random.rand(n_samples) < (rho / rho0.item())
                
                samples.extend(x[accept].numpy())
                densities.extend( rho[accept])
            
            samples = np.array(samples[:n_samples])
            samples -= samples.mean(axis=0)
            densities,_ = barenblatt_solution_porous(samples, C=c, m=args.porous_m)       

            # Convert to tensors (assuming PyTorch)
            Data_list.append(torch.tensor(samples, dtype=torch.float32).unsqueeze(0))  # Shape: (1, B_sample)
            V_list.append( torch.tensor(densities, dtype=torch.float32).unsqueeze(0) )
            C_list.append(c)

        # Stack along batch dimension (B_dist, B_sample, d)
        Data_list = torch.cat(Data_list, dim=0)
        V_list = torch.cat(V_list, dim=0)
        
        #torch.set_rng_state(torch_state)
        #random.setstate(random_state)
        #np.random.set_state(np_state)

        yield Data_list, V_list,C_list

'''
def porous_sym_generator(B_dist, n_samples, dim, args):
    assert dim==1, "data dimension not fit"
    
    while True:
        

        Data_list = []
        V_list = []
        C_list = []

        for _ in range(B_dist):    
            samples = []
            densities = []
        
            # Get support radius first
            c = torch.rand(1).mul(0.9).add(0.1).item()
            #print("------C",c)
            #c = 0.8
            rho0, r = barenblatt_solution_porous(torch.zeros(1,dim), C=c)
            
            while len(samples) < n_samples :
                # Generate random points in 2D
                x = torch.rand(n_samples, dim) * 2 * r - r
                
                # Get densities
                rho, _ = barenblatt_solution_porous(x, C=c)
                rho = rho.numpy()
                
                # Rejection sampling
                accept = np.random.rand(n_samples) < (rho / rho0.item())
                
                samples.extend(x[accept].numpy())
                densities.extend( rho[accept])
            
            samples = samples[:n_samples]
            samples.extend([-x for x in samples])
            samples = np.array(samples)
           
            densities,_ = barenblatt_solution_porous(samples, C=c)       

            # Convert to tensors (assuming PyTorch)
            Data_list.append(torch.tensor(samples, dtype=torch.float32).unsqueeze(0))  # Shape: (1, B_sample)
            V_list.append( torch.tensor(densities, dtype=torch.float32).unsqueeze(0) )
            C_list.append(c)

        # Stack along batch dimension (B_dist, B_sample, d)
        Data_list = torch.cat(Data_list, dim=0)
        V_list = torch.cat(V_list, dim=0)
        
        

        yield Data_list, V_list,C_list
'''

def aggregation_generator(B_dist, sample_size, dim, args):
    while True:        

        #samples = torch.rand(B_dist, sample_size, dim) * 2 - 1
        #samples = torch.rand(B_dist, sample_size, dim)
        #center = samples.mean(dim=1, keepdim=True)  
        #samples = samples - center

        # 生成10个矩形，每个100个点
        #half_B_dist = B_dist // 2
        #rect_points = generate_random_rectangles(B=half_B_dist, sample_size=sample_size)
        #tri_points = generate_random_triangles(B=B_dist-half_B_dist, sample_size=sample_size)
        #samples = torch.cat([rect_points, tri_points], dim=0)  # Shape: (B_dist, sample_size, dim)
        samples_list = []
        batch_list = []
        for b in range(B_dist):
            samples_list.append(get_random_shape_data())
            batch_list.append(torch.full((sample_size,), b))
        samples = torch.cat(samples_list, dim=0)  # Shape: (B_dist, sample_size, dim)
        batch = torch.cat(batch_list, dim=0)


        p = 0.5*torch.ones(B_dist,1)
        q = 3*torch.ones(B_dist,1)
#        p = torch.rand(B_dist,1)
#        q = p + (10 - p) * torch.rand(B_dist, 1)
        pq = torch.cat([p, q], dim=1)
        
        yield samples, pq, batch


def nonlocal_generator(B_dist, batch_size, dim, args):
    while True:
        

        samples = []
        densities = []
        while len(samples) < batch_size:
            # Step 1: Sample uniformly in the unit ball
            x = (np.random.rand(dim) * 2 - 1)  # Uniformly in [-1, 1]
            if np.linalg.norm(x) <= 1:  # Ensure it's within the unit ball
                # Step 2: Compute rho_0(x) for this point
                rho0_x = (3 / 4) * (1 - np.linalg.norm(x) ** 2)
                # Step 3: Accept with probability rho0_x
                if np.random.rand() <= rho0_x:
                    samples.append(x)
                    densities.append(rho0_x)

        dataset = np.array(samples)
        V  = np.array( densities )

        # Add a dimension to dataset and V
        dataset = np.expand_dims(dataset, axis=0)  # Shape becomes (1, batch_size, dim)
        V = np.expand_dims(V, axis=0)  # Shape becomes (1, batch_size)

        # Convert to tensors (assuming PyTorch)
        dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
        V_tensor = torch.tensor(V, dtype=torch.float32)

        yield dataset_tensor, V_tensor

def kalman_generator(B_dist, batch_size, dim, args):
    while True:        

        normal_samples = np.random.normal(0, 1, batch_size)
        uniform_samples = np.random.uniform(90, 110, batch_size)
        samples = np.column_stack((normal_samples, uniform_samples))
        
        # Calculate density values
        normal_density = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * normal_samples**2)
        uniform_density = np.full(batch_size, 1 / 20)  # Density is constant within the range for uniform distribution
        densities = normal_density * uniform_density

        dataset = samples
        V  = densities

        # Add a dimension to dataset and V
        dataset = np.expand_dims(dataset, axis=0)  # Shape becomes (1, batch_size, dim)
        V = np.expand_dims(V, axis=0)  # Shape becomes (1, batch_size)

        # Convert to tensors (assuming PyTorch)
        dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
        V_tensor = torch.tensor(V, dtype=torch.float32)

        yield dataset_tensor, V_tensor
 

# =================================================================================== #
#                                       MNIST                                         #
# =================================================================================== #

def process_padded_points(X, n_repeat):
    # this is not the most efficient implementation, but we only call it once in the beginning :P
    X_out = []
    for i in range(len(X)):
        x = X[i] # n x 2
        x = np.repeat(x, n_repeat, axis=0) # n_repeat*n x 2
        I = x[:,0] == -1
        x_normal_pts = x[~I]
        x[I] = x_normal_pts[np.random.choice(len(x_normal_pts), np.sum(I))]
        X_out.append(x)

    return np.stack(X_out)

def MNIST_dataset(fname, B, args):
    df = pd.read_csv(fname)

    X = df[df.columns[1:]].to_numpy()
    Y = df[df.columns[0]].to_numpy() # N x 1
    X = X.reshape(X.shape[0], -1, 3) # N x n x 3

    # remove intensity information
    X = X[:,:,:-1]
    X = process_padded_points(X, args.MNIST_n_repeat)
    # normalize coordinates to be in [0,1]^2
    min_val = 0
    max_val = 27
    X = (X - min_val) / (max_val - min_val)

    if args.MNIST_single_digit:
        # in this case, P_0 is a distribution of all images of a single chosen digit, instead of all digits
        # same for P_1
        X_0 = X[Y == args.MNIST_digit_P_0]
        X_1 = X[Y == args.MNIST_digit_P_1]
        # P_0_loader = DataLoader(X_0, batch_size=B, shuffle=True, drop_last=True)
        # P_1_loader = DataLoader(X_1, batch_size=B, shuffle=True, drop_last=True)
    else:
        X_0 = X
        X_1 = X.copy()
        # P_0_loader = DataLoader(X, batch_size=B, shuffle=True, drop_last=True)
        # P_1_loader = DataLoader(X, batch_size=B, shuffle=True, drop_last=True)

    return torch.tensor(X_0), torch.tensor(X_1)

def MNIST_generator(X_0, X_1, B):
    # This implementation is wack..I had to do this back then as torch had a bug with dataloaders.
    # It has since been fixed.
    while True:
        x_0 = X_0[torch.multinomial(torch.ones(len(X_0)), B, replacement=False)]
        x_1 = X_1[torch.multinomial(torch.ones(len(X_1)), B, replacement=False)]

        yield x_0, x_1


# Adapted from: https://github.com/hamrel-cxu/JKO-iFlow
def sample_from_img(img, data_size, size):
    def gen_data_from_img(image_mask, train_data_size):
        def sample_data(train_data_size):
            inds = np.random.choice(
                int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds] 
            samples = np.random.randn(*m.shape) * std + m 
            return samples
        img = image_mask
        h, w = img.shape
        xx = np.linspace(-size, size, w)
        yy = np.linspace(-size, size, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1) # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum() 
        std = np.array([2*size / w / 2, 2*size / h / 2])
        full_data = sample_data(train_data_size)
        return full_data, means, probs
    # image_mask = np.array(Image.open(f'{img_name}.png').rotate(
    #     180).transpose(0).convert('L'))
    img = torchvision.transforms.functional.rotate(img.unsqueeze(0), 270)[0].transpose(0,1)
    dataset = gen_data_from_img(1-img, data_size)

    return dataset # N x 2

# https://github.com/YijiangPang/Implementation-of-Meta-OT-between-discrete-measures
class MNIST(torchvision.datasets.MNIST):
    def __init__(self, train=True, n_sample=1024, w=4):
        # self.flag_split = "train" if train else "test"
        torchvision.datasets.MNIST.__init__(self, train = train, download=True, root="./data")
        data = self.data
        data = data.double()/255.
        #data = data.float() / 255.

        data = data.reshape(-1, 784)
        data = data/data.sum(axis=1, keepdims=True)
        self.data = data
        self.n_sample = n_sample
        self.w = w
    
    def __getitem__(self, index: int):
        np_state = np.random.get_state()
        np.random.seed(42) 
        (id_a, id_b) = np.random.randint(0, len(self.data), 2)
        np.random.seed(np_state) 
        img_0, img_1 = self.data[id_a], self.data[id_b]
        # get samples from the img
        x_0, _, _ = sample_from_img(img_0.reshape(28,28), self.n_sample, self.w)
        x_1, target_grids_batch,target_values_batch = sample_from_img(img_1.reshape(28,28), self.n_sample, self.w)
        return x_0, x_1, np.ones(self.n_sample),  np.ones(self.n_sample), target_grids_batch, target_values_batch 


class MNIST_full(torchvision.datasets.MNIST):
    def __init__(self, train=True, n_sample=1024, w=4):
        # self.flag_split = "train" if train else "test"
        torchvision.datasets.MNIST.__init__(self, train = train, download=True, root="./data")
        data = self.data
        data = data.double()/255.
        #data = data.float() / 255.

        data = data.reshape(-1, 784)
        data = data/data.sum(axis=1, keepdims=True)
        self.data = data
        self.n_sample = n_sample
        self.w = w
    
    def __getitem__(self, index: int):
        #np.random.seed(42) 
        (id_a, id_b) = np.random.randint(0, len(self.data), 2)
        img_0, img_1 = self.data[id_a], self.data[id_b]
        # get samples from the img
        x_0, _, _ = sample_from_img(img_0.reshape(28,28), self.n_sample, self.w)
        x_1, target_grids_batch,target_values_batch = sample_from_img(img_1.reshape(28,28), self.n_sample, self.w)
        return x_0, x_1, np.ones(self.n_sample),  np.ones(self.n_sample), target_grids_batch, target_values_batch

