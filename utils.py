import torch, os, sys, shutil, json
import numpy as np
# from sklearn.datasets import make_moons, make_s_curve, make_swiss_roll
import matplotlib.pyplot as plt
import torch.distributions as D
import torch.nn as nn
# from arguments import parse_arguments
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.datasets import MNIST, CIFAR10, ImageNet
# from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.distributions as D
import matplotlib.animation as animation
import pylab
from scipy import interpolate
import plotly.graph_objects as go
import plotly.express as px
import math
import scipy
from loss_jko import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# torch.set_default_tensor_type('torch.FloatTensor')

def chamfer_distance_batch(batch_points1, batch_points2):
    """
    Compute the Chamfer distance for batches of point sets using PyTorch tensors.

    Parameters:
    - batch_points1: torch tensor of shape (B, N, d)
    - batch_points2: torch tensor of shape (B, M, d)

    Returns:
    - chamfer_dists: torch tensor of shape (B,) containing Chamfer distances
    """
    B, N, _ = batch_points1.shape
    _, M, _ = batch_points2.shape

    batch_points1_expanded = batch_points1.unsqueeze(2).expand(B, N, M, -1)
    batch_points2_expanded = batch_points2.unsqueeze(1).expand(B, N, M, -1)

    distances = torch.norm(batch_points1_expanded - batch_points2_expanded, dim=-1)

    dist1 = distances.min(dim=2)[0]
    dist2 = distances.min(dim=1)[0]

    return dist1.pow(2).sum(dim=1)+dist2.pow(2).sum(dim=1)

def kl_standard_gaussian(D):
    # D: (N, d)
    mu    = D.mean(axis=0)
    Sigma = np.cov(D, rowvar=False)           # shape (d,d)
    d     = D.shape[1]
    trace = np.trace(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)   # for numerical stability
    return 0.5*( trace + mu@mu - d - logdet )

def check_gradUpdate(net, loss,i):
    with torch.no_grad():
        # after loss.backward(), before optimizer.step()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN or Inf in the loss")

        for name, param in net.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise RuntimeError(f"NaN or Inf in gradients of net before optimizer.step()")

        # ========== compute gradNormRatio ==========
        # 1) collect all grad-norms (each is a 0-dim tensor)
        grad_norms = [p.grad.norm(2) for p in net.parameters() if p.grad is not None]
        total_grad_norm   = torch.norm(torch.stack(grad_norms),   2)

        # 2) collect all weight-norms
        weight_norms = [p.data.norm(2) for p in net.parameters()]
        total_weight_norm = torch.norm(torch.stack(weight_norms), 2)

        # 3) ratio
        ratio = total_grad_norm / (total_weight_norm + 1e-12)

        # convert to Python floats for printing
        g_norm = total_grad_norm.item()
        w_norm = total_weight_norm.item()
        r_val  = ratio.item()

        print(f"Step {i} — grad_norm={g_norm:.1f}, weight_norm={w_norm:.1f}, ratio={r_val:.3f}")
    return g_norm, w_norm, r_val

# =================================================================================== #
#                                      monitoring                                       #
# =================================================================================== #

def writeLossToFile(loss, iteration, loss_log_path, args):
    if args.monitor == 'cumulative': 
        loss_vector = torch.cumsum(loss.reshape( -1,args.B_dist), dim=0).T
    else:
        loss_vector = loss.reshape( -1, args.B_dist).T
    loss_vector = loss_vector.detach().cpu().numpy().flatten()
    with open(loss_log_path, 'a') as f:
        # Write iteration header and values in one line
        f.write(f"# Iter {iteration}: ")
        np.savetxt(f, [loss_vector], delimiter=' ', fmt='%.3f', newline=' ')  # Saves without linebreak
        f.write("\n Ref:")
        if args.ReferenceLoss is None:
            f.write(" None \n")
        else:
            ref_loss = args.ReferenceLoss.detach().cpu().numpy().flatten()
            np.savetxt(f, [ref_loss], delimiter=' ', fmt='%.3f', newline=' ')
            f.write("\n")

def should_generate_new_data(k, loss_,args):
    with torch.no_grad():
        loss_copy = loss_.detach().clone()
        print("loss_ detached")

        if args.ReferenceLoss is None:
            if args.monitor == 'cumulative':
                args.ReferenceLoss = torch.cumsum(loss_copy.reshape(-1, args.B_dist), dim=0).T 
            else:
                args.ReferenceLoss = loss_copy.reshape(-1, args.B_dist).T
            return False

        if args.TrainingData is None:
            print("TrainingData is None")
            return True
        if k < args.WarmUpEpoch:
            return True
        if args.current_stall_count > args.max_stall_epochs:
            return True


        
        if args.monitor == 'cumulative':
            loss = torch.cumsum(loss_copy.reshape(-1, args.B_dist), dim=0).T  
        else:
            loss = loss_copy.reshape(-1, args.B_dist).T

        return torch.all( loss <= args.ReferenceLoss)


def GenerateNewAggregationData(samples, pq, batch, net, args):  # 注意参数名是pq不是pg
    B, N, d = samples.shape

    # 修正批处理索引生成逻辑
    if batch is None or batch.numel() != B * N:  # 统一判断条件
        batch = torch.arange(B, device=samples.device).repeat_interleave(N)

    # 处理pq参数
    pq = pq.unsqueeze(1).repeat(1, N, 1).to(device)  # 从[B,2]扩展到[B,N,2]

    # 生成数据路径
    path = []
    X_k = samples.to(device)
    path.append(X_k.clone())

    net.eval()
    with torch.no_grad():
        for i in range(args.jko_T):
            X_k += net(X_k, batch)  # 确保传入batch参数
            path.append(X_k.detach().clone())

    # 组装训练数据
    path = torch.cat(path, dim=0)
    args.TrainingData = [
        path,  # 路径数据
        pq.repeat(args.jko_T + 1, 1, 1),  # 扩展pq参数
        batch.repeat(args.jko_T + 1)  # 扩展批处理索引
    ]
    args.ReferenceLoss = None

def GenerateNewAggregationData_withdensity(sample,density, net, args):

    #generate the path of data
    path = []
    X_k = sample
    
    #path_density = []
    #density_new = density
    path.append(X_k.clone())
    #path_density.append( density_new.clone()  )

    net.eval()
    with torch.no_grad():
        for i in range(args.jko_T):
            X_k = X_k.to(device)
            code = net.encode(X_k,None, None)
            V_k = net.decode(code,X_k)           
            
            #div_V = compute_divergence(X_k, V_k, code, net,args).detach()
            #density_new = (density_new / (torch.exp(div_V) ).clamp(min= 1e-10)).detach()            
            #path_density.append(density_new.detach().clone())            
            X_k = (X_k +V_k).detach()            
            path.append(X_k.detach().clone())


    path = torch.cat(path, dim=0)
    #path_density = torch.cat(path_density, dim=0)   

   
    #args.TrainingData = [ path,  path_density   ]
    args.TrainingData = [ path   ]
    args.ReferenceLoss = None


def GenerateNewPorousData(sample,density, net, args):

    #generate the path of data
    path = []
    path_density = []
    X_k = sample
    density_new = density
    path.append(X_k.clone())
    path_density.append( density_new.clone()  )

    net.eval()
    with torch.no_grad():
        for i in range(args.jko_T):
            X_k = X_k.to(device)
            code = net.encode(X_k,density_new, None)
            V_k = net.decode(code,X_k)           
            
            div_V = compute_divergence(X_k, V_k, code, net,args).detach()
            density_new = (density_new / (torch.exp(div_V) ).clamp(min= 1e-10)).detach()
            
            X_k = (X_k +V_k).detach()

            if X_k.abs().max().item() > 1000 or density_new.abs().max().item() > 10000:
                print("data generating stopped early")
                break


            path.append(X_k .clone())
            path_density.append(density_new.detach().clone())


    path = torch.cat(path, dim=0)
    path_density = torch.cat(path_density, dim=0)   

   
    args.TrainingData = [ path,  path_density   ]
    args.ReferenceLoss = None








def GenerateNewGaussianData(initials_generator, net, args):
    #del args.TrainingData
    # fetch initial measures
    X = next(initials_generator)

    X_0 = X[0].clone().float().to(device) # B x n x d; samples from P_0
    X_0_density = X[2].clone().float().to(device) # B x n x d; samples density values
    X_1 = X[1].clone().float().to(device) # B x n x d; samples from P_1
    X_1_density = X[3].clone().float().to(device) # B x n x d; samples density values
    target_means = X[4].clone().float().repeat(1+args.jko_T,1,1).to(device) #(1,K,d) --> (B,N,d)
    target_weights = X[5].clone().float().repeat(1+args.jko_T,1).to(device)
    Std = X[6].clone().float().repeat(1+args.jko_T,1,1).to(device)

    assert X_0_density.dim() == 2, f"Gaussian density : Expected 2D tensor, got {X_0_density.dim()}D tensor"
    assert X_1_density.dim() == 2, f"Gaussian density : Expected 2D tensor, got {X_1_density.dim()}D tensor"

    print(X_0.shape,X_0_density.shape,X_1_density.shape, target_means.shape,  target_weights.shape, 1+args.jko_T, "New Data Generated")
    
    #generate the path of data
    path = []
    path_density = []
    X_k = X_0
    density_new = X_0_density

    path.append(X_k.clone())
    path_density.append( density_new.clone()  )
    
    net.eval()
    with torch.no_grad():
        for i in range(args.jko_T):
            
            #code = net.encode(X_k)
            if args.dataset == 'gaussianmix_merge':
                code = net.encode(X_k)
            else:
                code = net.encode(X_k,X_0_densityvalue=density_new, X_1=X_1,X_1_densityvalue = X_1_density)
            V_k = net.decode(code,X_k)
        
            div_V = compute_divergence(X_k, V_k, code, net,args) 
            density_new = (density_new/  (torch.exp(div_V.detach().clamp(min=-10, max=15)) + 1e-13)).detach()
            X_k = (X_k + V_k).detach()
            path.append(X_k.detach().clone())
            path_density.append(density_new.detach().clone())

    path = torch.cat(path, dim=0).detach().to(device)
    path_density = torch.cat(path_density, dim=0).detach().to(device)
    path_X_1 = X_1.repeat(1+args.jko_T,1,1).detach().to(device)
    path_X_1_density = X_1_density.repeat(1+args.jko_T,1).detach().to(device)

    print("path date shape", path.shape, path_X_1.shape, path_density.shape, path_X_1_density.shape, target_means.shape,  target_weights.shape, Std.shape )
    args.TrainingData = [ path,   path_X_1, path_density, path_X_1_density, target_means,  target_weights, Std   ]
    args.ReferenceLoss = None

# =================================================================================== #
#                                      Porous                                       #
# =================================================================================== #

def barenblatt_solution_porous(x, dt=0,  C=0.8, m = 2):
#def Barenblatt_LS(x, t=0, m=2):
    '''
    Barenblatt solution to the porous medium equation OF liuShu's paper

    Input:
    x: (N, d) array
    t: time
    m: parameter in the porous medium equation
    d: dimension

    Output:
    u: (N, d) array
    '''
    device = "cpu"    
    
    print("utils: m = ",m)
    t = 0.001 + dt
 
    if torch.is_tensor(x):
        device = x.device
        x = x.detach().cpu().numpy()

    if x.ndim == 1:
        x = x.reshape(-1, 1)
        print("Warnining the intput is treated as 1d points")

    d = x.shape[1]
    #print(d)
    alpha = d/(d*(m-1)+2)
    k = (m-1)*alpha/(2*m*d)
    #numer = (k/math.pi)**(d/2)*scipy.special.gamma(d/2) # the numerator for computing C
    #denom = d*scipy.special.beta(d/2, m/(m-1)) # the denominator for computing C
    #C = math.pow(0.8*numer/denom, 2*(m-1)*alpha/d)
    #print(C)
    #C = 0.8
    beta = alpha/d

    u = C - (k * np.linalg.norm(x, axis = 1) ** 2)/(t ** (2*beta))
    u[u<0] = 0
    u = (t**(-alpha)) * u**(1/(m-1))
    u = np.reshape(u, (len(u),1))
    
    
    r = np.sqrt(   C*(t ** (2*beta)) /k        )
    #return u
    return torch.from_numpy(u).flatten().to(device), r


def barenblatt_solution_porous_xtBatch(x, C, args):
    """
    Batch-compatible Barenblatt solution to the porous medium equation.
    C must be a scalar
    """
    assert x.ndim == 3, "Input must be batched (B, N, d) for porous density computation"
    
    device = x.device
    B, N, d = x.shape
    m=args.porous_m
    
    t = (0.001 + torch.arange(1, x.shape[0]+1) * args.deltat).reshape(x.shape[0],-1)
    
    
    # Compute solution parameters
    alpha = d / (d * (m - 1) + 2)  # scalar
    k = (m - 1) * alpha / (2 * m * d)  # scalar
    beta = alpha / d  # scalar
    
    # Compute norm across spatial dimensions
    x_norm = torch.norm(x, dim=2)  # (B, N)
    
    # Compute solution
    u = C - (k * x_norm ** 2) / (t ** (2 * beta))
    u = torch.clamp_min(u, 0)
    u = (t ** (-alpha)) * (u ** (1 / (m - 1)))
    assert u.ndim == 2, "The density shape is wrong"
    
    return u.to(device)


def compute_porous_error( X, rho_pred, C , args):
    
    error_list = []
    error_list2 = []

    for k in range(args.B_dist):
        X_pred = X[k::args.B_dist]
        rho_new = rho_pred[k::args.B_dist]
        Truedensity = barenblatt_solution_porous_xtBatch(X_pred, C[k], args)
        assert Truedensity.shape == rho_new.shape, f"Shapes do not match: density.shape={Truedensity.shape}, color.shape={color.shape}"
        diff = Truedensity - rho_new
        error = diff.abs().sum(dim=1) / Truedensity.abs().sum(dim=1)
        error_list.append(error.flatten())

        error2 = (diff**2).sum(dim=1) / (Truedensity**2).sum(dim=1)
        error_list2.append(np.sqrt(error2).flatten())

    error_ave = torch.stack(error_list, dim=0).mean(dim=0)
    error_ave2 = torch.stack(error_list2, dim=0).mean(dim=0)
    #assert error_ave.numel() == args.jko_T+1, "the error shape is wrong"
    return error_ave.flatten(),error_ave2.flatten()

        



from scipy.integrate import trapezoid  # Modern replacement for trapz


def generate_1d_barenblatt_samples(n_samples=1000):
    # Create evaluation grid (focus on supported region)
    x_grid = np.linspace(-0.35, 0.35, 10000)  # Tight grid around theoretical r=0.3098
    
    # Compute density and theoretical radius
    rho_values, r = barenblatt_1d_solution_porous(torch.from_numpy(x_grid))
    print(f"Theoretical support radius: r = {r:.4f}")
    
    # Normalize density
    rho_np = rho_values.numpy()
    total_mass = trapezoid(rho_np, x_grid)
    print("total_mass:", total_mass  )
    normalized_rho = rho_np / total_mass
    
    # Generate samples using inverse transform
    cdf = np.cumsum(normalized_rho)
    cdf /= cdf[-1]
    u = np.random.rand(n_samples)
    samples = np.interp(u, cdf, x_grid)
    densities,_  = barenblatt_1d_solution_porous(torch.from_numpy(samples))
    
    
    return torch.from_numpy(samples).float(), densities

#the code is from WL
def barenblatt_1d_solution_porous(x, dt=0, m=2, C=0.8):
    device = "cpu"
 
    if torch.is_tensor(x):
        device = x.device
        x = x.detach().cpu().numpy()

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    t = 0.001 + dt
    alpha = 1/(m + 1)
    term = C - alpha*(m-1)/(2*m) * (np.linalg.norm(x, axis = 1) ** 2)/t**(2*alpha)
    rho = (1/t**alpha) * term * (term > 0)**(1/(m-1))

    d = 1
    alpha = d / (d * (m - 1) + 2)
    r = t**alpha * np.sqrt(2 * m * C / (alpha * (m - 1)))

    return torch.from_numpy(rho).to(device), r



# =================================================================================== #
#                                      plotting                                       #
# =================================================================================== #


import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def plot_loss(loss_hist, args, log_dir):
    zeros_points = np.zeros_like(args.ChangePoint)
    changepoints = np.array(args.ChangePoint) -1
    second_stage = int(args.WarmUpEpoch / 2)
    changepoints_400 = changepoints[changepoints>second_stage]

    def toarray(v):
        return np.array([t.item() for t in v])  # Fixed: `l` → `t` (avoid confusion with `1`)

    def tensors_to_numpy(v: list[torch.Tensor]) -> np.ndarray:
        # Ensure tensors are moved to CPU and converted to numpy
        arrs = [t.detach().cpu().numpy() for t in v]
        return np.stack(arrs, axis=0)  # Stacks along axis=0 (samples)

    # Extract and validate loss data
    loss = toarray(loss_hist['loss'])
    loss = loss - loss.min() + 0.00001
    n = len(loss)
    loss_W2 = tensors_to_numpy(loss_hist['W2'])  # Shape: (n_steps, jko_T+1)
    loss_E = tensors_to_numpy(loss_hist['E'])   # Shape: (n_steps, jko_T+1)

    # --- Plot 1: Total Loss ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(n)
    ax.scatter(x, loss, s=10, alpha=0.7, label='Total Loss')
    if len(changepoints)>0:
        ax.scatter(changepoints, loss[changepoints], c='red',s=5, alpha=0.7, label='data generating')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss')
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'loss.png'), dpi=150)
    plt.close()

    # --- Plot 1: Total Loss ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(n)
    ax.scatter(x[second_stage:], loss[second_stage:], s=10, alpha=0.7, label='Total Loss')
    if len(changepoints_400)>0:
        ax.scatter(changepoints_400, loss[changepoints_400], c='red',s=5, alpha=0.7, label='data generating')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(f"Training Loss(after {second_stage})")
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"loss_Truncted({second_stage}).png"), dpi=150)
    plt.close()

    # --- Plot 2: Per-Step Loss (W2 + 2ΔtE) ---
    for i in range(args.jko_T + 1):
        y = loss_W2[:, i] /( 2 * args.deltat) + loss_E[:, i]
        y = y - y.min() + 0.00001
        x = np.arange(n)   
        mask = (y != 0) 
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(x[mask], y[mask], s=10, alpha=0.7, color='green', label=f'Step {i}')
        if len(changepoints)>0:
            changepoints = np.round(changepoints).astype(int)
            common = np.intersect1d(changepoints, np.nonzero(mask)[0])
            ax.scatter(common, y[common], c='red', s=5, alpha=0.7, label='data generating')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'Loss on {i}th data points')
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'Loss_Step{i}.png'), dpi=150)
        plt.close()

        if len(changepoints_400)>0:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            common = np.where((np.arange(n) >= second_stage) & mask)[0]
            ax.scatter(x[common], y[common], s=10, alpha=0.7, color='green', label=f'Loss')
            common = np.intersect1d(changepoints_400, np.nonzero(mask)[0])
            ax.scatter(common, y[common], c='red', s=5, alpha=0.7, label='data generating')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss (log scale)')
            ax.set_title(f'Loss on {i}th data points_after400')
            ax.grid(True, which="both", linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'Loss_Step{i}_truncted.png'), dpi=150)
            plt.close()



def plot_porous_withError(x_next, color, t,c, sPath,args):
    Truedensity, r = barenblatt_solution_porous(x_next, t,C=c,m=args.porous_m)
    Truedensity = Truedensity.cpu().numpy()

    

    assert Truedensity.shape == color.shape, f"Shapes do not match: density.shape={Truedensity.shape}, color.shape={color.shape}"
    density_diff = np.sqrt((((Truedensity - color.cpu().numpy()))**2).mean())
    #density_error = np.linalg.norm( (Truedensity - color.cpu().numpy()).flatten()  ) / np.linalg.norm( Truedensity.flatten()  )
    assert Truedensity.shape == color.shape, "Truedensity and color shape do not match"
    density_error = np.abs(Truedensity - color.cpu().numpy()).sum() / np.abs( Truedensity  ).sum()

    circle1 = plt.Circle((0, 0), r, color='g', fill=False)
    circle2 = plt.Circle((0, 0), r, color='g', fill=False)    
    

    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Create 1 row and 2 columns of subplots
    if x_next.shape[1]>1:
        # First subplot: Prediction
        scatter1 = axes[0].scatter(
            x_next.cpu()[:, 0], 
            x_next.cpu()[:, 1], 
            c=color.cpu(), 
            marker='.'
        )
        axes[0].set_xlim([args.plot_x_min,args.plot_x_max])
        axes[0].set_ylim([args.plot_y_min,args.plot_y_max])
        axes[0].add_patch(circle1)  # Circle specific to the first plot
        axes[0].set_title("Prediction")
        axes[0].set_aspect('equal')

        # Add color bar to the first subplot
        cbar1 = plt.colorbar(scatter1, ax=axes[0])

        # Second subplot: Density Difference
        scatter2 = axes[1].scatter(
            x_next.cpu()[:, 0], 
            x_next.cpu()[:, 1], 
            c=Truedensity - color.cpu().numpy(), 
            marker='.'
        )
        axes[1].set_xlim([args.plot_x_min,args.plot_x_max])
        axes[1].set_ylim([args.plot_y_min,args.plot_y_max])
        axes[1].add_patch(circle2)  # Circle specific to the second plot
        axes[1].set_title(f"Density error: {density_error:.4f}")
        axes[1].set_aspect('equal')

        # Add color bar to the second subplot
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
    else:
        # First subplot: Prediction
        axes[0].scatter(x_next.cpu().numpy()[:, 0], color.cpu().numpy(), marker='.')
        axes[0].axvline(x=r, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axes[0].axvline(x=-r, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axes[0].set_xlim([-1, 1])
        axes[0].set_ylim([0, color.cpu().numpy().max()*1.1])
        axes[0].set_title("Predicted Density function")
        axes[0].set_xlabel("Position")
        axes[0].set_ylabel("Density")
        axes[0].grid(True)
        
        
        # Second subplot: Density Difference
        diff = Truedensity - color.cpu().numpy()
        axes[1].scatter(x_next.cpu().numpy()[:, 0], diff, marker='.')
        axes[1].set_xlim([-1, 1])
        axes[1].set_title(f"Density Error: {density_error:.4f}")
        axes[1].set_xlabel("Position")
        axes[1].set_ylabel("True - Predicted")
        axes[1].grid(True)
        

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(sPath)
    plt.close('all')

    return density_error



def plot_nonlocal(x_next, color, t, sPath):
    circle1 = plt.Circle((0, 0), 1, color='g', fill=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Create a scatter plot with color mapping
    scatter = ax.scatter(x_next.cpu()[:, 0], x_next.cpu()[:, 1], c=color.cpu(), cmap='viridis', marker='.' ,vmin=0, vmax=1)
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Color Intensity', rotation=270, labelpad=15)
    
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.add_patch(circle1)
    
    plt.tight_layout()
    ax.set_aspect('equal')
    plt.savefig(sPath)
    plt.close('all')

from scipy.special import gamma
def compute_radius(p, q):
    """Compute r0 analytically using Gamma functions."""
    numerator = gamma((p + 2)/2) * gamma((q + 3)/2)
    denominator = gamma((q + 2)/2) * gamma((p + 3)/2)
    return 0.5 * (numerator / denominator) ** (1 / (q - p))

def plot_aggregation(x_next, p,q,  sPath, args, color=None):
    r = compute_radius(p, q)
    center = x_next.mean(dim=0).cpu().numpy()
    circle_ref = plt.Circle((center[0], center[1]), r, color='red', fill=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.add_patch(circle_ref)
    
    if color is not None:
        # Create a scatter plot with color mapping
        scatter = ax.scatter(x_next.cpu()[:, 0], x_next.cpu()[:, 1], c=color.cpu(), cmap='viridis', marker='.' )#,vmin=0, vmax=1)
        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Color Intensity', rotation=270, labelpad=15)
    else:
        scatter = ax.scatter(x_next.cpu()[:, 0], x_next.cpu()[:, 1], marker='.' )
    
    
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.title(f"p_{p:.1f},q_{q:.1f}__radius {r:.2f}")
    plt.tight_layout()
    ax.set_aspect('equal')
    plt.savefig(sPath)
    plt.close('all')

def plot_kalman(x_next, color, t, sPath):
   
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.scatter(x_next.cpu()[:,0], x_next.cpu()[:,1], c=color.cpu(), marker='.')
    ax.set_xlim([-5,5])
    ax.set_ylim([90,110])
    
    ax.set_aspect('equal', 'box') 
    plt.tight_layout()
    plt.savefig(sPath)
    plt.close('all')


def create_grid(x_min, x_max, y_min, y_max, n_pts, dim, n_pts_y=None):
    """Returns points on the regular grid on [-width, width]^2, with n_pts on each dimension
    dim is used to pad the grid into the desired dimension with 0's

    Returns:
        grid: n_pts^2 x 2
        grid_pad: 
        xx (grid_x): n_pts x n_pts
        yy (grid_y): n_pts x n_pts
    """
    if n_pts_y is None:
        n_pts_y = n_pts
    x = np.arange(x_min, x_max, (x_max - x_min)/n_pts)
    y = np.arange(y_min, y_max, (y_max - y_min)/n_pts_y)

    xx, yy = np.meshgrid(x,y) # both have shape n x n
    grid = torch.Tensor(np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=-1)) # n^2 x 2
    # if dim > 2, pad the remaining coordinates as 0's
    n_sqr = grid.shape[0]
    d_pad = dim - 2
    grid_pad = torch.cat((grid, torch.zeros(n_sqr, d_pad)), dim=-1)

    return grid, grid_pad, xx, yy




def plot_gaussian(X_0, X_1_pred,  X_1=None, X_0_color='blue', X_1_pred_color='red', X_1_color='green', marker_size=8, save_dir='fig', N_corr=10, obs=None, net=None, args=None):
    # X: n x d
    f = plt.figure(figsize=(5,5/0.8))
    ax = f.add_subplot(111)

    for i in range(N_corr):
        ax.plot([X_0[i][0],X_1_pred[i][0]], [X_0[i][1],X_1_pred[i][1]], color='k')

    # plot samples
    ax.scatter(X_0[:,0], X_0[:,1], marker='.', s=marker_size, c=X_0_color, alpha=0.3, label='X_0')

    if X_1 is not None: 
        ax.scatter(X_1[:,0], X_1[:,1], marker='.', c=X_1_color, s=marker_size, alpha=0.5, label='X_1')

    ax.scatter(X_1_pred[:,0], X_1_pred[:,1], marker='.', s=marker_size, c=X_1_pred_color, label='Predicted X_1')
    


    ax.set_xlabel(r'$x$', loc='right')
    ax.set_ylabel(r'$y$', loc='top', rotation=0)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    ax.set_xlim([args.plot_x_min,args.plot_x_max])
    ax.set_ylim([args.plot_y_min,args.plot_y_max])

    f.savefig(save_dir)
    plt.close()

def plot_kl_jko_gif(X_0, X_t, X_1,  X_0_color='blue', X_t_color='red', X_1_color='green', marker_size=8, save_dir='fig', d=2,  net=None, args=None, N_corr=10):
    print("plotting: ", X_0.shape, X_t.shape, X_1.shape,  X_0_color.shape, X_t_color.shape, X_1_color.shape)
    print("plotting loss: ", args.loss_W2.shape, args.loss_E.shape)
    # Create the main figure and subplots
    f, axes = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplots in a row
    ax1, ax2, ax3 = axes  # Unpack the axes for easier use

    assert args is not None and args.plot_x_min != 0  # Ensure plot ranges are specified
    assert X_1 is not None

    # Compute trajectories
    X_0_GPU = X_0.unsqueeze(0).to(device)
    X_1_GPU = X_1.unsqueeze(0).to(device)
    if X_t is None:
        print("Warning generating X_t for gif... there might be something wrong")
        paths = []
        X_t = X_0_GPU
        with torch.no_grad():
            for i in range(args.jko_T):
                X_t += net(X_t, X_1_GPU)
                paths.append(X_t.clone())
        X_t = torch.cat(paths)
    
    X_t_cat0 = torch.cat( [ X_0_GPU.cpu(), X_t.cpu() ], dim=0 ).permute(1, 0, 2).cpu()
    X_t = X_t.permute(1, 0, 2).cpu()  # N_t x N_sample x d --> N_sample x N_t x d
    if not isinstance(X_t_color, str):
        if not isinstance(X_0_color, str):
            X_t_color = torch.cat( [ X_0_color.unsqueeze(0).cpu(), X_t_color.cpu() ], dim=0 )
        X_t_color = X_t_color.permute(1,0).cpu()

    # Update function for animation
    def update_scatter(frame):
        # Clear the axes before updating
        for ax in axes:
            ax.clear()

        #KL_true = kl_standard_gaussian(X_t_cat0[:, frame, :].detach().cpu().numpy())
        #print(KL_true)

        # Subplot 1: Scatter plot of X_0
        ax1.scatter(X_0[:, 0], X_0[:, 1], marker='.', s=marker_size, c=X_0_color, label='X_0')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_xlim([args.plot_x_min, args.plot_x_max])
        ax1.set_ylim([args.plot_y_min, args.plot_y_max])
        ax1.set_title('Initial distribution (X_0)')
        ax1.legend()

        # Subplot 2: Scatter plot of X_t at current frame
        if isinstance(X_t_color, str):            
            ax2.scatter(X_t[:, frame, 0], X_t[:, frame, 1], marker='.', s=marker_size, c=X_t_color, label=f'X_t (Frame {frame})', cmap='viridis')#, vmin=0, vmax=1)
        else:
            ax2.scatter(X_t_cat0[:, frame, 0], X_t_cat0[:, frame, 1], marker='.', s=marker_size, c=X_t_color[:, frame], label=f'X_t (Frame {frame})', cmap='viridis')#, vmin=0, vmax=1)
        # Plot some trajectories
        for i in range(N_corr):
            ax2.plot(X_t_cat0[i, :frame+1, 0], X_t_cat0[i, :frame+1, 1], color=(91 / 255, 195 / 255, 220 / 255), lw=0.8)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_xlim([args.plot_x_min, args.plot_x_max])
        ax2.set_ylim([args.plot_y_min, args.plot_y_max]) 
        if args.loss_E is not None:
            ax2.set_title(f'X_t: W2 {args.loss_W2[frame].item():.4f}, E {args.loss_E[frame].item():.3f}')
        else:
            ax2.set_title(f'Current distribution (X_t)')
        ax2.legend()

        # Subplot 3: Scatter plot of X_true (if available)
        if X_1 is not None:
            ax3.scatter(X_1[:, 0], X_1[:, 1], marker='.', c=X_1_color, s=marker_size, label='X_1 (Target)')
        ax3.set_xlabel(r'$x$')
        ax3.set_ylabel(r'$y$')
        ax3.set_xlim([args.plot_x_min, args.plot_x_max])
        ax3.set_ylim([args.plot_y_min, args.plot_y_max])
        ax3.set_title('Target distribution (X_1)')
        ax3.legend()

        # Ensure layout is clean
        plt.tight_layout()

    # Create the animation
    ani = animation.FuncAnimation(f, update_scatter, frames=args.jko_T+1, interval=500)

    # Save the animation as a GIF
    ani.save(save_dir, writer='pillow')  # You can set dpi with dpi=300 if needed
    plt.close()






# =================================================================================== #
#                                       logging                                       #
# =================================================================================== #
def prepare_Test_loggers(args):
    log_dir = os.path.join('..', 'results', args.dataset, f'step{args.deltat:.4f}_T{args.jko_T}_{args.exp_name}')
    if args.bestmodel:
        model_dir = os.path.join('..', 'savedModels')
    else:
        model_dir = os.path.join(log_dir, 'model')

    #model_dir = os.path.join('..', 'savedModels', args.dataset)
    Test_result_dir = os.path.join('..', 'results', args.dataset, f'Test_step{args.deltat:.4f}_T{args.jko_T}_{args.Test_name}')
    if os.path.exists(Test_result_dir):
        shutil.rmtree(Test_result_dir)
    os.makedirs(Test_result_dir)

    return log_dir, model_dir, Test_result_dir

def prepare_loggers(args):
    log_dir, plot_dir, model_dir, hist_dir = make_log_dir(args)
    # plot_dir = os.path.join(log_dir, 'plot')
    # model_dir = os.path.join(log_dir, 'model')
    tbx = SummaryWriter(log_dir=log_dir, max_queue=20)
    # log args
    tbx.add_text(tag='args', text_string='CUDA_VISIBLE_DEVICES=0 python ' + " ".join(x for x in sys.argv))
    filename = os.path.join(log_dir, 'config.json')
    with open(filename, 'w') as file:
        json.dump(vars(args), file)
    print_args(sys.argv, save_dir=os.path.join(log_dir, 'args.txt'))

    with open(os.path.join(log_dir, 'args.txt'), 'a') as f:
        f.write(f'\n\n\n\n')
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    txt_logger = Logger(os.path.join(log_dir, 'log.txt'), hist_dir=hist_dir)

    return log_dir, plot_dir, model_dir, txt_logger, tbx

def make_log_dir(args):
    log_dir = os.path.join('..', 'results', args.dataset, f'step{args.deltat:.4f}_T{args.jko_T}_{args.exp_name}')
    plot_dir = os.path.join(log_dir, 'plot')
    model_dir = os.path.join(log_dir, 'model')
    hist_dir = os.path.join(log_dir, 'hist')
    if os.path.exists(log_dir):
        #choice = input("Directory already exists. Delete previous results? (y/n)")
        choice = 'y'
        if choice == 'y':
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)
            os.makedirs(plot_dir)
            os.makedirs(model_dir)
            os.makedirs(hist_dir)
    else:
        os.makedirs(log_dir)
        os.makedirs(plot_dir)
        os.makedirs(model_dir)
        os.makedirs(hist_dir)

    return log_dir, plot_dir, model_dir, hist_dir

def print_args(args, save_dir='args.txt'):
    """Intended usage: print_args(sys.argv)
    """
    # If this is not the first time writing to the same args.txt file
    # don't overwrite it, just keep writing below.
    if os.path.exists(save_dir):
        with open(save_dir, 'a') as f:
            f.writelines('\n')
            f.writelines('python ')
            for s in args:
                f.writelines(s + ' ')
    else:
        with open(save_dir, 'w') as f:
            f.writelines('python ')
            for s in args:
                f.writelines(s + ' ')

class Logger(object):
    def __init__(self, fpath=None, hist_dir=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'a')

        self.hist_dir = hist_dir

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        # self.console.write(msg)
        print(msg)
        if self.file is not None:
            self.file.write(msg + '\n')

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


    def write_to_file(self, fname, line_list, mode):
        file_dir = os.path.join(self.hist_dir, mode + '_' + fname+'.txt')
        file = open(file_dir, 'a')
        for a in line_list:
            file.write(str(float(a)) + '\n')
        file.close()
    
    # def print_and_write(self, msg):
    #     print(msg)
    #     self.write(msg + '\n')


# =================================================================================== #
#                                        Misc.                                        #
# =================================================================================== #

# def AL(net, mmd, X_0, X_1, X_0_2, v, beta):
#     X_1_pred = net(X_0, X_1, X_0_2) # B x n x d
#     loss_mmd = mmd(X_1_pred, X_1)
#     loss_OT  = L(X_0, X_1_pred)
#     AL_value = loss_OT.mean() + v*loss_mmd + beta/2*loss_mmd**2

#     return AL_value
    

def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters

def reset_loss_hist(d):
    for key in d:
        d[key] = []
    
    return d

def log_loss_hist(loss_hist, logger, iter=0, mode='Training'):
    for l_name in loss_hist:
        l_hist = torch.cat(loss_hist[l_name])
        l_std, l_mean = torch.std_mean(l_hist)
        logger.write("{}, i={}, avg {}: {:.10f} +- {:.10f}".format(\
                    mode, iter, l_name, l_mean, 2*l_std/np.sqrt(len(l_hist))))
        logger.write_to_file(l_name, l_hist, mode)
        if l_name == 'loss':
            loss_mean = l_mean

    return loss_mean
