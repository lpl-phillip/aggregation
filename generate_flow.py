import torch, os, sys, shutil
main_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, main_dir)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Important for parallel processing
import matplotlib.pyplot as plt
import torch.distributions as D
import torch.nn as nn
from arguments import parse_arguments
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from time import sleep
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from models import *
from utils import *
from loss_jko import *
from dataset_utils import *
import imageio.v2 as imageio
from scipy.special import gamma
from functools import partial
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor,as_completed
from shape_data import *

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')


def aggregation_V(x, p, q):
    """
    x : (N,d) torch.Tensor  (on CPU or CUDA)
    Returns velocity tensor of the same shape.
    """
    N  = x.size(0)
    diff = x.unsqueeze(1) - x.unsqueeze(0)           # (N,N,d)
    dist = diff.norm(dim=-1)                         # (N,N)
    dist.fill_diagonal_(1.0)                         # in-place safe guard
    F = dist.pow(p)-dist.pow(q)
    coef = F / dist
    v = (coef.unsqueeze(-1) * diff).sum(dim=1) / N
    return v

# Keep the helper functions the same
def generate_ring(r, n=1000):
    r = np.asarray(r)
    theta = np.linspace(0, 2 * np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    points = np.stack((x, y), axis=-1)
    if r.ndim == 0:
        return r * points
    else:
        return r[:, None, None] * points[None, :, :]

def compute_radius_batch(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    numerator = gamma((p + 2) / 2) * gamma((q + 3) / 2)
    denominator = gamma((q + 2) / 2) * gamma((p + 3) / 2)
    r0 = 0.5 * (numerator / denominator) ** (1 / (q - p))
    return r0

def detect_ring(X, pq):
    assert X.ndim == 3, "Input tensor X must have shape (B, n, d)"
    
    # Compute the center of the ring (centroid of points)
    center = X.mean(dim=1, keepdim=True)  
    X_centered = X - center  # Shape: (B, n, d) - now centered at origin
    
    p = pq[:, 0].cpu().numpy()
    q = pq[:, 1].cpu().numpy()
    r = compute_radius_batch(p, q)
    points = generate_ring(r, n=X.shape[1])
    X_ref = torch.tensor(points, dtype=X.dtype, device=X.device)
    
    # Use centered points for chamfer calculation
    return chamfer_distance_batch(X_centered, X_ref)

#def process_single_pair(p_val, q_val, net, args, device, Test_result_dir):
def process_single_pair(p_val, q_val,  args, Test_result_dir):
    import os
    import time
    process_id = os.getpid()
    start_time = time.time()
    print(f"Process {process_id} started processing: p={p_val:.2f}, q={q_val:.2f}")
    with torch.no_grad():
        batch_p = torch.tensor([p_val])
        batch_q = torch.tensor([q_val])
        
        # Initialize samples
        #if torch.rand(1).item() < 0.5:
        #samples = generate_random_rectangles(B=1, sample_size=1024)
        #else:
        #    samples = generate_random_triangles(B=1, sample_size=1024)
        samples = get_random_shape_data()
        samples = samples.to(device)
        pq = torch.stack([batch_p, batch_q], dim=1).to(device)
        batch = torch.zeros(samples.shape[1], dtype=torch.long, device=device)  # shape: (N,)

        pq_rep = pq.unsqueeze(1).repeat(1, samples.shape[1], 1)
        
        X_t = samples
        trajectory = [X_t[0].clone()]
        W2s = []
        V_errors = []
        
        # Evolution loop
        for k in range(args.jko_T):            

            code = net.encode(X_t, pq_rep, None)
            V_t = net.decode(code, X_t.squeeze(0), batch)   # 输入改为 (N, d), batch
            V_t = V_t.unsqueeze(0)                          # 恢复成 (1, N, d)
            V_t = V_t - V_t.mean(dim=1, keepdim=True)

            W2 = (V_t**2).sum(dim=[1, 2]) / V_t.shape[1]
            W2s.append(W2.max().item())
            
            V_ref = args.deltat* aggregation_V(X_t[0].detach().cpu(), p_val, q_val)
            V_diff = V_t[0].detach().cpu() - V_ref            
            V_errors.append( (torch.norm(V_diff.flatten()) / torch.norm(V_ref.flatten())).item() )

            # Visualize velocities with better analysis
            fig = plt.figure(figsize=(20, 12))
            
            # Plot 1: V_ref
            ax1 = plt.subplot(2, 4, 1)
            X_np = X_t[0].detach().cpu().numpy()
            V_ref_np = V_ref.numpy()
            q1 = ax1.quiver(X_np[:, 0], X_np[:, 1], V_ref_np[:, 0], V_ref_np[:, 1]) 
                           #scale_units='xy', scale=1, alpha=0.7)
            ax1.set_title(f'Reference Velocity (max: {np.max(np.linalg.norm(V_ref_np, axis=1)):.3f})')
            ax1.set_aspect('equal')
            
            # Plot 2: V_t
            ax2 = plt.subplot(2, 4, 2)
            V_t_np = V_t[0].detach().cpu().numpy()
            q2 = ax2.quiver(X_np[:, 0], X_np[:, 1], V_t_np[:, 0], V_t_np[:, 1])
                          # scale_units='xy', scale=1, alpha=0.7)
            ax2.set_title(f'Predicted Velocity (max: {np.max(np.linalg.norm(V_t_np, axis=1)):.3f})')
            ax2.set_aspect('equal')
            
            # Plot 3: V_diff (scaled for visibility)
            ax3 = plt.subplot(2, 4, 3)
            V_diff_np = V_diff.numpy()
            diff_magnitude = np.linalg.norm(V_diff_np, axis=1)
            q3 = ax3.quiver(X_np[:, 0], X_np[:, 1], V_diff_np[:, 0], V_diff_np[:, 1]) 
                           #scale_units='xy', scale=1, alpha=0.7, color='red')
            ax3.set_title(f'Velocity Difference (max: {np.max(diff_magnitude):.3f})')
            ax3.set_aspect('equal')
            
            # Plot 4: Magnitude comparison
            ax4 = plt.subplot(2, 4, 4)
            ref_magnitude = np.linalg.norm(V_ref_np, axis=1)
            pred_magnitude = np.linalg.norm(V_t_np, axis=1)
            ax4.scatter(ref_magnitude, pred_magnitude, alpha=0.6, s=20)
            ax4.plot([0, max(ref_magnitude.max(), pred_magnitude.max())], 
                    [0, max(ref_magnitude.max(), pred_magnitude.max())], 'r--', label='Perfect match')
            ax4.set_xlabel('Reference Magnitude')
            ax4.set_ylabel('Predicted Magnitude')
            ax4.set_title('Magnitude Correlation')
            ax4.legend()
            
            # Plot 5: Error heatmap
            ax5 = plt.subplot(2, 4, 5)
            pointwise_error = diff_magnitude / (ref_magnitude + 1e-8)  # Relative error per point
            scatter = ax5.scatter(X_np[:, 0], X_np[:, 1], c=pointwise_error, 
                                 cmap='hot', s=30, alpha=0.8)
            ax5.set_title('Pointwise Relative Error')
            ax5.set_aspect('equal')
            plt.colorbar(scatter, ax=ax5)
            
            # Plot 6: Angle difference
            ax6 = plt.subplot(2, 4, 6)
            # Compute angle difference
            ref_angles = np.arctan2(V_ref_np[:, 1], V_ref_np[:, 0])
            pred_angles = np.arctan2(V_t_np[:, 1], V_t_np[:, 0])
            angle_diff = np.abs(ref_angles - pred_angles)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Take smaller angle
            
            scatter = ax6.scatter(X_np[:, 0], X_np[:, 1], c=angle_diff, 
                                 cmap='plasma', s=30, alpha=0.8)
            ax6.set_title('Angle Difference (rad)')
            ax6.set_aspect('equal')
            plt.colorbar(scatter, ax=ax6)
            
            
            # Plot 7: Angle difference
            ax7 = plt.subplot(2, 4,7 )
                      
            scatter = ax7.scatter(X_np[:, 0], X_np[:, 1], c=angle_diff, 
                                 cmap='plasma', s=30, alpha=0.8, vmax=0.5)
            ax7.set_title('Angle Difference (max 0.5)')
            ax7.set_aspect('equal')
            plt.colorbar(scatter, ax=ax7)
            
            # Plot 8: Error histogram
            ax8 = plt.subplot(2, 4, 8)
            ax8.hist(pointwise_error, bins=50, alpha=0.7, edgecolor='black')
            ax8.set_xlabel('Pointwise Relative Error')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Error Distribution')
            ax8.axvline(pointwise_error.mean(), color='red', linestyle='--', label=f'Mean: {pointwise_error.mean():.3f}')
            ax8.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(Test_result_dir, f'{process_id}_iter_{k}.png'), dpi=150)
            plt.close()



            X_t = X_t + args.deltat * V_t

            trajectory.append(X_t[0].clone())

            
            
            #if W2.max() < 1e-5:
            #    break

        # Plot W2s using semilogy with scatter
        plt.figure()
        plt.semilogy()  # Set y-axis to log scale
        plt.scatter(range(len(V_errors)), V_errors, marker='o')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('V_error')
        plt.title(f'V_errors evolution: p={p_val:.1f}, q={q_val:.1f}')
        plt.savefig(os.path.join(Test_result_dir, f'V_errors_p_{p_val:.1f}_q_{q_val:.1f}_{process_id}.png'))
        plt.close()
        
        
        # Compute chamfer 
        chamfer = detect_ring(X_t, pq)        
        
        # Generate GIF and images
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        r = compute_radius_batch(p_val, q_val)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal')
        plt.title(f"p_{p_val:.1f},q_{q_val:.1f}__radius {r:.2f}")
        plt.tight_layout()

        images = []
        # Create directory for single images
        img_dir = os.path.join(Test_result_dir, f"p_{p_val:.1f}_q_{q_val:.1f}_{process_id}")
        os.makedirs(img_dir, exist_ok=True)

        for i, X_frame in enumerate(trajectory):
            X_np = X_frame.cpu().numpy()

            ax.clear()
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_aspect('equal')
            plt.title(f"p_{p_val:.1f},q_{q_val:.1f}__radius {r:.2f}")

            # ✅ 固定在 (0,0) 的红圈
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = r * np.cos(theta)
            circle_y = r * np.sin(theta)
            ax.plot(circle_x, circle_y, 'r', linewidth=1)

            ax.scatter(X_np[:, 0], X_np[:, 1], marker='.')

            fig.canvas.draw()

            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)

            frame_path = os.path.join(img_dir, f"frame_{i:03d}.png")
            plt.savefig(frame_path)

            
            # Convert to image array
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            
            # Save individual frame
            frame_path = os.path.join(img_dir, f"frame_{i:03d}.png")
            plt.savefig(frame_path)

        plt.close(fig)
        
        # Save GIF
        gif_path = os.path.join(Test_result_dir, f"p_{p_val:.1f},q_{q_val:.1f}_{process_id}.gif")
        imageio.mimsave(gif_path, images, duration=1000)  # Increased duration to 200ms per frame
        
        end_time = time.time()
        print(f"Process {process_id} finished p={p_val:.2f}, q={q_val:.2f} in {end_time-start_time:.2f}s, chamfer={chamfer.item():.4f}")
        
        return p_val, q_val, chamfer.item()

def init_worker(shared_model_state):
    """Initialize worker with shared model"""
    # Initialize CUDA once per process
    torch.cuda.init()
    global net
    net = torch.load(shared_model_state,  weights_only=False,map_location=device)
    net.share_memory()  # Enable sharing for potential further parallelization
    net.eval()
    torch.cuda.empty_cache()

def process_batch(pq_batch,  args, Test_result_dir):
    chamfer_results = []
    for p_val, q_val in pq_batch:
        _,_,chamfer = process_single_pair(p_val, q_val,  args, Test_result_dir)
        chamfer_results.append((p_val, q_val, chamfer))
    return chamfer_results

def main():
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()
    # Use the global device variable set at the top of the file
    log_dir, model_dir, Test_result_dir = prepare_Test_loggers(args)

    # Data and Network setup
    assert args.dataset in ('aggregation',), "dataset mismatch" 
    train_generator, test_generator, d = prepare_data(args)

    print("Loading saved Model")
    saved_model_path = os.path.join(log_dir+'_saved', 'model', args.savedModelName)
    #model = torch.load(saved_model_path, weights_only=False)
    #net = model.to(device)
    
    with torch.no_grad():
        #net.eval()
        args.jko_T = args.jko_T_test

        # Manually define p, q pairs
        #work_items = [ (0.5, 1.0), (0.2, 2.0), (0.5, 3.0), (0.5, 6.0) ]
        #work_items = [ (-0.5,0.5), (-0.5,0.5),(-0.5,0.5), (0.5, 3.0),(0.5, 3.0), (0.5, 3.0),(0.5, 3.0),(0.5, 3.0),(0.5, 3.0), (0.5, 3.0),(0.5, 3.0),(0.5, 3.0),(0.5, 3.0),(0.5, 3.0),(0.5, 3.0),(0.5, 3.0),(0.5, 3.0)]
        #work_items = [ (-0.5,0.5), (0.2, 2.0), (0.5, 3.0), (0.5, 6.0), (2,50), (-0.5,0.5), (0.2, 2.0), (0.5, 3.0), (0.5, 6.0),  (-0.5,0.5), (0.2, 2.0), (0.5, 3.0), (0.5, 6.0), (2,50)]
        work_items = [ (0.5, 4.0), (0.5, 4.0)]

        print(f"Total work items to process: {len(work_items)}")
        num_workers = 20
        print(f"{num_workers} workers processing in parallel")
        batch_size = len(work_items) // num_workers + 1
        # Split work items into batches
        work_batches = [work_items[i:i + batch_size] for i in range(0, len(work_items), batch_size)]

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(saved_model_path,)  # Pass path instead of model object
        ) as executor:
            futures = [
                executor.submit(
                    process_batch, work_batch,  args, Test_result_dir)
                for work_batch in work_batches
            ]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())        


    print('Testing Done!')

if __name__ == '__main__':
    main()
