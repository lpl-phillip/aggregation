import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def compute_JKO_aggregation_cost_NoDatagenerating(path, pq, net, args, batch=None):
    B, N, d = path.shape
    net.train()

 
    X_t = path.to(device)
    # Network forward pass
    V_t = net(X_t, batch)                       # 把 dataloader 给你的 batch 直接传进网络
    V_t = V_t - V_t.mean(dim=1, keepdim=True)   # 消掉整体平移（和数据生成阶段保持一致）

    print(path.shape,V_t.shape)
    path_T_pred = path + V_t                    # (B, N, d)

    ########### transport cost ###########
    W2 = (V_t ** 2).sum(dim=[1, 2]) / N        # (B,)

    ########### potential energy #########
    distances = torch.cdist(path_T_pred, path_T_pred)  # (B, N, N)
    p = (pq[:, 0,0]+1).view(-1, 1, 1)        # (B, 1, 1)
    q = (pq[:, 0,1]+1).view(-1, 1, 1)        # (B, 1, 1)
    print(distances.shape,p.shape)
    # Aggregation energy term E
    E = (distances.pow(q) / q - distances.pow(p) / p).mean(dim=[1, 2])
   
    print("Energy", (W2 / (2 * args.deltat) + E).mean())

    return (
        (W2 / (2 * args.deltat) + E).mean(),   # Scalar loss
        W2.detach().cpu(),                      # Wasserstein loss
        E.detach().cpu(),                       # Potential energy
        path_T_pred.detach().cpu()              # Updated points
    )




#import torch
#import numpy as np
#from scipy.special import gamma
#
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#
#def compute_radius_batch(p_arr, q_arr):
#    p = np.asarray(p_arr, dtype=float)
#    q = np.asarray(q_arr, dtype=float)
#    num = gamma((p + 2) / 2) * gamma((q + 3) / 2)
#    den = gamma((q + 2) / 2) * gamma((p + 3) / 2)
#    return 0.5 * (num / den) ** (1.0 / (q - p))
#
#def compute_JKO_aggregation_cost_NoDatagenerating(path, pq, net, args, batch=None):
#    B, N, d = path.shape
#    net.train()
#
#    # 1) predict velocity
#    X_t = path.to(device)
#    V_t = net(X_t, batch)
#    V_t = V_t - V_t.mean(dim=1, keepdim=True)
#
#    # 2) transport cost W2
#    W2 = (V_t ** 2).sum(dim=[1, 2]) / N
#
#    # 3) potential energy E
#    path_T_pred = path + V_t
#    distances = torch.cdist(path_T_pred, path_T_pred)
#    p = pq[:, 0, 0].view(-1, 1, 1)
#    q = pq[:, 0, 1].view(-1, 1, 1)
#    E = (distances.pow(q) / q - distances.pow(p) / p).mean(dim=[1, 2])
#
#    # 4) boundary‐overflow penalty
#    p_np = p.view(-1).cpu().numpy()
#    q_np = q.view(-1).cpu().numpy()
#    r_np = compute_radius_batch(p_np, q_np)
#    r = torch.from_numpy(r_np).float().to(device).view(-1, 1)  # float32 for MPS
#
#    radial = path_T_pred.norm(dim=2)              # (B, N)
#    overflow = torch.relu(radial - r)             # (B, N)
#    penalty = overflow.pow(2).mean(dim=1)         # (B,)
#
#    # 5) combine into final loss
#    λ = 2
#    loss = (W2 / (2 * args.deltat) + E + λ * penalty).mean()
#
#    return loss, W2.detach().cpu(), E.detach().cpu(), path_T_pred.detach().cpu()

