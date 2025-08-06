import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import knn_graph

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



class ParticleGNN(nn.Module):
    def __init__(self, d, args):
        super().__init__()
        self.args = args
        self.act_dict = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'mish': nn.Mish(),
            'leakyrelu': nn.LeakyReLU(0.2)
        }
        self.act = self.act_dict[args.nn_act]


        # Adjust input dimension if concatenating pq parameters
        self.res_linear = nn.Identity() if d == args.h else nn.Linear(d, args.h)

        # GATConv Layers
        input_dim = d
        self.conv1 = pyg.nn.GATConv(input_dim, args.h // 2, heads=4)  # output: 4 * (h//2) = 2h
        self.conv2 = pyg.nn.GATConv(2 * args.h, args.h // 2, heads=4) # output: 4 * (h//2) = 2h
        self.conv3 = pyg.nn.GATConv(2 * args.h, args.h, heads=1)      # output: h

        # MLP: h → h → h//2 → d
        self.mlp = nn.Sequential(
            nn.Linear(args.h, args.h),
            self.act,
            nn.Dropout(args.dropout_p),
            nn.Linear(args.h, args.h // 2),
            self.act,
            nn.Linear(args.h // 2, d)
        )
    def forward(self, x, batch=None):
        
        original_shape = None
        
        B, N, d = x.shape
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        
        if batch is None or batch.numel() != B*N:
            batch = torch.arange(B, device=x.device).repeat_interleave(N)

        # 在当前设备上构建 k-NN 图；k 用可调参数 args.k_nn
        # 1 · 取 k
        k = getattr(self.args, "k_nn", 16)

        # 2 · 若调用方没给 batch，就自己造
        if batch is None or batch.numel() != x.size(0) * x.size(1):
            B, N, _ = x.size()
            batch = torch.arange(B, device=x.device).repeat_interleave(N)

        # 3 · knn_graph 只能吃 CPU Tensor —— 先搬到 CPU，算完再搬回原设备
        x_cpu    = x.reshape(-1, x.size(-1)).cpu()
        batch_cpu = batch.cpu()
        edge_index = knn_graph(x_cpu, k=k, batch=batch_cpu)

        # 4 · 放回原设备，供后续 message passing 使用
        edge_index = edge_index.to(x.device)

        
        
#       edge_index = pyg.utils.dense_to_sparse(torch.ones(x.size(0),x.size(0),device=x.device))[0]


        y = self.conv1(x_flat, edge_index)  # 输出维度: 2h
        y = self.act(y)

        y = self.conv2(y, edge_index)       # 输出维度: 2h
        y = self.act(y)

        y = self.conv3(y, edge_index)       # 输出维度: h
        y = self.act(y)

# 残差连接（可选）
        res = self.res_linear(x_flat)
        y = y + res

# MLP 输出（维度: h → h → h//2 → d）
        y = self.mlp(y)

        return y.reshape(B, N, -1)
