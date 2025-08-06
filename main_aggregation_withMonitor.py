import torch, os, sys, shutil
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.distributions as D
import torch.nn as nn
from arguments import parse_arguments
#？？
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from time import sleep
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
#？？
from GNN import *
#from models import *
from utils import *
from loss_jko import *
from dataset_utils import *


#####THIS IS MAINLY DEALT WITH ONE SINGLE INSTANCE########
# =================================================================================== #
#                                        Meta                                         #
# =================================================================================== #
args = parse_arguments()
args.concat_densityvalue = True
print("args.concat_densityvalue is set to True")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log_dir, plot_dir, model_dir, txt_logger, tbx = prepare_loggers(args)



loss_log_path = os.path.join(log_dir, 'loss_vector_log.txt')
with open(loss_log_path, 'w') as f:
    f.write("Iteration, LossVector\n") 

# =================================================================================== #
#                                        Data                                         #
# =================================================================================== #
assert args.dataset in ( 'aggregation'), "dataset mismatch" 
train_generator, test_generator, d = prepare_data(args)



#The initial data is fixed
X = next(train_generator)

print("sample shape", X[0].shape) #B,n,d
print("density shape", X[1].shape) #B,2

# =================================================================================== #
#                                       Network                                       #
# =================================================================================== #
if args.savedModelName != '':
    print("Loading saved Model")
    saved_model_path = os.path.join(log_dir+'_saved', 'model',args.savedModelName)
    model = torch.load(saved_model_path,weights_only=False)
    ## 加载路径是什么
    net = model.to(device)
    args.WarmUpEpoch = 0
else:
    print("creating model from scratch")
    model = eval(args.model)
    #eval 是什么
    net = model(d, args).to(device)
net.train()
net_modules = [module for k, module in net._modules.items()]
txt_logger.write('There are {} trainable parameters in the network.'.format(get_num_parameters(net)))


decay, no_decay = [], []
for n, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if any(nd in n for nd in ["bias", "LayerNorm.weight", "embeddings"]):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [
      {"params": decay,    "weight_decay": 0.0},
      {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=args.lr,
)
grad_log_path = os.path.join(log_dir, 'grad_log.txt')
with open(grad_log_path, 'w') as f:
    f.write("Iteration, grad_norm, weights_norm, ratio\n") 
#optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)



# scheduler
if args.lr_scheduler == 'cyclic':
    scheduler = CosineAnnealingLR(optimizer, args.N_iter_final, 1e-6, last_epoch=-1)
else:
    print("No schduler is applied")
    scheduler = None



# =================================================================================== #
#                                       Training                                      #
# =================================================================================== #


# In main_aggregation_withMonitor.py, line 109 should be:
GenerateNewAggregationData(X[0].to(device), X[1].to(device), X[2].to(device), net, args)
##调用 GenerateNewAggregationData 函数，用当前模型 net 在输入数据 X 上生成新的训练数据，并将其存储在 args.TrainingData 中。

loss_hist = {'loss':[], 'W2':[], 'E':[]}
loss_hist_all = {'loss':[], 'W2':[], 'E':[]}
for i in tqdm(range(args.N_iter)):
    print(args.TrainingData[0].shape, args.TrainingData[1].shape)
    # save model
    if (i+1) % args.save_interval == 0:
        torch.save(net, os.path.join(model_dir, '{}.pth'.format(i)))
##每隔 args.save_interval 次迭代就保存一次模型，保存为如 100.pth, 200.pth 等。

    loss, loss_W2, loss_E, X_pred = compute_JKO_aggregation_cost_NoDatagenerating(
    args.TrainingData[0],
    args.TrainingData[1],
    net,
    args,
    batch=args.TrainingData[2]  # Pass the batch info
)
    
    writeLossToFile(loss_W2/(2*args.deltat)+loss_E,  i, loss_log_path, args)
    

        
#with 进入 no_grad 模式：不会记录计算图，加快执行速度、节省内存，适合用于评估或数据处理。

    with torch.no_grad():
        if should_generate_new_data(i, (loss_W2/(2*args.deltat)+loss_E).detach(), args):

            with open(loss_log_path, 'a') as f:
                f.write("New data generated..\n")
            
            
            # Explicit cleanup before generating new data
            del loss, loss_W2, loss_E, X_pred
            
            
            
        
            GenerateNewAggregationData(X[0].to(device), X[1].to(device), X[2], net, args)


            
            args.current_stall_count = 0
            args.ChangePoint.append( len( loss_hist_all['loss'] )  )
            continue
    

    

    # optimize
    optimizer.zero_grad()
    loss.backward() 

    g_norm, w_norm, r_val = check_gradUpdate(net,loss,i)
    with open(grad_log_path, 'a') as f:
        f.write(f"# Iter {i}: {g_norm:3f}   {w_norm:3f}   {r_val:3f}\n")

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    
    g_norm, w_norm, r_val = check_gradUpdate(net,loss,i)
    with open(grad_log_path, 'a') as f:
        f.write(f"# Iter {i}: {g_norm:3f}   {w_norm:3f}   {r_val:3f}\n\n")


    optimizer.step()
    args.current_stall_count +=1
    if args.lr_scheduler == 'cyclic' and i < args.N_iter_final:  
        scheduler.step()  # LR decays smoothly to eta_min by i=2000
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current LR: {current_lr:.6f}")
    


    # log
    loss_hist['loss'].append(loss.unsqueeze(-1))
    loss_hist['E'].append(loss_E)
    loss_hist['W2'].append(loss_W2)

    loss_hist_all['loss'].append(loss.flatten())
    
    if loss_E.numel()< args.B_dist*(args.jko_T+1):
        dd =  args.B_dist*(args.jko_T+1)- loss_E.numel()
        E = torch.cat([loss_E.detach().cpu().flatten(), torch.zeros(dd)])
        W2 = torch.cat([loss_W2.detach().cpu().flatten(), torch.zeros(dd)])
        loss_hist_all['E'].append( E)
        loss_hist_all['W2'].append(W2)
    else:
        loss_hist_all['E'].append(loss_E.detach().cpu().flatten())
        loss_hist_all['W2'].append(loss_W2.detach().cpu().flatten())
    
    tbx.add_scalar(tag='loss', scalar_value=float(loss), global_step=i)
    tbx.add_scalar(tag='E', scalar_value=float(loss_E.mean()), global_step=i)
    tbx.add_scalar(tag='W2', scalar_value=float(loss_W2.mean()), global_step=i)

    # log
    
    if i % args.log_interval == 0 or i == args.N_iter - 1:
        with torch.no_grad():
            X_pred = args.TrainingData[0][::args.B_dist]
            p = X[1][0,0]
            q = X[1][0,1]

            epoch_dir = os.path.join( plot_dir, f"Epoch_{i}_{len(loss_hist_all['loss'])}" )
            os.makedirs(epoch_dir, exist_ok=True)  # This creates the directory if it doesn't exist

            t = 0           

            for j in range(X_pred.shape[0]           ):
                plot_aggregation(X_pred[j].detach(), p,q, os.path.join(epoch_dir, f'Epoch_{i}_T_{t:.3f}.png'),args)                     
                t += args.deltat            
                
                
        loss_hist = reset_loss_hist(loss_hist)
    

changePoint_log_path = os.path.join(log_dir, 'Changepoint.txt')
with open(changePoint_log_path, 'w') as f:
    f.write(", ".join(map(str, args.ChangePoint)))

print(f"Attempting to save plot in: {log_dir}")
print(f"Directory exists: {os.path.exists(log_dir)}")
print(f"Directory writable: {os.access(log_dir, os.W_OK)}")
plot_loss(loss_hist_all, args, log_dir)

print ('Done!')

net.eval()
with torch.no_grad():
    args.jko_T = args.jko_T_test
    
    for i in range(args.B_dist_test):
        X = next(test_generator)
        X0 = X[0].to(device)
        X1 = X[1].to(device)
        # Generate data
        batch_test = torch.zeros(X0.shape[1], dtype=torch.long, device=X0.device)
        GenerateNewAggregationData(X0, X1, batch_test, net, args)
        X_pred = args.TrainingData[0]
        p = X[1][0,0]
        q = X[1][0,1]  
        
        # Create test-specific directory
        test_dir = os.path.join(plot_dir, f"Test_{i}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Plotting loop
        for j in range(args.jko_T):
            t = j * args.deltat
            # Use consistent zero-padding for filenames
            filename = os.path.join(test_dir, f'{t:.1f}.png')
            plot_aggregation(
                X_pred[j].detach().cpu(),  # Ensure tensor is on CPU for plotting
                p, q, 
                filename,
                args
            )
            
print('Testing Done!')
