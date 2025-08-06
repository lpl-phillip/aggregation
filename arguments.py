import argparse
import warnings

def parse_arguments():
    parser = argparse.ArgumentParser()

    #for Testing
    parser.add_argument(
    '--k_nn',
    type=int,
    default=16,
    help='number of neighbours used when building knn_graph'
)

    parser.add_argument('--TestJKO', type=int, default=10)
    parser.add_argument('--ModelPath', type=str, default="")
    parser.add_argument('--num_workers', type=int, default=0)


    # meta
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--Test_name', type=str, default='1')
    parser.add_argument('--savedModelName', type=str, default='')
    #parser.add_argument('--savedModelList',  default=[])
    parser.add_argument(  '--savedModelList',  nargs='+',  help='one or more model-basename(s) (no .pth)')
    parser.add_argument('--TestPlotDetail', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--TestGridDetail', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--Test_n_grid_pts', type=int, default=100)
    parser.add_argument('--Test_grid_range', type=float, default=1)

    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--optimizer_dir', type=str, default='')
    # data, there might be problem in main.py for modelnet, mnist, mnist_img
    parser.add_argument('--dataset', type=str, default='gaussian_gaussian',
                                    choices=['gaussianmix_1to2','gaussian_gaussian', 'mixgaussianMoon_bidirection', 'mixgaussianMoon_2to1','gaussianmix_2to1','gaussianmix_split','gaussianmix_merge',
                                     'gaussian_test', 'gaussian_gaussian_FixedVariance','gaussian_gaussian_SameVariance', 'gaussianmix_single_wlinv','gaussian_gaussian_DiffVariance',
                                             'gaussian_gaussianFixed', 'gaussianMixture_gaussianFixed', 'gaussianmix_single', 'gaussianmix_single_wl', 'gaussianmix','gaussianmix_1to2_bidirection',
                                             'MNIST_full', 'MNIST_img', 'MNIST_SAMPLE', 'crowd_motion', 'crowd_motion_single', 'aggregation', 'porous', 'nonlocal','kalman',
                                             'crowd_motion_two', 'modelnet'])
    parser.add_argument('--dim',         type=int, default=2, help='data dimension')
    parser.add_argument('--B_dist',       type=int, default=1, help='Batchsize of the distributions')
    parser.add_argument('--B_sample',       type=int, default=1024, help='the number of sample points from one distribution')
    parser.add_argument('--N_test',           type=int, default=100)
    parser.add_argument('--B_dist_test',       type=int, default=1)
    parser.add_argument('--B_sample_test',       type=int, default=1024)
    parser.add_argument('--n_landmk',       type=int, default=16)
    parser.add_argument('--jko_T',       type=int, default=16)
    parser.add_argument('--jko_T_test',       type=int, default=16)
    parser.add_argument('--bestmodel', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--gamma',         type=float, default=1, help='path weight')


    #save the tarining data
    parser.add_argument('--TrainingData',  default=None, help='Generated training data, which is empty at the begining')
    parser.add_argument('--ReferenceLoss',  default=None, help='ReferenceLoss in terms of --TrainingData ')
    parser.add_argument('--monitor',  type=str,  default='cumulative', choices=['inidividual', 'cumulative'], help='ReferenceLoss in terms of ? loss ')
    parser.add_argument('--WarmUpEpoch',  type=int, default=400, help='Regenerate data for the first WarmUpEpoch epochs ')
    parser.add_argument('--ChangePoint',  default=[], help='The epoch number when generate new data')
    parser.add_argument('--max_stall_epochs', type=int, default=88888888, help='Epochs to wait before moving on if training stalls (no progress)')
    parser.add_argument('--current_stall_count', type=int, default=0, help='Tracks consecutive stalled epochs on current data batch')


    parser.add_argument('--porous_m',       type=int, default=2)
    ## aggregation
    parser.add_argument('--aggregationChoice',   type=int, default=1)
    parser.add_argument('--aggregation_a',       type=float, default=4)
    parser.add_argument('--aggregation_b',       type=float, default=2)
    ## gaussian-gaussian
    parser.add_argument('--mu_1_offset',       type=float, default=5e-1)
    parser.add_argument('--sigma_max',       type=float, default=5e-1)
    ## gaussian-gaussian_mixture
    parser.add_argument('--ggm_sigma_min',       type=float, default=3e-1)
    parser.add_argument('--ggm_sigma_max',       type=float, default=5e-1)
    ## MNIST
    parser.add_argument('--MNIST_noise', type=float, default=1e-2)
    parser.add_argument('--MNIST_noise_type', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--MNIST_single_digit', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--MNIST_digit_P_0', type=int, default=0)
    parser.add_argument('--MNIST_digit_P_1', type=int, default=6)
    parser.add_argument('--MNIST_n_repeat', type=int, default=1)
    parser.add_argument('--MNIST_w', type=float, default=1)
    ## ModelNet, I don't understand it
    parser.add_argument('--modelnet_data', type=str, default='torch_geom', choices=['manual', 'torch_geom'])
    parser.add_argument('--modelnet_path', type=str, default='data/ModelNet40_torch_geom')

    
    
    # Model
    parser.add_argument('--model', type=str, default='JKO_op_net')
    parser.add_argument('--n_MHT', type=int, default=2, help='How many MHT in sequence')
    parser.add_argument('--h', type=int, default=1024, help='Embed dimension')
    parser.add_argument('--n_MHT_heads', type=int, default=4, help='Number of Heads in parallel')
    parser.add_argument('--MHT_bias', type=lambda x: (str(x).lower() == 'true'), default=False) #need to figure out why False
    parser.add_argument('--MHT_res_link', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--nn_act', type=str, default='gelu', choices=['relu', 'sigmoid', 'gelu', 'mish'])
    parser.add_argument('--concat_t', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--dropout_p', type=float, default=0.1, help='drop layer in MLP')
    parser.add_argument('--concat_densityvalue', type=lambda x: (str(x).lower() == 'true'), default=False)


    # Training
    parser.add_argument('--N_iter', type=int, default=50000, help='Training Epochs')
    parser.add_argument('--lr_scheduler', type=str, default='cyclic',  choices=['none', 'adaptive', 'cyclic'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--N_iter_final', type=int, default=10000, help='parameters in lr_scheduler.CosineAnnealingLR')
    parser.add_argument('--scheduler_last_iter', type=int, default=-1, help='parameters in lr_scheduler.CosineAnnealingLR')
    parser.add_argument('--patience', type=int, default=10, help='parameter in lr_scheduler.ReduceLROnPlateau')

    #JKO loss function
    parser.add_argument('--Deltax', type=float, default=1e-2, help='finite element girid')
    parser.add_argument('--Deltax_randomness', type=lambda x: (str(x).lower() == 'true'), default=False, help='randomness in finite difference')
    parser.add_argument('--deltat', type=float, default=1e-4, help='Weight of energy')
    parser.add_argument('--divChoice', type=str, default='forward',  choices=['forward', 'forwardAdaptive', 'central','adaptive_high','Hutchinson'])
    parser.add_argument('--divDecoderIndependent', type=lambda x: (str(x).lower() == 'true'), default=False, help='if the decoder is independant when computing divergence')




    # plotting
    parser.add_argument('--n_plot', type=int, default=10)
    parser.add_argument('--plot_x_min', type=float, default=-6)
    parser.add_argument('--plot_x_max', type=float, default=6)
    parser.add_argument('--plot_y_min', type=float, default=-6)
    parser.add_argument('--plot_y_max', type=float, default=6)
    parser.add_argument('--Vrange', type=float, default=0.3)
    parser.add_argument('--plot_n_time_pts', type=int, default=10)
    parser.add_argument('--n_test_plot', type=int, default=10)
    parser.add_argument('--plot_jko_gif', type=lambda x: (str(x).lower() == 'true'), default=True)

    # misc.
    parser.add_argument('--log_interval',       type=int, default=50)
    parser.add_argument('--save_interval',      type=int, default=1000)

    args = parser.parse_args()
    args = check_arguments(args)

    return args


def check_arguments(args):
    if args.N_iter_final < args.N_iter:
        args.N_iter_final = args.N_iter
        warnings.warn("(lr_scheduler.CosineAnnealingLR T_max = N_iter_final) is less than the training iterations (N_iter). Setting them to be equal. Please check that this is intended.")

    return args
