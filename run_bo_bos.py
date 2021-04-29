import argparse
import numpy as np
import os

# import the BO package
from bayesian_optimization import BayesianOptimization

# import the objetive function
from objective_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='untitled', type=str)
parser.add_argument('-a', '--alg', default='bobos+', type=str,
                    choices=['bo', 'bobos', 'bobos+'])
parser.add_argument('-t', '--task', default='mnist', type=str,
                    choices=['mnist', 'cifar', 'hopper'])
parser.add_argument('-r', '--repeats', default=10, type=int)
args = parser.parse_args()

if args.task == 'mnist':
    parameter_names = ["batch_size", "C", "learning_rate"]
    max_epoch = 50
    bos_epoch = 8
    interm_fid = [0, 9, 19, 29, 39]
    n_iter_bo = 50
    n_iter_bobos = 70
    if args.alg == 'bobos+':
        objective_function = MNIST_plus_train_loss
    else:
        objective_function = objective_function_LR_MNIST
elif args.task == 'cifar':
    max_epoch = 25
    bos_epoch = 8
    interm_fid = [0, 9, 19]
    n_iter_bo = 30
    n_iter_bobos = 42
    parameter_names = ["batch_size", "lr", "lr_decay",
                       "l2", "conv_filters", "dense_unit"]
    if args.alg == 'bobos+':
        objective_function = CIFAR_10_plus_train_loss
    else:
        objective_function = objective_function_CNN_CIFAR_10
else:
    raise Exception('Invalid Task')

np.random.seed(0)
iterations_list = np.arange(1, args.repeats+1)

for run_iter in iterations_list:
    '''
    The input arguments to "BayesianOptimization" are explained in the script "bayesian_optimization.py";
    In particular, set "no_BOS=True" if we want to run standard GP-UCB, and "no_BOS=False" if we want to run the BO-BOS algorithm;
    When running the "maximize" function, the intermediate results are saved after every BO iteration, under the file name log_file; the content of the log file is explained in the "analyze_results" ipython notebook script.
    '''
    log_file_dir = os.path.join("saved_results", args.name)
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_file_dir, f"{args.alg}_{args.task}_{run_iter}.p")
    if args.alg == 'bo':
        # run without BOS
        BO_no_BOS = BayesianOptimization(f=objective_function,
                                         dim=len(parameter_names), gp_opt_schedule=10,
                                         no_BOS=True, use_init=None,
                                         log_file=log_file_path, save_init=False,
                                         save_init_file=None, N=max_epoch, N_init_epochs=bos_epoch,
                                         parameter_names=parameter_names)
        # "parameter_names" are dummy variables whose correspondance in the display is not guaranteed
        BO_no_BOS.maximize(n_iter=n_iter_bo, init_points=len(parameter_names), kappa=2,
                           use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
    elif args.alg == 'bobos':
        # run with BOS, using the same initializations as above
        BO_BOS = BayesianOptimization(f=objective_function,
                                      dim=len(parameter_names), gp_opt_schedule=10, no_BOS=False, use_init=None,
                                      log_file=log_file_path, save_init=False,
                                      save_init_file=None, N=max_epoch, N_init_epochs=bos_epoch,
                                      add_interm_fid=interm_fid, parameter_names=parameter_names)
        BO_BOS.maximize(n_iter=n_iter_bobos, init_points=len(parameter_names), kappa=2,
                        use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
    elif args.alg == 'bobos+':
        # run with BOS, using the same initializations as above
        BO_BOS = BayesianOptimization(f=objective_function,
                                      dim=len(parameter_names), gp_opt_schedule=10, no_BOS=False, use_init=None,
                                      log_file=log_file_path, save_init=False,
                                      save_init_file=None, N=max_epoch, N_init_epochs=bos_epoch,
                                      add_interm_fid=interm_fid, parameter_names=parameter_names)
        BO_BOS.maximize(n_iter=n_iter_bobos, init_points=len(parameter_names), kappa=2,
                        use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
