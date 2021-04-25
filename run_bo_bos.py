import argparse
import numpy as np
import os

# import the BO package
from bayesian_optimization import BayesianOptimization

# import the objetive function
from objective_functions import *
from bc_objective_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='untitled', type=str)
parser.add_argument('-a', '--alg', default='bobos+', type=str,
                    choices=['bo', 'bobos', 'bobos+'])
parser.add_argument('-t', '--task', default='mnist', type=str,
                    choices=['mnist', 'hopper'])
parser.add_argument('-r', '--repeats', default=10, type=int)
args = parser.parse_args()

if args.task == 'mnist':
    parameter_names = ["batch_size", "C", "learning_rate"]
    if args.alg == 'bobos+':
        objective_function = MNIST_plus_train_loss
    else:
        objective_function = objective_function_LR_MNIST

elif args.task == 'hopper':
    parameter_names = ["batch_size", "log_lr", "bandwidth"]
    if args.alg == 'bobos+':
        objective_function = objective_function_bc
    else:
        raise Exception('Not implemented')
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
                                         dim=3, gp_opt_schedule=10,
                                         no_BOS=True, use_init=None,
                                         log_file=log_file_path, save_init=False,
                                         save_init_file=None,
                                         parameter_names=parameter_names)
        # "parameter_names" are dummy variables whose correspondance in the display is not guaranteed
        BO_no_BOS.maximize(n_iter=50, init_points=3, kappa=2,
                           use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
    elif args.alg == 'bobos':
        # run with BOS, using the same initializations as above
        BO_BOS = BayesianOptimization(f=objective_function,
                                      dim=3, gp_opt_schedule=10, no_BOS=False, use_init=None,
                                      log_file=log_file_path, save_init=False,
                                      save_init_file=None,
                                      add_interm_fid=[0, 9, 19, 29, 39], parameter_names=parameter_names)
        BO_BOS.maximize(n_iter=70, init_points=3, kappa=2,
                        use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
    elif args.alg == 'bobos+':
        # run with BOS, using the same initializations as above
        if args.task == 'hopper':
            BO_BOS = BayesianOptimization(f=objective_function,
                                        dim=3, gp_opt_schedule=10, no_BOS=False, use_init=None,
                                        log_file=log_file_path, save_init=False,
                                        save_init_file=None,
                                        add_interm_fid=[0, 9, 19, 29, 39], parameter_names=parameter_names, N=1000, N_init_epochs=80)
            BO_BOS.maximize(n_iter=70, init_points=3, kappa=2,
                            use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')

        else:
            BO_BOS = BayesianOptimization(f=objective_function,
                                        dim=3, gp_opt_schedule=10, no_BOS=False, use_init=None,
                                        log_file=log_file_path, save_init=False,
                                        save_init_file=None,
                                        add_interm_fid=[0, 9, 19, 29, 39], parameter_names=parameter_names)
            BO_BOS.maximize(n_iter=70, init_points=3, kappa=2,
                            use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
