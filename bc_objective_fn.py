import argparse
import datetime
import gym
import numpy as np
import itertools
import torch

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
from bc import BC
from mpc import MPC
from replay_memory import ReplayMemory
import time
from copy import deepcopy
import pickle


from bos_function import run_BOS, run_BOS_plus
import scipy.io as sio
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

kappa = 2.0 # the \kappa parameter to be used in the second criteria for early stopping

class Hardarguments:
    def __init__(self, lr=0.0003, batch_size=256, hidden_size=256, log_lr=-3., log_eps=-5., use_log_lr=1, nonlin='relu', bandwidth=0.):
        # hardcoded arguments
        self.env_name = 'Hopper-v2'
        self.policy = "Gaussian"
        self.eval = 1
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.automatic_entropy_tuning = 0
        self.seed = 123456

        self.num_steps = 1000001
        self.updates_per_step = 1
        self.start_steps = 10000
        self.target_update_interval =1
        self.replay_size = 10000
        self.cuda = 1 
        self.replay_filter_coeff = 0.5
        self.iter_filter_coeff = 0.5
        self.policy_sampler_type = 'uniform'
        self.algo ='BC'
        self.num_particles = 256
        self.num_clusters = 256
        self.expert_params_path = 'models/'
        
        self.sls_beta_b =0.9
        self.sls_c = 0.5
        self.sls_gamma = 2.0
        self.sls_beta_f = 2.0
        self.log_init_step_size = 4
        self.sls_eta_max = 100
        
        self.project = 'optimizers-in-bc'
        self.group = 'static-args'
        self.behavior_type = 'static-args'
        self.bandwidth= bandwidth

        self.transform_dist = 1
        self.clamp = 1

        self.use_torch_dataloader = 0
        self.sps_eps = 0
        self.batch_in_step = 0

        self.train_world_model = 0
        self.mpc_algo = 'WM'
        self.dyna_optim = 'Adam'
        self.log_gan_scale = 0
        self.marginal_steps = 15
        self.policy_updates = 1
        self.dyna_model_type = 'nn'
        self.dyna_bandwidth =0.


        # TBD
        self.policy_optim = 'Adam'
        self.critic_optim = 'Adam'
        self.model_type = 'rbf'


        # self defined arguements
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_size = 1024
        self.log_lr = log_lr
        self.log_eps=-5.
        self.use_log_lr =1
        self.nonlin = 'relu'



def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)


def check_args_bc(args):

    # presets
    args.expert_params_path = args.expert_params_path+'sac_actor_'+args.env_name+'_expert'
    args.critic_sampler_type = None
    args.combined_update = None
    args.beta = torch.tensor(1.)
    args.beta_update = torch.tensor(1.)

    # me being a big dummy on my grid-search
    if args.use_log_lr:
        args.lr = np.exp(args.log_lr)

    # assert args.log_lr <= 0.
    # assert args.log_lr == np.log(args.lr)
    args.init_step_size = np.exp(args.log_init_step_size)

    # set batch size
    args.n_batches_per_epoch = np.floor(args.replay_size / args.batch_size)

    # # fix dist if neccesary
    # if args.algo in ['BC_det', 'MLE', 'RKL', 'FKL']:
    #     args.transform_dist = 0
    #     args.clamp = 0
    # return em
    return args


def map(x, in_min, in_max, out_min, out_max):

    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



def dump_expert_rewards(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=200001, N_init_epochs=8):
    
    # should use the param instead but for demonstration just use the Hardarguemnts directly
    args = Hardarguments()
# (batch_size=batch_size, log_lr=log_lr, bandwidth=bandwidth)
#     """
#     bandwidth: 0.7075
#     hidden size: 1024
#     log lr: -4.433
#     num steps: 150000
#     replay_size: 10000
#     """
    # update
    args = check_args_bc(args)

    # make sure the expert params exist
    expert_params_dict = torch.load(args.expert_params_path)

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = BC(env.observation_space.shape[0], env.action_space, args)
    world_model = MPC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(10000, args.seed, args)

    # Training Loop
    total_numsteps = 0
    updates = 1
    start = time.time()



    # Expert setup
    # evaluate the expert
    avg_reward = 0.
    episodes = 25
    for _  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(state, eval_expert=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes
    exp_reward = avg_reward

    print("----------------------------------------")
    print("Time-elapsed: {}, Expert Avg. Reward: {}".format(timer(start,time.time()), round(avg_reward, 2)))
    print("----------------------------------------")

    # fill the buffer
    avg_reward = 0.
    episodes = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        episode_reward = 0.
        while not done:
            action, log_prob = agent.select_action(state, eval_expert=True)  # Sample action from policy
            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            mask = float(not done)
            #mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask, log_prob) # Append transition to memory
            state = next_state
            episode_reward += reward
        avg_reward += episode_reward
        episodes += 1
        if total_numsteps > len(memory):
            break
    avg_reward /= episodes
    # do we want a pytorch-data loader
    if args.use_torch_dataloader:
        train_set = memory.create_torch_dataset(tensor_data_set=True)
        data_generator = memory.create_torch_datalaoder(train_set, sampler='with_replacement')

    print("----------------------------------------")
    print("Time-elapsed: {}, Expert Avg. Reward: {}".format(timer(start,time.time()), round(avg_reward, 2)))
    print("----------------------------------------")
    # Expert setup done
    data = {
        'exp_reward': exp_reward,
        'memory': memory
    }
    with open('bc_memory.pickle', 'wb') as handle:
        pickle.dump(data, handle)
    # print(len(memory))
    # print(exp_reward)


def objective_function_bc(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=200001, N_init_epochs=8):
    
    # setup hyperparameter range 
    # transform the input to the real range of the hyper-parameters, to be used for model training
    parameter_range = [[128, 256], [-2.0, -8.0], [0.5, 0.9]]
    # parameter_range = [[256, 256], [-9.2, -9.2], [1e-3, 1e-3]]
    # parameter_range = [[256, 256], [-9.2, -9.2], [0.9, 0.9]]
    # parameter_range = [[256, 256], [-4.433,-4.433], [0.7075, 0.7075]]

#     """
#     bandwidth: 0.7075
#     hidden size: 1024
#     log lr: -4.433
#     num steps: 150000
#     replay_size: 10000
#     """

    
    batch_size_ = param[0]
    batch_size = int(batch_size_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0])
    log_lr = param[1]
    log_lr = log_lr * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
    bandwidth = param[2]
    bandwidth = bandwidth * (parameter_range[2][1] - parameter_range[2][0]) + parameter_range[2][0]
    
    # batch_size = 256
    # log_lr = -4.433
    # bandwidth = 0.7075

    print("[Evaluating parameters: batch size={0}/log_lr={1}/bandwidth={2}]".format(batch_size, log_lr, bandwidth))


    # should use the param instead but for demonstration just use the Hardarguemnts directly
    args = Hardarguments(batch_size=batch_size, log_lr=log_lr, bandwidth=bandwidth)


    # update
    args = check_args_bc(args)

    # make sure the expert params exist
    expert_params_dict = torch.load(args.expert_params_path)

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = BC(env.observation_space.shape[0], env.action_space, args)
    world_model = MPC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    # memory = ReplayMemory(args.replay_size, args.seed, args)

    # Training Loop
    total_numsteps = 0
    updates = 1
    start = time.time()


    # initialize weights and biases
    # import wandb
    # wandb.init(project=args.project, group=args.policy_optim)

    with open('bc_memory.pickle', 'rb') as handle:
        data = pickle.load(handle)

    exp_reward = data['exp_reward']
    memory = data['memory']

    training_epochs = N
    num_init_curve=N_init_epochs
    time_BOS = -1 # the time spent in solving the BOS problem, just for reference    
        
    train_epochs = []
    time_func_eval = []
    val_epochs = []
    updates = 1
    avg_loss = 0
    policy_loss = 0.
    accumulated_avg_loss = 0.
    early_stopped = False

    
    for epoch in tqdm(range(training_epochs)):

        iter_start=time.time()
        accumulated_avg_loss = 0.
        max_inner_ep = 1000
        for inner_ep in range(max_inner_ep):
            
            for _ in range(args.policy_updates):
                policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1
            if args.policy_updates == 0:
                updates += 1
            # log / print
            avg_loss = ((updates-1)/updates) * avg_loss + (1/updates)*policy_loss
            # wandb.log({'loss_policy': policy_loss, 'log_loss_policy': np.log(policy_loss),
            #             'step_size': agent.policy_optim.state['step_size'],
            #             'avg_loss': avg_loss, 'avg_log_loss': np.log(avg_loss)}, step=updates)
            
            accumulated_avg_loss  += avg_loss
        print("------ ORG Updates: {}, Avg. Loss: {}".format(updates, round(avg_loss, 4)))        
        accumulated_avg_loss /= max_inner_ep
        train_epochs.append(accumulated_avg_loss)
            
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        if (len(memory) > args.batch_size) and args.train_world_model:
            world_model.log_world_model(env, memory, updates, agent, len(memory), generate_hist=True, verbose=True)

        print("---------Test Episodes: {}, ORG Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        ## need to clip this awards TODO
        max_rewards = exp_reward
        # min_rewards = -50 # TBD

        diff_rewards = max_rewards-avg_reward
        # normalized
        normalized_avg_reward = map(diff_rewards, 0, max_rewards, 0., 1.)  
        print("------ Normalized Updates: {}, Avg. Reward: {}".format(updates, round(normalized_avg_reward, 4)))  
        val_epochs.append(normalized_avg_reward)
        time_func_eval.append(time.time())
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(diff_rewards, 2)))
        print("Updates: {}, Avg. Loss: {}".format(updates, round(avg_loss, 4)))
        print("Time-elapsed: {}, Avg. Log Loss: {}".format(timer(start,time.time()), round(np.log(policy_loss), 4)))
        print("Iteration-time: {}, Step-size: {}".format(timer(iter_start,time.time()), agent.policy_optim.state['step_size']))
        print("----------------------------------------")

        # run BOS after observing "num_init_curve" initial number of training epochs
        if (epoch+1 == num_init_curve) and (not no_stop):
            print("initial training losses: ", np.array(train_epochs))
            print("initial learning errors: ", np.array(val_epochs))
            time_start = time.time()
            # add BOBOS-PLUS specific info, AKA training loss. Must be a 2D array of shape <num_data x num_features>
            train_info = np.array(train_epochs).reshape(-1,1)
            print("train_epochs shape {}".format(train_info.shape))
            print("val epochs shape {}".format(np.array(val_epochs).shape))
            action_regions, grid_St = run_BOS_plus(np.array(val_epochs), train_info,incumbent, training_epochs, bo_iteration)
                
            time_BOS = time.time() - time_start
            
        if (epoch >= num_init_curve) and (not no_stop):
            state = np.sum(1 - np.array(val_epochs[num_init_curve:])) / (epoch - num_init_curve + 1)
            ind_state = np.max(np.nonzero(state > grid_St)[0])
            action_to_take = action_regions[epoch - num_init_curve, ind_state]
                
            # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
            if action_to_take == 2:
            # condition 2: the second criteria used in the BO-BOS algorithm
                if (kappa * stds[epoch] >= stds[-1]) or (stds == []):
                    early_stopped = True
                    break
    print("============================================================")
    print(f"Objective Function Complete:")
    print(f"- Final Performance: {val_epochs[-1]}")
    print(f"- Early Stopped: {early_stopped}")
    print("============================================================")
    env.close()            
    return val_epochs[-1], (epoch+1) / training_epochs, time_BOS, val_epochs, time_func_eval
    








def objective_function_bc_org(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=200001, N_init_epochs=8):
    
    # setup hyperparameter range 
    # transform the input to the real range of the hyper-parameters, to be used for model training
    parameter_range = [[128, 256], [-2.0, -8.0], [0.5, 0.9]]
    batch_size_ = param[0]
    batch_size = int(batch_size_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0])
    log_lr = param[1]
    log_lr = log_lr * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
    bandwidth = param[2]
    bandwidth = bandwidth * (parameter_range[2][1] - parameter_range[2][0]) + parameter_range[2][0]
    
    print("[Evaluating parameters: batch size={0}/log_lr={1}/bandwidth={2}]".format(batch_size, log_lr, bandwidth))


    # should use the param instead but for demonstration just use the Hardarguemnts directly
    args = Hardarguments(batch_size=batch_size, log_lr=log_lr, bandwidth=bandwidth)


    # update
    args = check_args_bc(args)

    # make sure the expert params exist
    expert_params_dict = torch.load(args.expert_params_path)

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = BC(env.observation_space.shape[0], env.action_space, args)
    world_model = MPC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    # memory = ReplayMemory(args.replay_size, args.seed, args)

    # Training Loop
    total_numsteps = 0
    updates = 1
    start = time.time()


    # initialize weights and biases
    # import wandb
    # wandb.init(project=args.project, group=args.policy_optim)

    with open('bc_memory.pickle', 'rb') as handle:
        data = pickle.load(handle)

    exp_reward = data['exp_reward']
    memory = data['memory']

    training_epochs = N
    num_init_curve=N_init_epochs
    time_BOS = -1 # the time spent in solving the BOS problem, just for reference    
        
    train_epochs = []
    time_func_eval = []
    val_epochs = []
    updates = 1
    avg_loss = 0
    policy_loss = 0.
    accumulated_avg_loss = 0.
    early_stopped = False



    for epoch in tqdm(range(training_epochs)):

        iter_start=time.time()
        accumulated_avg_loss = 0.
        max_inner_ep = 1000
        for inner_ep in range(max_inner_ep):
            
            for _ in range(args.policy_updates):
                policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1
            if args.policy_updates == 0:
                updates += 1
            # log / print
            avg_loss = ((updates-1)/updates) * avg_loss + (1/updates)*policy_loss
            # wandb.log({'loss_policy': policy_loss, 'log_loss_policy': np.log(policy_loss),
            #             'step_size': agent.policy_optim.state['step_size'],
            #             'avg_loss': avg_loss, 'avg_log_loss': np.log(avg_loss)}, step=updates)
            
            accumulated_avg_loss  += avg_loss

        accumulated_avg_loss /= max_inner_ep
        train_epochs.append(accumulated_avg_loss)
            
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        if (len(memory) > args.batch_size) and args.train_world_model:
            world_model.log_world_model(env, memory, updates, agent, len(memory), generate_hist=True, verbose=True)

        ## need to clip this awards TODO
        max_rewards = exp_reward
        # min_rewards = -50 # TBD

        diff_rewards = max_rewards-avg_reward

        # if avg_reward > max_rewards:
        #     avg_reward = 1.
        # if avg_reward < min_rewards:
        #     avg_reward = 0.0000001
        # normalized_avg_reward = map(diff_rewards, 0, max_rewards+abs(min_rewards), 0., 1.)
        normalized_avg_reward = map(diff_rewards, 0, max_rewards, 0., 1.)  
        val_epochs.append(normalized_avg_reward)
        time_func_eval.append(time.time())
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(normalized_avg_reward, 2)))
        print("Updates: {}, Avg. Loss: {}".format(updates, round(avg_loss, 4)))
        print("Time-elapsed: {}, Avg. Log Loss: {}".format(timer(start,time.time()), round(np.log(policy_loss), 4)))
        print("Iteration-time: {}, Step-size: {}".format(timer(iter_start,time.time()), agent.policy_optim.state['step_size']))
        print("----------------------------------------")

        # run BOS after observing "num_init_curve" initial number of training epochs
        if (epoch+1 == num_init_curve) and (not no_stop):
            print("initial learning errors: ", np.array(val_epochs))
            time_start = time.time()
            action_regions, grid_St = run_BOS(np.array(val_epochs), incumbent, training_epochs, bo_iteration)
            time_BOS = time.time() - time_start

            
        if (epoch >= num_init_curve) and (not no_stop):
            state = np.sum(1 - np.array(val_epochs[num_init_curve:])) / (epoch - num_init_curve + 1)
            ind_state = np.max(np.nonzero(state > grid_St)[0])
            action_to_take = action_regions[epoch - num_init_curve, ind_state]
                
            # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
            if action_to_take == 2:
            # condition 2: the second criteria used in the BO-BOS algorithm
                if (kappa * stds[epoch] >= stds[-1]) or (stds == []):
                    early_stopped = True
                    break
        print("============================================================")
        print(f"Objective Function Complete:")
        print(f"- Final Performance: {val_epochs[-1]}")
        print(f"- Early Stopped: {early_stopped}")
        print("============================================================")            
    return val_epochs[-1], (epoch+1) / training_epochs, time_BOS, val_epochs, time_func_eval




if __name__ == "__main__":
    param = {'fake': 0}
    # dump_expert_rewards(param)
    objective_function_bc(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=10, N_init_epochs=8)