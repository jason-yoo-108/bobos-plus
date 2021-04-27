import random
import numpy as np
from collections import deque
from torch.optim import SGD, Adam, RMSprop
from copy import deepcopy
import torch
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
# torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(b))

# class Dataset(torch.utils.data.Dataset):
#     'Characterizes a dataset for PyTorch'
#     # def __init__(self, state, action, reward, next_state, done, traj_id, timestep):
#     def __init__(self, data_tuples):
#         'Initialization'
#         self.data_tuples = data_tuples
#         self.tuple_names = ('state', 'action', 'reward', 'next_state', \
#                             'done', 'traj_id', 'timestep')
#         # self.state = state
#         # self.action = action
#         # self.reward = reward
#         # self.next_state = next_state
#         # self.done = done
#         # self.traj_id = traj_id
#         # self.timestep timestep
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.data_tuples)
#
#     def __getitem__(self, index, disk=False):
#         'Generates one sample of data'
#         if disk:
#             # Select sample
#             ID = self.list_IDs[index]
#             # Load data and get label
#             X = torch.load('data/' + ID + '.pt')
#             y = self.labels[ID]
#             return X, y
#         else:
#             return self.data_tuples[index]

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tuple_names = ('state', 'action', 'reward', 'next_state', \
                            'done', 'traj_id', 'timestep')
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_tuple = self.dataset[index]
        data_tuple_names = self.tuple_names
        dict_wrapper = {data_tuple_names[i]: data_tuple[i] for i in range(len(data_tuple))}
        dict_wrapper.update({'meta':{'indices':index}})

        return dict_wrapper

class ReplayMemory:
    def __init__(self, capacity, seed, args):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.args = args
        #
        self.traj_id, self.timestep = 0, -1
        self.traj_rew, self.unif_log_prob = 0., None
        self.target_entropy = None
        #
        self.buffer = deque(maxlen=self.capacity)
        self.alpha = 1.
        self.gamma = args.gamma
        self.q_function = None
        self.policy = None
        #
        self.sampler = None
        self.filter = args.replay_filter_coeff
        self.filter_coeff = args.replay_filter_coeff
        self.iter_filter_coeff = args.iter_filter_coeff
        self.sample_map = {}
        self.sample_gen_info = {}
        self.strat_buffer = []
        self.policy_sampler_type = args.policy_sampler_type
        self.critic_sampler_type = args.critic_sampler_type
        self.sampler_types = [self.policy_sampler_type, self.critic_sampler_type]
        self.total_examples = 0
        #
        self.initial_state_clusters = args.num_clusters
        self.state_cluster_map = {}
        #
        self.total_timesteps = args.num_steps
        self.k_means_clusters = None
        # check nothing is funny.
        assert self.filter <= 1. and self.iter_filter_coeff <= 1.
        # plotter/logger stuff
        self.dict_plotter = {"return": None, "traj_idx": None,
                            "elbo": None, "time_step": None }
        self.policy_sample_info, self.critic_sample_info = None, None
        self.policy_sampler, self.critic_sampler = None, None

    def push(self, state, action, reward, next_state, done, log_prob):

        if self.unif_log_prob is None:
            self.unif_log_prob = len(action) * np.log((1/2))
            self.target_entropy = 0.
        # update strat sampling info
        self.timestep += 1
        self.total_examples = min(self.total_examples + 1, self.capacity)
        self.traj_rew += reward

        if type(log_prob) == np.ndarray:
            log_prob = log_prob.item()

        if 'uniform' in self.sampler_types:
            self.buffer.append((state, action, reward, next_state, done, self.traj_id, self.timestep))
        elif self.traj_id < 2 * self.initial_state_clusters or self.sampler is None:
            self.buffer.append((state, action, reward, next_state, done, self.traj_id, self.timestep))
        elif 'priority-exp' in self.sampler_types:
            raise Exception
            td_err = self.q_function(state, action) \
                - (reward + self.gamma * self.q_function(next_state, self.policy.sample(next_state)))
            self.buffer.append((state, action, reward, next_state, done, self.traj_id, self.timestep, torch.abs(td_err)))


        if (self.policy_sampler_type != 'uniform') or (self.critic_sampler_type != 'uniform'):
            self.strat_buffer.append((state, action, reward, next_state, done, \
                    log_prob, self.traj_id, self.timestep))

        # remove partial trajectories once we are full
        if self.total_examples >= self.capacity:
            if (self.traj_id <  2 * self.initial_state_clusters):
                pass
            elif (self.policy_sampler_type == 'uniform') or (self.critic_sampler_type == 'uniform'):
                remove_id = self.buffer[-1][-2]
                current_buffer = self.buffer[-1][-2]
                while remove_id == current_buffer:
                    self.buffer.pop()
                    current_buffer = self.buffer[-1][-2]
            else:
                self.sample_map.pop(str(remove_id), None)

        # if the last example finished off a traj.
        if not done and ((self.policy_sampler_type != 'uniform') or (self.critic_sampler_type != 'uniform')):

            # update reward associated with example
            cum_log_prob = sum([self.strat_buffer[p][5] for p in range(self.timestep+1)])
            cum_unif_prob = self.timestep * self.unif_log_prob
            traj_prob = cum_unif_prob - cum_log_prob
            self.sample_gen_info[str(self.traj_id)] = (self.traj_id, self.traj_rew, traj_prob, self.timestep)
            # now add it
            self.sample_map[str(self.traj_id)] = deepcopy(self.strat_buffer)
            # update importance weight for sampling
            if (self.traj_id > 2 * self.initial_state_clusters) and ((self.policy_sampler_type != 'uniform') or (self.critic_sampler_type != 'uniform')):
                self.policy_sampler, self.critic_sampler = self.update_sampler()

        # more general resets at done
        if not done:
            # now reset it all
            self.traj_id += 1
            self.timestep = -1
            self.traj_rew = 0.
            self.strat_buffer = []

    def update_sampler(self):

        samplers = {}

        # if uniform
        if ('uniform' in self.sampler_types) or (self.traj_id < 2 * self.initial_state_clusters - 1):
            samplers['uniform'] = lambda : None
        # uniform - stratified sampling
        if ('stratified' in self.sampler_types):
            k_prob = len(self.sample_map.keys())
            sampling_prob = torch.tensor(1./k_prob)*torch.ones(k_prob)
            traj_sampler_ = torch.distributions.Categorical(sampling_prob)
            traj_sampler = lambda : traj_sampler_.sample().item()
            samplers['stratified'] = traj_sampler
        # RLAI - stratified setup (where filter shifts from uniform to greedy)
        if ('filtered' in self.sampler_types):
            # update filter
            self.filter *= self.iter_filter_coeff
            # create log_iw score
            info = self.sample_gen_info.values()
            r_log_iw = torch.tensor([s[1] for s in info])
            k_log_iw = torch.tensor([s[2] for s in info])
            # compute RLAI weights
            iw = torch.exp(r_log_iw + self.alpha * k_log_iw
                - torch.max(r_log_iw + self.alpha * k_log_iw))
            iw_prob = iw / iw.sum()
            # comute unif weights
            unif_prob = (1 / len(iw_prob)) * torch.ones(iw_prob.size())
            # now set sampling distribution
            assert self.filter <= 1. and self.filter >= 0.
            sampling_prob = self.filter * unif_prob + (1.-self.filter) * iw_prob
            traj_sampler_ = torch.distributions.Categorical(sampling_prob)
            traj_sampler = lambda : traj_sampler_.sample().item()
            samplers['filtered'] = traj_sampler
        # uniform initial conditions - stratified setup
        if ('init-stratified' in self.sampler_types):
            if self.traj_id <= 2 * self.initial_state_clusters:
                k_prob = len(self.sample_map.keys())
                sampling_prob = torch.tensor(1./k_prob)*torch.ones(k_prob)
                traj_sampler_ = torch.distributions.Categorical(sampling_prob)
                traj_sampler = lambda : traj_sampler_.sample().item()
            else:
                initial_states = np.array([self.sample_map[key][0][0] for key in self.sample_map.keys()])
                kmeans = KMeans(n_clusters=self.initial_state_clusters, random_state=0).fit(initial_states)
                p = 1./np.bincount(kmeans.labels_)
                probs = (1./self.initial_state_clusters) * np.array([p[c] for c in kmeans.labels_])
                traj_sampler_ = torch.distributions.Categorical(torch.tensor(probs))
                traj_sampler = lambda : traj_sampler_.sample().item()
            samplers['init-stratified'] = traj_sampler
        # uniform initial conditions - with RLAI filtering + variable filter coeff
        if ('init-filtered' in self.sampler_types):
            # now do the thing
            if self.traj_id <= 2 * (self.initial_state_clusters):
                k_prob = len(self.sample_map.keys())
                sampling_prob = torch.tensor(1./k_prob)*torch.ones(k_prob)
                traj_sampler_ = torch.distributions.Categorical(sampling_prob)
                traj_sampler = lambda : traj_sampler_.sample().item()
            else:
                # update filter
                self.filter *= self.iter_filter_coeff
                # compute init-state probs
                initial_states = np.array([self.sample_map[key][0][0] for key in self.sample_map.keys()])
                kmeans = KMeans(n_clusters=self.initial_state_clusters, random_state=0).fit(initial_states)
                p = 1./np.bincount(kmeans.labels_)
                probs = (1./self.initial_state_clusters) * np.array([p[c] for c in kmeans.labels_])
                # now compute un-normalized probs of RLAI stuff with pandas
                info = self.sample_gen_info.values()
                r_log_iw = torch.tensor([s[1] for s in info]).numpy()
                k_log_iw = torch.tensor([s[2] for s in info]).numpy()
                #
                elbo_df = pd.DataFrame({'kmeans_labels':kmeans.labels_, 'elbo':r_log_iw + self.alpha * k_log_iw,
                    'state_prob':np.array([p[c] for c in kmeans.labels_])})
                elbo_df['elbo_max'] = elbo_df['elbo'].groupby(elbo_df['kmeans_labels']).transform('max')
                elbo_df['elbo_sm'] = elbo_df['elbo'] - elbo_df['elbo_max']
                elbo_df['elbo_lse'] = np.exp(elbo_df['elbo_sm']).groupby(elbo_df['kmeans_labels']).transform('sum')
                elbo_probs = (np.exp(elbo_df['elbo_sm']) / elbo_df['elbo_lse']) * elbo_df['state_prob']
                # compute uniform
                unif_probs = (1./self.initial_state_clusters) * np.array(elbo_df['state_prob'])
                # compute interpolent
                sampling_prob = self.filter * torch.tensor(unif_probs) + (1.-self.filter) * torch.tensor(elbo_probs)
                # compute normalized probs across different clusters
                traj_sampler_ = torch.distributions.Categorical(sampling_prob)
                traj_sampler = lambda : traj_sampler_.sample().item()
            samplers['init-filtered'] = traj_sampler
        #
        if ('priority-exp' in self.sampler_types):
            raise Exception

        #
        if not bool(samplers):
            raise Exception('provide valid sampler types.')

        # now create sampler for critic and policy
        def critic_sampler(batch_size):
            # sample which trajectories
            traj_samples = [samplers[self.critic_sampler_type]() for _ in range(batch_size)]
            # sample which times
            t_samples = [torch.randint(0, len(self.sample_map[str(sample)]), (1,)).item() for sample in traj_samples]
            # return a hash of where to find these
            data_samples = [self.sample_map[str(data[0])][data[1]] for data in list(zip(traj_samples, t_samples))]
            # store sample info (self.traj_id, self.traj_rew, traj_prob, self.timestep)
            self.critic_sample_info = [self.sample_gen_info[str(id)] for id in traj_samples]
            # return the list you made
            return data_samples
        # now create sampler for critic and policy
        def policy_sampler(batch_size):
            # sample which trajectories
            traj_samples = [samplers[self.policy_sampler_type]() for _ in range(batch_size)]
            # sample which times
            t_samples = [torch.randint(0, len(self.sample_map[str(sample)]), (1,)).item() for sample in traj_samples]
            # return a hash of where to find these
            data_samples = [self.sample_map[str(data[0])][data[1]] for data in list(zip(traj_samples, t_samples))]
            # store sample info (self.traj_id, self.traj_rew, traj_prob, self.timestep)
            self.policy_sample_info = [self.sample_gen_info[str(id)] for id in traj_samples]
            # return the list you made
            return data_samples

        # return the generated function
        return policy_sampler, critic_sampler

    def sample(self, batch_size, sample_type='combined', verbose=True):

        if sample_type=='combined':
            if self.policy_sampler_type == 'uniform':
                batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
                state, action, reward, next_state, done,_,_ = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done
            elif self.policy_sampler_type in ['priority-exp']:
                batch = self.sampler(batch_size)
                state, action, reward, next_state, done,_,_,_ = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done
            elif (self.traj_id < 2 * self.initial_state_clusters) or self.sampler is None:
                batch = random.sample(self.buffer, min(batch_size,  self.total_examples-1))
                state, action, reward, next_state, done,_,_ = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done
            elif self.policy_sampler_type in \
                ['stratified','filtered','init-stratified', 'init-filtered'] and self.traj_id > self.initial_state_clusters:
                batch = self.sampler(batch_size)
                state, action, reward, next_state, done,_,_,_ = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done
            else:
                raise Exception('oof')
        else:
            # critic
            if self.critic_sampler_type == 'uniform':
                batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
                state, action, reward, next_state, done,_,_ = map(np.stack, zip(*batch))
                critic_sample = (state, action, reward, next_state, done)
            elif self.critic_sampler_type in ['priority-exp']:
                batch = self.critic_sampler(batch_size)
                state, action, reward, next_state, done,_,_,_ = map(np.stack, zip(*batch))
                critic_sample = (state, action, reward, next_state, done)
            elif (self.traj_id < 2 * self.initial_state_clusters) or (self.critic_sampler is None):
                batch = random.sample(self.buffer, min(batch_size,  self.total_examples-1))
                state, action, reward, next_state, done,_,_ = map(np.stack, zip(*batch))
                critic_sample = (state, action, reward, next_state, done)
            elif self.critic_sampler_type in \
                ['stratified','filtered','init-stratified', 'init-filtered'] and self.traj_id > self.initial_state_clusters:
                batch = self.critic_sampler(batch_size)
                state, action, reward, next_state, done,_,_,_ = map(np.stack, zip(*batch))
                critic_sample = (state, action, reward, next_state, done)
            else:
                raise Exception('oof')
            # policy
            if self.policy_sampler_type == 'uniform':
                batch = random.sample(self.buffer, min(batch_size,  self.total_examples))
                state, action, reward, next_state, done,_,_ = map(np.stack, zip(*batch))
                policy_sample = (state, action, reward, next_state, done)
            elif self.policy_sampler_type in ['priority-exp']:
                batch = self.policy_sampler(batch_size)
                state, action, reward, next_state, done,_,_,_ = map(np.stack, zip(*batch))
                policy_sample = state, action, reward, next_state, done
            elif (self.traj_id < 2 * self.initial_state_clusters) or self.policy_sampler is None:
                batch = random.sample(self.buffer, min(batch_size,  self.total_examples-1))
                state, action, reward, next_state, done,_,_ = map(np.stack, zip(*batch))
                policy_sample = (state, action, reward, next_state, done)
            elif self.policy_sampler_type in \
                ['stratified','filtered','init-stratified', 'init-filtered'] and (self.traj_id > 2*self.initial_state_clusters):
                batch = self.policy_sampler(batch_size)
                state, action, reward, next_state, done,_,_,_ = map(np.stack, zip(*batch))
                policy_sample = (state, action, reward, next_state, done)
            else:
                raise Exception('oof')
            #
            if verbose:
                self.viz_dist(self.policy_sample_info ,'policy_sample')
                self.viz_dist(self.critic_sample_info ,'critic_sample')
            #
            return policy_sample, critic_sample

    def __len__(self):
        return self.total_examples

    def viz_dist(self,x,sample_type):
        if x is not None:
            df = pd.DataFrame({
                'id': torch.tensor(x)[:,0].numpy(), 'return': torch.tensor(x)[:,1].numpy(),
                'log_prob': torch.tensor(x)[:,2].numpy(), 'T': torch.tensor(x)[:,3].numpy(),})
            fig = df.hist(bins=min(25,len(x)))
            plt.savefig(sample_type+'_samples_figure.pdf')
            plt.close('all')

    def save_examples(self,dir):
        state, action, reward, next_state, done, traj_id, timestep = zip(*self.buffer)
        state =torch.stack([torch.tensor(s) for s in state]).float().detach()
        action = torch.stack([torch.tensor(s) for s in action]).float().detach()
        reward = torch.stack([torch.tensor(s) for s in reward]).float().detach()
        next_state = torch.stack([torch.tensor(s) for s in next_state]).float().detach()
        done = torch.stack([torch.tensor(s) for s in done]).float().detach()
        traj_id = torch.stack([torch.tensor(s) for s in traj_id]).float().detach()
        timestep = torch.stack([torch.tensor(s) for s in timestep]).float().detach()
        save_dict = {'state':state, 'action': action,
        'reward': reward, 'next_state': next_state,
        'done': done, 'timestep':timestep, 'traj_id':traj_id}
        torch.save(save_dict, dir)
        # print(save_dict)
        print('saved dataloader to '+dir)
        return None

    def load_examples(self,dir):
        state, action, reward, next_state, done, traj_id, timestep = zip(*self.buffer)
        state =torch.stack([torch.tensor(s) for s in state]).float()
        action = torch.stack([torch.tensor(s) for s in action]).float()
        reward = torch.stack([torch.tensor(s) for s in reward]).float()
        next_state = torch.stack([torch.tensor(s) for s in next_state]).float()
        done = torch.stack([torch.tensor(s) for s in done]).float()
        traj_id = torch.stack([torch.tensor(s) for s in traj_id]).float()
        timestep = torch.stack([torch.tensor(s) for s in timestep]).float()
        dataset = torch.utils.data.TensorDataset(state, action, reward, next_state, done, traj_id, timestep)
        return None

    def create_torch_dataset(self, tensor_data_set=True):
        if tensor_data_set:
            state, action, reward, next_state, done, traj_id, timestep = zip(*self.buffer)
            state = torch.stack([torch.tensor(s) for s in state]).float()
            action = torch.stack([torch.tensor(s) for s in action]).float()
            reward = torch.stack([torch.tensor(s) for s in reward]).float()
            next_state = torch.stack([torch.tensor(s) for s in next_state]).float()
            done = torch.stack([torch.tensor(s) for s in done]).float()
            traj_id = torch.stack([torch.tensor(s) for s in traj_id]).float()
            timestep = torch.stack([torch.tensor(s) for s in timestep]).float()
            dataset = torch.utils.data.TensorDataset(state, action, reward, next_state, done, traj_id, timestep)
            self.torch_dataset = DatasetWrapper(dataset)
        else:
            self.torch_dataset = DatasetWrapper(Dataset(self.buffer))
        return self.torch_dataset

    def create_torch_datalaoder(self, train_set, sampler='with_replacement'):
        if sampler=='with_replacement':
            rand_sampler = torch.utils.data.RandomSampler(train_set, num_samples=self.args.batch_size, replacement=True)
            self.torch_dataloader = DataLoader(train_set, drop_last=False,
                    sampler=rand_sampler, batch_size=self.args.batch_size)
        else:
            self.torch_dataloader = DataLoader(train_set, drop_last=True, shuffle=True,
                    sampler=None, batch_size=self.args.batch_size)
        self.data_generator = iter(self.torch_dataloader)
        return self.torch_dataloader
