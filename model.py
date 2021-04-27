import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import numpy as np
from copy import deepcopy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class DiscrNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DiscrNetwork, self).__init__()
        #
        self.discriminator = nn.Sequential(
         nn.Linear(num_inputs, hidden_dim), nn.Tanh(),
         nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
         nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.apply(weights_init_)

    def get_labels(self, state):
        probs = self.discriminator(state)
        return probs

    def forward(self, state):
        return self.discriminator(state)

class DynaNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, model_type='nn', bandwidth=-6.):
        super(DynaNetwork, self).__init__()

        # dynamics models
        self.model_type = model_type
        self.hidden_dim =hidden_dim
        if self.model_type == 'nn':
            self.model_means = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, num_inputs))
            self.model_stds = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, num_inputs))
        # linear / rbf model
        elif self.model_type == 'rbf':
            self.hidden_dim = hidden_dim
            self.state_size = num_inputs + num_actions
            self.linear_model_mean = nn.Linear(hidden_dim, num_inputs)
            self.linear_model_sd = nn.Linear(hidden_dim, num_inputs)
            self.bandwidth = torch.exp(torch.tensor(bandwidth))
            self.scale = torch.randn((self.hidden_dim, self.state_size))
            self.shift = torch.FloatTensor(self.hidden_dim, 1).uniform_(-np.pi, np.pi)
        elif self.model_type == 'linear':
            self.linear_model_mean = nn.Linear(num_inputs + num_actions, num_inputs)
            self.linear_model_sd = nn.Linear(num_inputs + num_actions, num_inputs)
        else:
            raise Exception()

        #
        self.mask_model = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        #
        self.reward_model = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))

        # self.apply(weights_init_)

    def get_mean(self, state, action):
        xu = torch.cat([state, action], 1)
        if self.model_type == 'nn':
            return self.model_means(xu)
        elif self.model_type == 'rbf':
            y = torch.matmul(self.scale.to(xu.device), xu.t()) / self.bandwidth
            y += self.shift.to(xu.device)
            y = torch.sin(y).detach()
            mean = self.linear_model_mean(y.t())
            return mean
        elif self.model_type == 'linear':
            mean = self.linear_model_mean(xu)
            return mean
        else:
            raise Exception()

    def get_log_std(self, state, action):
        xu = torch.cat([state, action], 1)
        if self.model_type == 'nn':
            return torch.clamp(self.model_stds(xu), min=-12, max=4)
        elif self.model_type == 'rbf':
            y = torch.matmul(self.scale.to(xu.device), xu.t()) / self.bandwidth
            y += self.shift.to(xu.device)
            y = torch.sin(y).detach()
            log_std = self.linear_model_sd(y.t())
            return torch.clamp(log_std, min=-14, max=4)
        elif self.model_type == 'linear':
            log_std = self.linear_model_sd(xu)
            return torch.clamp(log_std, min=-14, max=4)
        else:
            raise Exception()

    def get_probs(self, state, action):
        xu = torch.cat([state, action], 1)
        probs = self.mask_model(xu) +1e-8
        return probs

    def get_reward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.reward_model(xu)

    def get_dists(self, state, action):
        mean = self.get_mean(state, action)
        std = self.get_log_std(state, action).exp()
        probs = self.get_probs(state, action)
        mask_dist = Bernoulli(logits=probs)
        dyna_dist = Normal(mean, std)
        return dyna_dist, mask_dist

    def step(self, state, action, reparam=False):
        dyna_dist, mask_dist = self.get_dists(state, action)
        reward = self.get_reward(state, action)
        if not reparam:
            next_state = dyna_dist.sample()
        else:
            next_state = dyna_dist.rsample()
        mask = mask_dist.sample().reshape(-1)
        info = {'mask_dist':mask_dist, 'dyna_dist':dyna_dist}
        return next_state, reward, mask, info

    def log_prob(self, state, action, next_state, mask):

        dyna_dist, mask_dist = self.get_dists(state, action)
        mask_logprob = mask_dist.log_prob(mask.unsqueeze(1))
        dyna_logprob = dyna_dist.log_prob(next_state).sum(dim=1)
        return mask_logprob, dyna_logprob

    def sample_marginal(self, state, agent, max_steps=25):

        # info storage
        dyna_log_prob =torch.zeros(state.size()[0]).to(agent.device)
        expected_reward = torch.zeros(state.size()[0]).to(agent.device)
        mask_log_prob = torch.zeros(state.size()[0]).to(agent.device)
        policy_log_prob = torch.zeros(state.size()[0]).to(agent.device)
        system_mask = torch.zeros(state.size()[0]).to(agent.device)
        horizon = torch.zeros(state.size()[0]).to(agent.device)

        # state storage
        steady_state = deepcopy(state)

        # iterate
        for i in range(max_steps):
            # get policy info
            action, log_prob, mean = agent.expert.sample(state, reparam=True)
            # step system
            (state, reward, mask, info), (dlp, mlp) = self.forward(state, action)
            # update log probs
            dyna_log_prob += system_mask*dlp.reshape(-1)
            mask_log_prob += system_mask*mlp.reshape(-1)
            policy_log_prob += system_mask*log_prob.reshape(-1)
            # update expected reward
            expected_reward += system_mask*reward.reshape(-1)
            # store horizon for loggin
            horizon += system_mask
            # store end states
            log_states = torch.nonzero(system_mask - system_mask*mask)
            steady_state[log_states.reshape(-1),...] = state[log_states.reshape(-1),...]
            # update system mask
            system_mask = system_mask*mask.detach()
            # early break
            if torch.nonzero(system_mask).sum() == 0:
                break

        # if the model was not set, then set final state
        if torch.nonzero(system_mask).sum() > 0:
            steady_state[system_mask.nonzero(),...] = state[system_mask.nonzero(),...]

        # compute total log_prob
        log_prob = dyna_log_prob + mask_log_prob + policy_log_prob

        # return marginal state dist and other info
        return (steady_state, expected_reward, horizon, log_prob)

    def forward(self, state, action):
        # get next state_state/reward/mask
        next_state, reward, mask, info = self.step(state, action)
        # get scores
        mask_log_prob = info['mask_dist'].log_prob(mask.unsqueeze(1)).squeeze(1)
        dyna_log_prob = info['dyna_dist'].log_prob(next_state).sum(dim=1)
        # return scores and samples
        return (next_state, reward, mask, info), (dyna_log_prob, mask_log_prob)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                    model_type='nn', bandwidth=0., transform_rv=True, nonlin='relu',
                    clamp=True, init_model=True):
        super(GaussianPolicy, self).__init__()
        self.model_type = model_type
        self.transform_rv = transform_rv
        self.clamp = clamp
        # nn model
        if self.model_type == 'nn':
            self.nonlin = nonlin
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.mean_linear = nn.Linear(hidden_dim, num_actions)
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        # linear / rbf model
        elif self.model_type == 'rbf':
            self.hidden_dim = hidden_dim
            self.state_size = num_inputs
            self.linear_model_mean = nn.Linear(hidden_dim, num_actions)
            self.linear_model_sd = nn.Linear(hidden_dim, num_actions)
            self.bandwidth = torch.exp(torch.tensor(bandwidth))
        elif self.model_type == 'linear':
            self.linear_model_mean = nn.Linear(num_inputs, num_actions)
            self.linear_model_sd = nn.Linear(num_inputs, num_actions)
        else:
            raise Exception()

        if init_model:
            self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        # generate random features transforms
        if self.model_type == 'rbf':
            self.scale = torch.randn((self.hidden_dim, self.state_size))
            self.shift = torch.FloatTensor(self.hidden_dim, 1).uniform_(-np.pi, np.pi)

    def forward(self, state):
        if self.model_type == 'nn':
            if self.nonlin == 'relu':
                x = F.relu(self.linear1(state))
                x = F.relu(self.linear2(x))
            elif self.nonlin == 'tanh':
                x = torch.tanh(self.linear1(state))
                x = torch.tanh(self.linear2(x))
            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        elif self.model_type == 'linear':
            mean = self.linear_model_mean(state)
            log_std = self.linear_model_sd(state)
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        elif self.model_type == 'rbf':
            y = torch.matmul(self.scale.to(state.device), state.t()) / self.bandwidth
            y += self.shift.to(state.device)
            y = torch.sin(y).detach()
            mean = self.linear_model_mean(y.t())
            log_std = self.linear_model_sd(y.t())
            if self.clamp:
                log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            raise Exception()

        return mean, log_std

    def sample(self, state, reparam=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if reparam == False:
            x_t = normal.sample()
        else:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        if self.transform_rv:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)
        return action, log_prob, mean

    def log_prob(self, state, action):

        # get dist
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # get log prob
        #assert not self.transform_rv
        if self.transform_rv:
            u = torch.atanh((action - self.action_bias) / self.action_scale)
            log_prob = normal.log_prob(u).sum(1, keepdim=True)
            log_prob -= torch.log(self.action_scale * (1 - u.pow(2)) + epsilon).sum(1, keepdim=True)
        else:
            log_prob = normal.log_prob(action).sum(1, keepdim=True)
        # return
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        # generate random features transforms
        if self.model_type == 'rbf':
            self.scale = self.scale.to(device)
            self.shift = self.shift.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        raise Exception()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
