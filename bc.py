import os
import torch
import torch.nn.functional as F
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
#
from torch.distributions import Normal
from utils import select_optizers
from copy import deepcopy

class BC(object):
    def __init__(self, num_inputs, action_space, args):

        # hyper-parameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.algo = args.algo
        self.args = args
        self.num_particles = args.num_particles
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.lr = args.lr
        self.transform_dist = args.transform_dist

        # set expert policy type
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.expert = GaussianPolicy(num_inputs, action_space.shape[0], 256, action_space).to(self.device)
        else:
            assert 1==0
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.expert = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

        # set critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # load expert parameters
        self.expert.load_state_dict(torch.load(args.expert_params_path))

        # set policy type
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, \
                    action_space=action_space, model_type=args.model_type, bandwidth=args.bandwidth, \
                    transform_rv=args.transform_dist, nonlin=args.nonlin, clamp=args.clamp, init_model=False).to(self.device)
        else:
            assert 1==0
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

        #
        self.policy_optim, self.critic_optim = select_optizers(self.policy, self.critic, args)

        #
        self.call_backwards_critic = True if args.critic_optim not in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        self.call_backwards_policy = True if args.policy_optim not in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        #
        self.update_stored_lr = True if args.policy_optim in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        self.step_size = args.lr
        # include batch in step
        self.include_batch = True if args.policy_optim in ['Sps'] else False
        #
        self.beta = args.beta
        self.beta_update = args.beta_update

    # model interactions
    def select_action(self, state, evaluate=False, eval_expert=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # if we just want to take expert action
        if eval_expert:
            _, log_prob, action = self.expert.sample(state)
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()
        # sample policy data
        if evaluate is False:
            action, agent_log_prob, _ = self.policy.sample(state)
        else:
            # if we are evaluating just return
            _, log_prob, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()
        # sample expert
        if torch.rand(1)[0] <= self.beta:
            action, expert_log_prob, _ = self.expert.sample(state)
        # recompute log-prob if we made it to here.
        expert_log_prob = self.expert.log_prob(state,action)
        agent_log_prob = self.policy.log_prob(state,action)
        # compute log_probs
        if (self.beta < 1) and (self.beta > 0):
            scores = torch.stack([torch.log(self.beta) + expert_log_prob, \
                                  torch.log(1.-self.beta) + agent_log_prob], dim=1)
            log_prob = torch.logsumexp(scores,dim=1)
        elif self.beta == 1.:
            log_prob = expert_log_prob
        elif self.beta == 0.:
            log_prob = agent_log_prob
        else:
            raise Exception()
        # return it all
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()

    # Parameter info
    def update_parameters(self, memory, batch_size, updates):

        # Sample a batch from memory
        if not self.args.use_torch_dataloader:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        else:
            batch = next(iter(memory.torch_dataloader))
            state_batch = batch['state'].to(self.device)
            action_batch = batch['action'].to(self.device)
            reward_batch = batch['reward'].to(self.device)
            next_state_batch = batch['next_state'].to(self.device).unsqueeze(1)
            mask_batch = batch['done'].to(self.device).unsqueeze(1)

        # format
        self.policy.to(self.device)

        # call the one IL thing we want to do
        if self.algo == 'BC':
            def policy_closure(backwards=self.call_backwards_policy):
                policy_pi, _, _ = self.policy.sample(state_batch, reparam=True)
                policy_loss = (policy_pi - action_batch.detach()).pow(2).mean()
                self.policy_optim.zero_grad()
                if backwards:
                    policy_loss.backward()
                return policy_loss
        elif self.algo == 'FKL':
            def policy_closure(backwards=self.call_backwards_policy):
                p_mean, log_p_var = self.policy(state_batch)
                q_mean, q_var = action_batch.detach(), 0.15*torch.ones(action_batch.size(), device=self.device).detach()
                p_var = log_p_var.exp() + torch.tensor(0.15)
                detp = torch.prod(p_var,dim=1)
                detq = torch.prod(q_var,dim=1)
                diff = q_mean - p_mean
                log_quot_frac = torch.log(detq) - torch.log(detp)
                tr = (p_var / q_var).sum()
                quadratic = ((diff / q_var) * diff).sum(dim=1)
                d = q_mean.size()[1]*torch.ones(quadratic.size(),device=self.device)
                kl_div = 0.5 * (log_quot_frac - d + tr + quadratic)
                policy_loss = kl_div.mean()
                self.policy_optim.zero_grad()
                if backwards:
                    policy_loss.backward()
                assert not torch.isnan(policy_loss)
                return policy_loss
        elif self.algo == 'RKL':
            def policy_closure(backwards=self.call_backwards_policy):
                q_mean, log_q_var = self.policy(state_batch)
                p_mean, p_var = action_batch.detach(), 0.15*torch.ones(action_batch.size(), device=self.device).detach()
                q_var = log_q_var.exp() + torch.tensor(0.15) #0.5*torch.tanh(log_q_var) + 0.51
                detp = torch.prod(p_var,dim=1)
                detq = torch.prod(q_var,dim=1)
                diff = q_mean - p_mean
                log_quot_frac = torch.log(detq) - torch.log(detp)
                tr = (p_var / q_var).sum()
                quadratic = ((diff / q_var) * diff).sum(dim=1)
                d = q_mean.size()[1]*torch.ones(quadratic.size(),device=self.device)
                kl_div = 0.5 * (log_quot_frac - d + tr + quadratic)
                policy_loss = kl_div.mean()
                self.policy_optim.zero_grad()
                if backwards:
                    policy_loss.backward()
                assert not torch.isnan(policy_loss)
                return policy_loss
        elif self.algo == 'MLE':
            def policy_closure(backwards=self.call_backwards_policy):
                q_mean, log_q_var = self.policy(state_batch)
                # make this numerically stable
                q_var = log_q_var.exp() + torch.tensor(0.15)
                diff = q_mean - action_batch.detach()
                quadratic = ((diff / q_var) * diff).sum(dim=1)
                log_prob = quadratic + torch.sum(log_q_var,dim=1)
                policy_loss = (0.5*log_prob).mean()
                # MAP estimation
                policy_loss += (q_mean.pow(2) / 2).mean()
                # backwards
                self.policy_optim.zero_grad()
                if backwards:
                    policy_loss.backward()
                assert not torch.isnan(policy_loss)
                return policy_loss #torch.exp(policy_loss)
        elif self.algo == 'BC_det':
            assert self.transform_dist == 0
            def policy_closure(backwards=self.call_backwards_policy):
                policy_pi, policy_std = self.policy(state_batch)
                policy_loss = (policy_pi - action_batch.detach()).pow(2).mean()
                policy_loss += (policy_std - torch.ones(policy_pi.size(), device=policy_pi.device)).pow(2).mean()
                self.policy_optim.zero_grad()
                if backwards:
                    policy_loss.backward()
                assert not torch.isnan(policy_loss)
                return policy_loss
        else:
            raise Exception('Provide valid optimizer')

        # check if we need batch info
        policy_loss = policy_closure(backwards=False)
        if self.args.batch_in_step:
            self.policy_optim.step(policy_closure, batch)
        else:
            self.policy_optim.step(policy_closure)
        if self.update_stored_lr:
            self.step_size = self.policy_optim.state['step_size']
        else:
            self.step_size = self.lr
        #
        return policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    # something to check interpolation
    def solver_info(self, memory):
        # get features
        state_batch, action_batch, _, _, _ = memory.sample(batch_size=memory.total_examples)
        # format
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        # create rff features
        y = torch.matmul(self.policy.scale.to(state_batch.device), state_batch.t()) / self.policy.bandwidth
        y += self.policy.shift.to(state_batch.device)
        y = torch.sin(y).detach().double()
        # solve system
        Soln = torch.mm(y, y.t())
        Soln = torch.inverse(Soln + 1e-8 * torch.eye(Soln.size()[0], device=state_batch.device))
        Soln = torch.mm(Soln,y)
        Soln = torch.mm(Soln, action_batch.double())
        # check we have interpolation
        pred = torch.mm(y.t(), Soln)
        loss = (pred - action_batch).pow(2).mean()
        # return it all
        return Soln, pred, loss
