import os
import torch
import torch.nn.functional as F
from utils import soft_update, hard_update
from model import DynaNetwork,DiscrNetwork
from collections import deque
#
from torch.distributions import Normal
from utils import select_optizers, select_optizer
from copy import deepcopy
import operator
import matplotlib.pyplot as plt
import numpy as np
import wandb
from functools import reduce
import pandas as pd
import imageio
import glob
import natsort

class MPC(object):
    def __init__(self, num_inputs, action_space, args):

        # hyper-parameters
        self.algo = args.mpc_algo
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.lr = args.lr
        self.gan_scale = torch.tensor(10**args.log_gan_scale, device=self.device)
        assert self.gan_scale <= 1.
        self.marginal_steps = args.marginal_steps
        #
        self.dyna_model = DynaNetwork(num_inputs, action_space.shape[0], args.hidden_size,
                            model_type=args.dyna_model_type)
        self.dyna_optim = select_optizer(self.dyna_model, args, args.dyna_optim)

        #
        self.discr_model = DiscrNetwork(num_inputs, action_space.shape[0], args.hidden_size)
        self.discr_optim = select_optizer(self.discr_model, args, 'Adam')

        #
        self.call_backwards = True if args.dyna_optim not in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        self.update_stored_lr = True if args.dyna_optim in ['Sls','Sps','Ssn','SlsEg','SlsAcc'] else False
        self.step_size = args.lr
        self.include_batch = True if args.dyna_optim in ['Sps'] else False

        #
        self.transition_hist = None
        self.done_hist = None
        self.reward_hist = None
        self.dyna_updates = 0

    # something to do it all
    def log_world_model(self, env, memory, updates, agent, total_examples,
            generate_hist=True, verbose=True, sweep_data=False):
        # update model
        if sweep_data:
            (mask_logprob, dyna_logprob, reward_loss, gen_loss), disc_loss = \
                self.sweep_data(memory, self.args.batch_size, updates, agent, total_examples)
        else:
            # single step
            (mask_logprob, dyna_loss, reward_loss, gen_loss), disc_loss = \
                self.update_parameters(memory, self.args.batch_size, updates, agent)
            self.dyna_updates += 1
            # update wandb
            wandb.log({'mask_loss': mask_logprob, 'dyna_loss': dyna_loss,
                       'reward_loss': reward_loss, 'total_examples':total_examples,
                       'disc_loss':disc_loss, 'gen_loss':gen_loss,
                       'dyna_updates': self.dyna_updates}, step=updates)
        # save hist of errorss
        if generate_hist:
            # self.likelihood_test(env, agent, duplicates=25)
            self.mse_test(env, agent, duplicates=25)
            self.update_histogram(fig_path=None,suffix=str(self.dyna_updates))
        # save models
        # self.save_model(self.args.env_name, suffix="", dyna_path=None)

        if verbose:
            print("=========================================")
            print("mask logprob: {}, dyna loss: {}".format(mask_logprob, dyna_loss))
            print("reward loss: {}, total examples: {}".format(reward_loss, total_examples))
            print("discriminator loss: {}, generator loss: {}".format(disc_loss, gen_loss))
            print("dynamics updates: {}, policy updates: {}".format(self.dyna_updates, updates))
            print("=========================================")

    # Parameter info
    def update_parameters(self, memory, batch_size, updates, agent):

        # Sample a batch from memory
        if not self.args.use_torch_dataloader:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        else:
            raise Exception('not functioning properly')
            batch = next(iter(memory.torch_dataloader))
            state_batch = batch['state'].float().to(self.device)
            action_batch = batch['action'].float().to(self.device)
            reward_batch = batch['reward'].float().to(self.device)
            next_state_batch = batch['next_state'].float().to(self.device).unsqueeze(1)
            mask_batch = batch['done'].float().to(self.device).unsqueeze(1)

        # format
        self.dyna_model.to(self.device)
        self.discr_model.to(self.device)

        # call the one IL thing we want to do
        if self.algo == 'WM':
            def dyna_closure(backwards=self.call_backwards, return_info=False):
                # compute prob losses
                mask_logprob, _ = self.dyna_model.log_prob(state_batch, action_batch, next_state_batch, mask_batch)
                # compute reward loss + repara dyna loss
                pred_state, pred_reward, _, _ = self.dyna_model.step(state_batch, action_batch, reparam=True)
                reward_loss = (reward_batch - pred_reward).pow(2).mean()
                state_loss = (next_state_batch - pred_state).pow(2).mean()
                # combine em
                dyna_loss = -1 * mask_logprob.mean() + reward_loss + state_loss
                self.dyna_optim.zero_grad()
                if backwards:
                    dyna_loss.backward()
                assert not torch.isnan(dyna_loss)
                if return_info:
                    return -1.*mask_logprob.mean().item(), state_loss.item(), reward_loss.item(), 0.
                else:
                    return dyna_loss
        elif self.algo == 'WM-MLE':
            def dyna_closure(backwards=self.call_backwards, return_info=False):
                # compute prob losses
                mask_logprob, state_logprob = self.dyna_model.log_prob(state_batch, action_batch, next_state_batch, mask_batch)
                # compute reward loss + repara dyna loss
                _, pred_reward, _, _ = self.dyna_model.step(state_batch, action_batch, reparam=False)
                reward_loss = (reward_batch - pred_reward).pow(2).mean()
                state_loss = - state_logprob.mean()
                # combine em
                dyna_loss = reward_loss - mask_logprob.mean() - state_logprob.mean()
                self.dyna_optim.zero_grad()
                if backwards:
                    dyna_loss.backward()
                assert not torch.isnan(dyna_loss)
                if return_info:
                    return -1.*mask_logprob.mean().item(), state_loss.item(), reward_loss.item(), 0.
                else:
                    return dyna_loss
        elif self.algo == 'WM-MSE':
            def dyna_closure(backwards=self.call_backwards, return_info=False):
                # compute prob losses
                mask_logprob, state_logprob = self.dyna_model.log_prob(state_batch, action_batch, next_state_batch, mask_batch)
                # compute reward loss + repara dyna loss
                _, pred_reward, _, _ = self.dyna_model.step(state_batch, action_batch, reparam=False)
                pred_state = self.dyna_model.get_mean(state_batch, action_batch)
                reward_loss = (reward_batch - pred_reward).pow(2).mean()
                state_loss = (next_state_batch - pred_state).pow(2).mean()
                # combine em
                dyna_loss = reward_loss - mask_logprob.mean() - state_logprob.mean()
                self.dyna_optim.zero_grad()
                if backwards:
                    dyna_loss.backward()
                assert not torch.isnan(dyna_loss)
                if return_info:
                    return -1.*mask_logprob.mean().item(), state_loss.item(), reward_loss.item(), 0.
                else:
                    return dyna_loss
        elif self.algo == 'C-WM':
            def dyna_closure(backwards=self.call_backwards, return_info=False):
                # compute prob losses
                mask_logprob, _ = self.dyna_model.log_prob(state_batch, action_batch, next_state_batch, mask_batch)
                # compute reward loss + repara dyna loss
                pred_state, pred_reward, _, _ = self.dyna_model.step(state_batch, action_batch, reparam=True)
                reward_loss = (reward_batch - pred_reward).pow(2).mean()
                state_loss = (next_state_batch - pred_state).pow(2).mean()
                # # get steady loss (generator)
                steady_state = self.dyna_model.sample_marginal(state_batch, agent, max_steps=self.marginal_steps)
                (generator_states, expected_reward, horizon, log_prob) = steady_state
                # compute loss from descriminator
                fake_state_pred = self.discr_model.get_labels(generator_states)
                # get steady loss
                lowerbound =  1e-8 * torch.ones(fake_state_pred.size(), device=fake_state_pred.device)
                GEN_loss = torch.log(torch.max(1.-fake_state_pred,lowerbound))
                # combine em
                dyna_loss = -1 * mask_logprob.mean() + reward_loss + state_loss
                dyna_loss = (1.-self.gan_scale) * dyna_loss + self.gan_scale * GEN_loss.mean()
                self.dyna_optim.zero_grad()
                if backwards:
                    dyna_loss.backward()
                assert not torch.isnan(dyna_loss)
                if return_info:
                    return -1.*mask_logprob.mean().item(), state_loss.item(), reward_loss.item(), GEN_loss.mean().item()
                else:
                    return dyna_loss
        else:
            raise Exception('Provide valid optimizer')

        # check if we need batch info
        dyna_losses  = dyna_closure(backwards=False, return_info=True)

        # step model
        if self.args.batch_in_step:
            self.dyna_optim.step(dyna_closure, batch)
        else:
            self.dyna_optim.step(dyna_closure)
        if self.update_stored_lr:
            self.step_size = self.dyna_optim.state['step_size']
        else:
            self.step_size = self.lr

        # step discriminator model
        if self.algo == 'C-WM':

            # get generated states from previous batch
            steady_state = self.dyna_model.sample_marginal(state_batch, agent, max_steps=self.marginal_steps)
            (generator_states, expected_reward, horizon, log_prob) = steady_state

            # re-Sample a batch from memory for disc
            if not self.args.use_torch_dataloader:
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            else:
                raise Exception('')
                batch = next(iter(memory.torch_dataloader))
                state_batch = batch['state'].float().to(self.device)
                action_batch = batch['action'].float().to(self.device)
                reward_batch = batch['reward'].float().to(self.device)
                next_state_batch = batch['next_state'].float().to(self.device).unsqueeze(1)
                mask_batch = batch['done'].float().to(self.device).unsqueeze(1)

            # compute loss from descriminator
            fake_state_pred = self.discr_model.get_labels(generator_states.detach())
            real_state_pred = self.discr_model.get_labels(state_batch)

            # get steady loss
            lowerbound =  1e-8 * torch.ones(fake_state_pred.size(), device=fake_state_pred.device)
            GAN_loss = torch.log(torch.max(1.-fake_state_pred,lowerbound))
            GAN_loss += torch.log(torch.max(real_state_pred,lowerbound))
            GAN_loss *= -1.
            GAN_loss = GAN_loss.mean()

            # combine
            GAN_loss = GAN_loss.mean()
            self.discr_optim.zero_grad()
            GAN_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discr_model.parameters(), 1.)
            self.discr_optim.step()
        else:
            GAN_loss=0.

        #
        return dyna_losses, GAN_loss

    def sweep_data(self, memory, batch_size, updates, agent, total_examples):
        # eterate over the data nlogn+n times to see all examples with high prob
        iters = memory.total_examples * (np.floor(np.log(memory.total_examples)) + 1)
        iters /= batch_size
        for i in range(int(iters+1)):
            self.dyna_updates += 1
            # single step
            (mask_logprob, dyna_loss, reward_loss, gen_loss), disc_loss = \
                self.update_parameters(memory, self.args.batch_size, updates, agent)
        # update wandb
        wandb.log({'mask_logprob': mask_logprob, 'dyna_loss': dyna_loss,
                   'reward_loss': reward_loss, 'total_examples':total_examples,
                   'disc_loss':disc_loss, 'gen_loss':gen_loss}, step=updates)

        return (mask_logprob, dyna_logprob, reward_loss, gen_loss), disc_loss

    def autocorrelation_test(self, env, agent, samples=25):

        # storage
        True_state, Pred_state = {}, {}
        True_done, Pred_done = {}, {}
        True_reward, Pred_reward = {}, {}

        # generate our samples
        for sample in range(samples):

            # starting state
            state = env.reset()
            pred_state = deepcopy(state)
            true_episode_reward, pred_episode_reward  = 0., 0.
            done, pred_done = False, False
            True_state[str(sample)], Pred_state[str(sample)] = {},{}
            True_done[str(sample)], Pred_done[str(sample)] = {},{}
            True_reward[str(sample)], Pred_reward[str(sample)] = {},{}

            # iterate real model
            t = 0
            while not done:

                # step through real env
                action, _ = agent.select_action(state)
                state, reward, done, _ = env.step(action)
                true_episode_reward += reward
                t += 1
                True_state[str(sample)][str(t)] = state
                True_done[str(sample)][str(t)] = done
                True_reward[str(sample)][str(t)] = reward

            # iterate fake model
            t = 0
            pred_state = torch.FloatTensor(pred_state).to(self.device).unsqueeze(0)
            while not pred_done:

                # step through fake env
                action, _ = agent.select_action(pred_state.to('cpu').squeeze(0).numpy())
                action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
                pred_state, pred_reward, pred_done, _ = self.dyna_model.step(pred_state, action)
                pred_episode_reward += pred_reward
                t += 1
                Pred_state[str(sample)][str(t)] = pred_state.detach().to('cpu').numpy()
                Pred_done[str(sample)][str(t)] = pred_done.detach().to('cpu').numpy()
                Pred_reward[str(sample)][str(t)] = pred_reward.detach().to('cpu').numpy()

        # create some pandas data frames
        done_df = pd.DataFrame.from_dict(Pred_done)
        reward_df = pd.DataFrame.from_dict(Pred_state)
        traj_df = pd.DataFrame.from_dict(Pred_reward)

        # now compute the auto-correlation

        #
        return done_df, reward_df, traj_df

    def likelihood_test(self, env, agent, duplicates=10):
        # storage
        self.trajectory_errors = deque([], maxlen=duplicates)
        for _ in range(duplicates):
            # starting state
            state = env.reset()
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            done, t = False, 0
            # storage
            model_tuples = deque([], maxlen=1000)
            # iterate real model
            while not done:
                # step through real env
                action, _ = agent.select_action(state.to('cpu').squeeze(0).numpy())
                next_state, reward, done, _ = env.step(action)
                #
                next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
                reward = torch.FloatTensor([reward]).to(self.device).unsqueeze(0)
                done = torch.FloatTensor([done]).to(self.device).unsqueeze(0)
                action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
                #
                t += 1
                # compute log-probs
                mask_logprob, dyna_logprob = self.dyna_model.log_prob(state, action, next_state, done)
                # get reward prediction
                _, pred_reward, _, _ = self.dyna_model.step(state, action)
                reward_error = (pred_reward - reward).pow(2).mean()
                #
                model_tuples.append((mask_logprob.item(), dyna_logprob.item(), reward_error.item(), t))
                # update state
                state = next_state
            # now add it to the stack
            self.trajectory_errors.append(model_tuples)
        # return it
        return self.trajectory_errors

    def mse_test(self, env, agent, duplicates=10):
        # storage
        self.trajectory_errors = deque([], maxlen=duplicates)
        for _ in range(duplicates):
            # starting state
            state = env.reset()
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            done, t = False, 0
            # storage
            model_tuples = deque([], maxlen=1000)
            # iterate real model
            while not done:
                # step through real env
                action, _ = agent.select_action(state.to('cpu').squeeze(0).numpy())
                next_state, reward, done, _ = env.step(action)
                #
                next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
                reward = torch.FloatTensor([reward]).to(self.device).unsqueeze(0)
                done = torch.FloatTensor([done]).to(self.device).unsqueeze(0)
                action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
                #
                t += 1
                # get reward prediction
                pred_state, pred_reward, pred_mask, _ = self.dyna_model.step(state, action)
                mask_logprob, dyna_logprob = self.dyna_model.log_prob(state, action, next_state, done)
                # copmpute errors
                reward_error = (pred_reward - reward).pow(2).mean()
                state_error = (pred_state - next_state).pow(2).mean()
                mask_error = (dyna_logprob.exp() - done).pow(2).mean()
                #
                model_tuples.append((mask_error.item(), state_error.item(), reward_error.item(), t))
                # update state
                state = next_state
            # now add it to the stack
            self.trajectory_errors.append(model_tuples)
        # return it
        return self.trajectory_errors

    def update_histogram(self, fig_path=None, suffix=''):

        # init
        examples = len(self.trajectory_errors)
        # format
        model_errors = [list(zip(*r)) for r in self.trajectory_errors]
        mask_errors = [r[0] for r in model_errors]
        dyna_errors = [r[1] for r in model_errors]
        reward_errors = [r[2] for r in model_errors]
        t = [r[3] for r in model_errors]
        # combine into single example set
        mask_errors = reduce(operator.concat,mask_errors)
        dyna_errors = reduce(operator.concat,dyna_errors)
        reward_errors = reduce(operator.concat,reward_errors)
        t = reduce(operator.concat,t)
        # combine with time_horizons
        mask_errors = np.array([vals for vals in zip(mask_errors,t)])
        dyna_errors = np.array([vals for vals in zip(dyna_errors,t)])
        reward_errors = np.array([vals for vals in zip(reward_errors,t)])
        # make sure save dir exists
        if not os.path.exists('models/'):
            os.makedirs('models/')
        # plot and save everything
        if fig_path is None:
            if not os.path.exists('world_models/'):
                os.makedirs('world_models/')
            trans_path = "world_models/hist_dyna_{}_{}.png".format(self.args.env_name, suffix)
            done_path = "world_models/hist_done_{}_{}.png".format(self.args.env_name, suffix)
            reward_path = "world_models/hist_reward_{}_{}.png".format(self.args.env_name, suffix)
        # transition figure
        fig1 = plt.subplots(figsize=(10,7))
        bins = [np.arange(0, 10., 0.5), np.arange(0, 1000, 10)]
        plt.hist2d(dyna_errors[:,0] , dyna_errors[:,1], bins=bins)
        plt.savefig(trans_path)
        plt.close()
        # mask figure
        fig1 = plt.subplots(figsize=(10,7))
        bins = [np.arange(0, 1., 0.05), np.arange(0, 1000, 10)]
        plt.hist2d(mask_errors[:,0] , mask_errors[:,1],bins=bins)
        plt.savefig(done_path)
        plt.close()
        # reward figure
        fig1 = plt.subplots(figsize=(10,7))
        bins = [np.arange(0, 15., 0.5), np.arange(0, 1000, 10)]
        plt.hist2d(reward_errors[:,0] , reward_errors[:,1], bins=bins)
        plt.savefig(reward_path)
        plt.close()
        # compute some stats
        ...
        # nothing to return
        return None

    def run_fake_env(self, env, agent):
        # generate a gif under true model
        env.render(mode='rgb_array')

    def save_model(self, env_name, suffix="", dyna_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if dyna_path is None:
            dyna_path = "models/sac_dyna_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(dyna_path))
        torch.save(self.dyna_model.state_dict(), dyna_path)

    def load_model(self, dyna_path):
        print('Loading models from {}'.format(dyna_path))
        if dyna_path is not None:
            self.dyna_model.load_state_dict(torch.load(dyna_path))

    def generate_gif(self, keyword, base_dir='./world_models'):

        filenames = glob.glob(base_dir+'/hist_'+keyword+'*.png')
        filenames = natsort.natsorted(filenames,reverse=False)
        with imageio.get_writer(keyword+'_movie.gif', mode='I', duration = 0.5) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        return None

    # something to check interpolation
    def solver_info(self, memory, hidden_dim=2048, bandwidth=1e-6):
        # get features
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=memory.total_examples)
        # format
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # combine
        transition_batch = torch.cat([state_batch, action_batch], 1)
        self.state_size = state_batch.size()[1]+action_batch.size()[1]
        # generate rff
        self.scale = torch.randn((hidden_dim, self.state_size))
        self.shift = torch.FloatTensor(hidden_dim, 1).uniform_(-np.pi, np.pi)
        self.bandwidth = bandwidth
        self.hidden_size = hidden_dim
        # create rff features
        y = torch.matmul(self.scale.to(transition_batch.device), transition_batch.t()) /bandwidth
        y += self.shift.to(transition_batch.device)
        y = torch.sin(y).detach().double()
        # solve system
        Soln = torch.mm(y, y.t())
        Soln = torch.inverse(Soln + 1e-8 * torch.eye(Soln.size()[0], device=transition_batch.device))
        Soln = torch.mm(Soln,y)
        Soln = torch.mm(Soln, next_state_batch.double())
        # check we have interpolation
        pred = torch.mm(y.t(), Soln)
        loss = (pred - next_state_batch).pow(2).mean()
        # return it all
        return Soln, pred, loss
