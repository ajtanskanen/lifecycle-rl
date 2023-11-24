# https://github.com/toshikwa/sac-discrete.pytorch/blob/master/sacd/agent/__init__.py

import os
import numpy as np
import torch
from torch.optim import Adam

from . base import BaseAgent
from . model import TwinnedQNetwork, CateoricalPolicy, MultidiscretePolicy, TwinnedQNetwork_multidiscrete
from . utils import disable_gradients

class SacdAgent(BaseAgent):
    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.92, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_episodes=2,
                 max_episode_steps=27000, log_interval=10, eval_interval=228*50,
                 cuda=False, seed=0, cont=False,loadname=None,multidiscrete=True):
        if multidiscrete:
            super().__init__(
                env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
                multi_step, target_entropy_ratio, start_steps, update_interval,
                target_update_interval, use_per, num_eval_episodes, max_episode_steps,
                log_interval, eval_interval, cuda, seed, env.action_space.nvec)
        else:
            self.actionspace_n=env.action_space.n

            super().__init__(
                env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
                multi_step, target_entropy_ratio, start_steps, update_interval,
                target_update_interval, use_per, num_eval_episodes, max_episode_steps,
                log_interval, eval_interval, cuda, seed, self.actionspace_n)

        # Define networks.
        if multidiscrete:
            self.policy = MultidiscretePolicy(
                self.env.observation_space.shape[0], env.action_space.nvec
                ).to(self.device)
            self.online_critic = TwinnedQNetwork_multidiscrete(
                self.env.observation_space.shape[0], env.action_space.nvec,
                dueling_net=dueling_net).to(device=self.device)
            self.target_critic = TwinnedQNetwork_multidiscrete(
                self.env.observation_space.shape[0], env.action_space.nvec,
                dueling_net=dueling_net).to(device=self.device).eval()
        else:
            self.policy = CateoricalPolicy(
                self.env.observation_space.shape[0], self.actionspace_n
                ).to(self.device)
            self.online_critic = TwinnedQNetwork(
                self.env.observation_space.shape[0], self.actionspace_n,
                dueling_net=dueling_net).to(device=self.device)
            self.target_critic = TwinnedQNetwork(
                self.env.observation_space.shape[0], self.actionspace_n,
                dueling_net=dueling_net).to(device=self.device).eval()

        if cont is not None:
            if cont:
                self.load(loadname)

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        self.multidiscrete = multidiscrete
        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        if multidiscrete:
            #A = np.sum(env.action_space.nvec)
            A = env.action_space.nvec.shape[0]
        else:
            A = self.actionspace_n

        self.target_entropy = \
            -np.log(1.0 / A) * target_entropy_ratio

        print('target_entropy',self.target_entropy)

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        # temperature for softmax
        #self.beta = torch.zeros(1, requires_grad=True, device=self.device)
        #self.beta_optim = Adam([self.log_alpha], lr=lr)

    def predict(self, state,deterministic=False,beta=1):
        state = torch.FloatTensor(state[None, ...]).to(self.device)#.float()
        act = self.policy.act(state)#.item()
        predstate = 0
        return act, predstate

    def explore(self, state):
        # Act with randomness.
        state = torch.FloatTensor(
            state[None, ...]).to(self.device)#.float()# / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action #.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.FloatTensor(
            state[None, ...]).to(self.device)#.float()# / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action #.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights, only used in PER
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        '''
        Pitäisikö min olla mean tässä?
        '''
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states,cat=True)
        #log_action_probs = torch.cat(log_action_probs,dim=0)
        #action_probs = torch.cat(action_probs,dim=0)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        # sign: original -
        if self.multidiscrete: # multidiscrete
            entropies = (-torch.sum(action_probs[:,:5] * log_action_probs[:,:5], dim=1, keepdim=True) 
                        -torch.sum(action_probs[:,5:10] * log_action_probs[:,5:10], dim=1, keepdim=True) 
                        -torch.sum(action_probs[:,10:13] * log_action_probs[:,10:13], dim=1, keepdim=True) 
                        -torch.sum(action_probs[:,13:16] * log_action_probs[:,13:16], dim=1, keepdim=True))
            #print(torch.mean(action_probs,dim=0))
        else:
            entropies = torch.sum(
                action_probs * log_action_probs, dim=1, keepdim=True) # dim=1

        #print('entropies',entropies)
        #print('action_probs',action_probs)

        # Expectations of Q.
        q = torch.sum(q * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        # NOTICE SIGN
        policy_loss = -(weights * (self.alpha * entropies - q)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        # etumerkki?? FIXME. Perus: - ; -
        # onko oikein? tämä laskee entropyn 
        entropy_loss = -torch.mean(
            self.log_alpha * (-entropies + self.target_entropy)
            * weights)
        #entropy_loss = -self.log_alpha * (self.target_entropy - torch.mean(entropies * weights))
        #print(entropy_loss.item(),torch.mean(entropies).item(),self.log_alpha.item(),self.alpha.item())
        return entropy_loss

    def save(self, save_dir):
        super().save(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))

    def load(self, save_dir):
        print('Loading from',save_dir)
        super().load(save_dir)
        self.policy.load(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.load(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.load(os.path.join(save_dir, 'target_critic.pth'))        