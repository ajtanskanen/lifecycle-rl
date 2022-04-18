'''
Runner for making fitting with stand-alone code without external libraries
'''

import gym, torch, numpy as np, torch.nn as nn

import argparse
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, AdamW #, RAdam
from torch.distributions import Categorical
from .utils import make_env

from . episodestats import EpisodeStats
from . simstats import SimStats

from kfoptim import KFOptimizer

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # for MountainCar-v0
        #self.affine1 = nn.Linear(2, 128)

        # actor's layer
        
        if False:
            self.affine2 = nn.Linear(8, 4)
            self.test_head = nn.Linear(4, 1)
        else:
            self.affine1 = nn.Linear(8, 16)
            self.action_head_hidden = nn.Linear(16, 16)
            self.action_head = nn.Linear(16, 4)
            self.value_head = nn.Linear(16, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
        self.float()

    def forward(self, x):
        """
        forward of both actor and critic
        """
        
        x = F.relu(self.affine1(x))
        z = F.relu(self.action_head_hidden(x))
        #z=F.relu(self.action_head_hidden_ii(z))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #z=F.relu(self.action_head_hidden_ii(z))
        action_prob = F.softmax(self.action_head(z), dim=-1)

        # critic: evaluates being in the state s_t
        #z = F.relu(self.value_head_hidden(x))
        #z = F.relu(self.value_head_hidden_ii(z))
        state_values = self.value_head(z)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values
        
class runner_standalone():
    def __init__(self,environment,gamma,timestep,n_time,n_pop,
                 minimal,min_age,max_age,min_retirementage,year,episodestats,gym_kwargs):
        self.gamma=gamma
        self.timestep=timestep
        self.environment=environment
        self.n_time=n_time
        self.n_pop=n_pop
        self.minimal=minimal
        self.min_age=min_age
        self.max_age=max_age
        self.min_retirementage=min_retirementage
        self.year=year
        self.gym_kwargs=gym_kwargs
        
        self.env = gym.make(self.environment)
        self.n_employment,self.n_acts=self.env.get_n_states()
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.action_shape = self.env.action_space.shape or self.env.action_space.n

        self.version = self.env.get_lc_version()

        self.episodestats=episodestats 
        #SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
        #                           self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
        #                           version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma)
                                   
        self.model = Policy()
        #optimizer = optim.AdamW(model.parameters(), lr=3e-3)

        #print(model.state_dict())

        #optimizer=KFACOptimizer(model,lr=1e-3)
        self.optimizer=KFOptimizer(model.parameters(),model,lr=1e-4,stat_decay=0.995)
        
    def load_model(self,PATH):
        # Model class must be defined somewhere
        model = torch.load(PATH)
        model.eval() 

    #def save_model(PATH):
        # Model class must be defined somewhere
    #    torch.save({
    #            'epoch': epoch,
    #            'model_state_dict': model.state_dict(),
    #            'optimizer_state_dict': optimizer.state_dict(),
    #            'loss': loss,
    #            ...
    #            }, PATH)

    def select_action(self,state,deterministic=True):
        state = torch.from_numpy(state).float()
        probs, state_value = model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        if deterministic:
            # and find the most probable action using the distribution
            action = probs.argmax()
        else:
            # and sample an action using the distribution
            action = m.sample()

        # save to action buffer
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def finish_episode(self,preconditioner=None):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            #print(R.item(),value.item(),advantage.item())
            R=R.type(torch.float)
            advantage=advantage.type(torch.float)
            #print(advantage)

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss # SIC! L1 not L2
            #value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            v=value.squeeze(dim=1)
            #print(v.size(),R.size())
            #value_losses.append(F.mse_loss(v, torch.tensor([R])))
            value_losses.append(F.mse_loss(v, R))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        # hmm, miksi tämä ajetaan vasta tässä vaiheessa? td? tässä mc
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        #if preconditioner is not None:
        #    preconditioner.step()
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.saved_actions[:]
        
    def train(self,train=False,debug=False,steps=20_000,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=None,learning_rate=0.25,
                start_from=None,max_n_cpu=1000,use_vecmonitor=False,
                bestname='tmp/best2',use_callback=False,log_interval=100,verbose=1,plotdebug=False,
                learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
        '''
        Opetusrutiini
        '''
        batch_size=1
        running_reward = -200
        
        self.best_mean_reward, self.n_steps = -np.inf, 0

        if pop is not None:
            self.n_pop=pop

        self.rlmodel=rlmodel
        self.bestname=bestname

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        self.savename=save
        n_cpu=min(max_n_cpu,n_cpu)

        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':True})
    
        policy=self.setup_training(self.rlmodel,start_from,env,batch,policy_kwargs,learning_rate,
                                cont,max_grad_norm=max_grad_norm,verbose=verbose,n_cpu=n_cpu,
                                learning_schedule=learning_schedule,vf=vf,gae_lambda=gae_lambda)

        if cont:
            policy.load_state_dict(torch.load(start_from))

        # run inifinitely many episodes
        for i_episode in count(episodes):

            for b in range(batch_size):
                # reset environment and episode reward
                state = env.reset()
                states=np.expand_dims(state, axis=0)
                ep_reward = 0

                # for each episode, only run 9999 steps so that we don't 
                # infinite loop while learning
                for t in range(1, 10000):
                    # select action from policy
                    action = select_action(states,deterministic=deterministic)

                    # take the action
                    state, reward, done, _ = env.step(action)
                
                    # make batch-like
                    states=np.expand_dims(state, axis=0)
                    rews=np.expand_dims(reward, axis=0)
                    dones=np.expand_dims(done, axis=0)

                    if render:
                        env.render()

                    model.rewards.append(rews)
                    ep_reward += reward
                    if done:
                        break

                # update cumulative reward
                running_reward = 0.01 * ep_reward + (1 - 0.01) * running_reward

                # perform backprop
                finish_episode(preconditioner=preconditioner)

            # log results
            if i_episode % log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))

            # check if we have "solved" the cart pole problem
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break
                
        print(f'Finished training! Use {result["duration"]}')

        torch.save(policy.state_dict(), save)
        
        print('done')

        del model,env
        
    def setup_simulation(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,
                    deterministic=False,arch=None,predict=False):

        if pop is not None:
            self.n_pop=pop

        if load is not None:
            self.loadname=load

        if rlmodel is not None:
            self.rlmodel=rlmodel
            
        print('simulate')
            
        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        print('simulating ',self.loadname)

        # multiprocess environment
        policy_kwargs,n_cpu=self.get_multiprocess_env(rlmodel,debug=debug,arch=arch,predict=predict)
        n_cpu=4

        nonvec=False
        if nonvec:
            env=self.env
        else:
            env = SubprocVecEnv([lambda: make_env(self.environment, i, self.gym_kwargs) for i in range(n_cpu)])

        normalize=False
        if normalize:
            normalize_kwargs={}
            env = VecNormalize(env, **normalize_kwargs)
            
        print('predicting...')

        return policy,env,collector,n_cpu

    def simulate(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,startage=None,
                 deterministic=False,save='results/testsimulate',arch=None):

        policy,env,collector,n_cpu=self.setup_simulation(debug=debug,rlmodel=rlmodel,plot=plot,load=load,pop=pop,
                 deterministic=deterministic,arch=arch,predict=True)

        states = env.reset()
        if self.version==4:  # increase by 2
            n_add=2
            pop_num=np.array([k for k in range(0,n_add*n_cpu,n_add)])
            n=n_add*(n_cpu-1)
        else:  # increase by 1
            pop_num=np.array([k for k in range(0,n_cpu,1)])
            n_add=1
            n=n_cpu-1
        
        tqdm_e = tqdm(range(int(self.n_pop/n_add)), desc='Population', leave=True, unit=" p")
        self.episodestats.init_variables()
        
        if startage is not None:
            self.env.set_startage(startage)

        print('predict')
        #print(n_add,pop_num.shape)
        while np.any(pop_num<self.n_pop):
            act, predstate = policy.predict(states,deterministic=deterministic)
            newstate, rewards, dones, infos = env.step(act)
            for k in range(n_cpu):
                if pop_num[k]<self.n_pop: # do not save extras
                    if dones[k]:
                        self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],infos[k]['terminal_observation'],infos[k],debug=debug)
                        tqdm_e.update(1)
                        #print(n,pop_num[k])
                        n+=n_add
                        tqdm_e.set_description("Pop " + str(n))
                        pop_num[k]=n
                    else:
                        self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],newstate[k],infos[k],debug=debug)
                #else:
                #    print(f'n{n} vs n_pop {self.n_pop}')
    
            states = newstate

        self.episodestats.save_sim(save)

        if plot:
            self.render()

        if False:
            return self.emp        