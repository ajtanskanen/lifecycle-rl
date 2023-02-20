'''
Runner for making fitting with stand-alone code without external libraries

Employ a3C with MLP + GRU
'''

from __future__ import print_function
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

import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
#from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
from cv2 import resize
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm as tqdm
import math

#from kfoptim import KFOptimizer

os.environ['OMP_NUM_THREADS'] = '1'

#SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: img #resize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.


class SharedAdam(torch.optim.AdamW): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

class A3CPolicy_no_attention(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(A3CPolicy, self).__init__()
        num_pt=num_actions[1]
        num_act=num_actions[0]
        self.conv1 = nn.Linear(channels, 256)
        self.conv2 = nn.Linear(256, 256)
        self.conv3 = nn.Linear(256, 256)
        self.gru = nn.GRUCell(256, memsize)
        self.critic_linear = nn.Linear(memsize, 1)
        self.actor_linear_act1, self.actor_linear_pt1 = nn.Linear(memsize, num_act), nn.Linear(memsize, num_pt)
        self.actor_linear_act2, self.actor_linear_pt2 = nn.Linear(memsize, num_act), nn.Linear(memsize, num_pt)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        hx = self.gru(x.view(-1, 256), (hx))
        return self.critic_linear(hx), self.actor_linear_act1(hx), self.actor_linear_pt1(hx), self.actor_linear_act2(hx), self.actor_linear_pt2(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

#@dataclass
class GPTConfig:
    block_size: int = 100
    n_layer: int = 12
    n_head: int = 1
    n_embd: int = 99
    dropout: float = 0#.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        x = x.view(1,1,-1)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class A3CPolicy_GRU(nn.Module): # an actor-critic neural network _with_Attention
    def __init__(self, channels, memsize, num_actions):
        super(A3CPolicy_GRU, self).__init__()
        num_pt=num_actions[1]
        num_act=num_actions[0]
        config = GPTConfig()
        self.memsize=memsize
        self.attn = CausalSelfAttention(config)
        self.conv1 = nn.Linear(99, 256)
        self.conv2 = nn.Linear(256, 128)
        self.conv3 = nn.Linear(128, 128)
        self.gru = nn.GRUCell(128, memsize)
        self.critic_linear = nn.Linear(memsize, 1)
        self.actor_linear_act1, self.actor_linear_pt1 = nn.Linear(memsize, num_act), nn.Linear(memsize, num_pt)
        self.actor_linear_act2, self.actor_linear_pt2 = nn.Linear(memsize, num_act), nn.Linear(memsize, num_pt)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = self.attn(inputs)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        hx = self.gru(x.view(-1, self.memsize), (hx))
        return self.critic_linear(hx), self.actor_linear_act1(hx), self.actor_linear_pt1(hx), self.actor_linear_act2(hx), self.actor_linear_pt2(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step        

class A3CPolicy_MLP(nn.Module): # an actor-critic neural network _with_Attention
    def __init__(self, channels, memsize, num_actions):
        super(A3CPolicy_MLP, self).__init__()
        num_pt=num_actions[1]
        num_act=num_actions[0]
        config = GPTConfig()
        self.conv1 = nn.Linear(99, 256)
        self.conv2 = nn.Linear(256, 256)
        self.conv3 = nn.Linear(256, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear_act1, self.actor_linear_pt1 = nn.Linear(256, num_act), nn.Linear(256, num_pt)
        self.actor_linear_act2, self.actor_linear_pt2 = nn.Linear(256, num_act), nn.Linear(256, num_pt)

    def forward(self, inputs, train=True, hard=False):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(1,-1)
        return self.critic_linear(x), self.actor_linear_act1(x), self.actor_linear_pt1(x), self.actor_linear_act2(x), self.actor_linear_pt2(x)#, x

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step                

class runner_standalone():
    def __init__(self,environment,gamma,timestep,n_time,n_pop,
                 minimal,min_age,max_age,min_retirementage,year,
                 episodestats,gym_kwargs,version,processes=5):
        self.minimal=minimal

        #try:
        #    mp.set_start_method('spawn') # this must not be in global scope
        #except:
        #    print('mp.set_start_method failed')

        self.env = gym.make(environment,kwargs=gym_kwargs)
        num_actions = self.env.action_space.shape or self.env.action_space.n
        n_employment,n_acts=self.env.get_n_states()
        #print('n_acts',n_acts)

        self.episodestats=episodestats
        
        self.args={'gamma': gamma, 
              'version': version,
              'tau': 1.0,
              'environment': environment,
              'test': False,
              'hidden': 128,
              'n_time': n_time, 
              'timestep': timestep, 
              'n_pop': n_pop, 
              'min_age': min_age, 
              'max_age': max_age, 
              'processes': processes,
              'seed': 1,
              'debug': False,
              'lr': 0.0001,
              'steps': 1e8,
              'gym_kwargs': gym_kwargs,
              'startage': None,
              'rnn_steps': 20,
              'horizon': 0.999,
              'min_retirementage': min_retirementage, 
              'num_actions': num_actions,
              'n_employment': n_employment,
              #'save_dir': 'a3c_results/',
              'save_dir': 'a3c_attn/',
              'simfile': 'results',
              'n_acts': n_acts,
              'minimal': minimal,
              'render': False,
              'state_shape': (self.env.observation_space.shape or self.env.observation_space.n)[0],
              'year': year}
        
        self.version = self.env.get_lc_version()
        #print('state_shape',self.args['state_shape'])
        #print('population',n_pop)

        self.episodestats=episodestats # global episodestats
        #SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
        #                           self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
        #                           version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma)
                                   
        #self.model = Policy()
        #optimizer = optim.AdamW(model.parameters(), lr=3e-3)

        #print(model.state_dict())

        #optimizer=KFACOptimizer(model,lr=1e-3)
        #self.optimizer=KFOptimizer(model.parameters(),model,lr=1e-4,stat_decay=0.995)
       
        
    def check_env(self,env):
        return

    def load_model(self,PATH):
        # Model class must be defined somewhere
        model = torch.load(PATH)
        model.eval() 
        
    def train(self,train=False,debug=False,steps=20_000,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=None,learning_rate=0.25,
                start_from=None,max_n_cpu=1000,use_vecmonitor=False,
                bestname='tmp/best2',use_callback=False,log_interval=100,verbose=1,plotdebug=False,
                learning_schedule='linear',vf=None,arch=None,gae_lambda=None,render=False, include_GRU=False,
                processes=None):
        '''
        Opetusrutiini
        '''
        self.test=False
        if render:  
            self.processes = 1 
            self.test = True # render mode -> test mode w one process

        if self.test:  
            self.lr = 0 # don't train in render mode

        self.args['steps']=steps

        os.makedirs(self.args['save_dir']) if not os.path.exists(self.args['save_dir']) else None # make dir to save models etc.

        torch.manual_seed(self.args['seed'])
        if include_GRU:
            shared_model = A3CPolicy_GRU(channels=self.args['state_shape'], memsize=self.args['hidden'], num_actions=self.args['n_acts']).share_memory()
        else:
            shared_model = A3CPolicy_MLP(channels=self.args['state_shape'], memsize=self.args['hidden'], num_actions=self.args['n_acts']).share_memory()

        shared_optimizer = SharedAdam(shared_model.parameters(), lr=self.args['lr'])

        info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames', 'pop']}
        if cont:
            info['frames'] += shared_model.try_load(self.args['save_dir']) * 1e6

        if int(info['frames'].item()) == 0: printlog(self.args,'', end='', mode='w') # clear log file
        
        self.rlmodel=rlmodel
        self.bestname=bestname

        gkwargs=self.args['gym_kwargs'].copy()
        gkwargs.update({'train':True})
    
        self.savename=save

        print('training...')

        processes = []
        for rank in range(self.args['processes']):
            p = mp.Process(target=self.train_single, args=(shared_model, shared_optimizer, rank, self.args, gkwargs, info, include_GRU))
            p.start()
            processes.append(p)
            
        for p in processes: 
            p.join()
        
        print('saving...')
        num_frames = int(info['frames'].item())
        torch.save(shared_model.state_dict(), self.args['save_dir']+'model.{:.0f}.tar'.format(num_frames/1e6))
        print('done')

        #del model,env

    def cost_func(self, args, values, logps_act1, logps_pt1, logps_act2, logps_pt2, actions_act1, actions_pt1, actions_act2, actions_pt2, rewards):
        np_values = values.view(-1).data.numpy()

        # generalized advantage estimation using \delta_t residuals (a policy gradient method)
        delta_t = np.asarray(rewards) + args['gamma'] * np_values[1:] - np_values[:-1]
        logpys_act1 = logps_act1.gather(1, torch.tensor(actions_act1).view(-1,1))
        logpys_act2 = logps_act2.gather(1, torch.tensor(actions_act2).view(-1,1))
        logpys_pt1 = logps_pt1.gather(1, torch.tensor(actions_pt1).view(-1,1))
        logpys_pt2 = logps_pt2.gather(1, torch.tensor(actions_pt2).view(-1,1))
        gen_adv_est = discount(delta_t, args['gamma'] * args['tau'])
        policy_loss = -(logpys_act1.view(-1) * torch.FloatTensor(gen_adv_est.copy()) + \
                        logpys_pt1.view(-1) * torch.FloatTensor(gen_adv_est.copy()) + \
                        logpys_act2.view(-1) * torch.FloatTensor(gen_adv_est.copy()) + \
                        logpys_pt1.view(-1) * torch.FloatTensor(gen_adv_est.copy()) \
                       ).sum()
        
        # l2 loss over value estimator
        rewards[-1] += args['gamma'] * np_values[-1]
        discounted_r = discount(np.asarray(rewards), args['gamma'])
        discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float)
        value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

        entropy_loss = (-logps_act1 * torch.exp(logps_act1) + \
                        -logps_act2 * torch.exp(logps_act2)).sum() + \
                    (-logps_pt1 * torch.exp(logps_pt1) + \
                        -logps_pt2 * torch.exp(logps_pt2) \
                    ).sum() # entropy definition, for entropy regularization
        return policy_loss + 0.25 * value_loss - 0.01 * entropy_loss

    def train_single(self,shared_model, shared_optimizer, rank, args, kwargs, info, include_GRU=False):
        print('train-single',rank)
        kwargs['silent']=True
        env = gym.make(args['environment'],kwargs=kwargs) # make a local (unshared) environment
        env.seed(args['seed'] + rank)
        torch.manual_seed(args['seed'] + rank) # seed everything
        if include_GRU:
            model = A3CPolicy_GRU(channels=self.args['state_shape'], memsize=args['hidden'], num_actions=args['n_acts']) # a local/unshared model
        else:
            model = A3CPolicy_MLP(channels=self.args['state_shape'], memsize=args['hidden'], num_actions=args['n_acts']) # a local/unshared model
        state = torch.tensor(prepro(env.reset()),dtype=torch.float) # get first state

        start_time = last_disp_time = time.time()
        episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping
        maxlen = args['steps']

        while info['frames'][0] <= maxlen: # openai baselines uses 40M frames...we'll use 80M
            model.load_state_dict(shared_model.state_dict()) # sync with shared model

            if include_GRU:
                hx = torch.zeros(1, args['hidden']) if done else hx.detach()  # rnn activation vector

            values, rewards = [], [] # save values for computing gradientss
            actions_act1, actions_act2, actions_pt1, actions_pt2 = [], [], [], []
            logps_act1, logps_pt1, logps_act2, logps_pt2 = [], [], [], []

            for step in range(args['rnn_steps']):
                episode_length += 1
                if include_GRU:
                    value, logit_act1, logit_pt1, logit_act2, logit_pt2, hx = model((state.view(1,1,1,args['state_shape']), hx))
                else:
                    value, logit_act1, logit_pt1, logit_act2, logit_pt2 = model(state.view(1,1,99))

                logp_act1 = F.log_softmax(logit_act1, dim=-1)
                logp_pt1 = F.log_softmax(logit_pt1, dim=-1)
                logp_act2 = F.log_softmax(logit_act2, dim=-1)
                logp_pt2 = F.log_softmax(logit_pt2, dim=-1)

                action1 = torch.exp(logp_act1).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                action2 = torch.exp(logp_act2).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                pt1 = torch.exp(logp_pt1).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                pt2 = torch.exp(logp_pt2).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                action=np.array([action1.numpy()[0],action2.numpy()[0],pt1.numpy()[0],pt2.numpy()[0]])
                state, reward, done, _ = env.step(action)
                if args['render']: env.render()

                state = torch.tensor(prepro(state),dtype=torch.float)
                epr += reward
                #reward = np.clip(reward, -1, 1) # reward
                reward = np.clip(reward, -10, 10) # reward
                #done = done or episode_length >= 1e4 # don't playing one ep for too long
                
                info['frames'].add_(1) ; num_frames = int(info['frames'].item())
                if num_frames % 1e6 == 0: # save every 1M frames
                    printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                    torch.save(shared_model.state_dict(), args['save_dir']+'model.{:.0f}.tar'.format(num_frames/1e6))

                if done: # update shared data
                    info['episodes'] += 1
                    interp = 1 if info['episodes'][0] < 2 else 1 - args['horizon']
                    info['run_epr'].mul_(1-interp).add_(interp * epr)
                    info['run_loss'].mul_(1-interp).add_(interp * eploss)

                if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                        .format(elapsed, info['episodes'].item(), num_frames/1e6,
                        info['run_epr'].item(), info['run_loss'].item()))
                    last_disp_time = time.time()

                if done: # maybe print info.
                    episode_length, epr, eploss = 0, 0, 0
                    state = torch.tensor(prepro(env.reset()),dtype=torch.float)

                values.append(value) ; rewards.append(reward)
                actions_act1.append(action1) ; actions_act2.append(action2) ; actions_pt1.append(pt1) ; actions_pt2.append(pt2)
                logps_act1.append(logp_act1) ; logps_pt1.append(logp_pt1) ; logps_act2.append(logp_act2) ; logps_pt2.append(logp_pt2)

            if include_GRU:
                next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
            else:
                next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0)))[0]
            values.append(next_value.detach())

            loss = self.cost_func(args, torch.cat(values), torch.cat(logps_act1), torch.cat(logps_pt1), torch.cat(logps_act2), torch.cat(logps_pt2),\
                    torch.cat(actions_act1), torch.cat(actions_pt1), torch.cat(actions_act2), torch.cat(actions_pt2), np.asarray(rewards))
            eploss += loss.item()
            shared_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

            for param, shared_param in zip(model.parameters(), shared_model.parameters()):
                if shared_param.grad is None: 
                    shared_param._grad = param.grad # sync gradients with shared model
            shared_optimizer.step()        

    def simulate_single_v0(self,shared_model, shared_optimizer, rank, args, kwargs, info, include_GRU=False):

        if args['version'] in set([4,5]):  # increase by 2
            n_add=2
            #pop_num=np.array([k for k in range(0,n_add*n_cpu,n_add)])
            #n=n_add*(n_cpu-1)
        else:  # increase by 1
            #pop_num=np.array([k for k in range(0,n_cpu,1)])
            n_add=1
            #n=n_cpu-1
        
        render = args['render']
        kwargs['silent']=True

        env = gym.make(args['environment'],kwargs=kwargs) # make a local (unshared) environment
        env.seed(args['seed'] + rank)

        if args['startage'] is not None:
            env.set_startage(args['startage'])

        oldstate = env.reset()

        savefile=args['save_dir']+args['simfile']+'_rank_'+str(rank)

        if render and rank == 0:
            env.render()

        single_n_pop=int(math.ceil(args['n_pop']/args['processes']))

        episodestats=SimStats(args['timestep'],args['n_time'],args['n_employment'],single_n_pop,
                            env,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True)
        episodestats.init_variables()

        torch.manual_seed(args['seed'] + rank) # seed everything
        model = A3CPolicy(channels=self.args['state_shape'], memsize=args['hidden'], num_actions=args['n_acts']) # a local/unshared model
        model.eval() # no backprop
        state = torch.tensor(prepro(env.reset()),dtype=torch.float) # get first state

        start_time = last_disp_time = time.time()
        episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        n_pop = args['n_pop']
        debug = args['debug']

        p=0
        pred=0
        last_proc=args['processes']-1

        if rank==last_proc:
            print('predict')
            tqdm_e = tqdm(range(int(n_pop)), desc='Population', leave=True, unit=" p")

        while p < single_n_pop: # openai baselines uses 40M frames...we'll use 80M
            hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
            values, rewards = [], [] # save values for computing gradientss
            actions_act1, actions_act2, actions_pt1, actions_pt2 = [], [], [], []
            logps_act1, logps_pt1, logps_act2, logps_pt2 = [], [], [], []

            for step in range(args['rnn_steps']):
                episode_length += 1
                value, logit_act1, logit_pt1, logit_act2, logit_pt2, hx = model((state.view(1,1,1,args['state_shape']), hx)) #,deterministic=deterministic
                logp_act1 = F.log_softmax(logit_act1, dim=-1)
                logp_pt1 = F.log_softmax(logit_pt1, dim=-1)
                logp_act2 = F.log_softmax(logit_act2, dim=-1)
                logp_pt2 = F.log_softmax(logit_pt2, dim=-1)

                action1 = torch.exp(logp_act1).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                action2 = torch.exp(logp_act2).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                pt1 = torch.exp(logp_pt1).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                pt2 = torch.exp(logp_pt2).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                action=np.array([action1.numpy()[0],action2.numpy()[0],pt1.numpy()[0],pt2.numpy()[0]])
                state, reward, done, infos = env.step(action)
                newstate = state
                if args['render']: env.render()

                state = torch.tensor(prepro(state),dtype=torch.float)
                epr += reward
                reward = np.clip(reward, -1, 1) # reward
                #done = done or episode_length >= 1e4 # don't playing one ep for too long
                
                info['frames'].add_(1) ; num_frames = int(info['frames'].item())
                
                if done: # update shared data
                    if p < n_pop:
                        interp = 1 if p == 1 else 1 - args['horizon']
                        info['run_epr'].mul_(1-interp).add_(interp * epr)
                        info['run_loss'].mul_(1-interp).add_(interp * eploss)
                        # correct:
                        #episodestats.add(pop_num,action,rew,oldstate,infos['terminal_observation'],infos,debug=debug)
                        # broken:
                        episodestats.add(p,action,rew,oldstate,newstate,infos,debug=debug)
                        newstate = env.reset()
                    else:
                        break
                    
                    info['pop_num'].add_(n_add)                      
                    p += n_add
                else:
                    if p < n_pop:
                        episodestats.add(p,action,rew,oldstate,newstate,infos,debug=debug)

                oldstate = newstate

                # if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                #     delta = (time.time() - start_time)
                #     elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(delta))
                #     printlog(args, 'time {}, pop_num {:.0f}/{}, speed {:.1f} pps, {} s, frames {}M, mean epr {:.2f}, run loss {:.2f}'
                #         .format(elapsed, info['pop_num'].item(),n_pop, info['pop_num'].item()/delta,delta,num_frames/1e6,
                #         info['run_epr'].item(), info['run_loss'].item()))
                #     last_disp_time = time.time()

                if done: # maybe print info.
                    episode_length, epr, eploss = 0, 0, 0
                    state = torch.tensor(prepro(env.reset()),dtype=torch.float)

                    if rank==last_proc:
                        num=info['pop_num'][0].item()
                        tqdm_e.update(int(num-pred))
                        tqdm_e.set_description("Pop " + str(int(num)))
                        pred=num

        episodestats.save_sim(savefile)

    def simulate_single(self,shared_model, shared_optimizer, rank, args, kwargs, info,include_GRU=True):
        '''
        v1
        '''

        if args['version'] in set([4,5]):  # increase by 2
            n_add=2
            #pop_num=np.array([k for k in range(0,n_add*n_cpu,n_add)])
            #n=n_add*(n_cpu-1)
        else:  # increase by 1
            #pop_num=np.array([k for k in range(0,n_cpu,1)])
            n_add=1
            #n=n_cpu-1
        
        render = args['render']
        kwargs['silent']=True

        env = gym.make(args['environment'],kwargs=kwargs) # make a local (unshared) environment
        env.seed(args['seed'] + rank)

        if args['startage'] is not None:
            env.set_startage(args['startage'])

        oldstate = env.reset()

        savefile=args['save_dir']+args['simfile']+'_rank_'+str(rank)

        if render and rank == 0:
            env.render()

        single_n_pop=int(math.ceil(args['n_pop']/args['processes']))

        episodestats=SimStats(args['timestep'],args['n_time'],args['n_employment'],single_n_pop,
                            env,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True)
        episodestats.init_variables()

        torch.manual_seed(args['seed'] + rank) # seed everything
        if include_GRU:
            model = A3CPolicy_GRU(channels=self.args['state_shape'], memsize=args['hidden'], num_actions=args['n_acts']) # a local/unshared model
        else:
            model = A3CPolicy_noGRU(channels=self.args['state_shape'], memsize=args['hidden'], num_actions=args['n_acts']) # a local/unshared model
        model.eval() # no backprop
        state = torch.tensor(prepro(env.reset()),dtype=torch.float) # get first state

        start_time = last_disp_time = time.time()
        episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        n_pop = args['n_pop']
        debug = args['debug']

        p=0
        pred=0
        last_proc=args['processes']-1

        if rank==last_proc:
            print('predict')
            tqdm_e = tqdm(range(int(n_pop)), desc='Population', leave=True, unit=" p")

        while p < single_n_pop: # openai baselines uses 40M frames...we'll use 80M
            if include_GRU:
                hx = torch.zeros(1, args['hidden']) if done else hx.detach()  # rnn activation vector
            values, rewards = [], [] # save values for computing gradientss
            actions_act1, actions_act2, actions_pt1, actions_pt2 = [], [], [], []
            logps_act1, logps_pt1, logps_act2, logps_pt2 = [], [], [], []
            done = False

            while not done: #for step in range(args['rnn_steps']):
                episode_length += 1
                if include_GRU:
                    value, logit_act1, logit_pt1, logit_act2, logit_pt2, hx = model((state.view(1,1,1,args['state_shape']), hx)) #,deterministic=deterministic
                else:
                    value, logit_act1, logit_pt1, logit_act2, logit_pt2, hx = model((state.view(1,1,1,args['state_shape']))) #,deterministic=deterministic
                logp_act1 = F.log_softmax(logit_act1, dim=-1)
                logp_pt1 = F.log_softmax(logit_pt1, dim=-1)
                logp_act2 = F.log_softmax(logit_act2, dim=-1)
                logp_pt2 = F.log_softmax(logit_pt2, dim=-1)

                action1 = logp_act1.max(1)[1].data # 
                action2 = logp_act2.max(1)[1].data # 
                pt1 = logp_pt1.max(1)[1].data # 
                pt2 = logp_pt2.max(1)[1].data # 
                action=np.array([action1.numpy()[0],action2.numpy()[0],pt1.numpy()[0],pt2.numpy()[0]])
                state, reward, done, infos = env.step(action)
                newstate = state
                if args['render']: env.render()

                state = torch.tensor(prepro(state),dtype=torch.float)
                epr += reward
                rew = reward
                reward = np.clip(reward, -1, 1) # reward
                
                info['frames'].add_(1)
                num_frames = int(info['frames'].item())
                
                if done: # update shared data
                    if p < n_pop:
                        interp = 1 if p == 1 else 1 - args['horizon']
                        info['run_epr'].mul_(1-interp).add_(interp * epr)
                        info['run_loss'].mul_(1-interp).add_(interp * eploss)
                        # correct:
                        #episodestats.add(pop_num,action,rew,oldstate,infos['terminal_observation'],infos,debug=debug)
                        # broken:
                        episodestats.add(p,action,rew,oldstate,newstate,infos,debug=debug)
                        newstate = env.reset()                        
                    else:
                        break
                    
                    info['pop_num'].add_(n_add)                      
                    p += n_add

                    episode_length, epr, eploss = 0, 0, 0
                    state = torch.tensor(prepro(env.reset()),dtype=torch.float)

                    if rank==last_proc:
                        num=info['pop_num'][0].item()
                        tqdm_e.update(int(num-pred))
                        tqdm_e.set_description("Pop " + str(int(num)))
                        pred=num
                else:
                    if p < n_pop:
                        episodestats.add(p,action,rew,oldstate,newstate,infos,debug=debug)

                oldstate = newstate

        episodestats.save_sim(savefile)        

    def combine_episodestats(self,args):
        save=args['save_dir']+args['simfile']+'_rank_'
        
        base=SimStats(args['timestep'],args['n_time'],args['n_employment'],args['n_pop'],
                            self.env,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True)
        base.load_sim(save+'0')
        eps=SimStats(args['timestep'],args['n_time'],args['n_employment'],args['n_pop'],
                            self.env,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True)
        for k in range(1,self.args['processes']):
            eps.load_sim(save+str(k))
            base.append_episodestat(eps)

        base.save_sim(args['save_dir']+args['simfile']+'_combined')

    def simulate(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,startage=None,processes=None,
                 deterministic=False,save='results/testsimulate',arch=None,render=False,include_GRU=True):
        '''
        Opetusrutiini
        '''
        print('simulate')

        if pop is not None:
            self.args['n_pop']=pop

        self.test=False
        if render:  
            self.processes = 1 
            self.test = True # render mode -> test mode w one process
        if self.test:  
            self.lr = 0 # don't train in render mode

        torch.manual_seed(self.args['seed'])
        if include_GRU:
            shared_model = A3CPolicy_GRU(channels=self.args['state_shape'], memsize=self.args['hidden'], num_actions=self.args['n_acts']).share_memory()
        else:
            shared_model = A3CPolicy_noGRU(channels=self.args['state_shape'], memsize=self.args['hidden'], num_actions=self.args['n_acts']).share_memory()
        shared_optimizer = SharedAdam(shared_model.parameters(), lr=self.args['lr'])

        info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'pop_num', 'frames', 'pop']}
        info['frames'] += shared_model.try_load(self.args['save_dir']) * 1e6
        if int(info['frames'].item()) == 0: printlog(self.args,'', end='', mode='w') # clear log file
        
        gkwargs=self.args['gym_kwargs'].copy()
        gkwargs.update({'train':False})
        self.args['gym_kwargs']=gkwargs.copy()
    
        processes = []
        print('spawning...')
        for rank in range(self.args['processes']):
            p = mp.Process(target=self.simulate_single, args=(shared_model, shared_optimizer, rank, self.args, gkwargs, info,include_GRU))
            p.start()
            processes.append(p)
            
        print('simulating...')
        for p in processes: 
            p.join()         

        print('combining...')
        self.combine_episodestats(self.args)
        
        print('done')

    def combine_results(self,results):
        self.combine_episodestats(self.args)

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args['save_dir']+'log.txt',mode) ; f.write(s+'\n') ; f.close()
