'''
Runner for making fitting with Stable Tianshou 0.4.5
'''

import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.policy import A2CPolicy,NPGPolicy
from tianshou.env import DummyVectorEnv,SubprocVectorEnv
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import A2CPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from tqdm import tqdm_notebook as tqdm

from . episodestats import EpisodeStats
from . simstats import SimStats
from . utils import make_env


from tianshou.data import Batch
from tianshou.utils.net.common import MLP
        

class runner_tianshou():
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
        
        self.gae_lambda=0.91
        self.max_grad_norm=0.1
        self.vf_coef=0.5
        self.ent_coef=0.01
        self.rew_norm=True
        self.bound_action_method=""
        
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
        self.n_employment,self.n_acts=self.env.get_n_states()
        
        self.version = self.env.get_lc_version()

        self.episodestats=episodestats
        #SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
        #                           self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
        #                           version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma)
        
    def check_env(self,env):
        return

    def setup_training(self,rlmodel,loadname,env,batch,cont):
        '''
        Alustaa RL-mallin ajoa varten
        
        '''
        
    def setup_policy(self,learning_rate):
        
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.action_shape = (5,5) #self.env.action_space.shape or self.env.action_space.n
        self.hidden_sizes=[128, 128, 128]
        self.device='cpu'
        
        net_a = Net(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_sizes=self.hidden_sizes,
            activation=nn.ReLU,
            device=self.device
        )
        actor = Actor(
            net_a,
            self.action_shape,
            device=self.device
        ).to(self.device)
        
        net_c = Net(
            state_shape=self.state_shape,
            hidden_sizes=self.hidden_sizes,
            activation=nn.ReLU,
            device=self.device
        )
        critic = Critic(net_c, device=self.device).to(self.device)

        net = Net(state_shape=self.state_shape, hidden_sizes=self.hidden_sizes)
        actor = Actor(net, action_shape=self.action_shape)
        critic = Critic(net)

        lr_scheduler = None

        optim = torch.optim.RMSprop(ActorCritic(actor, critic).parameters(), lr=learning_rate)
        dist = torch.distributions.Categorical
                
        policy = A2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=self.gamma,
            gae_lambda=self.gae_lambda,
            max_grad_norm=self.max_grad_norm,
            vf_coef=self.vf_coef,
            ent_coef=self.ent_coef,
            reward_normalization=self.rew_norm,
            action_scaling=False,
            action_bound_method=self.bound_action_method,
            lr_scheduler=lr_scheduler,
            action_space=self.env.action_space
        )

        return policy
        
    def train(self,train=False,debug=False,steps=20_000,n_episodes=100,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=None,learning_rate=0.25,
                start_from=None,max_n_cpu=1000,use_callback=None,use_vecmonitor=None,
                bestname='tmp/best2',log_interval=100,verbose=1,plotdebug=False,
                learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
        '''
        Opetusrutiini
        '''
        self.best_mean_reward, self.n_steps = -np.inf, 0
        self.learning_rate=learning_rate

        if pop is not None:
            self.n_pop=pop

        self.rlmodel=rlmodel
        self.bestname=bestname

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        self.savename=save
        n_cpu=8
        n_cpu=min(max_n_cpu,n_cpu)

        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':True})
            
        train_envs = SubprocVectorEnv([lambda: make_env(self.environment, i, gkwargs, use_monitor=False) for i in range(n_cpu)])
        test_envs = SubprocVectorEnv([lambda: make_env(self.environment, i, gkwargs, use_monitor=False) for i in range(n_cpu)])

        #train_envs = ts.env.DummyVectorEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for _ in range(n_cpu)])
        #test_envs = ts.env.DummyVectorEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for _ in range(n_cpu)]) 
        
        target_freq=1
        
        # seed
        seed=1
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
        
        # model
        policy=self.setup_policy(self.learning_rate)
        
        #policy = ts.policy.NPGPolicy(self.net, self.optim, self.gamma, steps)

        if cont:
            policy.load_state_dict(torch.load(start_from))

        #self.train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
        train_collector = ts.data.Collector(policy, train_envs)
        test_collector = ts.data.Collector(policy, train_envs)
        
        print('training..')
        step_per_epoch=210
        step_per_collect=210
        batch_size=8
        max_epoch=n_episodes
        result = ts.trainer.onpolicy_trainer(
            policy, train_collector, test_collector, max_epoch, step_per_epoch, step_per_collect,
            n_cpu, batch_size, verbose=True, test_in_train=False,episode_per_collect=1) #,logger=logger)
        print(f'Finished training! Use {result["duration"]}')

        torch.save(policy.state_dict(), save)
        
        print('done')

        del env
        
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

        policy=self.setup_policy(1e-6)
        policy.load_state_dict(torch.load(load))
        
        # multiprocess environment
        #policy_kwargs,n_cpu=self.get_multiprocess_env(rlmodel,debug=debug,arch=arch,predict=predict)
        n_cpu=1
        
        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':False})

        nonvec=True
        if nonvec:
            env=self.env
        else:
            env = SubprocVectorEnv([lambda: make_env(self.environment, i, gkwargs, use_monitor=False) for i in range(n_cpu)])
    
        print('predicting...')

        #policy = ts.policy.ACKTRPolicy(self.net, self.optim, self.gamma, n_step, deterministic_eval=deterministic)
        #policy.eval()
        collector = ts.data.Collector(policy, env, exploration_noise=True)
        collector.collect(n_episode=1, render=1 / 35)

        #model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)

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
            
        # assume env is a gym.Env
        obs, done = self.env.reset(), False
        while not done:
            batch = Batch(obs=[obs])  # the first dimension is batch-size
            act = policy(batch).act[0]  # policy.forward return a batch, use ".act" to extract the action
            print(act)
            obs, rew, done, info = env.step(act)

        print('predict')
        states = self.env.reset()
        while np.any(pop_num<self.n_pop):
            #act, predstate = policy.predict(states,deterministic=deterministic)
            
            batch = Batch(obs=[states])  # the first dimension is batch-size
            act = policy(batch).act[0]  # policy.forward return a batch, use ".act" to extract the action
            print(act)
            #obs, rew, done, info = env.step(act)
                
            #b=Batch(obs=states)<z
            #batch=policy.forward(b)
            #act=batch.act
            #print(act)
            newstate, rewards, dones, infos = env.step(act)
            for k in range(n_cpu):
                if pop_num[k]<self.n_pop: # do not save extras
                    #if dones[k]:
                    if dones:
                        #self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],infos[k]['terminal_observation'],infos[k],debug=debug)
                        self.episodestats.add(pop_num[k],act,rewards,states,states,infos,debug=debug)
                        tqdm_e.update(1)
                        n+=n_add
                        tqdm_e.set_description("Pop " + str(n))
                        pop_num[k]=n
                        states=self.env.reset()
                        dones=False
                    else:
                        #self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],newstate[k],infos[k],debug=debug)
                        self.episodestats.add(pop_num[k],act,rewards,states,newstate,infos,debug=debug)
    
            states = newstate

        self.episodestats.save_sim(save)

        if plot:
            self.render()

        if False:
            return self.emp        
