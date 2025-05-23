'''
Runner for making fitting with Stable Baselines 2.0
- revised model
'''

import gym, numpy as np
import os
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker  import check_env as env_checker_check_env
from stable_baselines import A2C, ACER, DQN, ACKTR, PPO2 #, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
#from .vec_monitor import VecMonitor
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from .utils import make_env
import tensorflow as tf

#from tqdm import tqdm_notebook as tqdm # jos notebookissa
from tqdm import tqdm

from . episodestats import EpisodeStats
from . simstats import SimStats

#from multiprocessing import shared_memory
from multiprocessing import Process,Manager


#Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 16],
                                                          vf=[512, 256, 128])],
                                           feature_extraction="mlp")
                                           #act_fun=tf.nn.relu)

class runner_stablebaselines2():
    def __init__(self,environment,gamma,timestep,n_time,n_pop,
                 minimal,min_age,max_age,min_retirementage,year,episodestats,
                 gym_kwargs,version,processes=10):
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
        self.gym_kwargs=gym_kwargs.copy()
        self.gym_kwargs['silent']=True
        
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
        self.n_employment,self.n_acts=self.env.get_n_states()
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.action_shape = self.env.action_space.shape or self.env.action_space.n
        self.save_pop = gym_kwargs['save_pop']

        self.version = self.env.get_lc_version()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # n_add = 2
        self.model_twoperson = set([4,5,6,7,8,9,104])

        self.args={'gamma': gamma, 
              'version': version,
              'tau': 1.0,
              'environment': environment,
              'test': False,
              'hidden': 256,
              'n_time': n_time, 
              'timestep': timestep, 
              'n_pop': n_pop, 
              'min_age': min_age, 
              'max_age': max_age, 
              'processes': processes,
              'seed': 1,
              'debug': False,
              'lr': 0.01,
              'n_employment': self.n_employment,
              'n_acts': self.n_acts,
              'gym_kwargs': gym_kwargs,
              'startage': None,
              'rnn_steps': 20,
              'horizon': 0.99,
              'min_retirementage': min_retirementage, 
              'save_dir': 'results/',
              'simfile': 'results_'+str(year),
              'minimal': minimal,
              'render': False,
              'parttime_actions': self.env.setup_parttime_actions(),
              'save_pop': self.save_pop,
              'state_shape': (self.env.observation_space.shape or self.env.observation_space.n)[0],
              'year': year}

        self.episodestats=episodestats
        #SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
        #                       self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
        #                       version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma)
        
    def check_env(self,env):
        env_checker_check_env(env, warn=True)

    def get_multiprocess_env(self,rlmodel,debug=False,arch=None,predict=False):

        if arch is not None:
            print('arch',arch)

        # multiprocess environment
        if rlmodel=='a2c':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='acer':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='deep_acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 512, 256, 128, 64]) 
            n_cpu = 10 # 12 # 20
        elif rlmodel=='acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 512, 256]) 
            n_cpu = 10 # 12 # 20
        elif  rlmodel=='custom_acktr': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu,net_arch=[dict(pi=[32, 32, 32],vf=[128, 128, 128])]) 
            if predict:
                n_cpu = 20
            else:
                n_cpu = 10 # 12 # 20
        elif rlmodel=='leaky_acktr': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[256, 256, 128]) 
            if predict:
                n_cpu = 10
            else:
                n_cpu = 10 # 12 
        elif rlmodel=='ppo': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[256, 256, 16]) 
            if predict:
                n_cpu = 20
            else:
                n_cpu = 8 # 12 # 20
        elif rlmodel=='small_leaky_acktr': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[64, 64, 16]) 
            if predict:
                n_cpu = 16 
            else:
                n_cpu = 8 # 12 # 20
        elif rlmodel=='small_acktr' or rlmodel=='small_lnacktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256, 128]) 
            n_cpu = 4 #8
        elif rlmodel=='large_acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256, 64, 16])
            n_cpu = 4 # 12
        elif rlmodel=='lstm' or rlmodel=='lnacktr':
            policy_kwargs = dict()
            n_cpu = 4
        elif rlmodel=='trpo':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='dqn': # DQN
            policy_kwargs = dict(act_fun=tf.nn.relu, layers=[64, 64])
            n_cpu = 1
            rlmodel='dqn'
        else:
            error('Unknown rlmodel')

        if debug:
            n_cpu=1
            
        return policy_kwargs,n_cpu

    def setup_rlmodel(self,rlmodel,loadname,env,batch,policy_kwargs,learning_rate,
                      cont,max_grad_norm=None,tensorboard=False,verbose=1,n_cpu=1,
                      learning_schedule='linear',vf=None,gae_lambda=0.9):
        '''
        Alustaa RL-mallin ajoa varten
        
        gae_lambda=0.9
        '''
        #n_cpu_tf_sess=16 #n_cpu #4 vai n_cpu??
        n_cpu_tf_sess=4 #n_cpu #4 vai n_cpu?? vai 8?
        batch=max(1,int(np.ceil(batch/n_cpu)))
        
        full_tensorboard_log=False
        #vf=0.10*50 # 50 kerroin uutta, vain versiolle 7! FIXME
        vf=0.1 #02 # 0.10*50#00 # vain versiolle 7! FIXME
        if vf is not None:
            vf_coef=vf
        ent_coef=0.005 #1 # 0.01 # default 0.01

        if max_grad_norm is None:
            max_grad_norm=0.05 # default 0.50
            
        max_grad_norm=0.1 # 0.05 # 0.01 # 0.001  was old
        kfac_clip=0.001

        #if cont and not os.path.isfile(loadname):
        #    cont=False
        #    print(f'File {loadname} does not exist, switching to cont=False')
        
        if cont:
            learning_rate=0.25*learning_rate
            #learning_rate=0.5*learning_rate
            
        #scaled_learning_rate=learning_rate*np.sqrt(batch)
        #scaled_learning_rate=learning_rate*batch
        #scaled_learning_rate=learning_rate*8
        scaled_learning_rate=learning_rate
        print('batch {} learning rate {} scaled {} n_cpu {}'.format(batch,learning_rate,
            scaled_learning_rate,n_cpu))

        if cont:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = A2C.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                     tensorboard_log=self.tenb_dir, policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,
                                     n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = A2C.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                     policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel in set(['custom_acktr']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                       max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess,
                                       policy_kwargs=policy_kwargs) # 
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                       learning_rate=np.sqrt(batch)*learning_rate,vf_coef=vf_coef,gae_lambda=gae_lambda,
                                       max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess,
                                       policy_kwargs=policy_kwargs)
            elif rlmodel in set(['small_acktr','acktr','large_acktr','deep_acktr','leaky_acktr','small_leaky_acktr']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                       ent_coef=ent_coef,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                       learning_rate=np.sqrt(batch)*learning_rate,vf_coef=vf_coef,gae_lambda=gae_lambda,
                                       ent_coef=ent_coef,policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,
                                       lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel in set(['ppo','PPO']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = PPO2.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = PPO2.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                       learning_rate=np.sqrt(batch)*learning_rate,vf_coef=vf_coef,gae_lambda=gae_lambda,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel=='small_lnacktr' or rlmodel=='lnacktr':
                from stable_baselines.common.policies import MlpLnLstmPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,kfac_clip=kfac_clip,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=np.sqrt(batch)*learning_rate,kfac_clip=kfac_clip,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir,learning_rate=scaled_learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=scaled_learning_rate, policy_kwargs=policy_kwargs,
                                       max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            else:
                if tensorboard:
                    from stable_baselines.deepq.policies import MlpPolicy # for DQN
                    model = DQN.load(loadname, env=env, verbose=verbose,gamma=self.gamma,
                                     batch_size=batch,tensorboard_log=self.tenb_dir,
                                     policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,
                                     full_tensorboard_log=full_tensorboard_log,learning_rate=learning_rate)
                else:
                    from stable_baselines.deepq.policies import MlpPolicy # for DQN
                    model = DQN.load(loadname, env=env, verbose=verbose,gamma=self.gamma,
                                     batch_size=batch,tensorboard_log=self.tenb_dir,
                                     policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,
                                     learning_rate=learning_rate)
        else:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy 
                model = A2C(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time, 
                            tensorboard_log=self.tenb_dir, policy_kwargs=policy_kwargs,lr_schedule=learning_schedule)
            elif rlmodel in set(['small_acktr','acktr','large_acktr','deep_acktr','leaky_acktr','small_leaky_acktr']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                ent_coef=ent_coef,
                                full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                learning_rate=scaled_learning_rate, max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                ent_coef=ent_coef,
                                policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel in set(['custom_acktr']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,policy_kwargs=policy_kwargs,
                                full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                learning_rate=scaled_learning_rate, max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess,policy_kwargs=policy_kwargs)
            elif rlmodel in set(['ppo','PPO']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = PPO2(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = PPO2(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                learning_rate=scaled_learning_rate, max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel=='small_lnacktr' or rlmodel=='lnacktr':
                from stable_baselines.common.policies import MlpLnLstmPolicy 
                if tensorboard:
                    model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,full_tensorboard_log=full_tensorboard_log,
                                lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                learning_rate=scaled_learning_rate,n_cpu_tf_sess=n_cpu_tf_sess, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy 
                model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                            tensorboard_log=self.tenb_dir, learning_rate=learning_rate,n_cpu_tf_sess=n_cpu_tf_sess, 
                            policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule)
            else:
                from stable_baselines.deepq.policies import MlpPolicy # for DQN
                if tensorboard:
                    model = DQN(MlpPolicy, env, verbose=verbose,gamma=self.gamma,batch_size=batch, 
                                tensorboard_log=self.tenb_dir,learning_rate=learning_rate,
                                policy_kwargs=policy_kwargs,full_tensorboard_log=full_tensorboard_log) 
                else:
                    model = DQN(MlpPolicy, env, verbose=verbose,gamma=self.gamma,batch_size=batch,
                                learning_rate=learning_rate,policy_kwargs=policy_kwargs) 
                            
        return model
        

    def train(self,train=False,debug=False,steps=20_000,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=None,learning_rate=0.25,
                start_from=None,max_n_cpu=1000,use_vecmonitor=False,
                bestname='tmp/best2',use_callback=False,log_interval=100,verbose=1,plotdebug=False,
                learning_schedule='linear',vf=None,arch=None,gae_lambda=None,processes=None):
        '''
        Opetusrutiini
        '''
        self.best_mean_reward, self.n_steps = -np.inf, 0

        if pop is not None:
            self.n_pop=pop

        self.rlmodel=rlmodel
        self.bestname=bestname

        if processes is not None:
            self.args['processes']=processes

        pop=1 # self.n_pop
        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        # multiprocess environment
        policy_kwargs,n_cpu=self.get_multiprocess_env(self.rlmodel,debug=debug,arch=arch)  

        self.savename=save
        n_cpu=min(max_n_cpu,n_cpu)

        if self.args['processes'] is not None:
            n_cpu = min(self.args['processes'],n_cpu)

        print('n_cpu',n_cpu)

        if debug:
            print('use_vecmonitor',use_vecmonitor)
            print('use_callback',use_callback)

        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':True})
    
        nonvec=False
        if nonvec:
            env=self.env
        else:
            #env = make_vec_env(self.environment, n_envs=n_cpu, seed=1, vec_env_cls=SubprocVecEnv)
            env = SubprocVecEnv([lambda: make_env(self.environment, i, gkwargs) for i in range(n_cpu)], start_method='spawn')
            #env = DummyVecEnv([lambda: make_env(self.environment, i, gkwargs) for i in range(n_cpu)])
            #env = ShmemVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='fork')

            #if False:
                #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=gkwargs) for i in range(n_cpu)])

        model=self.setup_rlmodel(self.rlmodel,start_from,env,batch,policy_kwargs,learning_rate,
                                cont,max_grad_norm=max_grad_norm,verbose=verbose,n_cpu=n_cpu,
                                learning_schedule=learning_schedule,vf=vf,gae_lambda=gae_lambda)
        print('training..')

        if use_callback: # tässä ongelma, vecmonitor toimii => kuitenkin monta callbackia
            model.learn(total_timesteps=steps, callback=self.callback,log_interval=log_interval)
        else:
            model.learn(total_timesteps=steps, log_interval=log_interval)

        model.save(save)
        print('done')

        del model,env

    def setup_model_v2(self,env,rank=1,debug=False,rlmodel='acktr',plot=True,load=None,
                    deterministic=False,arch=None,predict=False,n_cpu_tf_sess=1):

        if rank==0:    
            print('simulating ',load)

        # multiprocess environment
        policy_kwargs,_=self.get_multiprocess_env(rlmodel,debug=debug,arch=arch,predict=predict)

        if rlmodel=='a2c':
            model = A2C.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif rlmodel in set(['acktr','small_acktr','lnacktr','small_lnacktr','deep_acktr','leaky_acktr','small_leaky_acktr']):
            model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif rlmodel=='trpo':
            model = TRPO.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif rlmodel=='custom_acktr':
            model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs, n_cpu_tf_sess=n_cpu_tf_sess)
        elif rlmodel=='ppo':
            model = PPO2.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif rlmodel=='dqn':
            model = DQN.load(load, env=env, verbose=1,gamma=self.gamma,prioritized_replay=True,policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        else:
            error('unknown model')

        return model

    def simulate(self,debug=False,rlmodel='acktr',load=None,pop=None,startage=None,
                 deterministic=False,save='results/testsimulate',arch=None,set_seed=True,render=False,processes=None):

        args=self.args.copy()
        args['debug']=debug
        args['rlmodel']=rlmodel
        args['load']=load
        args['pop']=pop
        args['startage']=startage
        args['deterministic']=deterministic
        args['save']=save
        args['arch']=arch
        args['set_seed']=set_seed
        args['render']=render
        args['simfile']=save

        if processes is not None:
            self.args['processes']=processes

        if render:  
            self.args['processes'] = 1 

        gkwargs=self.args['gym_kwargs'].copy()
        gkwargs.update({'train':False})
        
        pop=2 # self.n_pop
        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        manager = Manager()
        info = manager.dict({'pop': 0})
        processes = []

        print('simulating with',self.args['processes'],'processes')
        pop_left = args['pop']
        per_process = int(np.ceil(args['pop']/args['processes']/2))*2
        for rank in range(self.args['processes']):
            sim_num = min(per_process,pop_left)
            pop_left -= sim_num
            print('********','rank',rank,'sim_num',sim_num,'pop_left',pop_left)
            p = Process(target=self.simulate_single, args=(rank, args, gkwargs, sim_num, info))
            print('started.',rank)
            p.start()
            processes.append(p)

        print('switching to join')
        print(processes)
            
        for p in processes: 
            print('exitcode',p.exitcode)
            p.join()         

        print('joined.')

        self.combine_episodestats(args)
        
        print('done')
            
    def simulate_single(self, rank, args, kwargs, n_pop_single, info):
        '''
        An own process for each simulation unit
        '''

        print('sim_single',rank)

        if args['version'] in self.model_twoperson:  # increase by 2
            n_add=2
        else:  # increase by 1
            n_add=1

        render = args['render']
        print('pop',args['pop'],'procs',args['processes'])
        deterministic = args['deterministic']
        debug = args['debug']
        #savefile=args['save_dir']+args['simfile']+'_rank_'+str(rank)
        savefile=args['simfile']+'_rank_'+str(rank)
        #kwargs['silent']=True
        #print(kwargs)

        if rank==0:
            print('savefile',savefile)

        #env = SubprocVecEnv([make_env(args['environment'], 0, kwargs=kwargs)], start_method='spawn')
        #env = gym.make(args['environment'],kwargs=kwargs) # make a local (unshared) environment
        #env = SubprocVecEnv(env)
        #print('Child',rank,'check point 3')
        seed = 20_000 + rank*100 # must not be identical to train seed
        #env = SubprocVecEnv([lambda: make_env(args['environment'],seed+i,kwargs=kwargs) for i in range(1)], start_method='spawn')
        envlist=[lambda: make_env(args['environment'],seed+i,kwargs=kwargs) for i in range(1)]
        #print('Child',rank,'check point 3b')
        env = DummyVecEnv(envlist)
        #env = SubprocVecEnv(envlist, start_method='spawn')
        #print('spawning',args['environment'])
        #env = SubprocVecEnv([lambda: make_env(args['environment'],seed,kwargs=kwargs) for i in range(1)], start_method='spawn')
        #print('Child',rank,'check point 4')

        env.seed(args['seed'] + rank)

        model=self.setup_model_v2(env,rank=rank,debug=args['debug'],rlmodel=args['rlmodel'],load=args['load'],
                 deterministic=args['deterministic'],arch=args['arch'],predict=True,n_cpu_tf_sess=1)

        if args['startage'] is not None:
            env.set_startage(args['startage'])

        #print('Child',rank,'check point 5')

        states = env.reset()

        if rank == 0:
            print('predict',rank)
            if render:
                env.render()

        epienv = gym.make(args['environment'],kwargs=kwargs) # make a local (unshared) environment
        episodestats = SimStats(args['timestep'],args['n_time'],args['n_employment'],n_pop_single,
                            epienv,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True,parttime_actions=args['parttime_actions'],save_pop=args['save_pop'])
        episodestats.init_variables()

        if rank==0:
            tqdm_e = tqdm(range(args['pop']), desc='Population', leave=True, unit=" p")

        k=0
        n=0
        pred=0
        while n<n_pop_single:
            act, predstate = model.predict(states,deterministic=deterministic)
            newstate, rewards, dones, infos = env.step(act)
            if n<n_pop_single: # do not save extras
                if dones[k]:
                    episodestats.add(n,act[k],rewards[k],states[k],infos[k]['terminal_observation'],infos[k])
                    n += n_add
                    info['pop'] += n_add
                    if rank==0:
                        tqdm_e.update(info['pop']-pred)
                        tqdm_e.set_description("Pop " + str(info['pop']))
                        pred=info['pop']
                else:
                    episodestats.add(n,act[k],rewards[k],states[k],newstate[k],infos[k])
    
            states = newstate

        if rank==0:
            print('saving results...')

        episodestats.scale_sim()
        episodestats.save_sim(savefile)

        return 1

    def check_cstate(self):
        main_empstate,g,spouse_g,main_pension,main_old_paid_wage,age,time_in_state,main_paid_pension,pinkslip,toe,\
            toekesto,tyoura,used_unemp_benefit,main_wage_reduction,unemp_after_ra,unempwage,\
            unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen,\
            puoliso,spouse_empstate,spouse_old_paid_wage,spouse_pension,spouse_wage_reduction,\
            puoliso_paid_pension,puoliso_next_wage,puoliso_used_unemp_benefit,\
            puoliso_unemp_benefit_left,puoliso_unemp_after_ra,puoliso_unempwage,\
            puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_toe,puoliso_toekesto,puoliso_tyoura,spouse_time_in_state,puoliso_pinkslip,\
            puoliso_ove_paid,kansanelake,spouse_kansanelake,tyoelake_maksussa,\
            spouse_tyoelake_maksussa,main_next_wage,\
            main_paid_wage,spouse_paid_wage,\
            pt_act,sp_pt_act,main_basis_wage,spouse_basis_wage,\
            main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,\
            time_to_marriage,time_to_divorce,until_birth,\
            main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
                 = self.states.state_decode(self.state)
        print(f'2.0: c3 {children_under3} c7 {children_under7} c18 {children_under18}')

    def combine_results(self,results=None):
        self.combine_episodestats(self.args,results=results)

    def combine_episodestats(self,args,results=None):
        if results is None:
            save=args['simfile']+'_rank_'
        else:
            save=results+'_rank_'
        
        base=SimStats(args['timestep'],args['n_time'],args['n_employment'],args['n_pop'],
                            self.env,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True,save_pop=args['save_pop'])
        base.load_sim(save+'0')
        eps=SimStats(args['timestep'],args['n_time'],args['n_employment'],args['n_pop'],
                            self.env,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True,save_pop=args['save_pop'])
        for k in range(1,self.args['processes']):
            eps.load_sim(save+str(k))
            base.append_episodestat(eps)

        base.rescale_sim_with_procs(self.args['processes'])

        if results is None:
            base.save_sim(args['simfile']+'_combined')
        else:
            base.save_sim(results+'_combined')

        # remove rank files
        for k in range(0,self.args['processes']):
            os.remove(save+str(k))
