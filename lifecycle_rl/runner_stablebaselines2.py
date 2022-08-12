'''
Runner for making fitting with Stable Baselines 2.0
'''

import gym, numpy as np
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines.common.env_checker  import check_env as env_checker_check_env
from stable_baselines import A2C, ACER, DQN, ACKTR, PPO2 #, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from .vec_monitor import VecMonitor
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from .utils import make_env
import tensorflow as tf

#from tqdm import tqdm_notebook as tqdm # jos notebookissa
from tqdm import tqdm_notebook as tqdm

from . episodestats import EpisodeStats
from . simstats import SimStats


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
                 gym_kwargs):
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

        self.version = self.env.get_lc_version()

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
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[256, 256, 16]) 
            if predict:
                n_cpu = 10
            else:
                n_cpu = 10 # 12 # 20
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
        n_cpu_tf_sess=16 #n_cpu #4 vai n_cpu??
        #batch=max(1,int(np.ceil(batch/n_cpu)))
        batch=max(1,int(np.ceil(batch/n_cpu)))
        
        full_tensorboard_log=True
        if vf is not None:
            vf_coef=vf
        else:
            vf_coef=0.10 # baseline 0.25, best 0.10

        if max_grad_norm is None:
            max_grad_norm=0.05
            
        max_grad_norm=0.001 # ok?
        kfac_clip=0.001
        
        if cont:
            learning_rate=0.25*learning_rate
            
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
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                       learning_rate=np.sqrt(batch)*learning_rate,vf_coef=vf_coef,gae_lambda=gae_lambda,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
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
                                full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                learning_rate=scaled_learning_rate, max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
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
                learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
        '''
        Opetusrutiini
        '''
        self.best_mean_reward, self.n_steps = -np.inf, 0

        if pop is not None:
            self.n_pop=pop

        self.rlmodel=rlmodel
        self.bestname=bestname

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        # multiprocess environment
        policy_kwargs,n_cpu=self.get_multiprocess_env(self.rlmodel,debug=debug,arch=arch)  

        self.savename=save
        n_cpu=min(max_n_cpu,n_cpu)

        if debug:
            print('use_vecmonitor',use_vecmonitor)
            print('use_callback',use_callback)

        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':True})
    
        nonvec=False
        if nonvec:
            env=self.env
        else:
            if use_vecmonitor:
                env = SubprocVecEnv([lambda: make_env(self.environment, i, gkwargs, use_monitor=False) for i in range(n_cpu)], start_method='spawn')
                env = VecMonitor(env,filename=self.log_dir+'monitor.csv')
            else:
                env = SubprocVecEnv([lambda: make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='spawn')
                #env = ShmemVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='fork')

            #if False:
                #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=gkwargs) for i in range(n_cpu)])

        normalize=False
        if normalize:
            normalize_kwargs={}
            env = VecNormalize(env, **normalize_kwargs)

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

#     def save_to_hdf(self,filename,nimi,arr,dtype):
#         f = h5py.File(filename, 'w')
#         dset = f.create_dataset(nimi, data=arr, dtype=dtype)
#         f.close()
# 
#     def load_hdf(self,filename,nimi):
#         f = h5py.File(filename, 'r')
#         val=f.get(nimi).value
#         f.close()
#         return val
        
    def setup_model(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,
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
        n_cpu_tf_sess=4

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

        if self.rlmodel=='a2c':
            model = A2C.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel in set(['acktr','small_acktr','lnacktr','small_lnacktr','deep_acktr','leaky_acktr','small_leaky_acktr']):
            model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel=='trpo':
            model = TRPO.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel=='custom_acktr':
            model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs, n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel=='ppo':
            model = PPO2.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel=='dqn':
            model = DQN.load(load, env=env, verbose=1,gamma=self.gamma,prioritized_replay=True,policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        else:
            error('unknown model')

        return model,env,n_cpu

    def simulate(self,debug=False,rlmodel='acktr',load=None,pop=None,startage=None,
                 deterministic=False,save='results/testsimulate',arch=None):

        model,env,n_cpu=self.setup_model(debug=debug,rlmodel=rlmodel,load=load,pop=pop,
                 deterministic=deterministic,arch=arch,predict=True)

        states = env.reset()
        if self.version in set([4,5,104]):  # increase by 2
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

        while np.any(pop_num<self.n_pop):
            act, predstate = model.predict(states,deterministic=deterministic)
            newstate, rewards, dones, infos = env.step(act)
            for k in range(n_cpu):
                if pop_num[k]<self.n_pop: # do not save extras
                    if dones[k]:
                        self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],infos[k]['terminal_observation'],infos[k],debug=debug)
                        tqdm_e.update(1)
                        n+=n_add
                        tqdm_e.set_description("Pop " + str(n))
                        pop_num[k]=n
                    else:
                        self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],newstate[k],infos[k],debug=debug)
    
            states = newstate

        print('saving results...')

        self.episodestats.scale_sim()
        self.episodestats.save_sim(save)

        print('done')

        if False:
            return self.emp        
