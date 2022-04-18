'''

    lifecycle.py

    implements the lifecycle model that predicts how people will act in the presence of
    social security
    
    recent

'''

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
#from fin_benefits import Benefits
import matplotlib.pyplot as plt
from matplotlib import ticker
import gym_unemployment
import h5py
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm_notebook as tqdm
import os
from . episodestats import EpisodeStats
from . simstats import SimStats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['OMP_NUM_THREADS'] = '4'  # or any {'0', '1', '2'}
OMP_NUM_THREADS=4

import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# use stable baselines
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines import A2C, ACER, DQN, ACKTR, PPO2 #, TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from .vec_monitor import VecMonitor
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

            
# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 16],
                                                          vf=[512, 256, 128])],
                                           feature_extraction="mlp")
                                           #act_fun=tf.nn.relu)

class Lifecycle():

    def __init__(self,**kwargs):
#     env=None,minimal=False,timestep=0.25,ansiopvraha_kesto300=None,
#                     ansiopvraha_kesto400=None,karenssi_kesto=None,max_age=None,min_age=None,
#                     ansiopvraha_toe=None,perustulo=None,mortality=None,
#                     randomness=None,include_putki=None,preferencenoise=None,
#                     callback_minsteps=None,pinkslip=True,plotdebug=None,
#                     use_sigma_reduction=None,porrasta_putki=None,perustulomalli=None,
#                     porrasta_1askel=None,porrasta_2askel=None,porrasta_3askel=None,
#                     osittainen_perustulo=None,gamma=None,exploration=None,exploration_ratio=None,
#                     additional_income_tax=None,additional_tyel_premium=None,
#                     additional_kunnallisvero=None,additional_income_tax_high=None,additional_vat=None,
#                     extra_ppr=None,perustulo_korvaa_toimeentulotuen=None,
#                     year=2018,scale_tyel_accrual=None,preferencenoise_level=None,
#                     scale_additional_tyel_accrual=None,valtionverotaso=None,perustulo_asetettava=None,
#                     porrasta_toe=None,include_halftoe=None,include_ove=None,min_retirementage=None,
#                     max_retirementage=None,unemp_limit_reemp=None,lang=None,startage=None,silent=None):
        '''
        Alusta muuttujat
        '''

        self.initial_parameters()
        self.setup_parameters(**kwargs)

        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.n_age = self.max_age-self.min_age+1
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+2

        # alustetaan gym-environment
        if self.minimal:
            #if EK:
            #    self.environment='unemploymentEK-v0'
            #else:
            #    self.environment='unemployment-v0'

            self.minimal=True
            #self.version=0
            self.gym_kwargs={'step': self.timestep,
            'gamma':self.gamma,
                'min_age': self.min_age, 
                'max_age': self.max_age,
                'plotdebug': self.plotdebug, 
                'min_retirementage': self.min_retirementage, 
                'max_retirementage':self.max_retirementage,
                'reset_exploration_go': self.exploration,
                'reset_exploration_ratio': self.exploration_ratio,
                'perustulomalli': perustulomalli,
                'perustulo': perustulo,
                'osittainen_perustulo': self.osittainen_perustulo, 
                'perustulo_korvaa_toimeentulotuen': self.perustulo_korvaa_toimeentulotuen,
                'startage': self.startage,
                'year': self.year,
                'silent': self.silent}
        else:
            #if EK:
            #    self.environment='unemploymentEK-v1'
            #else:
            #    self.environment='unemployment-v1'

            self.minimal=False
            self.gym_kwargs={'step': self.timestep,
                'gamma':self.gamma,
                'min_age': self.min_age, 
                'max_age': self.max_age,
                'min_retirementage': self.min_retirementage, 
                'max_retirementage':self.max_retirementage,
                'ansiopvraha_kesto300': self.ansiopvraha_kesto300,
                'ansiopvraha_kesto400': self.ansiopvraha_kesto400,
                'ansiopvraha_toe': self.ansiopvraha_toe,
                'include_pinkslip':self.include_pinkslip,
                'perustulo': self.perustulo, 
                'karenssi_kesto': self.karenssi_kesto,
                'mortality': self.mortality, 
                'randomness': self.randomness,
                'porrasta_putki': self.porrasta_putki, 
                'porrasta_1askel': self.porrasta_1askel,
                'porrasta_2askel': self.porrasta_2askel, 
                'porrasta_3askel': self.porrasta_3askel,
                'include_putki': self.include_putki, 
                'use_sigma_reduction': self.use_sigma_reduction,
                'plotdebug': self.plotdebug, 
                'include_preferencenoise': self.include_preferencenoise,
                'preferencenoise_level': self.preferencenoise_level,
                'perustulomalli': self.perustulomalli, 
                'osittainen_perustulo':self.osittainen_perustulo,
                'reset_exploration_go': self.exploration,
                'reset_exploration_ratio': self.exploration_ratio,
                'additional_income_tax': self.additional_income_tax,
                'additional_tyel_premium': self.additional_tyel_premium, 
                'additional_income_tax_high': self.additional_income_tax_high,
                'additional_kunnallisvero': self.additional_kunnallisvero,
                'additional_vat': self.additional_vat,
                'scale_tyel_accrual': self.scale_tyel_accrual,
                'scale_additional_tyel_accrual': self.scale_additional_tyel_accrual,
                'valtionverotaso': self.valtionverotaso,
                'perustulo_asetettava': self.perustulo_asetettava, 
                'include_halftoe': self.include_halftoe,
                'porrasta_toe': self.porrasta_toe,
                'include_ove': self.include_ove,
                'unemp_limit_reemp': self.unemp_limit_reemp,
                'extra_ppr': self.extra_ppr,
                'startage': self.startage,
                'year': self.year,
                'silent': self.silent}
                
        # Create log dir & results dirs
        self.log_dir = "tmp/" # +str(env_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tenb_dir = "tmp/tenb/" # +str(env_id)
        os.makedirs(self.tenb_dir, exist_ok=True)
        os.makedirs("saved/", exist_ok=True)
        os.makedirs("results/", exist_ok=True)
        os.makedirs("best/", exist_ok=True)
        
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
        self.n_employment,self.n_acts=self.env.get_n_states()
        self.version = self.env.get_lc_version()

        self.episodestats=SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                   self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
                                   version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma,
                                   lang=self.lang)
                                   
    def initial_parameters(self):
        self.min_age = 18
        self.max_age = 70
        self.figformat='pdf'
        self.minimal=False
        self.timestep=0.25
        self.year=2018
        
        self.lang='Finnish'
        
        if self.minimal:
            self.min_retirementage=63.5
        else:
            self.min_retirementage=63
        
        self.max_retirementage=68
        self.n_pop = 1000
        self.callback_minsteps = 1_000

        # apumuuttujia
        self.gamma = 0.92
        self.karenssi_kesto=0.25
        self.plotdebug=False
        self.include_pinkslip=True
        self.mortality=False
        self.perustulo=False
        self.perustulo_korvaa_toimeentulotuen=False
        self.perustulomalli=None
        self.ansiopvraha_kesto300=None
        self.ansiopvraha_kesto400=None
        self.include_putki=None
        self.ansiopvraha_toe=None
        self.environment='unemployment-v0'
        self.include_preferencenoise=False
        self.preferencenoise_level=0
        self.porrasta_putki=True
        self.porrasta_1askel=True
        self.porrasta_2askel=True
        self.porrasta_3askel=True
        self.osittainen_perustulo=True
        self.additional_income_tax=0.0
        self.additional_tyel_premium=0.0
        self.additional_kunnallisvero=0.0
        self.additional_income_tax_high=0.0
        self.additional_vat=0.0
        self.scale_tyel_accrual=False
        self.scale_additional_tyel_accrual=0
        self.perustulo_asetettava=None
        self.valtionverotaso=None
        self.include_halftoe=None
        self.porrasta_toe=None
        self.include_ove=False
        self.unemp_limit_reemp=False
        self.extra_ppr=0
        self.startage=self.min_age
        self.use_sigma_reduction=True
        self.silent=False
        self.exploration=False
        self.exploration_ratio=0.10
        self.randomness=True
        
    def setup_parameters(self,**kwargs):
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs

        for key, value in kwarg.items():
            if key=='callback_minsteps':
                if value is not None:
                    self.callback_minsteps=value
            elif key=='extra_ppr':
                if value is not None:
                    self.extra_ppr=value
            elif key=='karenssi_kesto':
                if value is not None:
                    self.karenssi_kesto=value
            elif key=='additional_income_tax':
                if value is not None:
                    self.additional_income_tax=value
            elif key=='additional_tyel_premium':
                if value is not None:
                    self.additional_tyel_premium=value
            elif key=='additional_kunnallisvero':
                if value is not None:
                    self.additional_kunnallisvero=value
            elif key=='additional_income_tax_high':
                if value is not None:
                    self.additional_income_tax_high=value
            elif key=='additional_tyel_premium':
                if value is not None:
                    self.additional_tyel_premium=value
            elif key=='additional_vat':
                if value is not None:
                    self.additional_vat=value
            elif key=='valtionverotaso':
                if value is not None:
                    self.valtionverotaso=value
            elif key=='perustulo_asetettava':
                if value is not None:
                    self.perustulo_asetettava=value
            elif key=='perustulomalli':
                if value is not None:
                    self.perustulomalli=value
            elif key=='perustulo':
                if value is not None:
                    self.perustulo=value
            elif key=='perustulo_korvaa_toimeentulotuen':
                if value is not None:
                    self.perustulo_korvaa_toimeentulotuen=value
            elif key=='osittainen_perustulo':
                if value is not None:
                    self.osittainen_perustulo=value
            elif key=='plotdebug':
                if value is not None:
                    self.plotdebug=value
            elif key=='pinkslip':
                if value is not None:
                    self.pinkslip=value
            elif key=='mortality':
                if value is not None:
                    self.mortality=value
            elif key=='randomness':
                if value is not None:
                    self.randomness=value
            elif key=='ansiopvraha_kesto300':
                if value is not None:
                    self.ansiopvraha_kesto300=value
            elif key=='ansiopvraha_kesto400':
                if value is not None:
                    self.ansiopvraha_kesto400=value
            elif key=='ansiopvraha_kesto500':
                if value is not None:
                    self.ansiopvraha_kesto500=value
            elif key=='exploration':
                if value is not None:
                    self.exploration=value
            elif key=='exploration_ratio':
                if value is not None:
                    self.exploration_ratio=value
            elif key=='ansiopvraha_toe':
                if value is not None:
                    self.ansiopvraha_toe=value
            elif key=='porrasta_putki':
                if value is not None:
                    self.porrasta_putki=value
            elif key=='porrasta_1askel':
                if value is not None:
                    self.porrasta_1askel=value
            elif key=='porrasta_2askel':
                if value is not None:
                    self.porrasta_2askel=value
            elif key=='porrasta_3askel':
                if value is not None:
                    self.porrasta_3askel=value
            elif key=='preferencenoise':
                if value is not None:
                    self.preferencenoise=value
            elif key=='preferencenoise_level':
                if value is not None:
                    self.preferencenoise_level=value
            elif key=='scale_tyel_accrual':
                if value is not None:
                    self.scale_tyel_accrual=value
            elif key=='scale_additional_tyel_accrual':
                if value is not None:
                    self.scale_additional_tyel_accrual=value
            elif key=='gamma':
                if value is not None:
                    self.gamma=value
            elif key=='lang':
                if value is not None:
                    self.lang=value
            elif key=='min_retirementage':
                if value is not None:
                    self.min_retirementage=value
            elif key=='max_retirementage':
                if value is not None:
                    self.max_retirementage=value
            elif key=='env':
                if value is not None:
                    self.environment=value
            elif key=='use_sigma_reduction':
                if value is not None:
                    self.use_sigma_reduction=value
            elif key=='include_halftoe':
                if value is not None:
                    self.include_halftoe=value
            elif key=='porrasta_toe':
                if value is not None:
                    self.porrasta_toe=value
            elif key=='include_ove':
                if value is not None:
                    self.include_ove=value
            elif key=='unemp_limit_reemp':
                if value is not None:
                    self.unemp_limit_reemp=value
            elif key=='startage':
                if value is not None:
                    self.startage=value
            elif key=='min_age':
                if value is not None:
                    self.min_age=value
            elif key=='max_age':
                if value is not None:
                    self.max_age=value
            elif key=='silent':
                if value is not None:
                    self.silent=value
            
    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        self.env.explain()
        #print('Parameters of lifecycle:\ntimestep {}\ngamma {} per anno\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.min_age,self.max_age,self.min_retirementage))
        #print('max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_toe {}'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_toe))
        #print('perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}'.format(self.perustulo,self.karenssi_kesto,self.mortality,self.randomness))
        #print('include_putki {}\ninclude_pinkslip {}\n'.format(self.include_putki,self.include_pinkslip))

    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)            

    def map_t(self,t,start_zero=False):
        return self.min_age+t*self.timestep

    def get_initial_reward(self,startage=None):
        return self.episodestats.get_initial_reward(startage=startage)

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
            n_cpu = 8 # 12 # 20
        elif rlmodel=='acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 512, 256]) 
            n_cpu = 8 # 12 # 20
        elif  rlmodel=='custom_acktr': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu,net_arch=[dict(pi=[32, 32, 32],vf=[128, 128, 128])]) 
            if predict:
                n_cpu = 16
            else:
                n_cpu = 8 # 12 # 20
        elif rlmodel=='leaky_acktr': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[256, 256, 16]) 
            if predict:
                n_cpu = 16
            else:
                n_cpu = 8 # 12 # 20
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
                      cont,max_grad_norm=None,tensorboard=False,verbose=1,n_cpu=1,learning_schedule='linear',
                      vf=None,gae_lambda=0.9):
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
        scaled_learning_rate=learning_rate*batch
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

    def make_env(self,env_id, rank, kwargs, seed=None, use_monitor=True):
        """
        Utility function for multiprocessed env.#

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id,kwargs=kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.env_seed(seed + rank + 100)

            # monitor enables various things, not used by default
            #print('monitor=',use_monitor)
            #if use_monitor:
            #    env = Monitor(env, self.log_dir, allow_early_resets=True)

            return env

        if seed is not None:
            set_global_seeds(seed)

        return _init()

    def callback(self,_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        #global n_steps, best_mean_reward
        # Print stats every 1000 calls, if needed

        min_steps=0
        mod_steps=1
        if self.callback_minsteps is not None:
            hist_eps=self.callback_minsteps
        else:
            hist_eps=5000
        if (self.n_steps + 1) % mod_steps == 0 and self.n_steps > min_steps:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) >= hist_eps:
                mean_reward = np.mean(y[-hist_eps:])
                print(x[-1], 'timesteps', len(y), 'episodes') #, 'mean', mean_reward, 'out of', y[-hist_eps:])

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    print("NEW Best mean reward: {:.2f} - Last best reward per episode: {:.2f}".format(mean_reward,self.best_mean_reward))
                    self.best_mean_reward = mean_reward
                    # Example for saving best model                    print("Saving new best model")
                    print('saved as ',self.bestname)
                    _locals['self'].save(self.bestname)
                else:
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
    
        self.n_steps += 1

        return True

    def train(self,train=False,debug=False,steps=20000,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=None,learning_rate=0.25,
                start_from=None,max_n_cpu=1000,plot=True,use_vecmonitor=False,
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
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=False) for i in range(n_cpu)], start_method='spawn')
                env = VecMonitor(env,filename=self.log_dir+'monitor.csv')
            else:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='spawn')
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

    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()

    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f.get(nimi).value
        f.close()
        return val
        
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
            #env.seed(4567)
            #env.env_seed(4567)
        else:
            env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs) for i in range(n_cpu)])
            #env = SubprocVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])
            #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])

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

    def simulate(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,startage=None,
                 deterministic=False,save='results/testsimulate',arch=None):

        model,env,n_cpu=self.setup_model(debug=debug,rlmodel=rlmodel,plot=plot,load=load,pop=pop,
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
            act, predstate = model.predict(states,deterministic=deterministic)
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
   
    def get_reward(self):
        return self.episodestats.get_reward()
   
    def render(self,load=None,figname=None,grayscale=False):
        if load is not None:
            self.episodestats.render(load=load,figname=figname,grayscale=grayscale)
        else:
            self.episodestats.render(figname=figname,grayscale=grayscale)
   
    def render_reward(self,load=None,figname=None):
        if load is not None:
            self.episodestats.load_sim(load)
            return self.episodestats.comp_total_reward()
        else:
            return self.episodestats.comp_total_reward()
   
    def render_laffer(self,load=None,figname=None,include_retwork=True,grouped=False,g=0):
        if load is not None:
            self.episodestats.load_sim(load)
            
        rew=self.episodestats.comp_total_reward(output=False)
            
        q=self.episodestats.comp_budget()
        if include_retwork:
            tyotulosumma=q['tyotulosumma']
        else:
            tyotulosumma=q['tyotulosumma_eielakkeella']
        
        if grouped:
            q2=self.episodestats.comp_participants(include_retwork=include_retwork,grouped=True,g=g)
            #kokotyossa,osatyossa=self.episodestats.comp_parttime_aggregate(grouped=grouped,g=g)
            htv=q2['htv']
            kokotyossa,osatyossa,tyot=self.episodestats.comp_employment_groupstats(g=g,include_retwork=include_retwork,grouped=True)
            palkansaajia=kokotyossa+osatyossa
            kokotyossa=kokotyossa/palkansaajia
            osatyossa=osatyossa/palkansaajia
        else:
            q2=self.episodestats.comp_participants(include_retwork=include_retwork,grouped=False)
            #kokotyossa,osatyossa=self.episodestats.comp_parttime_aggregate(grouped=False)
            htv=q2['htv']
            kokotyossa,osatyossa,tyot=self.episodestats.comp_employment_groupstats(grouped=False,include_retwork=include_retwork)
            palkansaajia=kokotyossa+osatyossa
            kokotyossa=kokotyossa/palkansaajia
            osatyossa=osatyossa/palkansaajia
        
        muut_tulot=q['muut tulot']
        tC=0.2*max(0,q['tyotulosumma']-q['verot+maksut'])
        if grouped:
            kiila,qc=self.episodestats.comp_verokiila()
            kiila2,qcb=self.episodestats.comp_verokiila_kaikki_ansiot()
        else:
            kiila,qc=self.episodestats.comp_verokiila()
            kiila2,qcb=self.episodestats.comp_verokiila_kaikki_ansiot()
            
        tyollaste,_=self.episodestats.comp_employed_aggregate(grouped=grouped,g=g)
        tyotaste=self.episodestats.comp_unemployed_aggregate(grouped=grouped,g=g)
        menot={}
        menot['etuusmeno']=q['etuusmeno']
        menot['tyottomyysmenot']=q['ansiopvraha']
        menot['kokoelake']=q['kokoelake']
        menot['asumistuki']=q['asumistuki']
        menot['toimeentulotuki']=q['toimeentulotuki']
        menot['muutmenot']=q['opintotuki']+q['isyyspaivaraha']+q['aitiyspaivaraha']+q['sairauspaivaraha']+q['perustulo']+q['kotihoidontuki']
        
        #print(tyollaste,tyotaste)
        #tyollaste,tyotaste=0,0
        #
        #qq={}
        #qq['tI']=q['verot+maksut']/q['tyotulosumma']
        #qq['tC']=tC/q['tyotulosumma']
        #qq['tP']=q['ta_maksut']/q['tyotulosumma']
        #
        #print(qq,qc)
            
        return rew,tyotulosumma,q['verot+maksut'],htv,muut_tulot,kiila,tyollaste,tyotaste,palkansaajia,osatyossa,kiila2,menot
   
    def load_sim(self,load=None):
        self.episodestats.load_sim(load)
   
    def run_optimize_x(self,target,results,n,startn=0,averaged=True):
        self.episodestats.run_optimize_x(target,results,n,startn=startn,averaged=averaged)
   
    def run_dummy(self,strategy='emp',debug=False,pop=None):
        '''
        Lasketaan työllisyysasteet ikäluokittain satunnaisella politiikalla
        hyödyllinen unemp-envin profilointiin
        '''
        
        self.n_pop=pop

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)
        if pop is not None:
            self.n_pop=pop

        print('simulating...')
        tqdm_e = tqdm(range(int(self.n_pop)), desc='Population', leave=True, unit=" p")
        
        self.env.seed(1234) 
        state = self.env.reset()
        age=self.min_age
        p=np.zeros(self.n_pop)
        n_cpu=1
        n=0
        pop_num=np.array([k for k in range(n_cpu)])
        v=2 # versio
        while n<self.n_pop:
            #print('agent {}'.format(n))
            tqdm_e.update(1)
            tqdm_e.set_description("Pop " + str(n))
            
            done=False
            pop_num[0]=n
            
            while not done:
                if v==1:
                    emp,_,_,_,age=self.env.state_decode(state) # current employment state
                elif v==3:
                    emp,_,_,_,age,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
                else:
                    emp,_,_,_,age,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state            

                if strategy=='random':
                    act=np.random.randint(3)
                elif strategy=='emp':
                    if age>=self.min_retirementage:
                        if emp!=2:
                            act=2
                        else:
                            act=0
                    else:
                        if emp==0:
                            act=1
                        else:
                            act=0
                elif strategy=='unemp':
                    if age>=self.retirement_age:
                        if emp!=2:
                            act=2
                        else:
                            act=0
                    else:
                        if emp==1:
                            act=1
                        else:
                            act=0
                elif strategy=='alt':
                    if age>=self.min_retirementage:
                        if emp!=2:
                            act=2
                        else:
                            act=0
                    else:
                        act=1

                newstate,r,done,_=self.env.step(act)
                
                if done:
                    n=n+1
                    if debug:
                        print('done')
                    newstate=self.env.reset()
                    break
                else:
                    self.episodestats.add(pop_num[0],act,r,state,newstate,debug=debug)
                    
                state=newstate

        #self.plot_stats(5)
        #self.plot_stats()
        #self.plot_reward()

    def compare_with(self,cc2,label1='vaihtoehto',label2='perus',figname=None,grayscale=False,dash=True):
        '''
        compare_with

        compare results obtained another model
        '''
        self.episodestats.compare_with(cc2.episodestats,label1=label1,label2=label2,grayscale=grayscale,figname=figname,dash=dash)

    def run_results(self,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',twostage=False,
               save='saved/perusmalli',debug=False,simut='simut',results='results/simut_res',
               stats='results/simut_stats',deterministic=True,train=True,predict=True,
               batch1=1,batch2=100,cont=False,start_from=None,callback_minsteps=None,
               verbose=1,max_grad_norm=None,learning_rate=0.25,log_interval=10,
               learning_schedule='linear',vf=None,arch=None,gae_lambda=None,plot=None,
               startage=None):
   
        '''
        run_results

        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
        
        if self.plotdebug:
            debug=True
   
        self.n_pop=pop
        if callback_minsteps is not None:
            self.callback_minsteps=callback_minsteps

        if train: 
            print('train...')
            if cont:
                self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,verbose=verbose,
                                  debug=debug,save=save,batch1=batch1,batch2=batch2,
                                  cont=cont,start_from=start_from,twostage=twostage,
                                  max_grad_norm=max_grad_norm,learning_rate=learning_rate,log_interval=log_interval,
                                  learning_schedule=learning_schedule,vf=vf,arch=arch,gae_lambda=gae_lambda)
            else:
                self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,verbose=verbose,
                                 debug=debug,batch1=batch1,batch2=batch2,cont=cont,
                                 save=save,twostage=twostage,
                                 max_grad_norm=max_grad_norm,learning_rate=learning_rate,log_interval=log_interval,
                                 learning_schedule=learning_schedule,vf=vf,arch=arch,gae_lambda=gae_lambda)
        if predict:
            #print('predict...')
            self.predict_protocol(pop=pop,rlmodel=rlmodel,load=save,startage=startage,
                          debug=debug,deterministic=deterministic,results=results,arch=arch)
          
    def run_protocol(self,steps1=2_000_000,steps2=1_000_000,rlmodel='acktr',
               debug=False,batch1=1,batch2=1000,cont=False,twostage=False,log_interval=10,
               start_from=None,save='best3',verbose=1,max_grad_norm=None,
               learning_rate=0.25,learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
        '''
        run_protocol

        train RL model in two steps:
        1. train with a short batch, not saving the best model
        2. train with a long batch, save the best model during the training
        '''
        
        if twostage:
            tmpname='tmp/simut_100'
        else:
            tmpname=save
            
        if steps1>0:
            print('phase 1')
            if cont:
                self.train(steps=steps1,cont=cont,rlmodel=rlmodel,save=tmpname,batch=batch1,debug=debug,
                           start_from=start_from,use_callback=False,use_vecmonitor=False,
                           log_interval=log_interval,verbose=1,vf=vf,arch=arch,gae_lambda=gae_lambda,
                           max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule)
            else:
                self.train(steps=steps1,cont=False,rlmodel=rlmodel,save=tmpname,batch=batch1,debug=debug,vf=vf,arch=arch,
                           use_callback=False,use_vecmonitor=False,log_interval=log_interval,verbose=1,gae_lambda=gae_lambda,
                           max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule)

        if twostage and steps2>0:
            print('phase 2')
            self.train(steps=steps2,cont=True,rlmodel=rlmodel,save=tmpname,
                       debug=debug,start_from=tmpname,batch=batch2,verbose=verbose,
                       use_callback=False,use_vecmonitor=False,log_interval=log_interval,bestname=save,plotdebug=plotdebug,
                       max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule)

    def predict_protocol(self,pop=1_00,rlmodel='acktr',results='results/simut_res',arch=None,
                         load='saved/malli',debug=False,deterministic=False,startage=None):
        '''
        predict_protocol

        simulate the three models obtained from run_protocol
        '''
 
        # simulate the saved best
        self.simulate(pop=pop,rlmodel=rlmodel,plot=False,debug=debug,arch=arch,
                      load=load,save=results,deterministic=deterministic,startage=startage)

    def run_distrib(self,n=5,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',
               save='saved/distrib_base_',debug=False,simut='simut',results='results/distrib_',
               deterministic=True,train=True,predict=True,batch1=1,batch2=100,cont=False,
               start_from=None,plot=False,twostage=False,callback_minsteps=None,
               stats_results='results/distrib_stats',startn=None,verbose=1,
               learning_rate=0.25,learning_schedule='linear',log_interval=100):
   
        '''
        run_verify

        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
   
        if startn is None:
            startn=0
    
        self.n_pop=pop

        # repeat simulation n times
        for num in range(startn,n):
            bestname2=save+'_'+str(100+num)
            results2=results+'_'+str(100+num)
            print('computing {}'.format(num))
        
            self.run_results(steps1=steps1,steps2=steps2,pop=pop,rlmodel=rlmodel,
               twostage=twostage,save=bestname2,debug=debug,simut=simut,results=results2,
               deterministic=deterministic,train=train,predict=predict,
               batch1=batch1,batch2=batch2,cont=cont,start_from=start_from,plot=False,
               callback_minsteps=callback_minsteps,verbose=verbose,learning_rate=learning_rate,
               learning_schedule=learning_schedule,log_interval=log_interval)

        #self.render_distrib(load=results,n=n,stats_results=stats_results)
            
    def comp_distribs(self,load=None,n=1,startn=0,stats_results='results/distrib_stats',singlefile=False):
        if load is None:
            return
            
        self.episodestats.run_simstats(load,stats_results,n,startn=startn,singlefile=singlefile)

    def render_distrib(self,stats_results='results/distrib_stats',plot=False,figname=None):
        self.episodestats.plot_simstats(stats_results,figname=figname)

        # gather results ...
        #if plot:
        #    print('plot')
            
    def compare_distrib(self,filename1,filename2,n=1,label1='perus',label2='vaihtoehto',figname=None):
        self.episodestats.compare_simstats(filename1,filename2,label1=label1,label2=label2,figname=figname)

    def plot_rewdist(self,age=20,sum=False,all=False):
        t=self.map_age(age)
        self.episodestats.plot_rewdist(t=t,sum=sum,all=all)

    def plot_saldist(self,age=20,sum=False,all=False,n=10):
        t=self.map_age(age)
        self.episodestats.plot_saldist(t=t,sum=sum,all=all,n=n)

    def plot_RL_act(self,t,rlmodel='acktr',load='perus',debug=True,deterministic=True,
                        n_palkka=80,n_emppalkka=80,deltapalkka=1000,deltaemppalkka=1000,n_elake=40,deltaelake=1500,
                        hila_palkka0=0,min_pension=0,deltapalkka_old=None,deltaemppalkka_old=None):
        model,env,n_cpu=self.setup_model(rlmodel=rlmodel,load=load,debug=debug)
        RL_unemp=self.RL_simulate_V(model,env,t,emp=0,deterministic=deterministic,n_palkka=n_palkka,n_emppalkka=n_emppalkka,
                        deltapalkka=deltapalkka,deltaemppalkka=deltaemppalkka,n_elake=n_elake,deltaelake=deltaelake,
                        min_wage=min_wage,min_pension=min_pension,deltapalkka_old=deltapalkka_old,deltaemppalkka_old=deltaemppalkka_old)
        RL_emp=self.RL_simulate_V(model,env,t,emp=1,deterministic=deterministic,n_palkka=n_palkka,n_emppalkka=n_emppalkka,
                        deltapalkka=deltapalkka,deltaemppalkka=deltaemppalkka,n_elake=n_elake,deltaelake=deltaelake,
                        min_wage=min_wage,min_pension=min_pension,deltapalkka_old=deltapalkka_old,deltaemppalkka_old=deltaemppalkka_old)
        self.plot_img(RL_emp,xlabel="Pension",ylabel="Wage",title='Töissä')
        self.plot_img(RL_unemp,xlabel="Pension",ylabel="Wage",title='Työttömänä')
        self.plot_img(RL_emp-RL_unemp,xlabel="Pension",ylabel="Wage",title='Työssä-Työtön')

    def get_RL_act(self,t,emp=0,time_in_state=0,rlmodel='acktr',load='perus',debug=True,deterministic=True,
                        n_palkka=80,n_emppalkka=80,deltapalkka=1000,deltaemppalkka=1000,n_elake=40,deltaelake=1500,
                        min_wage=1000,min_pension=0,deltapalkka_old=None,deltaemppalkka_old=None):
        model,env,n_cpu=self.setup_model(rlmodel=rlmodel,load=load,debug=debug)
        return self.RL_simulate_V(model,env,t,emp=emp,deterministic=deterministic,time_in_state=time_in_state,
                        n_palkka=n_palkka,n_emppalkka=n_emppalkka,deltapalkka=deltapalkka,deltaemppalkka=deltaemppalkka,
                        n_elake=n_elake,deltaelake=deltaelake,min_wage=min_wage,min_pension=min_pension,
                        deltapalkka_old=deltapalkka_old,deltaemppalkka_old=deltaemppalkka_old)

    def get_rl_act(self,t,emp=0,time_in_state=0,rlmodel='acktr',load='perus',debug=True,deterministic=True):
        model,env,n_cpu=self.setup_model(rlmodel=rlmodel,load=load,debug=debug)
        return self.RL_pred_act(model,env,t,emp=emp,deterministic=deterministic,time_in_state=time_in_state)

    def plot_img(self,img,xlabel="Eläke",ylabel="Palkka",title="Employed",
                    vmin=None,vmax=None,figname=None):
        fig, ax = plt.subplots()
        #im = ax.imshow(img,interpolation='none',aspect='equal',origin='lower',resample=False,alpha=0)
        #img = ax.pcolor(img)
        if vmin is not None:
            heatmap = ax.pcolor(img,vmax=vmax,cmap='gray')
        else:
            heatmap = ax.pcolor(img) 
        fig.colorbar(heatmap,ax=ax)        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_img_old(self,img,xlabel="Eläke",ylabel="Palkka",title="Employed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img,origin='lower')
        heatmap = plt.pcolor(img) 
        plt.colorbar(heatmap)        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()
        
    def plot_twoimg(self,img1,img2,xlabel="Pension",ylabel1="Wage",ylabel2="",title1="Employed",title2="Employed",
                    vmin=None,vmax=None,figname=None,show_results=True,alpha=None):
        fig, axs = plt.subplots(ncols=2)
        if alpha is None:
            alpha=1.0
        if vmin is not None:
            im0 = axs[0].pcolor(img1,vmin=vmin,vmax=vmax,cmap='gray',alpha=alpha,linewidth=0,rasterized=True)
        else:
            im0 = axs[0].pcolor(img1,alpha=alpha,linewidth=0,rasterized=True)
        #im0.set_edgecolor('face')
        #if vmin is not None:
        #    im0 = axs[0].imshow(img1,vmin=vmin,vmax=vmax,cmap='gray')
        #else:
        #    im0 = axs[0].imshow(img1)
        #heatmap = plt.pcolor(img1) 
        #kwargs = {'format': '%.1f'}
        kwargs = {'format': '%.0f'}
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="20%", pad=0.05)
        cbar0 = plt.colorbar(im0, cax=cax0, **kwargs)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel1)
        axs[0].set_title(title1)
        #axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        tick_locator = ticker.MaxNLocator(nbins=3,integer=True)
        cbar0.locator = tick_locator
        cbar0.update_ticks()        
        if vmin is not None:
            im1 = axs[1].pcolor(img2,vmin=vmin,vmax=vmax,cmap='gray',alpha=alpha,linewidth=0,rasterized=True)
        else:
            im1 = axs[1].pcolor(img2,alpha=alpha,linewidth=0,rasterized=True)
        #im1.set_edgecolor('face')
        #if vmin is not None:
        #    im1 = axs[1].imshow(img2,vmin=vmin,vmax=vmax,cmap='gray')
        #else:
        #    im1 = axs[1].imshow(img2)
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="20%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1,  **kwargs)
        #heatmap = plt.pcolor(img2) 
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel2)
        axs[1].set_title(title2)
        #axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        tick_locator = ticker.MaxNLocator(nbins=3,integer=True)
        cbar1.locator = tick_locator
        cbar1.update_ticks()        
        #plt.colorbar(heatmap)        
        plt.subplots_adjust(wspace=0.3)
        if figname is not None:
            plt.savefig(figname+'.'+self.figformat, format=self.figformat)

        if show_results:
            plt.show()
        else:
            return fig,axs
        
    def filter_act(self,act,state):
        employment_status,pension,old_wage,age,time_in_state,next_wage=self.env.state_decode(state)
        if age<self.min_retirementage:
            if act==2:
                act=0
        
        return act

    def RL_pred_act(self,model,env,age,emp=0,elake=0,time_in_state=0,vanhapalkka=0,palkka=0,deterministic=True):
        # dynaamisen ohjelmoinnin parametrejä
    
        if emp==2:
            elake=max(780*12,elake)
            state=self.env.state_encode(emp,elake,0,age,time_in_state,0)
        else:
            state=self.env.state_encode(emp,elake,vanhapalkka,age,time_in_state,palkka)

        act, predstate = model.predict(state,deterministic=deterministic)
        act=self.filter_act(act,state)
                 
        return act

    def RL_simulate_V(self,model,env,age,emp=0,time_in_state=0,deterministic=True,
                        n_palkka=80,n_emppalkka=80,deltapalkka=1000,deltaemppalkka=1000,n_elake=40,deltaelake=1500,
                        min_wage=0,min_pension=0,deltapalkka_old=None,deltaemppalkka_old=None):
        # dynaamisen ohjelmoinnin parametrejä
        def map_elake(v,emp=1):
            return min_pension+deltaelake*v 

        def map_palkka_old(v,emp):
            if emp==0:
                return min_wage+max(0,deltapalkka_old*v)
            elif emp==1:
                return min_wage+max(0,deltaemppalkka_old*v) 

        def map_palkka(v,emp=1):
            if emp==0:
                return min_wage+max(0,deltapalkka*v) 
            elif emp==1:
                return min_wage+max(0,deltaemppalkka*v) 
    
        prev=0
        toe=0
        if emp==0:
            fake_act=np.zeros((n_palkka,n_elake))
            for el in range(n_elake):
                for p in range(n_palkka): 
                    palkka=map_palkka(p,emp=emp)
                    old_palkka=map_palkka_old(p,emp=emp)
                    elake=map_elake(el)
                    #if emp==2:
                    #    elake=max(780*12,elake)
                    #    state=self.env.state_encode(emp,elake,0,age,time_in_state,0)
                    #else:
                    state=self.env.state_encode(emp,elake,old_palkka,age,time_in_state,palkka)

                    act, predstate = model.predict(state,deterministic=deterministic)
                    act=self.filter_act(act,state)
                    fake_act[p,el]=act
        else: # emp = 1
            fake_act=np.zeros((n_emppalkka,n_elake))
            for el in range(n_elake):
                for p in range(n_emppalkka): 
                    palkka=map_palkka(p,emp=emp)
                    elake=map_elake(el)
                    old_palkka=map_palkka_old(p,emp=emp)
                    if emp==2:
                        elake=max(780*12,elake)
                        state=self.env.state_encode(emp,elake,0,age,time_in_state,0)
                    else:
                        state=self.env.state_encode(emp,elake,old_palkka,age,time_in_state,palkka)

                    act, predstate = model.predict(state,deterministic=deterministic)
                    act=self.filter_act(act,state)
                    fake_act[p,el]=act
        
                 
        return fake_act
        
    def L2error(self):
        '''
        Laskee L2-virheen havaittuun työllisyysasteen L2-virhe/vuosi tasossa
        Käytetään optimoinnissa
        '''
        L2=self.episodestats.comp_L2error()
        
        return L2
        
    def L2BudgetError(self,ref_muut,scale=1e10):
        '''
        Laskee L2-virheen budjettineutraaliuteen
        Käytetään optimoinnissa
        '''
        L2=self.episodestats.comp_budgetL2error(ref_muut,scale=scale)
        
        return L2
        
    def optimize_scale(self,target,averaged=False):
        '''
        Optimoi utiliteetin skaalaus
        Vertailupaperia varten
        '''
        
        res=self.episodestats.optimize_scale(target,averaged=averaged)
        
        print(res)
        
    def comp_aggkannusteet(self,n=None,savefile=None):
        self.episodestats.comp_aggkannusteet(self.env.ben,n=n,savefile=savefile)
        
    def plot_aggkannusteet(self,loadfile,baseloadfile=None,figname=None,label=None,baselabel=None):
        self.episodestats.plot_aggkannusteet(self.env.ben,loadfile,baseloadfile=baseloadfile,figname=figname,
                                             label=label,baselabel=baselabel)
        
    def comp_taxratios(self,grouped=True):
        return self.episodestats.comp_taxratios(grouped=grouped)
        
    def comp_verokiila(self,grouped=True):
        return self.episodestats.comp_verokiila(grouped=grouped)
    