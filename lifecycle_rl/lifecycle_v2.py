'''

    lifecycle.py

    implements the lifecycle model that predicts how people will act in the presence of
    social security
    
    uses stable baselines 2 instead of stable baselines
    - can use tianshou in principle (NOT WORKING)
    - can use stable baselines 3 (NOT TESTED)
    - can use self implemented algorithms
    
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
#import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm_notebook as tqdm
import os
from . episodestats import EpisodeStats
from . plotstats import PlotStats
from . simstats import SimStats

import profile
import re


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['OMP_NUM_THREADS'] = '4'  # or any {'0', '1', '2'}
OMP_NUM_THREADS=4

import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# use stable baselines

# 0 sb2, 1 standalone, 2 tianshou, 3 stable baselines 3
class Lifecycle():

    def __init__(self,**kwargs):
        '''
        Alusta muuttujat
        '''
        self.initial_parameters()
        self.setup_parameters(**kwargs)

        # alustetaan gym-environment
        if self.minimal:
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
                'random_returns': self.random_returns,
                'r_mean': self.r_mean,
                'r_std': self.r_std,
                'investment': self.include_investment,
                'wealth': self.include_wealth,
                'use_utility': self.use_utility,
                'CRRA_eta': self.CRRA_eta,
                'silent': self.silent,
                'include_emtr': self.include_emtr
                }
        else:
            self.gym_kwargs={'step': self.timestep,
                'gamma':self.gamma,
                'random_returns': self.random_returns,
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
                'investment': self.include_investment,
                'wealth': self.include_wealth,
                'year': self.year,
                'r_mean': self.r_mean,
                'r_std': self.r_std,
                'use_utility': self.use_utility,
                'CRRA_eta': self.CRRA_eta,
                'silent': self.silent,
                'include_emtr': self.include_emtr
                }
                
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
        self.minimal = self.env.get_minimal()

        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.n_age = self.max_age-self.min_age+1
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+1

        if self.version in set([4,5,104]):
            self.min_retirementage=self.env.get_retirementage()

        self.episodestats=SimStats(self.timestep,self.n_time,self.n_employment,1, #self.n_pop,
                                   self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
                                   version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma,
                                   silent=self.silent)
        
        self.plotstats=PlotStats(self.episodestats,self.timestep,self.n_time,self.n_employment,self.n_pop,
                                   self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
                                   version=self.version,params=self.gym_kwargs,year=self.year,gamma=self.gamma,
                                   lang=self.lang)
                                   
        self.setup_solver(self.library)
        
                                   
    def setup_solver(self,use_solver):
        self.use_tianshou=False
        self.use_standalone=False
        self.use_sb3=False
        self.use_sb2=False
        if use_solver==2: # not working
            from .runner_tianshou import runner_tianshou
            self.use_tianshou=True
        elif use_solver==3: # does not implement ACKTR
            from .runner_stablebaselines3 import runner_stablebaselines3
            self.use_sb3=True
        elif use_solver==1: # A2C with GRU, fast
            from .runner_standalone_a3c import runner_standalone
            self.use_standalone=True
        else:
            from .runner_stablebaselines2_v2 import runner_stablebaselines2 # newer, roughly 13 p/s, allows larger simulations
            #from .runner_stablebaselines2 import runner_stablebaselines2 # older, roughly 11 p/s
            self.use_sb2=True

        if self.use_tianshou:
            self.runner=runner_tianshou(self.environment,self.gamma,self.timestep,self.n_time,self.n_pop,
                 self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year,self.episodestats,self.gym_kwargs,self.version)
        elif self.use_standalone:
            print('pop',self.n_pop)
            self.runner=runner_standalone(self.environment,self.gamma,self.timestep,self.n_time,self.n_pop,
                 self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year,self.episodestats,self.gym_kwargs,self.version,self.processes)
        elif self.use_sb3:
            self.runner=runner_stablebaselines3(self.environment,self.gamma,self.timestep,self.n_time,self.n_pop,
                 self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year,self.episodestats,self.gym_kwargs,self.version)
        else:
            self.runner=runner_stablebaselines2(self.environment,self.gamma,self.timestep,self.n_time,self.n_pop,
                 self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year,self.episodestats,self.gym_kwargs,self.version,self.processes)
        

    def initial_parameters(self):
    
        self.min_age = 18
        self.max_age = 70
        self.figformat='pdf'
        self.minimal=False
        self.timestep=0.25
        self.year=2018
        
        self.profile=False
        
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
        self.include_investment=False
        self.include_wealth=False
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
        self.environment=None #'unemployment-v0'
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
        self.include_ove=True
        self.unemp_limit_reemp=True
        self.extra_ppr=0
        self.startage=self.min_age
        self.use_sigma_reduction=True
        self.silent=False
        self.exploration=False
        self.exploration_ratio=0.10
        self.randomness=True
        self.include_emtr=False
        self.library=0
        self.processes=10
        
        self.random_returns=False
        self.r_mean=0.0
        self.r_std=0
        self.CRRA_eta=1.5
        self.use_utility=0 # log; 1 CRRA
        
    def setup_parameters(self,**kwargs):
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs

        for key, value in kwarg.items():
            if key=='profile':
                if value is not None:
                    self.profile=value
            if key=='processes':
                if value is not None:
                    self.processes=value
                    print('processes',self.processes)
            if key=='callback_minsteps':
                if value is not None:
                    self.callback_minsteps=value
            elif key=='random_returns':
                if value is not None:
                    self.random_returns=value
            elif key=='use_utility':
                if value is not None:
                    self.use_utility=value
            elif key=='CRRA_eta':
                if value is not None:
                    self.CRRA_eta=value
            elif key=='use_utility':
                if value is not None:
                    self.use_utility=value
            elif key=='investment':
                if value is not None:
                    self.include_investment=value
            elif key=='wealth':
                if value is not None:
                    self.include_wealth=value
            elif key=='library':
                if value is not None:
                    self.library=value
            elif key=='r_std':
                if value is not None:
                    self.r_std=value
            elif key=='r_mean':
                if value is not None:
                    self.r_mean=value
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
            elif key=='timestep':
                if value is not None:
                    self.timestep=value
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
            elif key=='include_emtr':
                if value is not None:
                    self.include_emtr=value
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
            elif key=='year':
                if value is not None:
                    self.year=value
            elif key=='use_tianshou':
                if value is not None:
                    self.use_tianshou=value
            
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
   
    def get_reward(self):
        return self.episodestats.get_reward()
   
    def render(self,load=None,figname=None,grayscale=False):
        if load is not None:
            self.episodestats.load_sim(load)
            self.plotstats.render(figname=figname,grayscale=grayscale)
        else:
            self.plotstats.render(figname=figname,grayscale=grayscale)
            #self.episodestats.render(figname=figname,grayscale=grayscale)
   
    def render_reward(self,load=None,figname=None):
        if load is not None:
            self.episodestats.load_sim(load)
            
        return self.episodestats.get_average_discounted_reward(),self.episodestats.get_initial_reward()
        #return self.episodestats.comp_total_reward(),self.episodestats.comp_initial_npv()

    def render_reward_trajectory(self,load=None,figname=None):
        if load is not None:
            self.episodestats.load_sim(load)
            
        return self.episodestats.get_reward_trajectory()

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

    def compare_with(self,cc2,label1='vaihtoehto',label2='perus',figname=None,grayscale=False,dash=True):
        '''
        compare_with

        compare results obtained another model
        '''
        self.plotstats.compare_with(cc2,label1=label1,label2=label2,grayscale=grayscale,figname=figname,dash=dash)

    def run_results(self,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',twostage=False,
               save='saved/perusmalli',debug=False,simut='simut',results='results/simut_res',
               stats='results/simut_stats',deterministic=True,train=True,predict=True,
               batch1=1,batch2=100,cont=False,start_from=None,callback_minsteps=None,
               verbose=1,max_grad_norm=None,learning_rate=0.25,log_interval=10,
               learning_schedule='linear',vf=None,arch=None,gae_lambda=None,
               startage=None,processes=None):
   
        '''
        run_results

        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
        
        if self.plotdebug:
            debug=True
            self.runner.check_env(self.env)
   
        self.n_pop=pop
        if callback_minsteps is not None:
            self.callback_minsteps=callback_minsteps

        if train: 
            print('train...')
            if cont:
                self.train_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,verbose=verbose,
                                  debug=debug,save=save,batch1=batch1,batch2=batch2,
                                  cont=cont,start_from=start_from,twostage=twostage,
                                  max_grad_norm=max_grad_norm,learning_rate=learning_rate,log_interval=log_interval,
                                  learning_schedule=learning_schedule,vf=vf,arch=arch,gae_lambda=gae_lambda,processes=processes)
            else:
                self.train_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,verbose=verbose,
                                 debug=debug,batch1=batch1,batch2=batch2,cont=cont,
                                 save=save,twostage=twostage,
                                 max_grad_norm=max_grad_norm,learning_rate=learning_rate,log_interval=log_interval,
                                 learning_schedule=learning_schedule,vf=vf,arch=arch,gae_lambda=gae_lambda,processes=processes)
        if predict:
            #print('predict...')
            self.predict_protocol(pop=pop,rlmodel=rlmodel,load=save,startage=startage,
                          debug=debug,deterministic=deterministic,results=results,arch=arch,processes=processes)
          
    def train_protocol(self,steps1=2_000_000,steps2=1_000_000,rlmodel='acktr',
               debug=False,batch1=1,batch2=1000,cont=False,twostage=False,log_interval=10,
               start_from=None,save='best3',verbose=1,max_grad_norm=None,
               learning_rate=0.25,learning_schedule='linear',vf=None,arch=None,gae_lambda=None,processes=None):
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
                self.runner.train(steps=steps1,cont=cont,rlmodel=rlmodel,save=tmpname,batch=batch1,debug=debug,
                           start_from=start_from,use_callback=False,use_vecmonitor=False,
                           log_interval=log_interval,verbose=1,vf=vf,arch=arch,gae_lambda=gae_lambda,
                           max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule,processes=processes)
            else:
                self.runner.train(steps=steps1,cont=False,rlmodel=rlmodel,save=tmpname,batch=batch1,debug=debug,vf=vf,arch=arch,
                           use_callback=False,use_vecmonitor=False,log_interval=log_interval,verbose=1,gae_lambda=gae_lambda,
                           max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule,processes=processes)

        if twostage and steps2>0:
            print('phase 2')
            self.runner.train(steps=steps2,cont=True,rlmodel=rlmodel,save=tmpname,
                       debug=debug,start_from=tmpname,batch=batch2,verbose=verbose,
                       use_callback=False,use_vecmonitor=False,log_interval=log_interval,bestname=save,plotdebug=plotdebug,
                       max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule,processes=processes)

    def predict_protocol(self,pop=1_00,rlmodel='acktr',results='results/simut_res',arch=None,
                         load='saved/malli',debug=False,deterministic=False,startage=None,processes=None):
        '''
        predict_protocol

        simulate the three models obtained from run_protocol
        '''

        print('pop_protocol',pop)
 
        def run_sim():
            self.runner.simulate(pop=pop,rlmodel=rlmodel,debug=debug,arch=arch,
                        load=load,save=results,deterministic=deterministic,startage=startage,processes=processes)
 
        # simulate the saved best
        #if self.profile:
        #    profile.run('run_sim()')
        #else:
        print('run_sim')
        run_sim()
        

    def run_distrib(self,n=5,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',
               save='saved/distrib_base_',debug=False,simut='simut',results='results/distrib_',
               deterministic=True,train=True,predict=True,batch1=1,batch2=100,cont=False,
               start_from=None,twostage=False,callback_minsteps=None,
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
               batch1=batch1,batch2=batch2,cont=cont,start_from=start_from,
               callback_minsteps=callback_minsteps,verbose=verbose,learning_rate=learning_rate,
               learning_schedule=learning_schedule,log_interval=log_interval)

        #self.render_distrib(load=results,n=n,stats_results=stats_results)
            
    def comp_distribs(self,load=None,n=1,startn=0,stats_results='results/distrib_stats',singlefile=False):
        if load is None:
            return
            
        self.episodestats.run_simstats(load,stats_results,n,startn=startn,singlefile=singlefile)

    def render_distrib(self,stats_results='results/distrib_stats',figname=None):
        self.plotstats.plot_simstats(stats_results,figname=figname)

        # gather results ...
        #if plot:
        #    print('plot')
            
    def compare_distrib(self,filename1,filename2,n=1,label1='perus',label2='vaihtoehto',figname=None):
        self.plotstats.compare_simstats(filename1,filename2,label1=label1,label2=label2,figname=figname)

    def plot_rewdist(self,age=20,sum=False,all=False):
        t=self.map_age(age)
        self.plotstats.plot_rewdist(t=t,sum=sum,all=all)

    def plot_saldist(self,age=20,sum=False,all=False,n=10):
        t=self.map_age(age)
        self.plotstats.plot_saldist(t=t,sum=sum,all=all,n=n)

    def plot_RL_act(self,t,rlmodel='acktr',load='perus',debug=True,deterministic=True,
                        n_palkka=80,n_emppalkka=80,deltapalkka=1000,deltaemppalkka=1000,n_elake=40,deltaelake=1500,
                        hila_palkka0=0,min_pension=0,deltapalkka_old=None,deltaemppalkka_old=None):
        model,env,n_cpu=self.runner.setup_model(rlmodel=rlmodel,load=load,debug=debug)
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
        model,env,n_cpu=self.runner.setup_model(rlmodel=rlmodel,load=load,debug=debug)
        return self.RL_simulate_V(model,env,t,emp=emp,deterministic=deterministic,time_in_state=time_in_state,
                        n_palkka=n_palkka,n_emppalkka=n_emppalkka,deltapalkka=deltapalkka,deltaemppalkka=deltaemppalkka,
                        n_elake=n_elake,deltaelake=deltaelake,min_wage=min_wage,min_pension=min_pension,
                        deltapalkka_old=deltapalkka_old,deltaemppalkka_old=deltaemppalkka_old)

    def get_rl_act(self,t,emp=0,time_in_state=0,rlmodel='acktr',load='perus',debug=True,deterministic=True):
        model,env,n_cpu=self.runner.setup_model(rlmodel=rlmodel,load=load,debug=debug)
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

    def combine_results(self,results='results/simut_res'):
        '''
        combine_results
        '''
 
        self.runner.combine_results(results=results)    