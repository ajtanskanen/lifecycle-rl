'''
Runner for fitting the model with multidiscrete-SAC
- revised model
'''

import gym, numpy as np
import os
from .utils import make_env

#from tqdm import tqdm_notebook as tqdm # jos notebookissa
from tqdm import tqdm

from . episodestats import EpisodeStats
from . simstats import SimStats

from . sacd.sacd import SacdAgent

#from multiprocessing import shared_memory
from multiprocessing import Process,Manager


class runner_SAC():
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

        self.logdir = "log"
        
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
        self.n_employment,self.n_acts=self.env.get_n_states()
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.action_shape = self.env.action_space.shape or self.env.action_space.n

        self.version = self.env.get_lc_version()

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
              'state_shape': (self.env.observation_space.shape or self.env.observation_space.n)[0],
              'year': year}

        self.episodestats=episodestats
        
    def check_env(self,env):
        return

    def get_multiprocess_env(self,rlmodel,debug=False,arch=None,predict=False):

        if arch is not None:
            print('arch',arch)

        n_cpu = 4
        policy_kwargs = {}

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
        batch=max(1,int(np.ceil(batch/n_cpu)))
        
        #full_tensorboard_log=False
        #if max_grad_norm is None:
        #    max_grad_norm=0.05 # default 0.50
            
        #max_grad_norm=0.1 # 0.05 # 0.01 # 0.001  was old

        print('batch {} learning rate {}'.format(batch,learning_rate))

        if cont:
            print('continuing from ',loadname)
            model = SacdAgent(env,env,self.logdir,cont=True,loadname=loadname)
        else:
            model = SacdAgent(env,env,self.logdir)
                            
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

        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':True})
    
        #env = self.env
        #nonvec=False
        #if nonvec:
        #   env=self.env
        #else:
            #env = make_vec_env(self.environment, n_envs=n_cpu, seed=1, vec_env_cls=SubprocVecEnv)
            #env = SubprocVecEnv([lambda: make_env(self.environment, i, gkwargs) for i in range(n_cpu)], start_method='spawn')
            #env = DummyVecEnv([lambda: make_env(self.environment, i, gkwargs) for i in range(n_cpu)])
            #env = ShmemVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='fork')

            #if False:
                #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=gkwargs) for i in range(n_cpu)])

        model=self.setup_rlmodel(self.rlmodel,start_from,self.env,batch,policy_kwargs,learning_rate,
                                cont,max_grad_norm=max_grad_norm,verbose=verbose,n_cpu=n_cpu,
                                learning_schedule=learning_schedule,vf=vf,gae_lambda=gae_lambda)
        print('training..')

        model.run(cont=cont,total_timesteps=steps)

        model.save(save)
        print('done')

        del model #,env

    def setup_model_v2(self,env,rank=1,debug=False,rlmodel='acktr',plot=True,load=None,
                    deterministic=False,arch=None,predict=False,n_cpu_tf_sess=1):
        '''
        Set up simulation
        '''

        if rank==0:    
            print('simulating ',load)

        # multiprocess environment
        policy_kwargs,_=self.get_multiprocess_env(rlmodel,debug=debug,arch=arch,predict=predict)
        model = SacdAgent(env,env,self.logdir,cont=True,loadname=load,batch_size=1)

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

        print('switching to join', processes)
            
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

        if args['version'] in set([4,5,6,7,104]):  # increase by 2
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
        #envlist=[lambda: make_env(args['environment'],seed+i,kwargs=kwargs) for i in range(1)]
        #print('Child',rank,'check point 3b')
        #env = DummyVecEnv(envlist)
        #env = SubprocVecEnv(envlist, start_method='spawn')
        #print('spawning',args['environment'])
        #env = SubprocVecEnv([lambda: make_env(args['environment'],seed,kwargs=kwargs) for i in range(1)], start_method='spawn')
        #print('Child',rank,'check point 4')

        print(args['environment'])

        env = gym.make(args['environment'],kwargs=kwargs)
        #env = make_env(args['environment'],seed,kwargs=kwargs)
        env.seed(args['seed'] + rank)

        model=self.setup_model_v2(env,rank=rank,debug=args['debug'],rlmodel=args['rlmodel'],load=args['load'],
                                  deterministic=args['deterministic'],arch=args['arch'],predict=True,n_cpu_tf_sess=1)

        if args['startage'] is not None:
            env.set_startage(args['startage'])

        states = env.reset()

        if rank == 0:
            print('predict',rank)
            if render:
                env.render()

        epienv = gym.make(args['environment'],kwargs=kwargs) # make a local (unshared) environment
        episodestats = SimStats(args['timestep'],args['n_time'],args['n_employment'],n_pop_single,
                            epienv,args['minimal'],args['min_age'],args['max_age'],args['min_retirementage'],
                            version=args['version'],params=args['gym_kwargs'],year=args['year'],gamma=args['gamma'],
                            silent=True)
        episodestats.init_variables()

        if rank==0:
            tqdm_e = tqdm(range(args['pop']), desc='Population', leave=True, unit=" p")

        k=0
        n=0
        pred=0
        while n<n_pop_single:
            act, predstate = model.predict(states,deterministic=deterministic)
            #act2 = env.get_Qactions(act)
            newstate, rewards, dones, infos = env.step(act)
            if n<n_pop_single: # do not save extras
                if dones:
                    #episodestats.add(n,act2,rewards,states,infos['terminal_observation'],infos)
                    #episodestats.add(n,act2,rewards,states,newstate,infos)
                    episodestats.add(n,act,rewards,states,newstate,infos)
                    n += n_add
                    info['pop'] += n_add
                    newstate = env.reset()
                    #episodestats.add(n,0,0,newstate,newstates,infos)
                    if rank==0:
                        tqdm_e.update(info['pop']-pred)
                        tqdm_e.set_description("Pop " + str(info['pop']))
                        pred=info['pop']
                else:
                    #episodestats.add(n,act2,rewards,states,newstate,infos)
                    episodestats.add(n,act,rewards,states,newstate,infos)
    
            states = newstate

        if rank==0:
            print('saving results...')

        episodestats.scale_sim()
        episodestats.save_sim(savefile)

        return 1

    def combine_results(self,results):
        self.combine_episodestats(self.args)

    def combine_episodestats(self,args):
        #save=args['save_dir']+args['simfile']+'_rank_'
        save=args['save_dir'] + args['simfile']+'_rank_'
        
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

        base.save_sim(args['save_dir'] + args['simfile']+'_combined')