'''

    dyn_prog.py
    
    implements a scheme similar to solving valuation of american options for the life cycle model
    this is a kind of dynamic programming scheme
    

'''

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import Benefits
import gym_unemployment
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm_notebook as tqdm
from lifecycle_rl import Lifecycle
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines import A2C, ACER, DQN, ACKTR #, TRPO



class DynProgLifecycle(Lifecycle):

    def __init__(self,minimal=True,env=None,timestep=1.0,ansiopvraha_kesto300=None,
                    ansiopvraha_kesto400=None,karenssi_kesto=None,
                    ansiopvraha_toe=None,perustulo=None,plotdebug=False,mortality=None,
                    gamma=None,n_palkka=None,n_elake=None,n_tis=None):

        super().__init__(minimal=minimal,env=env,timestep=timestep,ansiopvraha_kesto300=ansiopvraha_kesto300,
                    ansiopvraha_kesto400=ansiopvraha_kesto400,karenssi_kesto=karenssi_kesto,
                    ansiopvraha_toe=ansiopvraha_toe,perustulo=perustulo,mortality=mortality,plotdebug=plotdebug,
                    gamma=gamma)
        
        '''
        Alusta muuttujat
        '''
        self.hila_palkka0 = 0
        self.hila_elake0 = 0
        
        # dynaamisen ohjelmoinnin parametrejä
        self.n_palkka = 20
        self.n_elake = 40
        self.n_tis = 5 # ei vaikutusta palkkaan

        if n_palkka is not None:
            self.n_palkka=n_palkka
        if n_elake is not None:
            self.n_elake=n_elake
        if n_tis is not None:
            self.n_tis=n_tis
        
        self.deltapalkka = 100000/(self.n_palkka-1)
        self.deltaelake = 80000/(self.n_elake-1)
        self.deltatis = 1


        # ei käytetä
        #self.n_palkka_future = 30 # 40
        #self.delta_palkka_future = 0.15
        #self.mid_palkka_future=5 # 20
        
        print('min',self.min_retirementage)
        
    def init_grid(self):
        self.Hila = np.zeros((self.n_time+2,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_palkka))
        self.actHila = np.zeros((self.n_time+2,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_palkka,self.n_acts))        

    def explain(self):
        print('n_palkka {} n_elake {}'.format(self.n_palkka,self.n_elake))
        print('hila_palkka0 {} hila_elake0 {}'.format(self.hila_palkka0,self.hila_elake0))
        print('deltapalkka {} deltaelake {}'.format(self.deltapalkka,self.deltaelake))
        print('n_tis {} deltatis {}'.format(self.n_tis,self.deltatis))
        print('gamma {} timestep {}'.format(self.gamma,self.timestep))

    def map_elake(self,v):
        return self.hila_elake0+self.deltaelake*v # pitäisikö käyttää exp-hilaa?

    def inv_elake(self,v):
        vmin=max(0,min(self.n_elake-2,int((v-self.hila_elake0)/self.deltaelake)))
        vmax=vmin+1
        w=(v-self.hila_elake0)/self.deltaelake-vmin # lin.approximaatio

        return vmin,vmax,w

    def map_exp_elake(self,v):
        return self.hila_elake0+self.deltaelake*(np.exp(v*self.expelakescale)-1)

    def inv_exp_elake(self,v):
        vmin=max(0,min(self.n_elake-2,int((np.log(v-self.hila_elake0)+1)/self.deltaelake)))
        vmax=vmin+1
        vmin_value=self.map_exp_elake(vmin)
        vmax_value=self.map_exp_elake(vmax)
        w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio

        return vmin,vmax,w

    def map_palkka(self,v,midpoint=False):
        if midpoint:
            return self.hila_palkka0+self.deltapalkka*(v+0.5) # pitäisikö käyttää exp-hilaa?
        else:
            return self.hila_palkka0+self.deltapalkka*v # pitäisikö käyttää exp-hilaa?

    def inv_palkka(self,v):
        vmin=int(max(0,min(self.n_palkka-2,int((v-self.hila_palkka0)/self.deltapalkka))))
        vmax=vmin+1
        w=(v-self.hila_palkka0)/self.deltapalkka-vmin # lin.approximaatio

        return vmin,vmax,w
# 
#     def map_palkka_future(self,v,palkka):
#         p=max(palkka,1000)
#         return (1.0+self.delta_palkka_future*(v-self.mid_palkka_future))*p
# 
#     def inv_palkka_future(self,v,palkka):
#         p=max(palkka,1000)    
#         if p>0:
#             vv=(v/p-1.0)/self.delta_palkka_future+self.mid_palkka_future
#             vmin=max(0,min(self.n_palkka_future-2,int(vv)))
#             vmax=vmin+1
#             w=vv-vmin
#         else:
#             vmin=0
#             vmax=1
#             w=0
# 
#         return vmin,vmax,w

    def map_exp_palkka(self,v):
        return self.hila_palkka0+self.deltapalkka*(np.exp(v*self.exppalkkascale)-1)

    def inv_exp_palkka(self,v):
        vmin=max(0,min(self.n_palkka-2,int((np.log(v-self.hila_palkka0)+1)/self.deltapalkka)))
        vmax=vmin+1
        vmin_value=self.map_exp_palkka(vmin)
        vmax_value=self.map_exp_palkka(vmax)
        w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio

        return vmin,vmax,w

    def map_tis(self,v):
        return v

    def inv_tis(self,v):
        return int(min(self.n_tis-1,v))
    
    # lineaarinen approksimaatio
    def get_V(self,t,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if t>self.n_time:
            return 0
        
        if emp is None:
            emp,elake,old_wage,ika,time_in_state,wage=self.env.state_decode(s)
            
        emin,emax,we=self.inv_elake(elake)
        pmin,pmax,wp=self.inv_palkka(old_wage)
        p2min,p2max,wp2=self.inv_palkka(wage)        
        tismax=self.inv_tis(time_in_state)
        emp=int(emp)
        
        #if wp2<0 or wp2>1:
        #    print('emp {} elake {} old_wage {} wage {} tis {}: wp2 {}'.format(emp,elake,old_wage,wage,time_in_state,wp2))
        #if wp<0 or wp>1:
        #    print('emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,wp))
        #if we<0 or we>1:
        #    print('emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,we))
        
        V1=(1-wp2)*((1-wp)*( (1-we)*(self.Hila[t,pmin,emin,emp,tismax,p2min])\
                            +we*(self.Hila[t,pmin,emax,emp,tismax,p2min]))+\
                    wp*(     (1-we)*(self.Hila[t,pmax,emin,emp,tismax,p2min])\
                            +we*(self.Hila[t,pmax,emax,emp,tismax,p2min])))+\
          wp2*(     (1-wp)*( (1-we)*(self.Hila[t,pmin,emin,emp,tismax,p2max])\
                            +we*(self.Hila[t,pmin,emax,emp,tismax,p2max]))+\
                    wp*(     (1-we)*(self.Hila[t,pmax,emin,emp,tismax,p2max])\
                            +we*(self.Hila[t,pmax,emax,emp,tismax,p2max])))                         

        V=max(0,V1)

        return V
        
    def plot_Hila(self,age,l=5,emp=1,time_in_state=1,diff=False):
        x=np.arange(0,100000,1000)
        q=np.zeros(x.shape)
        t=self.map_age(age)    
        
        fig,ax=plt.subplots()
        if diff:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l)
                for palkka in x:
                    q[k]=self.get_V(t,emp=1,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka)-self.get_V(t,emp=0,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka)
                    k=k+1
            
                plt.plot(x,q,label=elake)
        else:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l)
                for palkka in x:
                    q[k]=self.get_V(t,emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka)
                    k=k+1
            
                plt.plot(x,q,label=elake)
            
        ax.set_xlabel('palkka')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()
                
    def plot_actHila(self,age,l=5,emp=1,time_in_state=1,diff=False,act=0,emp2=1):
        x=np.arange(0,100000,1000)
        q=np.zeros(x.shape)
        t=self.map_age(age)    
        
        fig,ax=plt.subplots()
        if diff:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l)
                for palkka in x:
                    q[k]=self.get_actV(t,emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=emp2)-self.get_actV(t,emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=0)
                    k=k+1
            
                plt.plot(x,q,label=elake)
        else:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l)
                for palkka in x:
                    q[k]=self.get_actV(t,emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=act)
                    k=k+1
            
                plt.plot(x,q,label=elake)
            
        ax.set_xlabel('palkka')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()

    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_actV(self,t,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,act=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if not s is None:
            emp,elake,old_wage,ika,time_in_state,wage=self.env.state_decode(s)
            
        emin,emax,we=self.inv_elake(elake)
        pmin,pmax,wp=self.inv_palkka(old_wage)
        p2min,p2max,wp2=self.inv_palkka(wage)    
        tismax=self.inv_tis(time_in_state)
        emp=int(emp)
        tismax=int(tismax)
        
        #if wp2<0 or wp2>1:
        #    print('emp {} elake {} old_wage {} wage {} tis {}: wp2 {}'.format(emp,elake,old_wage,wage,time_in_state,wp2))
        #if wp<0 or wp>1:
        #    print('emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,wp))
        #if we<0 or we>1:
        #    print('emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,we))
        
        apx1=(1-wp2)*((1-wp)*((1-we)*(self.actHila[t,pmin,emin,emp,tismax,p2min,act])
                              +we*(self.actHila[t,pmin,emax,emp,tismax,p2min,act]))+\
                        wp*((1-we)*(self.actHila[t,pmax,emin,emp,tismax,p2min,act])
                              +we*(self.actHila[t,pmax,emax,emp,tismax,p2min,act])))+\
                wp2*((1-wp)*((1-we)*(self.actHila[t,pmin,emin,emp,tismax,p2max,act])
                                +we*(self.actHila[t,pmin,emax,emp,tismax,p2max,act]))+\
                        wp*((1-we)*(self.actHila[t,pmax,emin,emp,tismax,p2max,act])
                                +we*(self.actHila[t,pmax,emax,emp,tismax,p2max,act])))
        V=max(0,apx1)
            
        act=int(np.argmax(V))
        maxV=np.max(V)

        return V

    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_act(self,t,s,full=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        emp,elake,old_wage,ika,time_in_state,wage=self.env.state_decode(s)
        emin,emax,we=self.inv_elake(elake)
        pmin,pmax,wp=self.inv_palkka(old_wage)
        p2min,p2max,wp2=self.inv_palkka(wage)    
        tismax=self.inv_tis(time_in_state)
        emp=int(emp)
        tismax=int(tismax)
        
        n_emp=self.n_acts
        
        V=np.zeros(n_emp)
        emp_set=set([0,1])
        if emp in emp_set:
            if ika<=self.min_retirementage:
                n_emp=2
            else:
                n_emp=3
            
        for k in range(n_emp):
            apx1=(1-wp2)*((1-wp)*((1-we)*(self.actHila[t,pmin,emin,emp,tismax,p2min,k])
                                  +we*(self.actHila[t,pmin,emax,emp,tismax,p2min,k]))+\
                            wp*((1-we)*(self.actHila[t,pmax,emin,emp,tismax,p2min,k])
                                  +we*(self.actHila[t,pmax,emax,emp,tismax,p2min,k])))+\
                    wp2*((1-wp)*((1-we)*(self.actHila[t,pmin,emin,emp,tismax,p2max,k])
                                    +we*(self.actHila[t,pmin,emax,emp,tismax,p2max,k]))+\
                            wp*((1-we)*(self.actHila[t,pmax,emin,emp,tismax,p2max,k])
                                    +we*(self.actHila[t,pmax,emax,emp,tismax,p2max,k])))
            V[k]=max(0,apx1)
            
        act=int(np.argmax(V))
        maxV=np.max(V)

        if full:
            return act,maxV,V
        else:
            return act,maxV
        
    def get_actV_random(self,age):
        if age<self.min_retirementage:
            return np.random.randint(2)
        else:
            return np.random.randint(3)

    # this routine is needed for the dynamic programming
    def get_rewards_continuous(self,s,actions):
        rewards=[]
        sps=[]
        
        start_state=self.env.state_encode(*s)
        #self.env.render(state=start_state)
        for a in actions:
            self.env.state=start_state
            newstate, reward, dones, info = self.env.step(a,dynprog=True)
            #if dones:
            #    self.reset()
            sps.append(np.array(newstate))
            rewards.append(reward)
            
        return rewards,sps

    def backtrack(self,age,debug=False):
        '''
        Dynaaminen ohjelmointi hilan avulla
        '''
        t=self.map_age(age)    
        
        if age<self.min_retirementage:
            act_set=set([0,1])
        else:
            act_set=set([0,1,2])
        
        emp_set=set([0])
        
        pn_weight=np.zeros((self.n_palkka,self.n_palkka))
        ika2=age+1
        for p in range(self.n_palkka): 
            palkka=self.map_palkka(p)
            #palkka_mid=self.map_palkka(p,midpoint=True)
            weight_old=0
            for pnext in range(self.n_palkka-1): 
                #palkka_next=self.map_palkka(pnext)
                palkka_next_mid=self.map_palkka(pnext,midpoint=True)
                weight_new=self.env.wage_process_cumulative(palkka_next_mid,palkka,ika2)
                pn_weight[p,pnext]=weight_new-weight_old
                weight_old=weight_new
                
            pn_weight[p,self.n_palkka-1]=1-weight_old
                    #print('w{} p{}'.format(weight,palkka))
    
        for emp in range(self.n_employment):
            if emp==2:
                if age<self.min_retirementage:
                    self.Hila[t,:,el,emp,:,:]=0
                else:
                    for el in range(self.n_elake):
                        elake=self.map_elake(el)
                        time_in_state=self.map_tis(0)

                        # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                        rts,Sps=self.get_rewards_continuous((emp,elake,0,age,time_in_state,0),emp_set)
                    
                        for ind,a in enumerate(emp_set):
                            emp2,elake2,_,_,_,_=self.env.state_decode(Sps[ind])
                            self.actHila[t,:,el,emp,:,:,:]=rts[ind]+self.gamma*self.get_V(t+1,emp=emp2,elake=elake2,old_wage=0,wage=0,time_in_state=0)
                            #print('getV(emp{} e{} p{}): {}'.format(emp2,elake2,palkka,q))

                        self.Hila[t,:,el,emp,:,:]=self.actHila[t,0,el,emp,0,0,0]
            elif emp==1:
                for el in range(self.n_elake):
                    elake=self.map_elake(el)
                    for p in range(self.n_palkka): 
                        palkka=self.map_palkka(p)
                        palkka_mid=self.map_palkka(p,midpoint=True)
                        for p_old in range(self.n_palkka): 
                            palkka_vanha=self.map_palkka(p_old)
                            time_in_state=self.map_tis(0)

                            # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                            rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                            #print('(emp{} e{} p_old{} p{} ika{})'.format(emp,elake,palkka_vanha,palkka,age))
                
                            for ind,a in enumerate(act_set):
                                emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                #print('emp2:{} e2:{} ika2:{} r{}'.format(emp2,elake2,ika2,rts[ind]))
                                q=rts[ind]
                                for pnext in range(self.n_palkka): 
                                    palkka_next=self.map_palkka(pnext)
                                    q+=self.gamma*self.get_V(t+1,emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,wage=palkka_next)*pn_weight[p,pnext]
                                    #print('palkka_next{} w{}'.format(palkka_next,pn_weight[p,pnext]))
                                
                                #print('getV(emp{} e{} p{}): {}'.format(emp2,elake2,palkka,q))
                                self.actHila[t,p_old,el,emp,:,p,a]=q
                                
                            self.Hila[t,p_old,el,emp,:,p]=np.max(self.actHila[t,p_old,el,emp,0,p,:])
            else: # emp==0
                for el in range(self.n_elake):
                    elake=self.map_elake(el)
                    for p in range(self.n_palkka): 
                        palkka=self.map_palkka(p)
                        for p_old in range(self.n_palkka): 
                            palkka_vanha=self.map_palkka(p_old)
                            for tis in range(self.n_tis):
                                time_in_state=self.map_tis(tis)

                                # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                                rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                    
                                for ind,a in enumerate(act_set):
                                    emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                    q=rts[ind]
                                    for pnext in range(self.n_palkka): 
                                        palkka_next=self.map_palkka(pnext)
                                        q+=self.gamma*self.get_V(t+1,emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,wage=palkka_next)*pn_weight[p,pnext]
                                    
                                    self.actHila[t,p_old,el,emp,tis,p,a]=q
                                    
                                self.Hila[t,p_old,el,emp,tis,p]=np.max(self.actHila[t,p_old,el,emp,tis,p,:])

        if debug:
            self.print_actV(age)
            self.print_V(age)
            #print('at age {} mean V {} mean r {}'.format(age,np.mean(self.Hila[t,:,:,:,:]),qr/m),qw/m)

                                
    def train(self,debug=False,save='best/dynamic_prog_V.h5'):
        '''
        Lasketaan optimaalinen työllistyminen/työttömyys/eläköityminen valitulla valintametodilla
        '''
        self.init_grid()
        print('Optimizing behavior')
        tqdm_e = tqdm(range(int(self.n_time)), desc='Score', leave=True, unit=" year")

        for age in range(self.max_age,self.min_age-1,-1):
            t=self.map_age(age)    
            self.backtrack(age)
            tqdm_e.set_description("Year " + str(t))
            tqdm_e.update(1)

        self.save_V(save)

    def simulate(self,debug=False,pop=1_000,save=None,load=None):
        '''
        Lasketaan työllisyysasteet ikäluokittain
        '''
        if pop is not None:
            self.n_pop=pop

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)
        if load is not None:
            self.load_V(load)        

        #self.env.seed(1234)   
        #self.env.env_seed(4567)            
        tqdm_e = tqdm(range(int(pop)), desc='Population', leave=True, unit=" p")

        for n in range(pop):
            state=self.env.reset()
            
            for t in range(self.n_time):
                if debug:
                    act,maxV,v=self.get_act(t,state,full=True)
                else:
                    act,maxV=self.get_act(t,state)
                
                newstate,r,done,info=self.env.step(act,dynprog=False)
                if debug:
                    print(r,info['r'])
                    #r=info['r']
                    
                self.episodestats.add(n,act,r,state,newstate,debug=debug,aveV=maxV)
                state=newstate
                
                if done:
                    tqdm_e.update(1)                
                    if debug:
                        print('done')
                    break      
        
        if save is not None:
            self.episodestats.save_sim(save)
          
    def plot_statsV(self,aveV):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #x=range(self.min_age,self.min_age+self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('Keskiarvo V')
        mV=np.mean(aveV,axis=1)
        ax.plot(x,mV)
        plt.show()
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('palkka')
        ax.plot(x,mV[:,0:10])
        plt.show()                      
        
    def save_V(self,filename):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset("Hila", data=self.Hila, dtype='f16')
        f.create_dataset("actHila", data=self.actHila, dtype='f16')
        f.create_dataset("hila_palkka0", data=self.hila_palkka0, dtype='f16')
        f.create_dataset("hila_elake0", data=self.hila_elake0, dtype='f16')
        f.create_dataset("n_palkka", data=self.n_palkka, dtype='i4')
        f.create_dataset("deltapalkka", data=self.deltapalkka, dtype='f16')
        f.create_dataset("n_elake", data=self.n_elake, dtype='i4')
        f.create_dataset("deltaelake", data=self.deltaelake, dtype='f16')
        f.create_dataset("n_tis", data=self.n_tis, dtype='i4')
        f.create_dataset("deltatis", data=self.deltatis, dtype='f16')
        f.close()
        
    def load_V(self,filename):
        f = h5py.File(filename, 'r')
        self.Hila = f.get('Hila').value
        self.actHila = f.get('actHila').value
        self.hila_palkka0 = f.get('hila_palkka0').value
        self.hila_elake0 = f.get('hila_elake0').value
        self.n_palkka = f.get('n_palkka').value
        self.deltapalkka = f.get('deltapalkka').value
        self.n_elake = f.get('n_elake').value
        self.deltaelake = f.get('deltaelake').value
        self.n_tis = f.get('n_tis').value
        self.deltatis = f.get('deltatis').value
        f.close()
        
    def plot_act(self):
        act_ratio=np.sum(self.actions==1,axis=1)/2000
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,act_ratio)    
        ax.set_xlabel('Aika')
        ax.set_ylabel('Tilasta pois-siirtyneet')
        plt.show()

    def plot_higher(self,t):
        emp=1
        prev=0
        q=np.array(self.actHila[t,:,:,emp,prev,0]>self.actHila[t,:,:,emp,prev,1]).astype(int)
        self.plot_img(q,xlabel="Pension",ylabel="Wage",title='Employed, stay in state')
        emp=0
        q=np.array(self.actHila[t,:,:,emp,prev,0]>self.actHila[t,:,:,emp,prev,1]).astype(int)
        self.plot_img(q,xlabel="Pension",ylabel="Wage",title='Unemployed, stay in state')

    def print_V(self,age):
        t=self.map_age(age)    
        print('t=',t,'age=',age)
        print('töissä\n',self.get_diag_V(t,1))
        print('ei töissä\n',self.get_diag_V(t,0))
        print('eläke\n',self.get_diag_V(t,2))

    def get_diag_V(self,t,emp,tis=1):
        sh=self.Hila.shape
        h=np.zeros((sh[1],sh[2]))
        for k in range(sh[1]):
            for l in range(sh[2]):
                h[k,l]=self.Hila[t,k,l,emp,tis,k]
                
        return h
    
    def get_diag_actV(self,t,emp,act,time_in_state=1):
        sh=self.Hila.shape
        h=np.zeros((sh[1],sh[2]))
        for k in range(sh[1]):
            for l in range(sh[2]):
                # self.actHila = np.zeros((self.n_time+2,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_palkka,self.n_acts))  
                h[k,l]=self.actHila[t,k,l,emp,time_in_state,k,act]
                
        return h
    
    def plot_V(self,age):
        t=self.map_age(age)
        self.plot_img(self.get_diag_V(t,1),xlabel="Pension",ylabel="Wage",title='Töissä')
        self.plot_img(self.get_diag_V(t,0),xlabel="Pension",ylabel="Wage",title='Työttömänä')
        self.plot_img(self.get_diag_V(t,1)-self.get_diag_V(t,0),xlabel="Pension",ylabel="Wage",title='Työssä-Työtön')

    def print_actV(self,age,time_in_state=1):
        t=self.map_age(age)
        print('t={} age={}'.format(t,age))
        if age>self.min_retirementage:
            print('eläke: pysyy\n{}\n'.format(self.get_diag_actV(t,2,0,time_in_state=time_in_state)))
            print('töissä: pois töistä\n{}\ntöissä: pysyy\n{}\ntöissä: eläköityy\n{}\n'.format(self.get_diag_actV(t,1,1,time_in_state=time_in_state),self.get_diag_actV(t,1,0,time_in_state=time_in_state),self.get_diag_actV(t,1,2,time_in_state=time_in_state)))
            print('ei töissä: töihin\n{}\nei töissä: pysyy\n{}\nei töissä: eläköityy\n{}\n'.format(self.get_diag_actV(t,0,1,time_in_state=time_in_state),self.get_diag_actV(t,0,0,time_in_state=time_in_state),self.get_diag_actV(t,0,2,time_in_state=time_in_state)))
        else:
            print('töissä: pois töistä\n',self.get_diag_actV(t,1,1),'\ntöissä: pysyy\n',self.get_diag_actV(t,1,0,time_in_state=time_in_state))
            print('ei töissä: töihin\n',self.get_diag_actV(t,0,1),'\nei töissä: pysyy\n',self.get_diag_actV(t,0,0,time_in_state=time_in_state))

    def print_act(self,age,time_in_state=0):
        print('age=',age)
        if age>=self.min_retirementage:
            print('eläke (act)\n')
            display(self.get_act_q(age,2,time_in_state=time_in_state))

        print('töissä (act)\n')
        display(self.get_act_q(age,1,time_in_state=time_in_state))
        print('ei töissä (act)\n')
        display(self.get_act_q(age,0,time_in_state=time_in_state))

    def plot_actV_diff(self,age):
        t=self.map_age(age)
        self.plot_img(self.get_diag_actV(t,1,1)-self.get_diag_actV(t,1,0),xlabel="Pension",ylabel="Wage",title='Töissä (ero switch-stay)')
        self.plot_img(self.get_diag_actV(t,0,1)-self.get_diag_actV(t,0,0),xlabel="Pension",ylabel="Wage",title='Työttömänä (ero switch-stay)')

    def plot_act(self,age,time_in_state=0):
        q1=self.get_act_q(age,emp=1,time_in_state=time_in_state)
        q2=self.get_act_q(age,emp=0,time_in_state=time_in_state)

        self.plot_img(q1,xlabel="Pension",ylabel="Wage",title='Töissä')
        self.plot_img(q2,xlabel="Pension",ylabel="Wage",title='Työttömänä')
    
    def get_act_q(self,age,emp=1,time_in_state=0,debug=False):
        t=self.map_age(age)
        q=np.zeros((self.n_palkka,self.n_elake))
        for p in range(self.n_palkka): 
            for el in range(self.n_elake):
                palkka=self.map_palkka(p)
                elake=self.map_elake(el)

                q[p,el]=int(np.argmax(self.actHila[t,p,el,emp,time_in_state,p,:]))
                if debug:
                    print('p {} e {} a {} q {}'.format(palkka,elake,q[p,el],self.actHila[t,p,el,emp,time_in_state,p,:]))
        return q

    def print_q(self,age,emp=1,time_in_state=0):
        _=self.get_act_q(age=age,emp=emp,time_in_state=time_in_state,debug=True)

    def compare_act(self,age,cc,time_in_state=0,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                    deterministic=True,vmin=None,vmax=None):
        q1=self.get_act_q(age,emp=0,time_in_state=time_in_state)
        q2=cc.get_RL_act(age,emp=0,time_in_state=time_in_state,rlmodel=rlmodel,
            load=load,deterministic=deterministic,
            n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
            hila_palkka0=self.hila_palkka0,hila_elake0=self.hila_elake0)
        q3=self.get_act_q(age,emp=1,time_in_state=time_in_state)
        q4=cc.get_RL_act(age,emp=1,time_in_state=time_in_state,rlmodel=rlmodel,
            load=load,deterministic=deterministic,
            n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
            hila_palkka0=self.hila_palkka0,hila_elake0=self.hila_elake0)
            
        self.plot_twoimg(q3,q4,title1='Employed DP {}'.format(age),title2='Employed RL {}'.format(age),vmin=0,vmax=2,figname='emp_'+str(age))
        self.plot_twoimg(q1,q2,title1='Unemployed DP {}'.format(age),title2='Unemployed RL {}'.format(age),vmin=0,vmax=2,figname='unemp_'+str(age))
        
    def compare_ages(self,cc,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                     deterministic=True,time_in_state=0):
        for age in set([20,30,40,45,50,55,60,61,62,63,64,65,66,67,68,69,70]):
            self.compare_act(age,cc,rlmodel=rlmodel,load=load,deterministic=deterministic,time_in_state=time_in_state)