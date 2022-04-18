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
from scipy.interpolate import interpn,interp1d,interp2d,RectBivariateSpline,PchipInterpolator
from scipy.integrate import quad,trapz
from scipy.special import softmax
import time
from scipy.stats import gaussian_kde

class DynProgLifecycleRev(Lifecycle):

    def __init__(self,minimal=True,env=None,timestep=1.0,ansiopvraha_kesto300=None,min_retirementage=None,
                    ansiopvraha_kesto400=None,karenssi_kesto=None,osittainen_perustulo=None,
                    ansiopvraha_toe=None,plotdebug=False,mortality=None,max_wage_old=None,max_empwage_old=None,
                    gamma=None,n_palkka=None,n_palkka_old=None,n_elake=None,n_tis=None,n_palkka_future=None,
                    n_emppalkka=None,n_emppalkka_old=None,use_v4=False,
                    max_pension=None,min_wage=None,max_wage=None,max_empwage=None,perustulo=None,perustulomalli=None,
                    perustulo_korvaa_toimeentulotuen=None,startage=None,lang=None):

        super().__init__(minimal=minimal,env=env,timestep=timestep,ansiopvraha_kesto300=ansiopvraha_kesto300,
                    ansiopvraha_kesto400=ansiopvraha_kesto400,karenssi_kesto=karenssi_kesto,min_retirementage=min_retirementage,
                    ansiopvraha_toe=ansiopvraha_toe,mortality=mortality,plotdebug=plotdebug,
                    gamma=gamma,perustulo=perustulo,perustulomalli=perustulomalli,osittainen_perustulo=osittainen_perustulo,
                    perustulo_korvaa_toimeentulotuen=perustulo_korvaa_toimeentulotuen,startage=startage,lang=lang)
        
        '''
        Alusta muuttujat
        '''
        #self.min_salary=1000
        #self.hila_palkka0 = 1000 # = self.min_salary # 0
        self.spline=True
        self.extrapolate=True
        self.pw_bivariate=True
        if True:
            self.spline_approx='cubic' #'cubic'
            self.pw_maxspline=3 # 1 = linear, 3 = cubic
        else:
            self.spline_approx='linear'
            self.pw_maxspline=1 # 1 = linear, 3 = cubic
            
        self.minspline=True # rajoita splineille alarajaa
        self.minbispline=True # rajoita splineille alarajaa
        self.monotone_spline=False #True
        self.oamonotone_spline=False
        #self.spline_approx='quadratic'
        #self.spline_approx='linear'
        self.integmethod=0 # compare
        self.use_v4=False
        
        self.n_employment=3
        
        self.epsabs=1e-6
        
        # dynaamisen ohjelmoinnin parametrejä
        self.n_palkka = 21
        self.n_palkka_old = 21
        self.n_emppalkka = 21
        self.n_emppalkka_old = 21
        self.n_palkka_future = 21
        #self.n_palkka_future_tr = 201
        self.n_elake = 40
        self.n_oapension = 1000
        self.n_tis = 2 # ei vaikutusta palkkaan
        self.min_wage=1_000
        self.max_wage=85_000
        self.max_wage_old=self.max_wage
        self.max_empwage=self.max_wage
        self.max_empwage_old=self.max_wage_old

        self.min_pension=0 # for active
        self.max_pension=50_000 # for active
        self.min_oapension=500*12 # for retired
        self.max_oapension=80_000 # for retired
        #self.factor_future_wage=8
        
        if n_palkka is not None:
            self.n_palkka=n_palkka
        if n_palkka_old is not None:
            self.n_palkka_old=n_palkka_old
        if n_emppalkka is not None:
            self.n_emppalkka=n_emppalkka
        if n_emppalkka_old is not None:
            self.n_emppalkka_old=n_emppalkka_old
        if n_palkka_future is not None:
            self.n_palkka_future=n_palkka_future
        if n_elake is not None:
            self.n_elake=n_elake
        if n_tis is not None:
            self.n_tis=n_tis
        if max_wage is not None:
            self.max_wage=max_wage
        if max_wage_old is not None:
            self.max_wage_old=max_wage_old
        if max_empwage is not None:
            self.max_empwage=max_empwage
        if max_empwage_old is not None:
            self.max_empwage_old=max_empwage_old
        if min_wage is not None:
            self.min_wage=min_wage
        if max_pension is not None:
            self.max_pension=max_pension
        if use_v4 is not None:
            self.use_v4=use_v4
        #if factor_future_wage is not None:
        #    self.factor_future_wage=factor_future_wage
        
        self.deltapalkka = (self.max_wage-self.min_wage)/(self.n_palkka-1)
        self.deltapalkka_old = (self.max_wage_old-self.min_wage)/(self.n_palkka_old-1)
        self.deltaemppalkka = (self.max_empwage-self.min_wage)/(self.n_emppalkka-1)
        self.deltaemppalkka_old = (self.max_empwage_old-self.min_wage)/(self.n_emppalkka_old-1)
        self.deltaelake = (self.max_pension-self.min_pension)/(self.n_elake-1)
        self.delta_oapension = (self.max_oapension-self.min_oapension)/(self.n_oapension-1)
        self.deltatis = 1
        
        self.include_pt=False
        
        #self.midfuture=int(np.floor(self.n_palkka_future/2))
        #self.deltafuture=self.factor_future_wage*0.07*0.5/self.midfuture

        self.min_grid_age=self.min_age
        self.max_grid_age=self.max_age
        
        print('min',self.min_retirementage)
        
        if False:
            self.unemp_wageshock=1.0 #0.95
        else:
            self.unemp_wageshock=0.95
            
        if self.spline:
            self.get_V=self.get_V_spline
            self.get_V_vector=self.get_V_vector_spline
            self.get_actV=self.get_actV_spline
            self.get_actReward=self.get_actReward_spline
            self.get_act=self.get_act_spline
        else:
            self.get_V=self.get_V_nospline
            self.get_V_vector=self.get_V_vector_nospline
            self.get_actV=self.get_actV_nospline
            self.get_actReward=self.get_actReward_nospline
            self.get_act=self.get_act_nospline
        
    def init_grid(self):
        self.unempHila = np.zeros((self.n_time+1,self.n_palkka_old,self.n_elake,self.n_tis,self.n_palkka))
        self.unempActHila = np.zeros((self.n_time+1,self.n_palkka_old,self.n_elake,self.n_tis,self.n_palkka,self.n_acts))        
        self.unempActReward = np.zeros((self.n_time+1,self.n_palkka_old,self.n_elake,self.n_tis,self.n_palkka,self.n_acts))        
        self.empHila = np.zeros((self.n_time+1,self.n_emppalkka_old,self.n_elake,self.n_emppalkka))
        self.empActHila = np.zeros((self.n_time+1,self.n_emppalkka_old,self.n_elake,self.n_emppalkka,self.n_acts))        
        self.empActReward = np.zeros((self.n_time+1,self.n_emppalkka_old,self.n_elake,self.n_emppalkka,self.n_acts))        
        self.oaHila = np.zeros((self.n_time+1,self.n_oapension))
        self.oaactHila = np.zeros((self.n_time+1,self.n_oapension))        
        self.oaactReward = np.zeros((self.n_time+1,self.n_oapension))        

    def explain(self):
        print(f'n_palkka {self.n_palkka}\nn_palkka_old {self.n_palkka_old}\nn_elake {self.n_elake}\nn_palkka_future {self.n_palkka_future}')
        print(f'n_emppalkka {self.n_emppalkka}\nn_emppalkka_old {self.n_emppalkka_old}\nn_elake {self.n_elake}')
        print(f'n_oapension {self.n_oapension}')
        print(f'min_wage {self.min_wage} max_wage {self.max_wage} max_wage_old {self.max_wage_old}')
        print(f'min_empwage {self.min_wage} max_empwage {self.max_empwage} max_empwage_old {self.max_empwage_old}')
        print(f'min_pension {self.min_pension} max_pension {self.max_pension}')
        print(f'min_oapension {self.min_oapension} max_oapension {self.max_oapension}')
        print(f'deltapalkka {self.deltapalkka} deltapalkka_old {self.deltapalkka_old} deltaelake {self.deltaelake}')
        print(f'delta_oapension {self.delta_oapension} deltaemppalkka {self.deltaemppalkka} deltaemppalkka_old {self.deltaemppalkka_old}')
        print(f'n_tis {self.n_tis} deltatis {self.deltatis}')
        print(f'gamma {self.gamma} timestep {self.timestep}')
        self.env.explain()

    def map_elake(self,v,emp=1):
        if emp==2:
            return self.min_oapension+self.delta_oapension*v # pitäisikö käyttää exp-hilaa?
        else:
            return self.min_pension+self.deltaelake*v # pitäisikö käyttää exp-hilaa?

    def inv_elake(self,v,emp=1):
        if emp==2:
            vmin=max(0,min(self.n_oapension-2,int(np.floor((v-self.min_oapension)/self.delta_oapension))))
            vmax=vmin+1
            w=max(0,v-self.min_oapension)/self.delta_oapension-vmin # lin.approximaatio
        else:
            vmin=max(0,min(self.n_elake-2,int(np.floor((v-self.min_pension)/self.deltaelake))))
            vmax=vmin+1
            w=max(0,v-self.min_pension)/self.deltaelake-vmin # lin.approximaatio
        
        if w<0:
            print(f'w<0: {w} {v} {vmin}')
            w=0
            #raise ValueError('A very specific bad thing happened.')        

        return vmin,vmax,w

#     def map_exp_elake(self,v):
#         return self.min_pension+self.deltaelake*(np.exp(v*self.expelakescale)-1)
# 
#     def inv_exp_elake(self,v):
#         vmin=max(0,min(self.n_elake-2,int((np.log(v-self.min_pension)+1)/self.deltaelake)))
#         vmax=vmin+1
#         vmin_value=self.map_exp_elake(vmin)
#         vmax_value=self.map_exp_elake(vmax)
#         w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio
# 
#         return vmin,vmax,w

    def map_palkka_old(self,v,emp,midpoint=False):
        if emp==0:
            deltapalkka_old=self.deltapalkka_old
        elif emp==1:
            deltapalkka_old=self.deltaemppalkka_old
    
        if midpoint:
            return self.min_wage+max(0,deltapalkka_old*(v+0.5))
        else:
            return self.min_wage+max(0,deltapalkka_old*v) 

    def inv_palkka_old(self,v,emp):
        if emp==0:
            deltapalkka_old=self.deltapalkka_old
            n_palkka_old=self.n_palkka_old
        elif emp==1:
            deltapalkka_old=self.deltaemppalkka_old
            n_palkka_old=self.n_emppalkka_old
        
        q=int(np.floor((v-self.min_wage)/deltapalkka_old))
        vmin=int(max(0,min(n_palkka_old-2,q)))
        vmax=vmin+1
        w=max(0,v-self.min_wage)/deltapalkka_old-vmin # lin.approximaatio
        
        if w<0:
            print(f'w<0: {w} {v} {vmin}')
            w=0
            #raise ValueError('A very specific bad thing happened.')

        return vmin,vmax,w

    def map_palkka(self,v,emp,midpoint=False):
        if emp==0:
            deltapalkka=self.deltapalkka
        elif emp==1:
            deltapalkka=self.deltaemppalkka

        if midpoint:
            return self.min_wage+max(0,deltapalkka*(v+0.5))
        else:
            return self.min_wage+max(0,deltapalkka*v) 

    def inv_palkka(self,v,emp):
        if emp==0:
            deltapalkka=self.deltapalkka
            n_palkka=self.n_palkka
        elif emp==1:
            deltapalkka=self.deltaemppalkka
            n_palkka=self.n_emppalkka
        else:
            print('error: unknown state ',emp)
    
        q=int(np.floor((v-self.min_wage)/deltapalkka))
        vmin=int(max(0,min(n_palkka-2,q)))
        vmax=vmin+1
        w=max(0,v-self.min_wage)/deltapalkka-vmin # lin.approximaatio
        
        if w<0:
            print(f'w<0: {w} {v} {vmin}')
            w=0
            #raise ValueError('A very specific bad thing happened.')

        return vmin,vmax,w
        
    def test_map_palkka_old(self):
        '''
        debug function
        '''
        for k in range(1000,100000,1000):
            vmin,vmax,w=self.inv_elake(k,emp=2)
            p2=(1-w)*self.map_elake(vmin,emp=2)+w*self.map_elake(vmax,emp=2)
            print(k,p2,vmin,vmax,w)
            
        for p in range(self.n_palkka): 
            palkka=self.map_palkka(p)
            print(palkka)

    def test_map_palkka(self,emp=1):
        print('palkka')
        wage=np.linspace(self.min_wage, self.max_wage, self.n_palkka)
        wage_old=np.linspace(self.min_wage, self.max_wage_old, self.n_palkka_old)
        pension=np.linspace(0, self.max_pension, self.n_elake)
        for x in np.arange(self.min_wage-10_000,self.max_wage+10_000,1_000):
            vmin,vmax,w=self.inv_palkka(x,emp)
            p2=(1-w)*self.map_palkka(vmin,emp)+w*self.map_palkka(vmax,emp)
            p3=(1-w)*wage[vmin]+w*wage[vmax]
            print(x,p2,p3)
        print('palkka_old')
        for x in np.arange(self.min_wage-10_000,self.max_wage_old+10_000,1_000):
            vmin,vmax,w=self.inv_palkka_old(x,emp)
            p2=(1-w)*self.map_palkka_old(vmin,emp)+w*self.map_palkka_old(vmax,emp)
            p3=(1-w)*wage_old[vmin]+w*wage_old[vmax]
            print(x,p2,p3)
            
        print('elake')
        for x in np.arange(self.min_pension-10_000,self.max_pension+10_000,1_000):
            vmin,vmax,w=self.inv_elake(x,emp=emp)
            p2=(1-w)*self.map_elake(vmin,emp=emp)+w*self.map_elake(vmax,emp=emp)
            p3=(1-w)*pension[vmin]+w*pension[vmax]
            print(x,p2,p3)

#     def map_palkka_future(self,palkka,v,med,state=1,midpoint=False):
#         #if state==0:
#         #    kerroin=self.unemp_wageshock
#         #else:
#         #    kerroin=1.0
#         if midpoint:
#             return med*(1+(v+0.5-self.midfuture)*self.deltafuture)
#         else:
#             return med*(1+(v-self.midfuture)*self.deltafuture)

    def map_palkka_future_v4(self,palkka,age,state=1,midpoints=None):
        '''
        vaihtoehtoinen tapa laskea
        '''
        midp=[1.0*(v+1)/self.n_palkka_future for v in range(self.n_palkka_future)]
        midp_w=self.env.wage_process_map(midp,palkka,age,state=state)
        rho=1.0/self.n_palkka_future
        
        #print(f'midp_w {state}',midp_w,rho)
        
        w=np.zeros(self.n_palkka_future)
        pred=self.min_wage
        for pnext in range(self.n_palkka_future-1): 
            #print(pred,midp_w[pnext])
            w[pnext]=self.env.wage_process_expect(pred,midp_w[pnext],palkka,age,state=state)/rho
            #print(w[pnext])
            pred=midp_w[pnext]

        #print(pred,1e6)
        w[self.n_palkka_future-1]=self.env.wage_process_expect(pred,10*pred,palkka,age,state=state)/rho
        #print(w[self.n_palkka_future-1])
        #print('cum',self.env.wage_process_cumulative(pred,palkka,age,state=state))
            
        #w=self.env.wage_process_map(x,palkka,age,state=state)
        return midp_w,w

    def map_palkka_future_v2(self,palkka,age,state=1,midpoint=False):
        def f(x):
            if x>self.n_palkka_future/2:
                return (0.49+x)/self.n_palkka_future
            else:
                return (0.51+x)/self.n_palkka_future
        
        if midpoint:
            x=[1.0*(v+1)/self.n_palkka_future for v in range(self.n_palkka_future)]
        else:
            if state==1: # in the large limit, it does not really matter which addition here is as long as x\in (0,1)
                x=[f(v) for v in range(self.n_palkka_future)]
            else:
                #x=[1.0*(0.50+v)/self.n_palkka_future for v in range(self.n_palkka_future)]
                x=[f(v) for v in range(self.n_palkka_future)]

        w=self.env.wage_process_map(x,palkka,age,state=state)
        return w

    def map_palkka_future_v3(self,palkka,age,state=1,midpoint=False):
        x=0.0001+np.array([0.9998*v/self.n_palkka_future for v in range(self.n_palkka_future+1)])

        w=self.env.wage_process_map(x,palkka,age,state=state)
        return w

    def test_palkka_future(self):
        for s in range(2):
            for palkka in range(1000,50000,5000):
                for v in range(self.n_palkka_future):
                    p=self.map_palkka_future(palkka,v,s)
                    qmin,qmax,ql=self.inv_palkka_future(palkka,p,s)
                    print(f'{palkka}: {p} {qmin} {qmax} {ql} {v}')

    def map_tis(self,v):
        return v

    def inv_tis(self,v):
        return int(min(self.n_tis-1,v))

    # lineaarinen approksimaatio
    def get_V_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,show=False,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t = self.map_grid_age(age)
        
        if age>self.max_age:
            return 0.0

        tismax=self.inv_tis(time_in_state)
        
        if emp==2:
            tismax=0
            x = np.linspace(self.min_oapension,self.max_oapension,self.n_oapension)
            y = self.oaHila[t,:]
            if self.oamonotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x,y,fill_value="extrapolate",kind=self.spline_approx)
                    
            if self.minspline:
                V1=max(y[0],f(max(self.min_oapension,elake))) # max tarkoittaa käytännössä takuueläkettä
            else:
                V1=max(0,f(max(self.min_oapension,elake))) # max tarkoittaa käytännössä takuueläkettä
        else:    
            emin,emax,we=self.inv_elake(elake,emp=emp)
            pmin,pmax,wp=self.inv_palkka_old(old_wage,emp)

            if self.pw_bivariate:
                if emp==0:
                    values=(1-wp)*self.unempHila[t,pmin,:,tismax,:]+wp*self.unempHila[t,pmax,:,tismax,:]
                elif emp==1:
                    values=(1-wp)*self.empHila[t,pmin,:,:]+wp*self.empHila[t,pmax,:,:]
            else:
                if emp==0:
                    values=(1-we)*self.unempHila[t,:,emin,tismax,:]+we*self.unempHila[t,:,emax,tismax,:]
                elif emp==1:
                    values=(1-we)*self.empHila[t,:,emin,:]+we*self.empHila[t,:,emax,:]
            
            V1=self.get_V_values(emp,elake,wage,old_wage,values)

        #if show:      
        #    print(f'getV({emp},{elake},{old_wage},{wage}): p2min {p2min} p2max {p2max} wp2 {wp2})')
        #    print(self.Hila[t,pmin,emin,emp,tismax,p2min],self.Hila[t,pmin,emin,emp,tismax,p2max])
            
        V=np.maximum(0,V1)

        return V
    

    # lineaarinen approksimaatio
    def get_V_vector_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wages=None,show=False,age=None,debug=False,sum=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        #if t>self.n_time:
        #    return 0
            
        Vs=np.zeros(wages.shape)
        
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t=self.map_grid_age(age)
        
        if emp==2:
            tismax=0
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaHila[t,:]
            if self.oamonotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            if self.minspline:
                Vs[:]=np.maximum(y[0],f(elake))
            else:
                Vs[:]=np.maximum(0,f(elake))
        else:
            emin,emax,we=self.inv_elake(elake,emp=emp)
            pmin,pmax,wp=self.inv_palkka_old(old_wage,emp)
            tismax=self.inv_tis(time_in_state)
            p = np.linspace(0, self.max_pension, self.n_elake)
            if emp==0:
                max_wage=self.max_wage
                max_wage_old=self.max_wage_old
                n_palkka=self.n_palkka
                n_palkka_old=self.n_palkka_old
            elif emp==1:
                max_wage=self.max_empwage
                max_wage_old=self.max_empwage_old
                n_palkka=self.n_emppalkka
                n_palkka_old=self.n_emppalkka_old

            w = np.linspace(self.min_wage, max_wage, n_palkka)
            w_old = np.linspace(self.min_wage, max_wage_old, n_palkka_old)

            if self.pw_bivariate:
                if emp==0:
                    values=(1-wp)*self.unempHila[t,pmin,:,tismax,:]+wp*self.unempHila[t,pmax,:,tismax,:]
                elif emp==1:
                    values=(1-wp)*self.empHila[t,pmin,:,:]+wp*self.empHila[t,pmax,:,:]
                else:
                    error(1)
                    
                g=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                if self.monotone_spline:
                    g_extra = PchipInterpolator(w,g(elake,w)[0,:],extrapolate=True)
                else:
                    g_extra = interp1d(w,g(elake,w),fill_value="extrapolate",kind='linear')
                
                for ind,wage in enumerate(wages):
                    if elake>self.max_pension and self.extrapolate:
                        if wage>max_wage and self.extrapolate:
                            vh=np.zeros(n_palkka)
                            for k,wg in enumerate(w):
                                gw=interp1d(p,values[:,k],fill_value="extrapolate",kind='linear') #self.spline_approx)
                                vh[k]=gw(elake)
                            h=interp1d(w,vh,fill_value="extrapolate",kind='linear') #self.spline_approx)
                            if self.minbispline:
                                Vs[ind] = max(values[0,0],np.squeeze(h(wage)))
                            else:
                                Vs[ind] = max(0.0,np.squeeze(h(wage)))
                        else:
                            gq=interp1d(p,g(p,wage)[:,0],fill_value="extrapolate",kind='linear') #self.spline_approx)
                            if self.minbispline:
                                Vs[ind] = max(values[0,0],np.squeeze(gq(elake)))
                            else:
                                Vs[ind] = max(0.0,np.squeeze(gq(elake)))
                    else:
                        if wage>max_wage and self.extrapolate:
                            # entä jos old_wage>max_wage??
                            #g_extra=interp1d(w,g(elake,w),fill_value="extrapolate",kind='linear') #self.spline_approx)                
                            V1 = np.squeeze(g_extra(wage))
                        else:
                            V1 = np.squeeze(g(elake,wage))
                        
                        if self.minbispline:
                            Vs[ind]=max(values[0,0],V1)
                        else:
                            Vs[ind]=max(0,V1)
            else:
                if emp==0:
                    values=(1-we)*self.unempHila[t,:,emin,tismax,:]+we*self.unempHila[t,:,emax,tismax,:]
                elif emp==1:
                    values=(1-we)*self.empHila[t,:,emin,:]+we*self.empHila[t,:,emax,:]
                else:
                    error(1)
                g=RectBivariateSpline(w_old,w,values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                if self.monotone_spline:
                    g_extra = PchipInterpolator(w,g(old_wage,w)[0,:],extrapolate=True)
                else:
                    g_extra = interp1d(w,g(old_wage,w),fill_value="extrapolate",kind='linear')
                num=0
                for ind,wage in enumerate(wages):
                    if old_wage>max_wage_old and self.extrapolate:
                        if wage>max_wage:
                            num=num+1
                            #print(num)
                            vh=np.zeros(n_palkka)
                            for k,wg in enumerate(w):
                                gw=interp1d(w_old,values[:,k],fill_value="extrapolate",kind='linear') #self.spline_approx)
                                vh[k]=gw(old_wage)
                            h=interp1d(w,vh,fill_value="extrapolate",kind='linear') #self.spline_approx)
                            if self.minbispline:
                                Vs[ind] = max(values[0,0],np.squeeze(h(wage)))
                            else:
                                Vs[ind] = max(0.0,np.squeeze(h(wage)))
                        else:
                            if self.monotone_spline:
                                hwx = PchipInterpolator(w_old,g(w_old,wage)[:,0],extrapolate=True)
                            else:
                                hwx=interp1d(w_old,g(w_old,wage)[:,0],fill_value="extrapolate",kind='linear') #self.spline_approx)
                            if self.minbispline:
                                Vs[ind] = max(values[0,0],np.squeeze(hwx(old_wage)))
                            else:
                                Vs[ind] = max(0.0,np.squeeze(hwx(old_wage)))
                    else:
                        if wage>max_wage and self.extrapolate:
                            if self.minbispline:
                                Vs[ind] = max(values[0,0],np.squeeze(g_extra(wage)))
                            else:
                                Vs[ind] = max(0.0,np.squeeze(g_extra(wage)))
                        else:
                            V1 = np.squeeze(g(old_wage,wage))
                            if self.minbispline:
                                Vs[ind]=max(values[0,0],V1)
                            else:
                                Vs[ind]=max(0.0,V1)
                    #if debug:
                    #    print(f'w {wage} V1 {V1}')

        #if show:      
        #    print(f'getV({emp},{elake},{old_wage},{wage}): p2min {p2min} p2max {p2max} wp2 {wp2})')
        #    print(self.Hila[t,pmin,emin,emp,tismax,p2min],self.Hila[t,pmin,emin,emp,tismax,p2max])

        if sum:
            return np.sum(Vs)
        else:
            return Vs

    def map_grid_age(self,age):
        return int(np.round(age-self.min_grid_age))

    def plot_Hila(self,age,emp=1,time_in_state=1,diff=False):
        x=np.arange(self.min_wage,100000,1000)
        q=np.zeros(x.shape)
        t=self.map_grid_age(age)    
        
        fig,ax=plt.subplots()
        if emp==2:
            if diff:
                for l in range(self.n_oapension):
                    k=0
                    elake=self.map_elake(l,emp=2)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=emp,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)-self.get_V(age=age,emp=0,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
            else:
                for l in range(self.n_oapension):
                    k=0
                    elake=self.map_elake(l,emp=emp)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=emp,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
        else:
            if diff:
                for l in range(self.n_elake):
                    k=0
                    elake=self.map_elake(l)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=1,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)-self.get_V(age=age,emp=0,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
            else:
                for l in range(self.n_elake):
                    k=0
                    elake=self.map_elake(l,emp=emp)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=emp,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
                    #print(q)
            
        ax.set_xlabel('palkka')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()
                
    def plot_actHila(self,age,l=5,emp=1,time_in_state=1,diff=False,act=0,emp2=1):
        x=np.arange(0,100000,1000)
        q=np.zeros(x.shape)
        t=self.map_grid_age(age)    
        
        fig,ax=plt.subplots()
        if diff:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l,emp=emp)
                for palkka in x:
                    q[k]=self.get_actV(emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=emp2,age=age)-self.get_actV(emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=0,age=age)
                    k=k+1
            
                plt.plot(x,q,label=elake)
        else:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l,emp=emp)
                for palkka in x:
                    q[k]=self.get_actV(emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=act,age=age)
                    k=k+1
            
                plt.plot(x,q,label=elake)
            
        ax.set_xlabel('palkka')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()
        
    def get_V_values(self,emp,elake,wage,old_wage,values):
        if emp==0:
            max_wage=self.max_wage
            max_wage_old=self.max_wage_old
            n_palkka=self.n_palkka
            n_palkka_old=self.n_palkka_old
        elif emp==1:
            max_wage=self.max_empwage
            max_wage_old=self.max_empwage_old
            n_palkka=self.n_emppalkka
            n_palkka_old=self.n_emppalkka_old
    
        if self.pw_bivariate:
            p = np.linspace(0, self.max_pension, self.n_elake)
            w = np.linspace(self.min_wage, max_wage, n_palkka)
            w_old = np.linspace(self.min_wage, max_wage_old, n_palkka_old)
                
            if elake>self.max_pension and self.extrapolate:
                if wage>max_wage and self.extrapolate:
                    vh=np.zeros(n_palkka)
                    for k,wg in enumerate(w):
                        g=interp1d(p,values[:,k],fill_value="extrapolate",kind='linear') #self.spline_approx)
                        vh[k]=g(elake)
                    h=interp1d(w,vh,fill_value="extrapolate",kind='linear') #self.spline_approx)
                    if self.minbispline:
                        V1 = max(values[0,0],np.squeeze(h(wage)))
                    else:
                        V1 = max(0.0,np.squeeze(h(wage)))
                else:
                    f=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline) # default kx=3,ky=3 DOES NOT EXTRAPOLATE!
                    g=interp1d(p,f(p,wage)[:,0],fill_value="extrapolate",kind='linear') #self.spline_approx)
                    if self.minbispline:
                        V1 = max(values[0,0],np.squeeze(g(elake)))
                    else:
                        V1 = max(0.0,np.squeeze(g(elake)))
            else:
                if wage>max_wage and self.extrapolate:
                    f=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline) # default kx=3,ky=3
                    g=interp1d(w,f(elake,w),fill_value="extrapolate",kind='linear') #self.spline_approx)
                    if self.minbispline:
                        V1 = max(values[0,0],np.squeeze(g(wage)))
                    else:
                        V1 = max(0.0,np.squeeze(g(wage)))
                else:
                    try:
                        g=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline) # default kx=3,ky=3 DOES NOT EXTRAPOLATE!
                    except:
                        print(p.shape,w.shape,values.shape)
                        g=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline) # default kx=3,ky=3 DOES NOT EXTRAPOLATE!
                    if self.minbispline:
                        V1 = max(values[0,0],np.squeeze(g(elake,wage)))
                    else:
                        V1 = max(0.0,np.squeeze(g(elake,wage)))
        else:
            #p = np.linspace(0, self.max_pension, self.n_elake)
            w = np.linspace(self.min_wage, max_wage, n_palkka)
            w_old = np.linspace(self.min_wage, max_wage_old, n_palkka_old)
            f=RectBivariateSpline(w_old,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
            if old_wage>max_wage_old and self.extrapolate:
                if wage>max_wage:
                    vh=np.zeros(n_palkka)
                    for k,wg in enumerate(w):
                        g=interp1d(w_old,values[:,k],fill_value="extrapolate",kind='linear') #self.spline_approx)
                        vh[k]=g(old_wage)
                    h=interp1d(w,vh,fill_value="extrapolate",kind='linear') #self.spline_approx)
                    if self.minbispline:
                        V1 = np.maximum(values[0,0],np.squeeze(h(wage)))
                    else:
                        V1 = np.squeeze(h(wage))
                else:
                    g=interp1d(w_old,f(w_old,wage)[:,0],fill_value="extrapolate",kind='linear') #self.spline_approx)
                    if self.minbispline:
                        V1 = np.maximum(values[0,0],np.squeeze(g(old_wage)))
                    else:
                        V1 = np.squeeze(g(old_wage))
            else:
                if wage>max_wage and self.extrapolate:
                    # entä jos old_wage>max_wage??
                    if self.monotone_spline:
                        g_extra = PchipInterpolator(w,f(old_wage,w)[0,:],extrapolate=True)
                    else:
                        g_extra = interp1d(w,f(old_wage,w),fill_value="extrapolate",kind='linear')
            
                    if self.minbispline:
                        V1 = np.squeeze(np.maximum(values[0,0],g_extra(wage)))
                    else:
                        V1 = np.squeeze(g_extra(wage))
                else:
                    if self.minbispline:
                        V1 = np.squeeze(np.maximum(values[0,0],f(old_wage,wage)))
                    else:
                        V1 = np.squeeze(f(old_wage,wage))
                        
        return V1

    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_actV_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,act=None,age=None,debug=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if s is not None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
        
        t=self.map_grid_age(age)
        
        #print(t,emp,elake,old_wage,age,time_in_state,wage)
            
        if emp==2:
            tismax=0
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaactHila[t,:]
            if self.oamonotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            if self.minspline:
                V1=max(y[0],f(elake))
            else:
                V1=max(0,f(elake))
        else:
            emin,emax,we=self.inv_elake(elake,emp=emp)
            pmin,pmax,wp=self.inv_palkka_old(old_wage,emp)
            p2min,p2max,wp2=self.inv_palkka(wage,emp)        
            tismax=self.inv_tis(time_in_state)
            
            if self.pw_bivariate:
                if emp==0:
                    values=(1-wp)*self.unempActHila[t,pmin,:,tismax,:,act]+wp*self.unempActHila[t,pmax,:,tismax,:,act]
                elif emp==1:
                    values=(1-wp)*self.empActHila[t,pmin,:,:,act]+wp*self.empActHila[t,pmax,:,:,act]
            else:
                if emp==0:
                    values=(1-we)*self.unempActHila[t,:,emin,tismax,:,act]+we*self.unempActHila[t,:,emax,tismax,:,act]
                elif emp==1:
                    values=(1-we)*self.empActHila[t,:,emin,:,act]+we*self.empActHila[t,:,emax,:,act]
            
            V1=self.get_V_values(emp,elake,wage,old_wage,values)
            
        if debug:
            if wp2<0 or wp2>1:
                print('actV: emp {} elake {} old_wage {} wage {} tis {}: wp2 {}'.format(emp,elake,old_wage,wage,time_in_state,wp2))
            if wp<0 or wp>1:
                print('actV: emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,wp))
            if we<0 or we>1:
                print('actV: emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,we))
        
        #V=max(0.0,V1)
            
        #act=int(np.argmax(V))
        #maxV=np.max(V)

        return ax(0.0,V1)
        
    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_actReward_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,act=None,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t = self.map_grid_age(age)
        
        if age>self.max_age:
            return 0.0

        if emp==2:
            tismax=0
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaactReward[t,:]
            if self.oamonotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            #f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            R = np.squeeze(f(elake))
        else:
            emin,emax,we=self.inv_elake(elake,emp=emp)
            pmin,pmax,wp=self.inv_palkka_old(old_wage,emp)
            #p2min,p2max,wp2=self.inv_palkka(wage)        
            tismax=self.inv_tis(time_in_state)

            if self.pw_bivariate:
                if emp==0:
                    values=(1-wp)*self.unempActReward[t,pmin,:,tismax,:,act]+wp*self.unempActReward[t,pmax,:,tismax,:,act]
                elif emp==1:
                    values=(1-wp)*self.empActReward[t,pmin,:,:,act]+wp*self.empActReward[t,pmax,:,:,act]
            else:
                if emp==0:
                    values=(1-we)*self.unempActReward[t,:,emin,tismax,:,act]+we*self.unempActReward[t,:,emax,tismax,:,act]
                elif emp==1:
                    values=(1-we)*self.empActReward[t,:,emin,:,act]+we*self.empActReward[t,:,emax,:,act]
            
            R=self.get_V_values(emp,elake,wage,old_wage,values)
            
                    
        return R
    
    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_act_spline(self,s,full=False,debug=False,use_softmax=False,T=0.001):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''

        emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
        
        if debug:
            print(f'emp {emp} elake {elake} old_wage {old_wage} age {age} time_in_state {time_in_state} wage {wage}')
        
        t=self.map_grid_age(age)
        
        n_emp=self.n_acts
        
        V=np.zeros(n_emp)
        #emp_set=set([0,1,3])
        emp_set=set([0,1])
        if emp in emp_set:
            if age<self.min_retirementage-1:
                n_emp=3
                act_set=set([0,1])
                #n_emp=4
                #act_set=set([0,1,3])
            else:
                act_set=set([0,1,2])
                n_emp=3
                #act_set=set([0,1,2,3])
                #n_emp=4
        else:
            act_set=set([0])
            
        if emp == 2:
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaactHila[t,:]
            if self.oamonotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            
            #V[0] = f(elake)
            if self.minspline:
                V[0] = max(y[0],f(elake))
            else:
                V[0] = f(elake)
        else:
            emin,emax,we=self.inv_elake(elake,emp=emp)
            pmin,pmax,wp=self.inv_palkka_old(old_wage,emp)
            p2min,p2max,wp2=self.inv_palkka(wage,emp)
            tismax=self.inv_tis(time_in_state)
        
            for k in act_set:
                if self.pw_bivariate:
                    if emp==0:
                        values=(1-wp)*self.unempActHila[t,pmin,:,tismax,:,k]+wp*self.unempActHila[t,pmax,:,tismax,:,k]
                    elif emp==1:
                        values=(1-wp)*self.empActHila[t,pmin,:,:,k]+wp*self.empActHila[t,pmax,:,:,k]
                else:
                    if emp==0:
                        values=(1-we)*self.unempActHila[t,:,emin,tismax,:,k]+we*self.unempActHila[t,:,emax,tismax,:,k]
                    elif emp==1:
                        values=(1-we)*self.empActHila[t,:,emin,:,k]+we*self.empActHila[t,:,emax,:,k]
            
                V[k]=self.get_V_values(emp,elake,wage,old_wage,values)
            
        if use_softmax:
            actV=softmax(V/T)
            act=np.random.choice(len(V),p=actV)
        else:
            act=int(np.argmax(V))
            
        maxV=np.max(V)
        
        reward=self.get_actReward(s=s,act=act)
        
        if full:
            rs=[self.get_actReward(s=s,act=a) for a in act_set]
            return act,maxV,V,reward,rs
        else:
            return act,maxV,reward
            
    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_random_act(self,s,full=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
        #emin,emax,we=self.inv_elake(elake,emp=emp)
        #pmin,pmax,wp=self.inv_palkka(old_wage,emp)
        #p2min,p2max,wp2=self.inv_palkka(wage,emp)
        #tismax=self.inv_tis(time_in_state)
        
        #t=self.map_grid_age(age)
        
        n_emp=self.n_acts
        
        V=np.zeros(n_emp)
        #emp_set=set([0,1,3])
        emp_set=set([0,1])
        if emp in emp_set:
            if age<self.min_retirementage-1:
                n_emp=3
                act_set=set([0,1])
                #n_emp=4
                #act_set=set([0,1,3])
            else:
                act_set=set([0,1,2])
                n_emp=3
                #act_set=set([0,1,2,3])
                #n_emp=4
        else:
            act_set=set([0])
            
        a_set=list(act_set)
        act_set=set(act_set)

        act=a_set[np.random.randint(len(act_set))]
        #maxV=V[act]
        
        reward=0 #self.get_actReward(s=s,act=act)
        maxV=0
        
        if full:
            return act,maxV,V,reward
        else:
            return act,maxV,reward            
        
    def get_actV_random(self,age):
        if age<self.min_retirementage-1:
            return np.random.randint(2)
        else:
            return np.random.randint(3)

    def test_salaries_v3(self,age=25,n=100,wage=20000,state=1,next_state=1,tis=0,n_future=21):
        w=np.zeros(n)

        self.n_palkka_future=n_future
        elake=10000
        s0=self.env.state_encode(state,elake,wage,age-1,tis,wage)
        
        if state==next_state:
            act=0
        else:
            act=1

        for k in range(n):
            self.env.state=s0
            newstate,r,done,info=self.env.step(act,dynprog=False)
            _,_,_,_,_,next_wage=self.env.state_decode(newstate)
            w[k]=next_wage
        
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins=200
        
        ave=np.mean(w)
        

        # plot the cumulative histogram
        n, bins, patches = ax.hist(w, n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')        
        
        z=np.zeros(bins.shape)
        q=np.zeros(bins.shape)
        for k,x in enumerate(bins):
            z[k]=self.env.wage_process_cumulative(x,wage,age,state=next_state)
            
        n_b=self.n_palkka_future
        q=self.map_palkka_future_v2(wage,age,state=next_state,midpoint=False)
        q_mid=self.map_palkka_future_v2(wage,age,state=next_state,midpoint=True)
        
        ax.plot(bins, z, 'k--', linewidth=1.5, label='Theoretical')

        for k in range(n_b):
            ax.axvline(q[k],color='r',ls='--')
            ax.axvline(q_mid[k],color='r',ls='dotted')

        med=self.env.wage_process_mean(wage,age,state=next_state)
        ax.axvline(med,color='b')

        #print(f'mean predicted {med} obserrved {ave}')

        # tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('Cumulative step histograms')
        ax.set_xlabel('Wage (e/y)')
        ax.set_ylabel('Likelihood of occurrence')

        plt.show()

    def test_salaries_v2(self,age=25,n=100,wage=20000,state=1,tis=0):
        w=np.zeros(n)

        elake=10000
        s0=self.env.state_encode(state,elake,wage,age-1,tis,wage)
        act=0

        for k in range(n):
            self.env.state=s0
            newstate,r,done,info=self.env.step(act,dynprog=False)
            _,_,_,_,_,next_wage=self.env.state_decode(newstate)
            w[k]=next_wage
        
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins=200

        # plot the cumulative histogram
        n, bins, patches = ax.hist(w, n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')        
        
        z=np.zeros(bins.shape)
        for k,x in enumerate(bins):
            z[k]=self.env.wage_process_cumulative(x,wage,age,state=state)
        
        ax.plot(bins, z, 'k--', linewidth=1.5, label='Theoretical')

        # tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('Cumulative step histograms')
        ax.set_xlabel('Wage (e/y)')
        ax.set_ylabel('Likelihood of occurrence')

        plt.show()
            
    def test_salaries(self,age=25,n=100,wage=20000,state=1):
        w=np.zeros(n)
        for k in range(n):
            w[k]=self.env.get_wage_raw(age,wage,state)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins=200

        # plot the cumulative histogram
        n, bins, patches = ax.hist(w, n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')        
        
        z=np.zeros(bins.shape)
        for k,x in enumerate(bins):
            z[k]=self.env.wage_process_cumulative(x,wage,age,state=state)
        
        ax.plot(bins, z, 'k--', linewidth=1.5, label='Theoretical')

        # tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('Cumulative step histograms')
        ax.set_xlabel('Wage (e/y)')
        ax.set_ylabel('Likelihood of occurrence')
        med=self.env.wage_process_mean(wage,age,state=state)
        ax.axvline(med,color='r')
        for k in range(7):
            ax.axvline(med*(1-0.07*0.5*k),color='r',ls='--')
            ax.axvline(med*(1+0.07*0.5*k),color='r',ls='--')
        plt.show()

    def get_dpreward(self,emp=1,elake=10000,ow=10000,ika=50,tis=1,w=10000,action=0):
        actions=[action]
        r,sps=self.get_rewards_continuous((emp,elake,ow,ika,tis,w),actions,debug=False)
        return r

    # this routine is needed for the dynamic programming
    def print_rewards_continuous(self,age=19,emp=1,tis=1.0,elake=1000.0,palkka=2000.0,palkka_next=3000.0,a=1,debug=False):
        
        s=(emp,elake,palkka,age,tis,palkka_next)
        
        rewards=[]
        sps=[]
        
        start_state=self.env.state_encode(*s)
        #self.env.render(state=start_state)
        self.env.state=start_state
        newstate,reward,dones,info = self.env.step(a,dynprog=True)
        #if dones:
        #    self.reset()
        sps.append(np.array(newstate))
        rewards.append(reward)
        

        print('start_state '.format(self.env.render(state=start_state)))        
        print('end_state '.format(self.env.render(state=newstate)))        
        print('reward {}'.format(reward))
        
        #return rewards,sps

    # this routine is needed for the dynamic programming
    def get_rewards_continuous(self,s,actions,debug=False):
        rewards=[]
        sps=[]
        
        start_state=self.env.state_encode(*s)
        #self.env.render(state=start_state)
        for a in actions:
            self.env.state=start_state
            newstate,reward,dones,info = self.env.step(a,dynprog=True)
            #if dones:
            #    self.reset()
            sps.append(np.array(newstate))
            rewards.append(reward)
            
        return rewards,sps
        
    def get_wage_arrays(self,emp,age):
        if emp==0:
            max_wage=self.max_wage
            max_wage_old=self.max_wage_old
            n_palkka=self.n_palkka
            n_palkka_old=self.n_palkka_old
        elif emp==1:
            max_wage=self.max_empwage
            max_wage_old=self.max_empwage_old
            n_palkka=self.n_emppalkka
            n_palkka_old=self.n_emppalkka_old
    
        #tic=time.perf_counter()
        pn_weight=np.zeros((n_palkka,self.n_palkka_future,self.n_employment))
        wagetable=np.zeros(n_palkka)
        wagetable_old=np.zeros(n_palkka_old)
        wagetable_future=np.zeros((n_palkka,self.n_palkka_future,self.n_employment))
        tr_wagetable_future=np.zeros((n_palkka,self.n_palkka_future+1,self.n_employment))
        ika2=age+1
        #print('age',age)

        for p in range(n_palkka_old): 
            wagetable_old[p]=self.map_palkka_old(p,emp)
        
        for p in range(n_palkka): 
            palkka=self.map_palkka(p,emp)
            wagetable[p]=palkka
            weight_old_s0=0
            weight_old_s1=0

            if self.use_v4:        
                palkka_next_mid0_v,wagetable_future[p,:,0]=self.map_palkka_future_v4(palkka,ika2,state=0)
                palkka_next_mid1_v,wagetable_future[p,:,1]=self.map_palkka_future_v4(palkka,ika2,state=1)
                #vrt0=self.map_palkka_future_v2(palkka,ika2,state=0,midpoint=False)
                #vrt1=self.map_palkka_future_v2(palkka,ika2,state=1,midpoint=False)
                #print('future1',wagetable_future[p,:,1])
                #print('v2_1:',vrt1)
                #print('future0',wagetable_future[p,:,0])
                #print('v2_0:',vrt0)
            else:
                palkka_next_mid0_v=self.map_palkka_future_v2(palkka,ika2,state=0,midpoint=True)
                palkka_next_mid1_v=self.map_palkka_future_v2(palkka,ika2,state=1,midpoint=True)
                wagetable_future[p,:,0]=self.map_palkka_future_v2(palkka,ika2,state=0,midpoint=False)
                wagetable_future[p,:,1]=self.map_palkka_future_v2(palkka,ika2,state=1,midpoint=False)

            #print(wagetable_future.shape,palkka_next_mid0_v.shape)


            #rho=1.0/self.n_palkka_future

#             wns0=self.env.wage_process_cumulative(palkka_next_mid0_v,palkka,ika2,state=0)
#             wns1=self.env.wage_process_cumulative(palkka_next_mid1_v,palkka,ika2,state=1)
#             print(wns0)
#             print(wns1)
#             pred_sal0=0
#             for pnext in range(self.n_palkka_future-1): 
#                 palkka_next_mid0=palkka_next_mid0_v[pnext]
#                 weight_new_s0=wns0[pnext]
#                 pn_weight[p,pnext,0]=weight_new_s0-weight_old_s0
#                 weight_old_s0=weight_new_s0
#                 #print(f'p {p} pnext {pnext} mid {palkka_next_mid0} w {pn_weight[p,pnext,0]}')
#                 #print(f'wage {wagetable_future[p,pnext,0]} vs {palkka_next_mid0_v[pnext]}')
#                 #print(self.env.wage_process_expect(pred_sal0,palkka_next_mid0_v[pnext],palkka,ika2,state=0))
#                 pred_sal0=palkka_next_mid0
#                 
#                 palkka_next_mid1=palkka_next_mid1_v[pnext]
#                 weight_new_s1=wns1[pnext] # oli wns0 !!
#                 pn_weight[p,pnext,1]=weight_new_s1-weight_old_s1
#                 weight_old_s1=weight_new_s1
#     
#             pn_weight[p,self.n_palkka_future-1,0]=1.0-weight_old_s0
#             pn_weight[p,self.n_palkka_future-1,1]=1.0-weight_old_s1

            #ave=self.env.wage_process_mean(palkka,ika2,state=0)
            #ave2=np.sum(wagetable_future[p,:,0]*pn_weight[p,:,0])
            #print(f'p {p} pnext {self.n_palkka_future-1} mid {palkka_next_mid0_v[self.n_palkka_future-1]} w {pn_weight[p,self.n_palkka_future-1,0]}')
            #print(f'wage {wagetable_future[p,-1,0]}')
            #print(f'mean {ave} obs {ave2}')

        pn_weight[:,0,2]=1.0
    #toc=time.perf_counter()
    #print('time 1',toc-tic)    
    
        return pn_weight,wagetable,wagetable_old,wagetable_future,tr_wagetable_future


    def backtrack(self,age,debug=False):
        '''
        Dynaaminen ohjelmointi hilan avulla
        '''
        t=self.map_grid_age(age)
        
        if t==0:
            t0=0
        else:  
            t0=0
        
        #def func0(x,emp2,elake2,palkka,tis2,ika2):
        #    a=self.get_V(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
        #    b=self.env.wage_process_pdf(x,palkka,ika2,state=emp2)
        #    return a*b

        #def func1(x,emp2,elake2,palkka,tis2,ika2):
        #    a=self.get_V(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
        #    b=self.env.wage_process_pdf(x,palkka,ika2,state=emp2)
        #    c=a*b
        #    return a*b
        
        # actions when active
        if age<self.min_retirementage:
            if self.include_pt:
                act_set=set([0,1,3])
            else:
                act_set=set([0,1])
        else:
            if self.include_pt:
                act_set=set([0,1,2,3])
            else:
                act_set=set([0,1,2])
        
        ret_set=set([0]) # actions in retirement
        #stay_set=set([0]) # stay put
        
        #print('backtrack')
        
        if debug: # and age<70:
            tulosta=True
        else:
            tulosta=False
            
        #tic=time.perf_counter()
        unemp_pn_weight,unemp_wagetable,unemp_wagetable_old,unemp_wagetable_future,unemp_tr_wagetable_future=self.get_wage_arrays(0,age)
        emp_pn_weight,emp_wagetable,emp_wagetable_old,emp_wagetable_future,emp_tr_wagetable_future=self.get_wage_arrays(1,age)
        rho=1.0/self.n_palkka_future
        
        ika2=age+1
        #print('age',age)

        for emp in range(self.n_employment):
            if emp==2:
                if age<self.min_retirementage:
                    self.oaHila[t,:]=0
                    self.oaactHila[t,:]=0
                    self.oaactReward[t,:]=0
                else:
                    #tic=time.perf_counter()

                    time_in_state=self.map_tis(0)
                    for el in range(self.n_oapension):
                        elake=self.map_elake(el,emp=2)

                        # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                        rts,Sps=self.get_rewards_continuous((emp,elake,0,age,time_in_state,0),ret_set)
                        
                        for ind,a in enumerate(ret_set):
                            emp2,elake2,_,ika2,_,_=self.env.state_decode(Sps[ind])
                            self.oaactHila[t,el]=rts[ind]+self.gamma*self.get_V(emp=emp2,elake=elake2,old_wage=self.min_wage,wage=self.min_wage,time_in_state=0,age=ika2)
                            self.oaactReward[t,el]=rts[ind]

                        self.oaHila[t,el]=self.oaactHila[t,el]
                    #toc=time.perf_counter()
                    #print('time state 2',toc-tic)

            elif emp==1:
                #tic=time.perf_counter()
                time_in_state=self.map_tis(0)
                for el in range(self.n_elake):
                    elake=self.map_elake(el)
                    for p_old in range(self.n_emppalkka_old): 
                        palkka_vanha=emp_wagetable_old[p_old]
                        for p in range(self.n_emppalkka): 
                            palkka=emp_wagetable[p]
                            # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                            rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                            
                            for ind,a in enumerate(act_set):
                                emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                if emp2==2:
                                    d=self.get_V(emp=emp2,elake=elake2,old_wage=0,time_in_state=0,age=ika2,wage=0)
                                else:
                                    #if self.integmethod==0:
                                    gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,
                                            wages=emp_wagetable_future[p,:,emp2],sum=True)
                                    #w=emp_pn_weight[p,:,emp2]
                                    #d=np.sum(gw*w)
                                    
                                    d=gw*rho
                                    
                                    #if True:
                                    #    mean_sal=np.sum(w*emp_wagetable_future[p,:,emp2])
                                    #    mean_predsal=self.env.wage_process_mean(palkka,ika2,state=emp2)
                                    #    delta=mean_sal-mean_predsal
                                        #print('emp 1:',mean_sal,mean_predsal,delta,delta/mean_sal)
                                        
                                        #if p==self.n_palkka-1:
                                        #    gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2],debug=True)
                                        #    print(f'({p} {el}): d={d}\ngw={gw}\nw={w}\n')
                                    #elif self.integmethod==1:
                                    #    d,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                    #elif self.integmethod==2:
                                    #    gw2=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                    #    b=self.env.wage_process_pdf(tr_wagetable_future[p,:,emp2],palkka,ika2,state=emp2)
                                    #    d=trapz(gw2*b,x=tr_wagetable_future[p,:,emp2])
                                    #else: # compare
                                    #    d1,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                    #    gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                    #    w=pn_weight[p,:,emp2]
                                    #    d2=np.sum(gw*w) 
                                    #    delta=d2-d1
                                    #    d=d1
                                    #    #if age==69 and palkka==1000.0:
                                    #    #    print(wagetable_future[p,:,emp])
                                    #        
                                    #    if np.abs(delta)>0.0001:
                                    #        print(f'em1={emp},emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{d2:.5f} vs {d1:.5f} {delta}')
                                    #        d1,_=quad(func1, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))

                                q=rts[ind]+self.gamma*d
                                #if p==0 and p_old==0 and el>=6:
                                #    print(p,p_old,el,rts,self.env.state_decode(Sps[ind]),q)

                                self.empActHila[t,p_old,el,p,a]=q
                                self.empActReward[t,p_old,el,p,a]=rts[ind]
                                
                            #self.Hila[t,p_old,el,emp,:,p]=np.max(self.actHila[t,p_old,el,emp,0,p,:])
                            self.empHila[t,p_old,el,p]=np.max(self.empActHila[t,p_old,el,p,:])

                            #if p==0 and p_old==0:
                            #    print(t+self.min_age,p,p_old,el,np.max(self.actHila[t,p_old,el,emp,0,p,:]))
                                
                #toc=time.perf_counter()
                #print('time state 1',toc-tic)

            elif emp==0:
                #tic=time.perf_counter()
                for p_old in range(self.n_palkka_old):
                    palkka_vanha=unemp_wagetable_old[p_old]
                    for el in range(self.n_elake):
                        elake=self.map_elake(el)
                        for tis in range(t0,self.n_tis):
                            time_in_state=self.map_tis(tis)
                            for p in range(self.n_palkka): 
                                palkka=unemp_wagetable[p]
                                # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                                rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                    
                                for ind,a in enumerate(act_set):
                                    emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                    if emp2==2:
                                        d=self.get_V(emp=emp2,elake=elake2,old_wage=0,time_in_state=0,age=ika2,wage=0)
                                    else:
                                    #    if self.integmethod==0:
                                        gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,
                                                            wages=unemp_wagetable_future[p,:,emp2],sum=True)
                                        #w=unemp_pn_weight[p,:,emp2]
                                        #d=np.sum(gw*w) 
                                        
                                        d=gw*rho
                                        
                                        #if True:
                                        #    mean_sal=np.sum(w*unemp_wagetable_future[p,:,emp2])
                                        #    mean_predsal=self.env.wage_process_mean(palkka,ika2,state=emp2)
                                        #    delta=mean_sal-mean_predsal
                                            #print('emp 0:',mean_sal,mean_predsal,delta,delta/mean_sal)
                                        
                                        #elif self.integmethod==1:
                                        #    d,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                        #elif self.integmethod==2:
                                        #    gw2=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                        #    b=self.env.wage_process_pdf(tr_wagetable_future[p,:,emp2],palkka,ika2,state=emp2)
                                        #    d=trapz(gw2*b,x=tr_wagetable_future[p,:,emp2])
                                        #else: # compare
                                        #    d1,_=quad(func0, 1000.0, min(10*palkka,130_000.0),epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                        #    gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                        #    w=pn_weight[p,:,emp2]
                                        #    d2=np.sum(gw*w)
                                        #    delta=d2-d1
                                        #    d=d1
                                        #    if np.abs(delta)>0.0001:
                                        #        print(f'em1={emp},emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{d2:.5f} vs {d1:.5f} {delta}')
                                        #        d1,_=quad(func1, 1000.0,min(10*palkka,130_000.0),epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))

                                    q=rts[ind]+self.gamma*d

                                    #self.actHila[t,p_old,el,emp,tis,p,a]=q
                                    #self.actReward[t,p_old,el,emp,tis,p,a]=rts[ind]
                                    self.unempActHila[t,p_old,el,tis,p,a]=q
                                    self.unempActReward[t,p_old,el,tis,p,a]=rts[ind]
                                    
                                #self.Hila[t,p_old,el,emp,tis,p]=np.max(self.actHila[t,p_old,el,emp,tis,p,:])
                                self.unempHila[t,p_old,el,tis,p]=np.max(self.unempActHila[t,p_old,el,tis,p,:])
                #toc=time.perf_counter()
                #print('time state 0',toc-tic)
            else:
                print('unknown state ',emp)

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
        tqdm_e = tqdm(range(int(self.n_time-1)), desc='Score', leave=True, unit=" year")

        for age in range(self.max_age,self.min_age-1,-1):
            t=self.map_grid_age(age)    
            self.backtrack(age,debug=debug)
            tqdm_e.set_description("Year " + str(t))
            tqdm_e.update(1)

        self.save_V(save)

    def simulate(self,debug=False,pop=1_000,save=None,load=None,ini_pension=None,ini_wage=None,ini_age=None,random_act=False,softmax=False,T=0.001,
                 startage=None):
        '''
        Lasketaan työllisyysasteet ikäluokittain simuloimalla aineisto
        '''
        if pop is not None:
            self.n_pop=pop
            
        if startage is None:
            startage=self.startage
        
        self.env.set_startage(startage)
        t0=startage-self.min_age

        n_empl=4 #self.n_employment
        self.episodestats.reset(self.timestep,self.n_time,n_empl,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year,dynprog=True)
        if load is not None:
            self.load_V(load)        
            
        print('simulate debug',debug)
        print('simulate random_act',random_act)

        tqdm_e = tqdm(range(int(pop)), desc='Population', leave=True, unit=" p")
        
        rewards_pred=np.zeros((pop,self.n_time))
        rewards=np.zeros((pop,self.n_time))

        for n in range(pop):
            state=self.env.reset(pension=ini_pension,ini_wage=ini_wage,ini_age=ini_age)
            t=t0
            if random_act:
                act,maxV,rewards_pred[n,t]=self.get_random_act(state)
            else:
                act,maxV,v,rewards_pred[n,t],rs=self.get_act(state,full=True,use_softmax=softmax,T=T)

            self.episodestats.add(n,-1,0.0,state,state,{},debug=debug,aveV=maxV,pred_r=rewards_pred[n,t])
            
            
            for t in range(t0,self.n_time+1):
                
                if random_act:
                    act,maxV,rewards_pred[n,t+1]=self.get_random_act(state)
                else:
                    act,maxV,v,rewards_pred[n,t+1],rs=self.get_act(state,full=True,use_softmax=softmax,T=T)

                #predV[n,t+1]=maxV

                #if False and act==1:
                #    rw=rewards_pred[n,t]
                #    print(f'act {act} rws {rw} v {v}')

                #if True:
                #    self.env.render(state=state,pred_r=rewards_pred[n,t],pred_V=maxV)
                    
                newstate,r,done,info=self.env.step(act,dynprog=False)
                
                if rewards_pred[n,t+1]<0.001 and not random_act:
                    print('***',self.env.state_decode(state),'\n',self.env.state_decode(newstate))
                    act,maxV,v,pr,rs=self.get_act(state,full=True,use_softmax=softmax,T=T,debug=True)
                    print(rs,maxV)

                if debug:
                    self.env.render(state=newstate,reward=r,pred_r=rewards_pred[n,t+1])
                
                rewards[n,t+1]=r
                 
                if done: 
                    self.episodestats.add(n,act,r,state,newstate,info,debug=debug,aveV=maxV,pred_r=rewards_pred[n,t+1])
                    #print(info,r)
                    #print(newstate,info[0]['terminal_observation'])
                    tqdm_e.update(1)
                    tqdm_e.set_description("Pop " + str(n))
                    break
                else:
                    self.episodestats.add(n,act,r,state,newstate,info,debug=debug,aveV=maxV,pred_r=rewards_pred[n,t+1])
                    
                state=newstate
        
        age=max(1,startage-self.min_age)
        variance_pred=np.var(rewards_pred[:,age:])
        variance_real=np.var(rewards[:,age:])
        diff_variance=np.var(rewards[:,age:]-rewards_pred[:,age:])
        coef=1-diff_variance/variance_real
        print('Explained variance ',coef)
        print('Pred variance {} variance {} diff variance {}'.format(variance_pred,variance_real,diff_variance))
        absmax=np.abs(rewards-rewards_pred)
        if debug:
            unrind=np.unravel_index(absmax.argmax(),absmax.shape)
            print(unrind,absmax.argmax(),absmax.shape)
            age=self.map_t(unrind[1])
            print('real',rewards[unrind[0],:])
            print('pred',rewards_pred[unrind[0],:])
            print('mean real',np.mean(rewards,axis=0))
            print('mean pred',np.mean(rewards_pred,axis=0))
            print('Max diff in r {} at {} in {} state {}'.format(np.max(absmax),age,unrind,self.episodestats.popempstate[unrind[1],unrind[0]]))
            print('Max pension {} wage {}'.format(self.episodestats.infostats_pop_pension[unrind[1],unrind[0]],self.episodestats.infostats_pop_wage[unrind[1],unrind[0]],self.episodestats.infostats_poptulot_netto[unrind[1],unrind[0]]))
            
        #for n in range(pop):
        #    coef=(1-np.var(rewards[n,:-2]-rewards_pred[n,:-2]))/np.var(rewards[n,:-2])
        #    print(F'{n}: {coef}')

        
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
        ft='f16'
        f.create_dataset("unempHila", data=self.unempHila, dtype=ft)
        f.create_dataset("empHila", data=self.empHila, dtype=ft)
        f.create_dataset("min_grid_age", data=self.min_grid_age, dtype=ft)
        f.create_dataset("max_grid_age", data=self.max_grid_age, dtype=ft)
        f.create_dataset("max_wage", data=self.max_wage, dtype=ft)
        f.create_dataset("max_wage_old", data=self.max_wage_old, dtype=ft)
        f.create_dataset("max_empwage", data=self.max_empwage, dtype=ft)
        f.create_dataset("max_empwage_old", data=self.max_empwage_old, dtype=ft)
        f.create_dataset("min_wage", data=self.min_wage, dtype=ft)
        f.create_dataset("max_pension", data=self.max_pension, dtype=ft)
        f.create_dataset("unempActHila", data=self.unempActHila, dtype=ft)
        f.create_dataset("unempActReward", data=self.unempActReward, dtype=ft)
        f.create_dataset("empActHila", data=self.empActHila, dtype=ft)
        f.create_dataset("empActReward", data=self.empActReward, dtype=ft)
        f.create_dataset("n_palkka", data=self.n_palkka, dtype='i4')
        f.create_dataset("n_palkka_old", data=self.n_palkka_old, dtype='i4')
        f.create_dataset("n_emppalkka", data=self.n_emppalkka, dtype='i4')
        f.create_dataset("n_palkka_future", data=self.n_palkka_future, dtype='i4')
        f.create_dataset("n_emppalkka_old", data=self.n_emppalkka_old, dtype='i4')
        f.create_dataset("n_elake", data=self.n_elake, dtype='i4')
        f.create_dataset("n_oapension", data=self.n_oapension, dtype='i4')
        f.create_dataset("deltapalkka", data=self.deltapalkka, dtype=ft)
        f.create_dataset("deltapalkka_old", data=self.deltapalkka_old, dtype=ft)
        f.create_dataset("deltaemppalkka", data=self.deltaemppalkka, dtype=ft)
        f.create_dataset("deltaemppalkka_old", data=self.deltaemppalkka_old, dtype=ft)
        f.create_dataset("deltaelake", data=self.deltaelake, dtype=ft)
        f.create_dataset("delta_oapension", data=self.delta_oapension, dtype=ft)
        f.create_dataset("n_tis", data=self.n_tis, dtype='i4')
        f.create_dataset("deltatis", data=self.deltatis, dtype=ft)
        f.create_dataset("oaHila", data=self.oaHila, dtype=ft)
        f.create_dataset("oaactHila", data=self.oaactHila, dtype=ft)
        f.create_dataset("oaactReward", data=self.oaactReward, dtype=ft)
        f.create_dataset("min_oapension", data=self.min_oapension, dtype=ft)
        f.create_dataset("max_oapension", data=self.max_oapension, dtype=ft)
        
        f.close()
        
    def load_V(self,filename):
        f = h5py.File(filename, 'r')
        self.unempHila = f['unempHila'][()]
        self.empHila = f['empHila'][()]
        self.min_grid_age = f['min_grid_age'][()]
        self.max_grid_age = f['max_grid_age'][()]
        self.empActReward = f['empActReward'][()]
        self.empActHila = f['empActHila'][()]
        self.unempActHila = f['unempActHila'][()]
        self.unempActReward = f['unempActReward'][()]
        self.n_palkka = f['n_palkka'][()]
        self.n_palkka_old = f['n_palkka_old'][()]
        self.n_emppalkka = f['n_emppalkka'][()]
        if 'n_palkka_future' in f.keys():
            self.n_palkka_future = f['n_palkka_future'][()]
        self.n_emppalkka_old = f['n_emppalkka_old'][()]
        self.n_oapension = f['n_oapension'][()]
        self.n_elake = f['n_elake'][()]
        self.n_tis = f['n_tis'][()]
        self.deltapalkka = f['deltapalkka'][()]
        self.deltapalkka_old = f['deltapalkka_old'][()]
        self.deltaemppalkka = f['deltaemppalkka'][()]
        self.deltaemppalkka_old = f['deltaemppalkka_old'][()]
        self.delta_oapension = f['delta_oapension'][()]
        self.deltaelake = f['deltaelake'][()]
        self.deltatis = f['deltatis'][()]
        self.max_pension=f['max_pension'][()]
        self.max_wage=f['max_wage'][()]
        self.max_wage_old=f['max_wage_old'][()]
        self.max_empwage=f['max_empwage'][()]
        self.max_empwage_old=f['max_empwage_old'][()]
        self.min_wage=f['min_wage'][()]
        self.oaHila = f['oaHila'][()]
        self.oaactHila = f['oaactHila'][()]
        self.oaactReward = f['oaactReward'][()]
        self.min_oapension = f['min_oapension'][()]
        self.max_oapension = f['max_oapension'][()]
        
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
        t=self.map_grid_age(age)    
        print('t=',t,'age=',age)
        print('töissä\n',self.get_diag_V(t,1))
        print('ei töissä\n',self.get_diag_V(t,0))
        print('eläke\n',self.get_diag_V(t,2))
        #print('osatyö\n',self.get_diag_V(t,3))

    def get_diag_V(self,t,emp,tis=1):
        if emp == 0:
            sh=self.unempHila.shape
            h=np.zeros((sh[1],sh[2]))
            for k in range(sh[1]):
                for l in range(sh[2]):
                    h[k,l]=self.unempHila[t,k,l,tis,k]
        elif emp==1:
            sh=self.empHila.shape
            h=np.zeros((sh[1],sh[2]))
            for k in range(sh[1]):
                for l in range(sh[2]):
                    h[k,l]=self.empHila[t,k,l,k]
        else:
            sh=self.oaHila.shape
            h=np.zeros((sh[1]))
            for k in range(sh[1]):
                h[k]=self.oaHila[t,k]
                        
        return h
    
    def get_diag_actV(self,t,emp,act,time_in_state=1):
        if emp == 0:
            sh=self.unempActHila.shape
            h=np.zeros((sh[1],sh[2]))
            for k in range(sh[1]):
                for l in range(sh[2]):
                    h[k,l]=self.unempActHila[t,k,l,time_in_state,k,act]
        elif emp == 1:
            sh=self.empActHila.shape
            h=np.zeros((sh[1],sh[2]))
            for k in range(sh[1]):
                for l in range(sh[2]):
                    h[k,l]=self.empActHila[t,k,l,k,act]
        else:
            sh=self.oaactHila.shape
            h=np.zeros((sh[1]))
            for k in range(sh[1]):
                h[k]=self.oaactHila[t,k]
                        
        return h
    
    def plot_V(self,age):
        t=self.map_grid_age(age)
        self.plot_img(self.get_diag_V(t,1),xlabel="Pension",ylabel="Wage",title='Töissä')
        self.plot_img(self.get_diag_V(t,0),xlabel="Pension",ylabel="Wage",title='Työttömänä')
        self.plot_img(self.get_diag_V(t,1)-self.get_diag_V(t,0),xlabel="Pension",ylabel="Wage",title='Työssä-Työtön')

    def print_actV(self,age,time_in_state=1):
        t=self.map_grid_age(age)
        print('t={} age={}'.format(t,age))
        if age>=self.min_retirementage:
            print('eläke: pysyy\n{}\n'.format(self.get_diag_actV(t,2,0,time_in_state=time_in_state)))
            print('töissä: pois töistä\n{}\ntöissä: pysyy\n{}\ntöissä: eläköityy\n{}\n'.format(self.get_diag_actV(t,1,1,time_in_state=time_in_state),self.get_diag_actV(t,1,0,time_in_state=time_in_state),self.get_diag_actV(t,1,2,time_in_state=time_in_state)))
            print('ei töissä: töihin\n{}\nei töissä: pysyy\n{}\nei töissä: eläköityy\n{}\n'.format(self.get_diag_actV(t,0,1,time_in_state=time_in_state),self.get_diag_actV(t,0,0,time_in_state=time_in_state),self.get_diag_actV(t,0,2,time_in_state=time_in_state)))
        else:
            print('töissä: pois töistä\n',self.get_diag_actV(t,1,1),'\ntöissä: pysyy\n',self.get_diag_actV(t,1,0,time_in_state=time_in_state))
            #print('töissä: osatöihin\n',self.get_diag_actV(t,1,3),'\nei töissä: osatöihin\n',self.get_diag_actV(t,1,3,time_in_state=time_in_state))
            print('ei töissä: töihin\n',self.get_diag_actV(t,0,1),'\nei töissä: pysyy\n',self.get_diag_actV(t,0,0,time_in_state=time_in_state))
            #print('osatöissä: pysyy\n',self.get_diag_actV(t,3,0,time_in_state=time_in_state))
            #print('osatöissä: töihin\n',self.get_diag_actV(t,3,3),'\nosatöissä: työttömäksi\n',self.get_diag_actV(t,3,1,time_in_state=time_in_state))

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
        t=self.map_grid_age(age)
        self.plot_img(self.get_diag_actV(t,1,1)-self.get_diag_actV(t,1,0),xlabel="Pension",ylabel="Wage",title='Töissä (ero switch-stay)')
        self.plot_img(self.get_diag_actV(t,0,1)-self.get_diag_actV(t,0,0),xlabel="Pension",ylabel="Wage",title='Työttömänä (ero switch-stay)')

    def plot_act(self,age,time_in_state=0):
        #q3=self.get_act_q(age,emp=3,time_in_state=time_in_state)
        q1=self.get_act_q(age,emp=1,time_in_state=time_in_state)
        q2=self.get_act_q(age,emp=0,time_in_state=time_in_state)

        self.plot_img(q1,xlabel="Pension",ylabel="Wage",title='Töissä',vmin=0,vmax=3)
        self.plot_img(q2,xlabel="Pension",ylabel="Wage",title='Työttömänä',vmin=0,vmax=3)
        ##self.plot_img(q3,xlabel="Pension",ylabel="Wage",title='Osatyössä',vmin=0,vmax=3)
    
    def get_act_q(self,age,emp=1,time_in_state=1,debug=False):
        t=self.map_grid_age(age)
        q=np.zeros((self.n_palkka,self.n_elake))
        if self.n_palkka != self.n_palkka_old:
            print('error self.n_palkka != self.n_palkka_old')
        
        for p in range(self.n_palkka): # olettaa, että n_palkka=n_palkka_old!!
            for el in range(self.n_elake):
                palkka=self.map_palkka(p,emp=emp)
                elake=self.map_elake(el,emp=emp)

                if emp==1:
                    q[p,el]=int(np.argmax(self.empActHila[t,p,el,p,:]))
                elif emp==0:
                    q[p,el]=int(np.argmax(self.unempActHila[t,p,el,time_in_state,p,:]))
        return q

    def print_q(self,age,emp=1,time_in_state=0):
        _=self.get_act_q(age=age,emp=emp,time_in_state=time_in_state,debug=True)

    def compare_act(self,age,cc,time_in_state=1,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                    deterministic=True,vmin=None,vmax=None,dire='kuvat',show_results=False):
        q1=self.get_act_q(age,emp=0,time_in_state=time_in_state)
        q2=cc.get_RL_act(age,emp=0,time_in_state=time_in_state,rlmodel=rlmodel,
            load=load,deterministic=deterministic,deltaemppalkka=self.deltaemppalkka,n_emppalkka=self.n_emppalkka,
            n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
            min_wage=self.min_wage,min_pension=self.min_pension,deltapalkka_old=self.deltapalkka_old,deltaemppalkka_old=self.deltaemppalkka_old)
        q3=self.get_act_q(age,emp=1,time_in_state=time_in_state)
        q4=cc.get_RL_act(age,emp=1,time_in_state=time_in_state,rlmodel=rlmodel,
            load=load,deterministic=deterministic,deltaemppalkka=self.deltaemppalkka,n_emppalkka=self.n_emppalkka,
            n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
            min_wage=self.min_wage,min_pension=self.min_pension,deltapalkka_old=self.deltapalkka_old,deltaemppalkka_old=self.deltaemppalkka_old)
        #q5=self.get_act_q(age,emp=3,time_in_state=time_in_state)
        #q6=cc.get_RL_act(age,emp=3,time_in_state=time_in_state,rlmodel=rlmodel,
        #    load=load,deterministic=deterministic,
        #    n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
        #    min_wage=self.min_wage,min_pension=self.min_pension)
        
        self.plot_twoimg(q3,q4,title1='Employed DP {}'.format(age),title2='Employed RL {}'.format(age),vmin=0,vmax=2,figname=dire+'/emp_'+str(age),ylabel1='Wage')
        self.plot_twoimg(q1,q2,title1='Unemployed DP {}'.format(age),title2='Unemployed RL {}'.format(age),vmin=0,vmax=2,figname=dire+'/unemp_'+str(age),ylabel1='Wage')
        #self.plot_twoimg(q5,q6,title1='Parttime DP {}'.format(age),title2='Parttime RL {}'.format(age),vmin=0,vmax=2,figname=dire+'/parttime_'+str(age))
        
    def compare_ages(self,cc,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                     deterministic=True,time_in_state=0):
        for age in set([20,25,30,35,40,45,50,55,59,60,61,62,63,64,65,66,67,68,69,70]):
            self.compare_act(age,cc,rlmodel=rlmodel,load=load,deterministic=deterministic,time_in_state=time_in_state)
            
    def compare_age_and_real(self,cc,rlmodel=None,load=None,
                     deterministic=True,time_in_state=0,age=50,dire='kuvat',results=None,figname=None,emp1=0,emp2=1):
        self.load_sim(results)
        q1=cc.get_RL_act(age,emp=emp1,time_in_state=time_in_state,rlmodel=rlmodel,
           load=load,deterministic=deterministic,n_palkka=self.n_palkka,n_emppalkka=self.n_emppalkka,
           deltapalkka=self.deltapalkka,deltaemppalkka=self.deltaemppalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
           min_wage=self.min_wage,min_pension=self.min_pension,deltapalkka_old=self.deltapalkka_old,deltaemppalkka_old=self.deltaemppalkka_old)
        q2=cc.get_RL_act(age,emp=emp2,time_in_state=time_in_state,rlmodel=rlmodel,
           load=load,deterministic=deterministic,n_palkka=self.n_palkka,n_emppalkka=self.n_emppalkka,
           deltapalkka=self.deltapalkka,deltaemppalkka=self.deltaemppalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
           min_wage=self.min_wage,min_pension=self.min_pension,deltapalkka_old=self.deltapalkka_old,deltaemppalkka_old=self.deltaemppalkka_old)
        
        fig,axs=self.plot_twoimg(q1,q2,title1='Unemployed RL {}'.format(age),title2='Employed RL {}'.format(age),vmin=0,vmax=2,
           show_results=False)#,alpha=0.5)
        print('scatter...')
           
        if emp1 != 2:
            c1='w'
        else:
            c1='w'
        if emp2 != 2:
            c2='w'
        else:
            c2='w'
        
        t=self.map_age(age)
        xa=[]
        ya=[]
        xb=[]
        yb=[]
        for k in range(self.episodestats.n_pop):
            x0,x1,dx=self.inv_elake(self.episodestats.infostats_pop_pension[t,k],emp=emp1)
            y0,y1,dy=self.inv_palkka(self.episodestats.infostats_pop_wage[t,k],emp1)

            y2=min(self.n_palkka,y0+dy)

            if self.episodestats.popempstate[t,k]==0:
                xa.append(x0+dx)
                ya.append(y2)
                #axs[0].scatter(x0+dx,y0+dy,marker='.',s=2,c=c1)
            elif self.episodestats.popempstate[t,k]==1:
                xb.append(x0+dx)
                yb.append(y2)
                #axs[1].scatter(x0+dx,y0+dy,marker='.',s=2,c=c2)

        axs[0].scatter(xa,ya,marker='.',s=2,c=c1)#,alpha=0.5)
        axs[1].scatter(xb,yb,marker='.',s=2,c=c2)#,alpha=0.5)
        
        if figname is not None:
            plt.savefig(figname+'.pdf', format='pdf')

        fig.show()
                     
    def comp_pave(self):
    
        self.ave=np.zeros(self.n_time)
        self.sted=np.zeros(self.n_time)
        self.real=self.episodestats.comp_presentvalue()
        for t in range(1,self.n_time):
            self.ave[t]=np.mean(self.episodestats.aveV[t,:]-self.real[t,:])
            self.sted[t]=np.std(self.episodestats.aveV[t,:]-self.real[t,:])
            
        self.pave=np.zeros((self.n_time,3))
        self.nave=np.zeros((self.n_time,3))
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                s=int(self.episodestats.popempstate[t,k])
                self.pave[t,s]+=self.episodestats.aveV[t,k]-self.real[t,k]
                self.nave[t,s]+=1            

        self.tave=np.zeros((self.n_time,20))
        self.ntave=np.zeros((self.n_time,20))
        self.pvave=np.zeros((self.n_time,10))
        self.npvave=np.zeros((self.n_time,10))
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                wa=int(np.floor(self.episodestats.infostats_pop_wage[t,k]/5000))
                self.tave[t,wa]+=self.episodestats.aveV[t,k]-self.real[t,k]
                self.ntave[t,wa]+=1
                pa=int(np.floor(self.episodestats.infostats_pop_pension[t,k]/5000))
                self.pvave[t,pa]+=self.episodestats.aveV[t,k]-self.real[t,k]
                self.npvave[t,pa]+=1
        
        self.pave=self.pave/np.maximum(1.0,self.nave)
        self.pvave=self.pvave/np.maximum(1.0,self.npvave)
        self.tave=self.tave/np.maximum(1.0,self.ntave)

        self.oavave=np.zeros((self.n_time,10))
        self.n_oavave=np.zeros((self.n_time,10))
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                if int(self.episodestats.popempstate[t,k])==2:
                    wa=int(np.floor(self.episodestats.infostats_pop_pension[t,k]/5000))
                    self.oavave[t,wa]+=self.episodestats.aveV[t,k]-self.real[t,k]
                    self.n_oavave[t,wa]+=1
        self.oavave=self.oavave/np.maximum(1.0,self.n_oavave)

        self.pave_err_val=[]
        self.pave_err_t=[]
        self.pave_err_wage=[]
        self.pave_err_pen=[]
        self.pave_err_val_emp=[]
        self.pave_err_t_emp=[]
        self.pave_err_wage_emp=[]
        self.pave_err_pen_emp=[]
        self.pave_err_val_unemp=[]
        self.pave_err_t_unemp=[]
        self.pave_err_wage_unemp=[]
        self.pave_err_pen_unemp=[]
        self.pave_err_basis_emp=[]
        self.pave_err_basis_unemp=[]
        self.pave_err_basis=[]
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                err=self.episodestats.aveV[t,k]-self.real[t,k]
                if np.abs(err)>0.005:
                    s=int(self.episodestats.popempstate[t,k])
                    if s==0:
                        self.pave_err_val_unemp.append(err)
                        self.pave_err_t_unemp.append(t)
                        self.pave_err_wage_unemp.append(self.episodestats.infostats_pop_wage[t,k])
                        self.pave_err_pen_unemp.append(self.episodestats.infostats_pop_pension[t,k])
                        self.pave_err_basis_unemp.append(self.episodestats.infostats_unempwagebasis)
                    elif s==1:
                        self.pave_err_val_emp.append(err)
                        self.pave_err_t_emp.append(t)
                        self.pave_err_wage_emp.append(self.episodestats.infostats_pop_wage[t,k])
                        self.pave_err_pen_emp.append(self.episodestats.infostats_pop_pension[t,k])
                        self.pave_err_basis_emp.append(self.episodestats.infostats_unempwagebasis)
                        
                    self.pave_err_val.append(err)
                    self.pave_err_t.append(t)
                    self.pave_err_wage.append(self.episodestats.infostats_pop_wage[t,k])
                    self.pave_err_pen.append(self.episodestats.infostats_pop_pension[t,k])
                    self.pave_err_basis.append(self.episodestats.infostats_unempwagebasis)

    def plot_pave(self):
        self.episodestats.plot_Vstats()

        plt.plot(self.ave[1:])
        plt.show()
        
        plt.plot(self.pave[:,0],'r',label='unemp')
        plt.plot(self.pave[:,1],'k',label='emp')
        plt.legend()
        plt.title('emp/unemp aveV error')
        plt.show()
        
        plt.plot(self.pave[:,2],'k',label='ret')     
        plt.title('ret aveV error')
        plt.show()
        
        for k in range(10):
            plt.plot(self.tave[:,k],label=str(k))
        plt.legend()
        plt.title('aveV error by wage level')
        plt.show()        
        for k in range(10,20):
            plt.plot(self.tave[:,k],label=str(k))
        plt.legend()
        plt.title('aveV error by wage level')
        plt.show()        
        for k in range(10):
            plt.plot(self.ntave[:,k],label=str(k))
        plt.legend()
        plt.title('number of agents by wage level')
        plt.show()        

        for k in range(5):
            plt.plot(self.oavave[45:,k],label=str(k))
        plt.legend()
        plt.title('error by retirees by wage level')
        plt.show()        
        for k in range(5):
            plt.plot(self.n_oavave[45:,k],label=str(k))
        plt.legend()
        plt.title('number of retirees by wage level')
        plt.show()
        
        for k in range(5):
            plt.plot(self.pvave[:,k],label=str(k))
        plt.legend()
        plt.title('aveV error by pension level')
        plt.show()        
        for k in range(5):
            plt.plot(self.npvave[:,k],label=str(k))
        plt.legend()
        plt.title('number of agents by pension level')
        plt.show()        

        #plt.scatter(self.pave_err_t,self.pave_err_val)
        #plt.legend()
#        plt.title('number of agents by pension level')
        #plt.show()        

        if False:
            self.scatter_density(self.pave_err_t,self.pave_err_val,label1='t',label2='error')
            self.scatter_density(self.pave_err_t,self.pave_err_wage,label1='t',label2='wage')
            self.scatter_density(self.pave_err_t,self.pave_err_pen,label1='t',label2='pen')
            self.scatter_density(self.pave_err_wage,self.pave_err_val,label1='wage',label2='err')
            self.scatter_density(self.pave_err_wage,self.pave_err_pen,label1='wage',label2='pen')

        if True:
            #self.scatter_density(self.pave_err_basis_emp,self.pave_err_val_emp,label1='basis',label2='emp err')
            #self.scatter_density(self.pave_err_basis_unemp,self.pave_err_val_unemp,label1='basis',label2='unemp err')
            #self.scatter_density(self.pave_err_t_emp,self.pave_err_basis_emp,label1='t',label2='emp basis')
            #self.scatter_density(self.pave_err_t_unemp,self.pave_err_basis_unemp,label1='t',label2='unemp basis')
            self.scatter_density(self.pave_err_t_emp,self.pave_err_val_emp,label1='t',label2='emp error')
            self.scatter_density(self.pave_err_t_unemp,self.pave_err_val_unemp,label1='t',label2='unemp error')
            self.scatter_density(self.pave_err_t_emp,self.pave_err_wage_emp,label1='t',label2='emp wage')
            self.scatter_density(self.pave_err_t_unemp,self.pave_err_wage_unemp,label1='t',label2='unemp wage')
            self.scatter_density(self.pave_err_t_emp,self.pave_err_pen_emp,label1='t',label2='emp pen')
            self.scatter_density(self.pave_err_t_unemp,self.pave_err_pen_unemp,label1='t',label2='unemp pen')
            self.scatter_density(self.pave_err_wage_emp,self.pave_err_val_emp,label1='wage',label2='emp err')
            self.scatter_density(self.pave_err_wage_unemp,self.pave_err_val_unemp,label1='wage',label2='unemp err')
            self.scatter_density(self.pave_err_pen_emp,self.pave_err_val_emp,label1='pen',label2='emp err')
            self.scatter_density(self.pave_err_pen_unemp,self.pave_err_val_unemp,label1='pen',label2='unemp err')
            self.scatter_density(self.pave_err_wage_emp,self.pave_err_pen_emp,label1='wage',label2='emp pen')
            self.scatter_density(self.pave_err_wage_unemp,self.pave_err_pen_unemp,label1='wage',label2='unemp pen')
            
        
    def scatter_density(self,x,y,label1='',label2=''):
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        #idx = z.argsort()
        #x, y, z = x[idx], y[idx], z[idx]
        fig,ax=plt.subplots()
        ax.scatter(x,y,c=z)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        #plt.legend()
#        plt.title('number of agents by pension level')
        plt.show()        

    def plot_p_aveV(self,num=500,start=1):
        plt.plot(self.episodestats.aveV[start:,num],'r')
        plt.plot(self.real[start:,num])
        plt.show()
        plt.plot(self.episodestats.aveV[start:,num]-self.real[start:,num])
        plt.show()
        plt.plot(self.episodestats.popempstate[start:,num])                                
        
        
    def check_V(self):
        # check final age
        t=self.map_grid_age(70)
        diff=np.zeros((self.n_employment,self.n_elake,3))
        for emp in range(self.n_employment):
            for el in range(self.n_elake):
                for a in range(3):
                    diff[emp,el,a]=np.max(self.actReward[t,:,el,emp,:,:,a]-self.actHila[t,:,el,emp,:,:,a])
                
        print('max diff',np.max(diff))
        print('min diff',np.min(diff),' argmin',np.argmin(diff))
        
        return diff

    def check_actHila(self):
        # check final age
        for t in range(self.n_time):
            for emp in range(2):
                for el in range(self.n_elake-1):
                    for ow in range(self.n_palkka_old):
                        for w in range(self.n_palkka):
                            if np.min(self.actHila[t,ow,el,emp,:,w,0:2])<0.1:
                                print(t,ow,el,emp,w,self.actHila[t,ow,el,emp,:,w,:])
                
    def check_Hila(self,age=65):
        # check final age
        t=self.map_grid_age(age)
        diff=np.zeros((self.n_employment,self.n_elake,3))
        for emp in range(self.n_employment):
            for el in range(self.n_elake-1):
                for ow in range(self.n_palkka_old):
                    for w in range(self.n_palkka):
                        diff=np.min(self.Hila[t,ow,el+1,emp,:,w]-self.Hila[t,ow,el,emp,:,w])
                        dd=self.Hila[t,ow,el+1,emp,:,w]-self.Hila[t,ow,el,emp,:,w]
                        if diff<0:
                            print(f'elake Hila t={t} ow={ow} el={el} emp={emp} w={w}: {diff} {dd}')
                            print(self.Hila[t,ow,el,emp,:,w],self.Hila[t,ow,el+1,emp,:,w])
        for emp in range(self.n_employment):
            for el in range(self.n_elake):
                for ow in range(self.n_palkka_old):
                    for w in range(self.n_palkka-1):
                        diff=np.min(self.Hila[t,ow,el,emp,:,w+1]-self.Hila[t,ow,el,emp,:,w])
                        if diff<0:
                            dd=self.Hila[t,ow,el,emp,:,w+1]-self.Hila[t,ow,el,emp,:,w]
                            print(f'w Hila t={t} ow={ow} el={el} emp={emp} w={w}: {diff} {dd}')
                            print(self.Hila[t,ow,el,emp,:,w],self.Hila[t,ow,el,emp,:,w+1])
        for emp in range(self.n_employment):
            for el in range(self.n_elake):
                for ow in range(self.n_palkka_old-1):
                    for w in range(self.n_palkka):
                        diff=np.min(self.Hila[t,ow+1,el,emp,:,w]-self.Hila[t,ow,el,emp,:,w])
                        if diff<0:
                            dd=self.Hila[t,ow+1,el,emp,:,w]-self.Hila[t,ow,el,emp,:,w]
                            print(f'ow Hila t={t} ow={ow} el={el} emp={emp} w={w}: {diff} {dd}')
                            print(self.Hila[t,ow,el,emp,:,w],self.Hila[t,ow+1,el,emp,:,w])

    def check_oaHila(self,age=65):
        t=self.map_grid_age(age)
        for el in range(self.n_oapension-1):
            diff=self.oaHila[t,el+1]-self.oaHila[t,el]
            if diff<0:
                print(f'elakeella Hila t={t} el={el}: {diff}')
                print(self.oaHila[t,el+1],self.oaHila[t,el])
        
        
    def test_real_Hila(self,age,emp=1,elake=0,ow=15_000,tis=0):
        act=0
        actions=[act]
        ww=np.arange(self.min_wage,self.max_wage,1_00)
        obsr0=np.zeros_like(ww)
        predrew0=np.zeros_like(ww)
        obsr1=np.zeros_like(ww)
        predrew1=np.zeros_like(ww)
        for k,w in enumerate(ww):
            predrew0[k]=self.get_actReward_spline(emp=0,elake=elake,old_wage=ow,time_in_state=tis,wage=w,act=act,age=age)
            r,sps=self.get_rewards_continuous((0,elake,ow,age,tis,w),actions,debug=False)
            obsr0[k]=r[0]
            predrew1[k]=self.get_actReward_spline(emp=1,elake=elake,old_wage=ow,time_in_state=tis,wage=w,act=act,age=age)
            r,sps=self.get_rewards_continuous((1,elake,ow,age,tis,w),actions,debug=False)
            obsr1[k]=r[0]
            
        fig,ax=plt.subplots()
        plt.title('emp 0')
        ax.plot(ww,predrew0)
        ax.plot(ww,obsr0)
        ax.set_xlabel('wage')
        plt.show()

        fig,ax=plt.subplots()
        plt.title('emp 1')
        ax.plot(ww,predrew1)
        ax.plot(ww,obsr1)
        ax.set_xlabel('wage')
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(ww,predrew1,label='pred emp 1')
        ax.plot(ww,obsr1,label='obs emp 1')
        ax.plot(ww,predrew0,label='pred emp 0')
        ax.plot(ww,obsr0,label='obs emp 0')
        ax.set_xlabel('wage')
        plt.show()

        fig,ax=plt.subplots()
        plt.title('diff')
        ax.plot(ww,predrew0-obsr0,label='emp 0')
        ax.plot(ww,predrew1-obsr1,label='emp 1')
        ax.set_xlabel('old wage')
        plt.show()

        act=0
        actions=[act]
        ww=np.arange(self.min_wage,self.max_wage,1_00)
        obsr0=np.zeros_like(ww)
        predrew0=np.zeros_like(ww)
        obsr1=np.zeros_like(ww)
        predrew1=np.zeros_like(ww)
        for k,w in enumerate(ww):
            predrew0[k]=self.get_actReward_spline(emp=0,elake=elake,old_wage=w,time_in_state=tis,wage=w,act=act,age=age)
            r,sps=self.get_rewards_continuous((0,elake,w,age,tis,w),actions,debug=False)
            obsr0[k]=r[0]
            predrew1[k]=self.get_actReward_spline(emp=1,elake=elake,old_wage=w,time_in_state=tis,wage=w,act=act,age=age)
            r,sps=self.get_rewards_continuous((1,elake,w,age,tis,w),actions,debug=False)
            obsr1[k]=r[0]
            
        fig,ax=plt.subplots()
        plt.title('emp 0')
        ax.plot(ww,predrew0)
        ax.plot(ww,obsr0)
        ax.set_xlabel('old wage')
        plt.show()

        fig,ax=plt.subplots()
        plt.title('emp 1')
        ax.set_xlabel('old wage')
        ax.plot(ww,predrew1)
        ax.plot(ww,obsr1)
        plt.show()
        
        fig,ax=plt.subplots()
        ax.plot(ww,predrew1,label='pred emp 1')
        ax.plot(ww,obsr1,label='obs emp 1')
        ax.plot(ww,predrew0,label='pred emp 0')
        ax.plot(ww,obsr0,label='obs emp 0')
        ax.set_xlabel('wage')
        plt.show()

        fig,ax=plt.subplots()
        plt.title('diff')
        ax.plot(ww,predrew0-obsr0,label='emp 0')
        ax.plot(ww,predrew1-obsr1,label='emp 1')
        ax.set_xlabel('old wage')
        plt.show()
