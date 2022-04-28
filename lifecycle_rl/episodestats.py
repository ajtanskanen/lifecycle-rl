'''

    episodestats.py

    implements statistic that are used in producing employment statistics for the
    lifecycle model

'''

import h5py
import numpy as np
import numpy.ma as ma
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json
import ast
from scipy.stats import norm
#import locale
from tabulate import tabulate
import pandas as pd
import scipy.optimize
from tqdm import tqdm_notebook as tqdm
from . empstats import Empstats
from fin_benefits import Labels
from scipy.stats import gaussian_kde
from .utils import empirical_cdf,print_html,modify_offsettext


#locale.setlocale(locale.LC_ALL, 'fi_FI')

class EpisodeStats():
    def __init__(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year=2018,version=3,params=None,gamma=0.92,lang='English'):
        self.version=version
        self.gamma=gamma
        self.params=params
        self.params['n_time']=n_time
        self.params['n_emps']=n_emps
        self.lab=Labels()
        self.reset(timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,params=params,lang=lang)
        
        if self.version==0:
            self.add=self.add_v0
        elif self.version==1:
            self.add=self.add_v1
        elif self.version==2:
            self.add=self.add_v2
        elif self.version==3:
            self.add=self.add_v3
        elif self.version==4:
            self.add=self.add_v4
        elif self.version==101:
            self.add=self.add_v101
        elif self.version==104:
            self.add=self.add_v104
        else:
            print('Unknown version ',self.version)

    def reset(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,version=None,params=None,lang=None,dynprog=False):
        self.min_age=min_age
        self.max_age=max_age
        self.min_retirementage=min_retirementage
        self.minimal=minimal

        if params is not None:
            self.params=params

        if lang is None:
            self.language='English'
        else:
            self.language=lang

        if version is not None:
            self.version=version

        self.setup_labels()

        self.figformat='pdf'

        self.n_employment=n_emps
        self.n_time=n_time
        self.timestep=timestep # 0.25 = 3kk askel
        self.inv_timestep=int(np.round(1/self.timestep)) # pitää olla kokonaisluku
        self.n_pop=n_pop
        self.year=year
        self.env=env
        self.reaalinen_palkkojenkasvu=0.016
        self.palkkakerroin=(0.8*1+0.2*1.0/(1+self.reaalinen_palkkojenkasvu))**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/(1+self.reaalinen_palkkojenkasvu))**self.timestep
        self.dynprog=dynprog

        if self.version in set([0,101]):
            self.n_groups=1
        else:
            self.n_groups=6
            
        self.empstats=Empstats(year=self.year,max_age=self.max_age,n_groups=self.n_groups,timestep=self.timestep,n_time=self.n_time,
                                min_age=self.min_age)
        self.init_variables()

    def init_variables(self):
        n_emps=self.n_employment
        self.empstate=np.zeros((self.n_time,n_emps),dtype=np.int64)
        self.gempstate=np.zeros((self.n_time,n_emps,self.n_groups),dtype=np.int64)
        self.deceiced=np.zeros((self.n_time,1),dtype=np.int64)
        self.alive=np.zeros((self.n_time,1),dtype=np.int64)
        self.galive=np.zeros((self.n_time,self.n_groups),dtype=np.int64)
        self.rewstate=np.zeros((self.n_time,n_emps))
        self.poprewstate=np.zeros((self.n_time,self.n_pop))
        self.salaries_emp=np.zeros((self.n_time,n_emps))
        #self.salaries=np.zeros((self.n_time,self.n_pop))
        self.popempstate=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
        self.popunemprightleft=np.zeros((self.n_time,self.n_pop))
        self.popunemprightused=np.zeros((self.n_time,self.n_pop))
        self.tyoll_distrib_bu=np.zeros((self.n_time,self.n_pop))
        self.unemp_distrib_bu=np.zeros((self.n_time,self.n_pop))
        self.siirtyneet=np.zeros((self.n_time,n_emps),dtype=np.int64)
        self.siirtyneet_det=np.zeros((self.n_time,n_emps,n_emps),dtype=np.int64)
        self.pysyneet=np.zeros((self.n_time,n_emps),dtype=np.int64)
        if self.minimal:
            self.aveV=np.zeros((self.n_time,self.n_pop))
        self.time_in_state=np.zeros((self.n_time,n_emps))
        self.stat_tyoura=np.zeros((self.n_time,n_emps))
        self.stat_toe=np.zeros((self.n_time,n_emps))
        self.out_of_work=np.zeros((self.n_time,n_emps),dtype=np.int64)
        self.stat_unemp_len=np.zeros((self.n_time,self.n_pop))
        self.stat_wage_reduction=np.zeros((self.n_time,n_emps))
        self.stat_wage_reduction_g=np.zeros((self.n_time,n_emps,self.n_groups))
        self.infostats_group=np.zeros((self.n_pop,1),dtype=np.int8)
        self.infostats_taxes=np.zeros((self.n_time,1))
        self.infostats_wagetaxes=np.zeros((self.n_time,1))
        self.infostats_taxes_distrib=np.zeros((self.n_time,n_emps))
        self.infostats_etuustulo=np.zeros((self.n_time,1))
        self.infostats_etuustulo_group=np.zeros((self.n_time,self.n_groups))
        self.infostats_perustulo=np.zeros((self.n_time,1))
        self.infostats_palkkatulo=np.zeros((self.n_time,1))
        self.infostats_palkkatulo_eielakkeella=np.zeros((self.n_time,1))
        self.infostats_palkkatulo_group=np.zeros((self.n_time,self.n_groups))
        self.infostats_palkkatulo_eielakkeella_group=np.zeros((self.n_time,1))
        self.infostats_ansiopvraha=np.zeros((self.n_time,1))
        self.infostats_ansiopvraha_group=np.zeros((self.n_time,self.n_groups))
        self.infostats_asumistuki=np.zeros((self.n_time,1))
        self.infostats_asumistuki_group=np.zeros((self.n_time,self.n_groups))
        self.infostats_valtionvero=np.zeros((self.n_time,1))
        self.infostats_valtionvero_group=np.zeros((self.n_time,self.n_groups))
        self.infostats_kunnallisvero=np.zeros((self.n_time,1))
        self.infostats_kunnallisvero_group=np.zeros((self.n_time,self.n_groups))
        self.infostats_valtionvero_distrib=np.zeros((self.n_time,n_emps))
        self.infostats_kunnallisvero_distrib=np.zeros((self.n_time,n_emps))
        self.infostats_ptel=np.zeros((self.n_time,1))
        self.infostats_tyotvakmaksu=np.zeros((self.n_time,1))
        self.infostats_tyoelake=np.zeros((self.n_time,1))
        self.infostats_kansanelake=np.zeros((self.n_time,1))
        self.infostats_kokoelake=np.zeros((self.n_time,1))
        #self.infostats_takuuelake=np.zeros((self.n_time,1))
        self.infostats_lapsilisa=np.zeros((self.n_time,1))
        self.infostats_elatustuki=np.zeros((self.n_time,1))
        self.infostats_opintotuki=np.zeros((self.n_time,1))
        self.infostats_isyyspaivaraha=np.zeros((self.n_time,1))
        self.infostats_aitiyspaivaraha=np.zeros((self.n_time,1))
        self.infostats_kotihoidontuki=np.zeros((self.n_time,1))
        self.infostats_sairauspaivaraha=np.zeros((self.n_time,1))
        self.infostats_toimeentulotuki=np.zeros((self.n_time,1))
        self.infostats_tulot_netto=np.zeros((self.n_time,1))
        self.infostats_pinkslip=np.zeros((self.n_time,n_emps))
        self.infostats_pop_pinkslip=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
        #self.infostats_chilren18_emp=np.zeros((self.n_time,n_emps),dtype=np.int64)
        #self.infostats_chilren7_emp=np.zeros((self.n_time,n_emps),dtype=np.int64)
        #self.infostats_chilren18=np.zeros((self.n_time,1),dtype=np.int64)
        #self.infostats_chilren7=np.zeros((self.n_time,1),dtype=np.int64)
        self.infostats_tyelpremium=np.zeros((self.n_time,self.n_pop))
        self.stat_paidpension=np.zeros((self.n_time,n_emps))
        self.stat_pop_paidpension=np.zeros((self.n_time,self.n_pop))
        self.stat_pension=np.zeros((self.n_time,n_emps))
        self.stat_unemp_after_ra=np.zeros((self.n_time,n_emps))
        self.infostats_pop_pension=np.zeros((self.n_time,self.n_pop))
        self.infostats_paid_tyel_pension=np.zeros((self.n_time,self.n_pop))
        self.infostats_pop_tyoelake=np.zeros((self.n_time,self.n_pop))
        self.infostats_pop_kansanelake=np.zeros((self.n_time,self.n_pop))
        self.infostats_sairausvakuutus=np.zeros((self.n_time,1))
        self.infostats_pvhoitomaksu=np.zeros((self.n_time,self.n_pop))
        self.infostats_ylevero=np.zeros((self.n_time,1))
        self.infostats_ylevero_distrib=np.zeros((self.n_time,n_emps))
        self.infostats_irr_tyel_reduced=np.zeros((self.n_pop,1))
        self.infostats_irr_tyel_full=np.zeros((self.n_pop,1))
        self.infostats_npv0=np.zeros((self.n_pop,1))
        self.infostats_mother_in_workforce=np.zeros((self.n_time,1))
        #self.infostats_father_in_workforce=np.zeros((self.n_time,1))
        self.infostats_children_under3=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
        self.infostats_children_under7=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
        self.infostats_children_under18=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
        self.infostats_unempwagebasis=np.zeros((self.n_time,self.n_pop))
        self.infostats_unempwagebasis_acc=np.zeros((self.n_time,self.n_pop))
        self.infostats_toe=np.zeros((self.n_time,self.n_pop))
        self.infostats_ove=np.zeros((self.n_time,n_emps))
        self.infostats_ove_g=np.zeros((self.n_time,self.n_groups,n_emps))
        self.infostats_kassanjasen=np.zeros((self.n_time,1))
        self.infostats_poptulot_netto=np.zeros((self.n_time,self.n_pop))
        self.infostats_pop_wage=np.zeros((self.n_time,self.n_pop))
        self.infostats_equivalent_income=np.zeros((self.n_time,1))
        self.infostats_alv=np.zeros((self.n_time,1))
        self.pop_predrew=np.zeros((self.n_time,self.n_pop))
        if self.version in set([101,102,104]):
            self.infostats_savings=np.zeros((self.n_time,self.n_pop))
            self.sav_actions=np.zeros((self.n_time,self.n_pop))

        if self.version in set([4,104]):
            self.actions=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
            self.infostats_puoliso=np.zeros((self.n_time,self.n_pop),dtype=np.int8)
        else:
            self.actions=np.zeros((self.n_time,self.n_pop),dtype=np.int8)

    def add_taxes(self,t,q,newemp,n,g,person=''):
        scale=self.timestep*12
        self.infostats_taxes[t]+=q[person+'verot']*scale
        self.infostats_wagetaxes[t]+=q[person+'verot_ilman_etuuksia']*scale
        self.infostats_taxes_distrib[t,newemp]+=q[person+'verot']*scale
        self.infostats_etuustulo[t]+=q[person+'etuustulo_brutto']*scale
        self.infostats_etuustulo_group[t,g]+=q[person+'etuustulo_brutto']*scale
        self.infostats_perustulo[t]+=q[person+'perustulo']*scale
        self.infostats_palkkatulo[t]+=q[person+'palkkatulot']*scale
        self.infostats_palkkatulo_eielakkeella[t]+=q[person+'palkkatulot_eielakkeella']*scale
        self.infostats_ansiopvraha[t]+=q[person+'ansiopvraha']*scale
        self.infostats_asumistuki[t]+=q[person+'asumistuki']*scale
        self.infostats_valtionvero[t]+=q[person+'valtionvero']*scale
        self.infostats_valtionvero_distrib[t,newemp]+=q[person+'valtionvero']*scale
        self.infostats_kunnallisvero[t]+=q[person+'kunnallisvero']*scale
        self.infostats_kunnallisvero_distrib[t,newemp]+=q[person+'kunnallisvero']*scale
        self.infostats_ptel[t]+=q[person+'ptel']*scale
        self.infostats_tyotvakmaksu[t]+=q[person+'tyotvakmaksu']*scale
        self.infostats_kokoelake[t]+=q[person+'kokoelake']*scale
        self.stat_pop_paidpension[t,n]=q[person+'kokoelake']*scale
        self.infostats_lapsilisa[t]+=q[person+'lapsilisa']*scale
        self.infostats_elatustuki[t]+=q[person+'elatustuki']*scale
        self.infostats_opintotuki[t]+=q[person+'opintotuki']*scale
        self.infostats_isyyspaivaraha[t]+=q[person+'isyyspaivaraha']*scale
        self.infostats_aitiyspaivaraha[t]+=q[person+'aitiyspaivaraha']*scale
        self.infostats_kotihoidontuki[t]+=q[person+'kotihoidontuki']*scale
        self.infostats_sairauspaivaraha[t]+=q[person+'sairauspaivaraha']*scale
        self.infostats_toimeentulotuki[t]+=q[person+'toimeentulotuki']*scale
        
#         d1=q[person+'verot']
#         d2=q[person+'valtionvero']+q[person+'kunnallisvero']+q[person+'ptel']+q[person+'tyotvakmaksu']+\
#             q[person+'ylevero']+q[person+'sairausvakuutusmaksu']
#             
#         if np.abs(d2-d1)>1e-6:
#             print('add_taxes',person,d2-d1)
        
        if self.version in set([4,104]):
            self.infostats_tulot_netto[t]+=q[person+'netto']*scale # vuositasolla, huomioi alv:n
            self.infostats_poptulot_netto[t,n]=q[person+'netto']*scale
            self.infostats_tyoelake[t]+=q[person+'tyoelake']*scale
            self.infostats_kansanelake[t]+=q[person+'kansanelake']*scale
            self.infostats_pop_tyoelake[t,n]=q[person+'tyoelake']*scale
            self.infostats_pop_kansanelake[t,n]=q[person+'kansanelake']*scale
            self.infostats_pop_wage[t,n]=q[person+'palkkatulot']*12 # at annual level
            #self.infostats_takuuelake[t]+=q[person+'takuuelake']*scale
        else:
            self.infostats_tulot_netto[t]+=q['kateen']*scale # legacy for v1-3
            self.infostats_poptulot_netto[t,n]=q['kateen']*scale
            self.infostats_tyoelake[t]+=q[person+'elake_maksussa']*scale
            
        self.infostats_tyelpremium[t,n]=q[person+'tyel_kokomaksu']*scale
        self.infostats_paid_tyel_pension[t,n]=q[person+'puhdas_tyoelake']*scale
        self.infostats_sairausvakuutus[t]+=q[person+'sairausvakuutusmaksu']*scale
        self.infostats_pvhoitomaksu[t,n]=q[person+'pvhoito']*scale
        self.infostats_ylevero[t]+=q[person+'ylevero']*scale
        self.infostats_ylevero_distrib[t,newemp]=q[person+'ylevero']*scale
        self.infostats_npv0[n]=q[person+'multiplier']
        self.infostats_equivalent_income[t]+=q[person+'eq']*self.timestep # already at annual level
        if 'alv' in q:
            self.infostats_alv[t]+=q[person+'alv']*scale
            
    def add_v0(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,a,_,_=self.env.state_decode(state) # current employment state
        newemp,newpen,newsal,a2,tis,next_wage=self.env.state_decode(newstate)
        g=0
        bu=0
        ove=0
        jasen=0
        puoliso=0

        t=int(np.round((a2-self.min_age)*self.inv_timestep))#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            self.empstate[t,newemp]+=1
            self.alive[t]+=1
            self.rewstate[t,newemp]+=r
            self.infostats_tulot_netto[t]+=q['netto'] # already at annual level
            self.infostats_poptulot_netto[t,n]=q['netto']

            self.poprewstate[t,n]=r
            self.popempstate[t,n]=newemp
            #self.salaries[t,n]=newsal
            self.salaries_emp[t,newemp]+=newsal
            self.time_in_state[t,newemp]+=tis
            self.infostats_equivalent_income[t]+=q['eq']
            self.infostats_pop_wage[t,n]=newsal
            self.infostats_pop_pension[t,n]=newpen
            if self.dynprog and pred_r is not None:
                self.pop_predrew[t,n]=pred_r

            self.actions[t,n]=act

            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1
            
    def add_v1(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):
        emp,_,_,_,a,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,oof,bu,wr,p=self.env.state_decode(newstate)
        ove=0
        jasen=0
        puoliso=0

        t=int(np.round((a2-self.min_age)*self.inv_timestep))#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2
            if self.version in set([1,2,3,104]):
                self.empstate[t,newemp]+=1
                self.alive[t]+=1
                self.rewstate[t,newemp]+=r
                self.poprewstate[t,n]=r
                
                self.actions[t,n]=act
                self.popempstate[t,n]=newemp
                #self.salaries[t,n]=newsal
                self.salaries_emp[t,newemp]+=newsal
                self.time_in_state[t,newemp]+=tis
                if tis<=0.25 and newemp==5:
                    self.infostats_mother_in_workforce[t]+=1
                #if tis<=0.25 and newemp==6:
                #    self.infostats_father_in_workforce[t]+=1
                self.infostats_pinkslip[t,newemp]+=pink
                self.infostats_pop_pinkslip[t,n]=pink
                self.gempstate[t,newemp,g]+=1
                self.stat_wage_reduction[t,newemp]+=wr
                self.stat_wage_reduction_g[t,newemp,g]+=wr
                self.galive[t,g]+=1
                self.stat_tyoura[t,newemp]+=ura
                self.stat_toe[t,newemp]+=toe
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_len[t,n]=tis
                self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura)
                self.popunemprightused[t,n]=bu
                self.infostats_group[n]=int(g)
                self.infostats_unempwagebasis[t,n]=uw
                self.infostats_unempwagebasis_acc[t,n]=uwr
                self.infostats_toe[t,n]=toe
                self.infostats_ove[t,newemp]+=ove
                self.infostats_kassanjasen[t]+=jasen
                self.infostats_pop_wage[t,n]=newsal
                self.infostats_pop_pension[t,n]=newpen
                self.infostats_children_under3[t,n]=c3
                self.infostats_children_under7[t,n]=c7
                self.infostats_children_under18[t,n]=c18

                if q is not None:
                    self.add_taxes(t,q,newemp,n,g)
                    
            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1
            
    def add_v2(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):
        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58=self.env.state_decode(newstate)
        ove=0
        jasen=0
        puoliso=0

        t=int(np.round((a2-self.min_age)*self.inv_timestep))#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3 and self.version in set([1,2,3,4,104]):
                newemp=2
            self.empstate[t,newemp]+=1
            self.alive[t]+=1
            self.rewstate[t,newemp]+=r
            self.poprewstate[t,n]=r
            
            self.actions[t,n]=act
            self.popempstate[t,n]=newemp
            self.salaries_emp[t,newemp]+=newsal
            self.time_in_state[t,newemp]+=tis
            if tis<=0.25 and newemp==5:
                self.infostats_mother_in_workforce[t]+=1
            #if tis<=0.25 and newemp==6:
            #    self.infostats_father_in_workforce[t]+=1
            self.infostats_pinkslip[t,newemp]+=pink
            self.infostats_pop_pinkslip[t,n]=pink
            self.gempstate[t,newemp,g]+=1
            self.stat_wage_reduction[t,newemp]+=wr
            self.stat_wage_reduction_g[t,newemp,g]+=wr
            self.galive[t,g]+=1
            self.stat_tyoura[t,newemp]+=ura
            self.stat_toe[t,newemp]+=toe
            self.stat_pension[t,newemp]+=newpen
            self.stat_paidpension[t,newemp]+=paidpens
            self.stat_unemp_len[t,n]=tis
            self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura)
            self.popunemprightused[t,n]=bu
            self.infostats_group[n]=int(g)
            self.infostats_unempwagebasis[t,n]=uw
            self.infostats_unempwagebasis_acc[t,n]=uwr
            self.infostats_toe[t,n]=toe
            self.infostats_ove[t,newemp]+=ove
            self.infostats_kassanjasen[t]+=jasen
            self.infostats_pop_wage[t,n]=newsal
            self.infostats_pop_pension[t,n]=newpen
            self.infostats_children_under3[t,n]=c3
            self.infostats_children_under7[t,n]=c7
            self.infostats_children_under18[t,n]=c18

            if q is not None:
                self.add_taxes(t,q,newemp,n,g)

            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1
            
    def add_v3(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58,ove,jasen=self.env.state_decode(newstate)
        puoliso=0

        t=int(np.round((a2-self.min_age)*self.inv_timestep))#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2

            self.empstate[t,newemp]+=1
            self.alive[t]+=1
            self.rewstate[t,newemp]+=r
            self.poprewstate[t,n]=r
            
            self.actions[t,n]=act
            self.popempstate[t,n]=newemp
            #self.salaries[t,n]=newsal
            self.salaries_emp[t,newemp]+=newsal
            self.time_in_state[t,newemp]+=tis
            if tis<=0.25 and newemp==5:
                self.infostats_mother_in_workforce[t]+=1
            #if tis<=0.25 and newemp==6:
            #    self.infostats_father_in_workforce[t]+=1
            self.infostats_pinkslip[t,newemp]+=pink
            self.infostats_pop_pinkslip[t,n]=pink
            self.gempstate[t,newemp,g]+=1
            self.stat_wage_reduction[t,newemp]+=wr
            self.stat_wage_reduction_g[t,newemp,g]+=wr
            self.galive[t,g]+=1
            self.stat_tyoura[t,newemp]+=ura
            self.stat_toe[t,newemp]+=toe
            self.stat_pension[t,newemp]+=newpen
            self.stat_paidpension[t,newemp]+=paidpens
            self.stat_unemp_len[t,n]=tis
            self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura)
            self.popunemprightused[t,n]=bu
            self.infostats_group[n]=int(g)
            self.infostats_unempwagebasis[t,n]=uw
            self.infostats_unempwagebasis_acc[t,n]=uwr
            self.infostats_toe[t,n]=toe
            self.infostats_ove[t,newemp]+=ove
            self.infostats_kassanjasen[t]+=jasen
            self.infostats_pop_wage[t,n]=newsal
            self.infostats_pop_pension[t,n]=newpen
            self.infostats_children_under3[t,n]=c3
            self.infostats_children_under7[t,n]=c7
            self.infostats_children_under18[t,n]=c18

            if q is not None:
                self.add_taxes(t,q,newemp,n,g)

            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1                                    

    def add_v4(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p_tila_vanha,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_g,p_w,\
            p_newpen,p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,\
            p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,\
            kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,nw\
            =self.env.state_decode(newstate)

        t=int(np.round((a2-self.min_age)*self.inv_timestep))#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) # FIXME: tässä newemp>0
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2
                    
            self.empstate[t,newemp]+=1
            self.empstate[t,p_tila]+=1
            if newemp<15:
                self.alive[t]+=1
                self.galive[t,g]+=1
                #self.pop_alive[t,n]=1
                self.infostats_kassanjasen[t]+=jasen
                self.actions[t,n]=act[0]
                self.salaries_emp[t,newemp]+=newsal
                self.infostats_pinkslip[t,newemp]+=pink
                self.stat_toe[t,newemp]+=toe
                self.infostats_toe[t,n]=toe
                self.infostats_pop_pension[t,n]=newpen
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_after_ra[t,newemp]+=upr
                self.stat_unemp_len[t,n]=tis
                self.stat_wage_reduction[t,newemp]+=wr
                self.stat_wage_reduction_g[t,newemp,g]+=wr
                self.infostats_unempwagebasis[t,n]=uw
                self.infostats_unempwagebasis_acc[t,n]=uwr
                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila<15:
                self.alive[t]+=1
                self.galive[t,p_g]+=1
                #self.pop_alive[t,n+1]=1
                self.infostats_kassanjasen[t]+=jasen
                self.actions[t,n+1]=act[1]
                self.salaries_emp[t,p_tila]+=p_w
                self.infostats_pop_pinkslip[t,n]=pink
                self.infostats_pinkslip[t,p_tila]+=p_pink
                self.infostats_pop_pinkslip[t,n+1]=p_pink
                self.stat_toe[t,p_tila]+=p_toe
                self.infostats_toe[t,n+1]=p_toe
                self.infostats_pop_pension[t,n+1]=p_newpen
                self.stat_pension[t,p_tila]+=p_newpen
                self.stat_paidpension[t,p_tila]+=p_paidpens
                self.stat_unemp_after_ra[t,p_tila]+=p_unemp_after_ra
                self.stat_unemp_len[t,n+1]=p_tis
                self.stat_wage_reduction[t,p_tila]+=p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g]+=p_wr
                self.infostats_unempwagebasis[t,n+1]=p_uw
                self.infostats_unempwagebasis_acc[t,n+1]=p_uwr
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp]+=r
            self.rewstate[t,p_tila]+=r
            
            self.poprewstate[t,n]=r
            self.poprewstate[t,n+1]=r
            
            # spouse is a first-class citizen
            self.infostats_puoliso[t,n]=puoliso
            self.infostats_puoliso[t,n+1]=puoliso
            self.popempstate[t,n]=newemp
            self.popempstate[t,n+1]=p_tila
            self.time_in_state[t,newemp]+=tis
            self.time_in_state[t,p_tila]+=p_tis
            if tis<=0.25 and newemp==5:
                self.infostats_mother_in_workforce[t]+=1
            if p_tis<=0.25 and p_tila==5:
                self.infostats_mother_in_workforce[t]+=1
            #if tis<=0.25 and newemp==6:
            #    self.infostats_father_in_workforce[t]+=1
            #if p_tis<=0.25 and newemp==6:
            #    self.infostats_mother_in_workforce[t]+=1
            self.gempstate[t,newemp,g]+=1
            self.gempstate[t,p_tila,p_g]+=1
            self.stat_tyoura[t,newemp]+=ura
            self.stat_tyoura[t,p_tila]+=p_ura
            self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura)
            self.popunemprightleft[t,n+1]=-self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
            self.popunemprightused[t,n]=bu
            self.popunemprightused[t,n+1]=p_bu
            self.infostats_group[n]=int(g)
            self.infostats_group[n+1]=int(p_g)
            self.infostats_ove[t,newemp]+=ove
            self.infostats_ove[t,p_tila]+=p_ove
            self.infostats_ove_g[t,g,newemp]+=ove
            self.infostats_ove_g[t,p_g,p_tila]+=ove
            #self.infostats_pop_wage[t,n]=newsal
            #self.infostats_pop_wage[t,n+1]=p_w
            self.infostats_children_under3[t,[n,n+1]]=c3
            self.infostats_children_under7[t,[n,n+1]]=c7
            self.infostats_children_under18[t,[n,n+1]]=c18
            #if puoliso:
            #    self.infostats_children_under3[t,n+1]=c3
            #    self.infostats_children_under7[t,n+1]=c7
            #    self.infostats_children_under18[t,n+1]=c18

            # fixme
            #self.infostats_tyoelake[t]+=q[person+'elake_maksussa']*scale+q[person+'elake_maksussa']*scale

            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha]+=1
                self.siirtyneet_det[t,p_tila_vanha,p_tila]+=1
            else:
                self.pysyneet[t,p_tila_vanha]+=1
        elif newemp<0:
            self.deceiced[t]+=1                                    

    def add_v101(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,a,_,_=self.env.state_decode(state) # current employment state
        newemp,newpen,newsal,a2,next_wage,savings=self.env.state_decode(newstate)
        g=0
        bu=0
        ove=0
        jasen=0
        
        if q is not None:
            mod_sav_action=q['mod_sav_action']
        else:
            mod_sav_action=a[1]

        t=int(np.round((a2-self.min_age)*self.inv_timestep))
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            self.empstate[t,newemp]+=1
            self.alive[t]+=1
            self.rewstate[t,newemp]+=r
            self.infostats_tulot_netto[t]+=q['netto'] # already at annual level
            self.infostats_poptulot_netto[t,n]=q['netto']

            self.poprewstate[t,n]=r
            self.popempstate[t,n]=newemp
            #self.salaries[t,n]=newsal
            self.salaries_emp[t,newemp]+=newsal
            #self.time_in_state[t,newemp]+=tis
            self.infostats_equivalent_income[t]+=q['eq']
            self.infostats_pop_wage[t,n]=newsal
            self.infostats_pop_pension[t,n]=newpen
            if self.dynprog and pred_r is not None:
                self.pop_predrew[t,n]=pred_r

            self.infostats_savings[t,n]=savings
            self.actions[t,n]=act[0]
            self.sav_actions[t,n]=mod_sav_action

            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
        elif newemp<0:
            self.deceiced[t]+=1        
            

    def add_v104(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p_tila_vanha,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_g,p_w,\
            p_newpen,p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,\
            p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,\
            kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,\
            nw,savings,p_savings\
            =self.env.state_decode(newstate)

        t=int(np.round((a2-self.min_age)*self.inv_timestep))#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) # FIXME: tässä newemp>0
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2
                    
            if q is not None:
                mod_sav_action=q['mod_sav_action']
                mod_p_sav_action=q['mod_p_sav_action']
            else:
                mod_sav_action=a[2]
                mod_p_sav_action=a[3]
                    
            self.empstate[t,newemp]+=1
            self.empstate[t,p_tila]+=1
            if newemp<15:
                self.alive[t]+=1
                self.galive[t,g]+=1
                #self.pop_alive[t,n]=1
                self.infostats_kassanjasen[t]+=jasen
                self.actions[t,n]=act[0]
                self.salaries_emp[t,newemp]+=newsal
                self.infostats_pinkslip[t,newemp]+=pink
                self.infostats_pop_pinkslip[t,n]=pink
                self.stat_toe[t,newemp]+=toe
                self.infostats_toe[t,n]=toe
                self.infostats_pop_pension[t,n]=newpen
                self.stat_pension[t,newemp]+=newpen
                self.stat_paidpension[t,newemp]+=paidpens
                self.stat_unemp_after_ra[t,newemp]+=upr
                self.stat_unemp_len[t,n]=tis
                self.stat_wage_reduction[t,newemp]+=wr
                self.stat_wage_reduction_g[t,newemp,g]+=wr
                self.infostats_unempwagebasis[t,n]=uw
                self.infostats_unempwagebasis_acc[t,n]=uwr
                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                self.infostats_savings[t,n]=savings
                self.sav_actions[t,n]=mod_sav_action
                
            if p_tila<15:
                self.alive[t]+=1
                self.galive[t,p_g]+=1
                #self.pop_alive[t,n+1]=1
                self.infostats_kassanjasen[t]+=jasen
                self.actions[t,n+1]=act[1]
                self.salaries_emp[t,p_tila]+=p_w
                self.infostats_pinkslip[t,p_tila]+=p_pink
                self.infostats_pop_pinkslip[t,n+1]=p_pink
                self.stat_toe[t,p_tila]+=p_toe
                self.infostats_toe[t,n+1]=p_toe
                self.infostats_pop_pension[t,n+1]=p_newpen
                self.stat_pension[t,p_tila]+=p_newpen
                self.stat_paidpension[t,p_tila]+=p_paidpens
                self.stat_unemp_after_ra[t,p_tila]+=p_unemp_after_ra
                self.stat_unemp_len[t,n+1]=p_tis
                self.stat_wage_reduction[t,p_tila]+=p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g]+=p_wr
                self.infostats_unempwagebasis[t,n+1]=p_uw
                self.infostats_unempwagebasis_acc[t,n+1]=p_uwr
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                self.infostats_savings[t,n+1]=p_savings
                self.sav_actions[t,n+1]=mod_p_sav_action
                
            self.rewstate[t,newemp]+=r
            self.rewstate[t,p_tila]+=r
            
            self.poprewstate[t,n]=r
            self.poprewstate[t,n+1]=r
            
            # spouse is a first-class citizen
            self.infostats_puoliso[t,n]=puoliso
            self.infostats_puoliso[t,n+1]=puoliso
            self.popempstate[t,n]=newemp
            self.popempstate[t,n+1]=p_tila
            self.time_in_state[t,newemp]+=tis
            self.time_in_state[t,p_tila]+=p_tis
            if tis<=0.25 and newemp==5:
                self.infostats_mother_in_workforce[t]+=1
            if p_tis<=0.25 and p_tila==5:
                self.infostats_mother_in_workforce[t]+=1
            #if tis<=0.25 and newemp==6:
            #    self.infostats_father_in_workforce[t]+=1
            #if p_tis<=0.25 and newemp==6:
            #    self.infostats_mother_in_workforce[t]+=1
            self.gempstate[t,newemp,g]+=1
            self.gempstate[t,p_tila,p_g]+=1
            self.stat_tyoura[t,newemp]+=ura
            self.stat_tyoura[t,p_tila]+=p_ura
            self.popunemprightleft[t,n]=-self.env.unempright_left(newemp,tis,bu,a2,ura)
            self.popunemprightleft[t,n+1]=-self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
            self.popunemprightused[t,n]=bu
            self.popunemprightused[t,n+1]=p_bu
            self.infostats_group[n]=int(g)
            self.infostats_group[n+1]=int(p_g)
            self.infostats_ove[t,newemp]+=ove
            self.infostats_ove[t,p_tila]+=p_ove
            self.infostats_ove_g[t,g,newemp]+=ove
            self.infostats_ove_g[t,p_g,p_tila]+=ove
            #self.infostats_pop_wage[t,n]=newsal
            #self.infostats_pop_wage[t,n+1]=p_w
            self.infostats_children_under3[t,[n,n+1]]=c3
            self.infostats_children_under7[t,[n,n+1]]=c7
            self.infostats_children_under18[t,[n,n+1]]=c18
            #if puoliso:
            #    self.infostats_children_under3[t,n+1]=c3
            #    self.infostats_children_under7[t,n+1]=c7
            #    self.infostats_children_under18[t,n+1]=c18

            # fixme
            #self.infostats_tyoelake[t]+=q[person+'elake_maksussa']*scale+q[person+'elake_maksussa']*scale

            if aveV is not None:
                self.aveV[t,n]=aveV

            if not emp==newemp:
                self.siirtyneet[t,emp]+=1
                self.siirtyneet_det[t,emp,newemp]+=1
            else:
                self.pysyneet[t,emp]+=1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha]+=1
                self.siirtyneet_det[t,p_tila_vanha,p_tila]+=1
            else:
                self.pysyneet[t,p_tila_vanha]+=1
        elif newemp<0:
            self.deceiced[t]+=1 
      
    def save_sim(self,filename):
        f = h5py.File(filename, 'w')
        ftype='float64'
        _ = f.create_dataset('version', data=self.version, dtype=np.int64)
        _ = f.create_dataset('n_pop', data=self.n_pop, dtype=np.int64)
        _ = f.create_dataset('empstate', data=self.empstate, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('gempstate', data=self.gempstate, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('deceiced', data=self.deceiced, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('rewstate', data=self.rewstate, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('salaries_emp', data=self.salaries_emp, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('actions', data=self.actions, dtype=np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('alive', data=self.alive, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('galive', data=self.galive, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('siirtyneet', data=self.siirtyneet, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('siirtyneet_det', data=self.siirtyneet_det, dtype=np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('pysyneet', data=self.pysyneet, dtype=np.int64,compression="gzip", compression_opts=9)
        if self.dynprog:
            _ = f.create_dataset('aveV', data=self.aveV, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('pop_predrew', data=self.pop_predrew, dtype=ftype,compression="gzip", compression_opts=9)

        _ = f.create_dataset('time_in_state', data=self.time_in_state, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_tyoura', data=self.stat_tyoura, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_toe', data=self.stat_toe, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_pension', data=self.stat_pension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_paidpension', data=self.stat_paidpension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_pop_paidpension', data=self.stat_pop_paidpension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_unemp_len', data=self.stat_unemp_len, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('popempstate', data=self.popempstate, dtype=np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_wage_reduction', data=self.stat_wage_reduction, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_wage_reduction_g', data=self.stat_wage_reduction_g, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('popunemprightleft', data=self.popunemprightleft, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('popunemprightused', data=self.popunemprightused, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_taxes', data=self.infostats_taxes, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_wagetaxes', data=self.infostats_wagetaxes, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_taxes_distrib', data=self.infostats_taxes_distrib, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_etuustulo', data=self.infostats_etuustulo, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_etuustulo_group', data=self.infostats_etuustulo_group, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_perustulo', data=self.infostats_perustulo, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_palkkatulo', data=self.infostats_palkkatulo, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_palkkatulo_eielakkeella', data=self.infostats_palkkatulo_eielakkeella, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_ansiopvraha', data=self.infostats_ansiopvraha, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_asumistuki', data=self.infostats_asumistuki, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_valtionvero', data=self.infostats_valtionvero, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_kunnallisvero', data=self.infostats_kunnallisvero, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_valtionvero_distrib', data=self.infostats_valtionvero_distrib, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_kunnallisvero_distrib', data=self.infostats_kunnallisvero_distrib, dtype=ftype)
        _ = f.create_dataset('infostats_ptel', data=self.infostats_ptel, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_tyotvakmaksu', data=self.infostats_tyotvakmaksu, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_tyoelake', data=self.infostats_tyoelake, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_kansanelake', data=self.infostats_kansanelake, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_kokoelake', data=self.infostats_kokoelake, dtype=ftype,compression="gzip", compression_opts=9)
        #_ = f.create_dataset('infostats_takuuelake', data=self.infostats_takuuelake, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_elatustuki', data=self.infostats_elatustuki, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_lapsilisa', data=self.infostats_lapsilisa, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_opintotuki', data=self.infostats_opintotuki, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_isyyspaivaraha', data=self.infostats_isyyspaivaraha, dtype=ftype)
        _ = f.create_dataset('infostats_aitiyspaivaraha', data=self.infostats_aitiyspaivaraha, dtype=ftype)
        _ = f.create_dataset('infostats_kotihoidontuki', data=self.infostats_kotihoidontuki, dtype=ftype)
        _ = f.create_dataset('infostats_sairauspaivaraha', data=self.infostats_sairauspaivaraha, dtype=ftype)
        _ = f.create_dataset('infostats_toimeentulotuki', data=self.infostats_toimeentulotuki, dtype=ftype)
        _ = f.create_dataset('infostats_tulot_netto', data=self.infostats_tulot_netto, dtype=ftype)
        _ = f.create_dataset('poprewstate', data=self.poprewstate, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pinkslip', data=self.infostats_pinkslip, dtype=int)
        if self.version<2:
            _ = f.create_dataset('infostats_pop_pinkslip', data=self.infostats_pop_pinkslip, dtype=np.int8)

        _ = f.create_dataset('infostats_paid_tyel_pension', data=self.infostats_paid_tyel_pension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pop_kansanelake', data=self.infostats_pop_kansanelake, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pop_tyoelake', data=self.infostats_pop_tyoelake, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_tyelpremium', data=self.infostats_tyelpremium, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_npv0', data=self.infostats_npv0, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_irr_tyel_full', data=self.infostats_irr_tyel_full, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_irr_tyel_reduced', data=self.infostats_irr_tyel_reduced, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_group', data=self.infostats_group, dtype=np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_sairausvakuutus', data=self.infostats_sairausvakuutus, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pvhoitomaksu', data=self.infostats_pvhoitomaksu, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_ylevero', data=self.infostats_ylevero, dtype=ftype)
        _ = f.create_dataset('infostats_ylevero_distrib', data=self.infostats_ylevero_distrib, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_mother_in_workforce', data=self.infostats_mother_in_workforce, dtype=ftype)
        #_ = f.create_dataset('infostats_father_in_workforce', data=self.infostats_father_in_workforce, dtype=ftype)
        _ = f.create_dataset('infostats_children_under3', data=self.infostats_children_under3, dtype=np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_children_under7', data=self.infostats_children_under7, dtype=np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_children_under18', data=self.infostats_children_under18, dtype=np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_unempwagebasis', data=self.infostats_unempwagebasis, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_unempwagebasis_acc', data=self.infostats_unempwagebasis_acc, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_toe', data=self.infostats_toe, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_ove', data=self.infostats_ove, dtype=ftype)
        _ = f.create_dataset('infostats_ove_g', data=self.infostats_ove_g, dtype=ftype)
        _ = f.create_dataset('infostats_kassanjasen', data=self.infostats_kassanjasen, dtype=np.int64)
        _ = f.create_dataset('infostats_poptulot_netto', data=self.infostats_poptulot_netto, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_equivalent_income', data=self.infostats_equivalent_income, dtype=ftype)
        _ = f.create_dataset('infostats_pop_wage', data=self.infostats_pop_wage, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pop_pension', data=self.infostats_pop_pension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_alv', data=self.infostats_alv, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_unemp_after_ra', data=self.stat_unemp_after_ra, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('params', data=json.dumps(self.params))
        if self.version in set([101,104]):
            _ = f.create_dataset('infostats_savings', data=self.infostats_savings, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('sav_actions', data=self.sav_actions, dtype=ftype,compression="gzip", compression_opts=9)

        if self.version in set([4,104]):
            _ = f.create_dataset('infostats_puoliso', data=self.infostats_puoliso,  dtype=ftype,compression="gzip", compression_opts=9)

        f.close()

    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()

    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f.get(nimi).value
        f.close()
        return val

    def load_sim(self,filename,n_pop=None,print_pop=False):
        f = h5py.File(filename, 'r')

        if 'version' in f.keys():
            version=int(f['version'][()])
        else:
            version=1

        self.empstate=f['empstate'][()]
        self.gempstate=f['gempstate'][()]
        self.deceiced=f['deceiced'][()]
        self.rewstate=f['rewstate'][()]
        if 'poprewstate' in f.keys():
            self.poprewstate=f['poprewstate'][()]

        if 'pop_predrew' in f.keys():
            self.pop_predrew=f['pop_predrew'][()]

        self.salaries_emp=f['salaries_emp'][()]
        self.actions=f['actions'][()]
        self.alive=f['alive'][()]
        self.galive=f['galive'][()]
        self.siirtyneet=f['siirtyneet'][()]
        self.pysyneet=f['pysyneet'][()]
        if 'aveV' in f.keys():
            self.aveV=f['aveV'][()]
        self.time_in_state=f['time_in_state'][()]
        self.stat_tyoura=f['stat_tyoura'][()]
        self.stat_toe=f['stat_toe'][()]
        self.stat_pension=f['stat_pension'][()]
        self.stat_paidpension=f['stat_paidpension'][()]
        if 'stat_pop_paidpension' in f.keys():
            self.stat_pop_paidpension=f['stat_pop_paidpension'][()]
        self.stat_unemp_len=f['stat_unemp_len'][()]
        self.popempstate=f['popempstate'][()]
        self.stat_wage_reduction=f['stat_wage_reduction'][()]
        self.popunemprightleft=f['popunemprightleft'][()]
        self.popunemprightused=f['popunemprightused'][()]

        if 'infostats_wagetaxes' in f.keys(): # older runs do not have these
            self.infostats_wagetaxes=f['infostats_wagetaxes'][()]

        if 'infostats_taxes' in f.keys(): # older runs do not have these
            self.infostats_taxes=f['infostats_taxes'][()]
            self.infostats_etuustulo=f['infostats_etuustulo'][()]
            self.infostats_perustulo=f['infostats_perustulo'][()]
            self.infostats_palkkatulo=f['infostats_palkkatulo'][()]
            self.infostats_ansiopvraha=f['infostats_ansiopvraha'][()]
            self.infostats_asumistuki=f['infostats_asumistuki'][()]
            self.infostats_valtionvero=f['infostats_valtionvero'][()]
            self.infostats_kunnallisvero=f['infostats_kunnallisvero'][()]
            self.infostats_ptel=f['infostats_ptel'][()]
            self.infostats_tyotvakmaksu=f['infostats_tyotvakmaksu'][()]
            self.infostats_tyoelake=f['infostats_tyoelake'][()]
            self.infostats_kokoelake=f['infostats_kokoelake'][()]
            self.infostats_opintotuki=f['infostats_opintotuki'][()]
            self.infostats_isyyspaivaraha=f['infostats_isyyspaivaraha'][()]
            self.infostats_aitiyspaivaraha=f['infostats_aitiyspaivaraha'][()]
            self.infostats_kotihoidontuki=f['infostats_kotihoidontuki'][()]
            self.infostats_sairauspaivaraha=f['infostats_sairauspaivaraha'][()]
            self.infostats_toimeentulotuki=f['infostats_toimeentulotuki'][()]
            self.infostats_tulot_netto=f['infostats_tulot_netto'][()]

        if 'infostats_kansanelake' in f.keys(): # older runs do not have these
            self.infostats_kansanelake=f['infostats_kansanelake'][()]
            #self.infostats_takuuelake=f['infostats_takuuelake'][()]

        if 'infostats_lapsilisa' in f.keys(): # older runs do not have these
            self.infostats_lapsilisa=f['infostats_lapsilisa'][()]
            self.infostats_elatustuki=f['infostats_elatustuki'][()]

        if 'spouseactions' in f.keys(): # older runs do not have these
            self.spouseactions=f['spouseactions'][()]
        if 'infostats_etuustulo_group' in f.keys(): # older runs do not have these
            self.infostats_etuustulo_group=f['infostats_etuustulo_group'][()]

        if 'infostats_valtionvero_distrib' in f.keys(): # older runs do not have these
            self.infostats_valtionvero_distrib=f['infostats_valtionvero_distrib'][()]
            self.infostats_kunnallisvero_distrib=f['infostats_kunnallisvero_distrib'][()]

        if 'infostats_taxes_distrib' in f.keys(): # older runs do not have these
            self.infostats_taxes_distrib=f['infostats_taxes_distrib'][()]
            self.infostats_ylevero_distrib=f['infostats_ylevero_distrib'][()]

        if 'infostats_pinkslip' in f.keys(): # older runs do not have these
            self.infostats_pinkslip=f['infostats_pinkslip'][()]

        if 'infostats_pop_pinkslip' in f.keys():
            self.infostats_pop_pinkslip=f['infostats_pop_pinkslip'][()]

        if 'infostats_paid_tyel_pension' in f.keys(): # older runs do not have these
            self.infostats_paid_tyel_pension=f['infostats_paid_tyel_pension'][()]
            self.infostats_tyelpremium=f['infostats_tyelpremium'][()]

        if 'infostats_pop_tyoelake' in f.keys(): # older runs do not have these
            self.infostats_pop_tyoelake=f['infostats_pop_tyoelake'][()]
            self.infostats_pop_kansanelake=f['infostats_pop_kansanelake'][()]

        if 'infostats_npv0' in f.keys(): # older runs do not have these
            self.infostats_npv0=f['infostats_npv0'][()]

        if 'infostats_irr_tyel_full' in f.keys(): # older runs do not have these
            self.infostats_irr=f['infostats_irr_tyel_full'][()]
            self.infostats_irr=f['infostats_irr_tyel_reduced'][()]

        #if 'infostats_chilren7' in f.keys(): # older runs do not have these
        #    self.infostats_chilren7=f['infostats_chilren7'][()]
        #if 'infostats_chilren18' in f.keys(): # older runs do not have these
        #    self.infostats_chilren18=f['infostats_chilren18'][()]

        if 'infostats_group' in f.keys(): # older runs do not have these
            self.infostats_group=f['infostats_group'][()]

        if 'infostats_sairausvakuutus' in f.keys():
            self.infostats_sairausvakuutus=f['infostats_sairausvakuutus'][()]
            self.infostats_pvhoitomaksu=f['infostats_pvhoitomaksu'][()]
            self.infostats_ylevero=f['infostats_ylevero'][()]

        if 'infostats_mother_in_workforce' in f.keys():
            self.infostats_mother_in_workforce=f['infostats_mother_in_workforce'][()]
        #if 'infostats_father_in_workforce' in f.keys():
        #    self.infostats_father_in_workforce=f['infostats_father_in_workforce'][()]

        if 'stat_wage_reduction_g' in f.keys():
            self.stat_wage_reduction_g=f['stat_wage_reduction_g'][()]

        if 'siirtyneet_det' in f.keys():
            self.siirtyneet_det=f['siirtyneet_det'][()]

        if 'infostats_kassanjasen' in f.keys():
            self.infostats_kassanjasen=f['infostats_kassanjasen'][()]

        if 'n_pop' in f.keys():
            self.n_pop=int(f['n_pop'][()])
        else:
            self.n_pop=np.sum(self.empstate[0,:])

        if 'infostats_puoliso' in f.keys():
            self.infostats_puoliso=f['infostats_puoliso'][()]

        if 'infostats_ove' in f.keys():
            self.infostats_ove=f['infostats_ove'][()]
        if 'infostats_ove_g' in f.keys():
            self.infostats_ove_g=f['infostats_ove_g'][()]

        if 'infostats_children_under3' in f.keys():
            self.infostats_children_under3=f['infostats_children_under3'][()]
            self.infostats_children_under7=f['infostats_children_under7'][()]
            self.infostats_unempwagebasis=f['infostats_unempwagebasis'][()]
            self.infostats_unempwagebasis_acc=f['infostats_unempwagebasis_acc'][()]
            self.infostats_toe=f['infostats_toe'][()]

        if 'infostats_children_under18' in f.keys():
            self.infostats_children_under18=f['infostats_children_under18'][()]
            
        if 'infostats_palkkatulo_eielakkeella' in f.keys():
            self.infostats_palkkatulo_eielakkeella=f['infostats_palkkatulo_eielakkeella'][()]

        if 'infostats_tulot_netto' in f.keys():
            self.infostats_tulot_netto=f['infostats_tulot_netto'][()]

        if 'infostats_poptulot_netto' in f.keys():
            self.infostats_poptulot_netto=f['infostats_poptulot_netto'][()]

        if 'infostats_equivalent_income' in f.keys():
            self.infostats_equivalent_income=f['infostats_equivalent_income'][()]

        if 'infostats_pop_wage' in f.keys():
            self.infostats_pop_wage=f['infostats_pop_wage'][()]
            self.infostats_pop_pension=f['infostats_pop_pension'][()]

        if 'infostats_savings' in f.keys():
            self.infostats_savings=f['infostats_savings'][()]
            self.sav_actions=f['sav_actions'][()]

        if 'infostats_alv' in f.keys():
            self.infostats_alv=f['infostats_alv'][()]
            
        if 'stat_unemp_after_ra' in f.keys():
            self.stat_unemp_after_ra=f['stat_unemp_after_ra'][()]

        if n_pop is not None:
            self.n_pop=n_pop

        if 'params' in f.keys():
            self.params=f['params'][()]
            try:
                self.params = json.loads(self.params)
            except:
                self.params = str(self.params)
        else:
            self.params=None

        if print_pop:
            print('n_pop {}'.format(self.n_pop))

        f.close()
        
    def scale_error(self,x,target=None,averaged=False):
        return (target-self.comp_scaled_consumption(x,averaged=averaged))


    def plot_various_groups(self,figname=None):
        empstate_ratio=100*self.empstate/self.alive
        ratio_label=self.labels['osuus']
        if figname is not None:
            self.plot_states(empstate_ratio,ylabel=ratio_label,stack=True,figname=figname+'_stack')
        else:
            self.plot_states(empstate_ratio,ylabel=ratio_label,stack=True)
        if figname is not None:
            self.plot_states(empstate_ratio,ylabel=ratio_label,start_from=60,stack=True,figname=figname+'_stack60')
            self.plot_states(empstate_ratio,ylabel=ratio_label,start_from=57,stack=True,figname=figname+'_stack60')
        else:
            self.plot_states(empstate_ratio,ylabel=ratio_label,start_from=60,stack=True)
            self.plot_states(empstate_ratio,ylabel=ratio_label,start_from=57,stack=True)

        if self.version in set([1,2,3,4,104]):
            self.plot_states(empstate_ratio,ylabel=ratio_label,ylimit=20,stack=False)
            self.plot_states(empstate_ratio,ylabel=ratio_label,unemp=True,stack=False)
            
    def compare_against(self,cc=None,cctext='toteuma'):
        if self.version in set([1,2,3,4,104]):
            q=self.comp_budget(scale=True)
            if cc is None:
                q_stat=self.empstats.stat_budget()
            else:
                q_stat=cc.comp_budget(scale=True)
                
            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['e/v'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=[cctext])
            df=df1.copy()
            df[cctext]=df2[cctext]
            df['ero']=df1['e/v']-df2[cctext]

            print('Rahavirrat skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

            q=self.comp_participants(scale=True,lkm=False)
            q_lkm=self.comp_participants(scale=True,lkm=True)
            if cc is None:
                q_stat=self.empstats.stat_participants()
                q_days=self.empstats.stat_days()
            else:
                q_stat=cc.comp_participants(scale=True,lkm=True)
                q_days=cc.comp_participants(scale=True,lkm=False)

            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['arvio (htv)'])
            df2 = pd.DataFrame.from_dict(q_days,orient='index',columns=[cctext+' (htv)'])
            df4 = pd.DataFrame.from_dict(q_lkm,orient='index',columns=['arvio (kpl)'])
            df5 = pd.DataFrame.from_dict(q_stat,orient='index',columns=[cctext+' (kpl)'])

            df=df1.copy()
            df[cctext+' (htv)']=df2[cctext+' (htv)']
            df['ero (htv)']=df['arvio (htv)']-df[cctext+' (htv)']

            print('Henkilövuosia tiloissa skaalattuna väestötasolle henkilövuosina')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.0f"))

            df=df4.copy()
            df[cctext+' (kpl)']=df5[cctext+' (kpl)']
            df['ero (kpl)']=df['arvio (kpl)']-df[cctext+' (kpl)']
            print('Henkilöiden lkm tiloissa skaalattuna väestötasolle (keskeneräinen)')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.0f"))
        else:
            q=self.comp_participants(scale=True)
            q_stat=self.empstats.stat_participants()
            q_days=self.empstats.stat_days()
            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['arvio (htv)'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=['toteuma'])
            df3 = pd.DataFrame.from_dict(q_days,orient='index',columns=['htv_tot'])

            df=df1.copy()
            df['toteuma (kpl)']=df2['toteuma']
            df['toteuma (htv)']=df3['htv_tot']
            df['ero (htv)']=df['arvio (htv)']-df['toteuma (htv)']

            print('Henkilöitä tiloissa skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.0f"))    

    def plot_results(self,grayscale=False,figname=None):

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        print_html('<h1>Statistics</h1>')
        
        self.comp_total_netincome()
        #self.plot_rewdist()

        if self.version in set([1,2,3,4,104]):
            self.compare_against()
        else:
            q=self.comp_participants(scale=True)
            q_stat=self.empstats.stat_participants(lkm=False)
            q_days=self.empstats.stat_days()
            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['arvio (htv)'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=['toteuma'])
            df3 = pd.DataFrame.from_dict(q_days,orient='index',columns=['htv_tot'])

            df=df1.copy()
            df['toteuma (kpl)']=df2['toteuma']
            df['toteuma (htv)']=df3['htv_tot']
            df['ero (htv)']=df['arvio (htv)']-df['toteuma (htv)']

            print('Henkilöitä tiloissa skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.0f"))

        print_html('<h2>Tilastot</h2>')

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.comp_employed_ratio(self.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1,osatyolliset1,kokotyolliset1,osata1,kokota1=self.comp_tyollisyys_stats(self.empstate/self.n_pop,scale_time=True,start=18,end=65,full=True)

        tyollaste=tyollaste1*100
        print('\nSic! Työllisyysaste vastaa työvoimatilaston laskutapaa!')
        print(f'Työllisyysaste 18-64: {tyollaste:.2f}%')

        self.comp_total_reward()
        self.comp_total_reward(discounted=True)
        
        gini=self.empstats.get_gini(self.year)

        print('Gini coefficient is {:.3f} (havainto {:.3f})'.format(self.comp_gini(),gini))

        print('\nSic! pienituloisuus lasketaan vain aikuisväestöstä!')
        abs_pienituloisuus=12000
        p50,p60,pt=self.comp_pienituloisuus(level=abs_pienituloisuus)
        print('Pienituloisuus 50%: {:.2f}%; 60%: {:.2f}%; abs 1000 e/kk {:.2f}%'.format(100*p50,100*p60,100*pt))

        print_html('<h2>Sovite</h2>')

        print('Real discounted reward {}'.format(self.comp_realoptimrew()))
        #print('vrt reward {}'.format(self.comp_scaled_consumption(np.array([0]))))
        real=self.comp_presentvalue()
        print('Initial discounted reward {}'.format(np.mean(real[1,:])))

        print_html('<h2>Työssä</h2>')
        self.plot_emp(figname=figname)
        if self.version in set([1,2,3,4,104]):
            self.plot_gender_emp(figname=figname)
            self.plot_group_emp()
            self.plot_parttime_ratio(figname=figname)
            self.plot_workforce()
            self.plot_various_groups()

        if self.version in set([4,104]):
            print_html('<h2>Lapset ja puolisot</h2>')
            self.plot_spouse()
            self.plot_children()
            self.plot_family()
            self.plot_parents_in_work()
            
        if self.version in set([101,104]):
            print_html('<h2>Säästöt</h2>')
            self.plot_savings()

        print_html('<h2>Tulot</h2>')
        if self.version in set([3,4,104]):
            self.plot_tulot()

        self.plot_sal()

        print_html('<h2>Työttömyys</h2>')
        self.plot_toe()
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        keskikesto=self.comp_unemp_durations()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        keskikesto=self.comp_unemp_durations_v2()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

        self.plot_unemp_after_ra()

        if self.version in set([1,2,3,4,104]):
            print('Lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki()))

        self.plot_unemp(unempratio=True,figname=figname)
        self.plot_unemp(unempratio=False)

        if self.version in set([1,2,3,4,104]):
            self.plot_unemp_shares()
            self.plot_kassanjasen()
            self.plot_pinkslip()

        #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, no max age',ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää',ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50,figname=figname)

        if self.version in set([1,2,3,4,104]):
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=True,ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50)
            self.plot_distrib(label='Jakauma ansiosidonnainen+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=False,ansiosid=True,tmtuki=False,putki=True,outsider=False,max_age=50)
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea',ansiosid=True,tmtuki=True,putki=False,outsider=False)
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea, max Ikä 50v',ansiosid=True,tmtuki=True,putki=False,outsider=False,max_age=50)
            self.plot_distrib(label='Jakauma tmtuki',ansiosid=False,tmtuki=True,putki=False,outsider=False)
            #self.plot_distrib(label='Jakauma työvoiman ulkopuoliset',ansiosid=False,tmtuki=False,putki=False,outsider=True)
            #self.plot_distrib(label='Jakauma laaja (ansiosidonnainen+tmtuki+putki+ulkopuoliset)',laaja=True)
            
        if self.version in set([1,2,3,4,104]):
            print_html('<h2>Verot</h2>')
            self.plot_taxes()
            print_html('<h2>Kestot</h2>')
            self.plot_ave_stay()
            self.plot_career()
            print_html('<h2>Ove</h2>')
            self.plot_ove()
            print_html('<h2>Palkan reduktio</h2>')
            self.plot_wage_reduction()
            print_html('<h2>Hyöty</h2>')
            self.plot_reward()
            print_html('<h2>Siirtymät</h2>')
            self.plot_alive()
            self.plot_moved()

    def plot_unemp_after_ra(self):
        self.plot_ratiostates(self.stat_unemp_after_ra,ylabel='Unemp after ret.age',stack=False,start_from=60,end_at=70)

    def plot_all_pensions(self):
        self.plot_group_disab()
        self.plot_group_disab(xstart=60,xend=67)
        alivemask=(self.popempstate==self.env.get_mortstate()) # pois kuolleet
        kemask=(self.infostats_pop_kansanelake<0.1)
        temask=(self.infostats_pop_kansanelake>0.1) # pois kansaneläkkeen saajat
        notemask=(self.infostats_pop_tyoelake>10.0) # pois kansaneläkkeen saajat
        self.plot_pensions()
        self.plot_pension_stats(self.stat_pop_paidpension/self.timestep,65,'kokoeläke ilman kuolleita',mask=alivemask)
        self.plot_pension_stats(self.infostats_pop_tyoelake/self.timestep,65,'työeläke')
        self.plot_pension_stats(self.infostats_paid_tyel_pension/self.timestep,65,'työeläkemaksun vastine')
        self.plot_pension_stats(self.infostats_paid_tyel_pension/self.timestep,65,'työeläkemaksun vastine, vain työeläke',mask=temask)
        self.plot_pension_stats(self.infostats_pop_tyoelake/self.timestep,65,'vain työeläke',mask=temask)
        self.plot_pension_stats(self.infostats_pop_kansanelake/self.timestep,65,'kansanelake kaikki',max_pen=10_000,plot_ke=True)
        self.plot_pension_stats(self.infostats_pop_kansanelake/self.timestep,65,'kansanelake>0',max_pen=10_000,mask=kemask,plot_ke=True)
        self.plot_pension_stats(self.infostats_pop_kansanelake/self.timestep,65,'kansaneläke, ei työeläkettä',max_pen=10_000,mask=notemask,plot_ke=True)
        self.plot_pension_stats(self.infostats_pop_tyoelake/self.timestep,65,'työeläke, jos kansanelake>0',max_pen=20_000,mask=kemask)
        self.plot_pension_stats(self.infostats_pop_pension,60,'tulevat eläkkeet')
        self.plot_pension_stats(self.infostats_pop_pension,60,'tulevat eläkkeet, vain elossa',mask=alivemask)
        self.plot_pension_time()

    def comp_employed_ratio_by_age(self,emp=None,grouped=False,g=0):
        if emp is None:
            if grouped:
                emp=np.squeeze(self.gempstate[:,:,g])
            else:
                emp=self.empstate

        nn=np.sum(emp,1)
        if self.minimal:
            tyoll_osuus=(emp[:,1]+emp[:,3])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,3])/nn
            tyoll_osuus=np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            htv_osuus=np.reshape(htv_osuus,(htv_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])
            htv_osuus=(emp[:,1]+0.5*emp[:,8]+emp[:,9]+0.5*emp[:,10])

            tyoll_osuus=np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            htv_osuus=np.reshape(htv_osuus,(htv_osuus.shape[0],1))

        return tyoll_osuus,htv_osuus

    def comp_employed_aggregate(self,emp=None,start=20,end=63.5,grouped=False,g=0):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        nn=np.sum(emp,1)

        if self.minimal:
            tyoll_osuus=(emp[:,1]+emp[:,3])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,3])/nn
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,8]+emp[:,9]+0.5*emp[:,10])/nn

        htv_osuus=self.comp_state_stats(htv_osuus,start=start,end=end,ratio=True)
        tyoll_osuus=self.comp_state_stats(tyoll_osuus,start=start,end=end,ratio=True)

        return tyoll_osuus,htv_osuus

    def comp_group_ps(self):
        return self.comp_palkkasumma(grouped=True)

    def comp_palkkasumma(self,start=19,end=68,grouped=False,scale_time=True):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1

        if grouped:
            scalex=demog2/self.n_pop*self.timestep
            ps=np.zeros((self.n_time,6))
            ps_norw=np.zeros((self.n_time,6))
            a_ps=np.zeros(6)
            a_ps_norw=np.zeros(6)
            for k in range(self.n_pop):
                g=int(self.infostats_group[k,0])
                for t in range(min_cage,max_cage):
                    e=int(self.popempstate[t,k])
                    if e in set([1,10]):
                        ps[t,g]+=self.infostats_pop_wage[t,k]
                        ps_norw[t,g]+=self.infostats_pop_wage[t,k]
                    elif e in set([8,9]):
                        ps[t,g]+=self.infostats_pop_wage[t,k]*self.timestep
            for g in range(6):
                a_ps[g]=np.sum(scalex[min_cage:max_cage]*ps[min_cage:max_cage,g])
                a_ps_norw[g]=np.sum(scalex[min_cage:max_cage]*ps_norw[min_cage:max_cage,g])
        else:
            scalex=demog2/self.n_pop*self.timestep
            ps=np.zeros((self.n_time,1))
            ps_norw=np.zeros((self.n_time,1))

            for k in range(self.n_pop):
                for t in range(min_cage,max_cage):
                    e=int(self.popempstate[t,k])
                    if e in set([1,10]):
                        ps[t,0]+=self.infostats_pop_wage[t,k]
                        ps_norw[t,0]+=self.infostats_pop_wage[t,k]
                    elif e in set([8,9]):
                        ps[t,0]+=self.infostats_pop_wage[t,k]

            a_ps=np.sum(scalex[min_cage:max_cage]*ps[min_cage:max_cage])
            a_ps_norw=np.sum(scalex[min_cage:max_cage]*ps_norw[min_cage:max_cage])

        return a_ps,a_ps_norw

    def comp_stats_agegroup(self,border=[19,35,50]):
        n_groups=len(border)
        low=border.copy()
        high=border.copy()
        high[0:n_groups-1]=border[1:n_groups]
        high[-1]=65
        employed=np.zeros(n_groups)
        unemployed=np.zeros(n_groups)
        ahtv=np.zeros(n_groups)
        parttimeratio=np.zeros(n_groups)
        unempratio=np.zeros(n_groups)
        empratio=np.zeros(n_groups)
        i_ps=np.zeros(n_groups)
        i_ps_norw=np.zeros(n_groups)
        for n in range(n_groups):
            l=low[n]
            h=high[n]
            htv,tyollvaikutus,tyollaste,tyotosuus,tyottomat,osatyollaste=\
                self.comp_tyollisyys_stats(self.empstate,scale_time=True,start=l,end=h,agegroups=True)
            ps,ps_norw=self.comp_palkkasumma(start=l,end=h)

            print(f'l {l} h {h}\nhtv {htv}\ntyollaste {tyollaste}\ntyotosuus {tyotosuus}\ntyottomat {tyottomat}\nosatyollaste {osatyollaste}\nps {ps}')

            employed[n]=tyollvaikutus
            ahtv[n]=htv
            unemployed[n]=tyottomat
            unempratio[n]=tyotosuus
            empratio[n]=tyollaste
            parttimeratio[n]=osatyollaste
            i_ps[n]=ps
            i_ps_norw[n]=ps_norw

        return employed,ahtv,unemployed,parttimeratio,i_ps,i_ps_norw,unempratio,empratio


    def comp_unemployed_ratio_by_age(self,emp=None,grouped=False,g=0):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        nn=np.sum(emp,1)
        if self.minimal:
            tyot_osuus=emp[:,0]/nn
            tyot_osuus=np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])[:,None]
            #tyot_osuus=np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))

        return tyot_osuus

    def comp_unemployed_aggregate(self,emp=None,start=20,end=63.5,scale_time=True,grouped=False,g=0):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        nn=np.sum(emp,1)

        if self.minimal:
            tyot_osuus=emp[:,0]/nn
        else:
            tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])/nn

        #print(f'tyot_osuus {tyot_osuus}')
        unemp=self.comp_state_stats(tyot_osuus,start=start,end=end,ratio=True)

        return unemp

    def comp_parttime_aggregate(self,emp=None,start=20,end=63.5,scale_time=True,grouped=False,g=0):
        '''
        Lukumäärätiedot (EI HTV!)
        '''

        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        nn=np.sum(emp,1)

        if not self.minimal:
            tyossa=(emp[:,1]+emp[:,10]+emp[:,8]+emp[:,9])/nn
            osatyossa=(emp[:,10]+emp[:,8])/nn
        else:
            tyossa=emp[:,1]/nn
            osatyossa=0*tyossa

        osatyo_osuus=osatyossa/tyossa
        osatyo_osuus=self.comp_state_stats(osatyo_osuus,start=start,end=end,ratio=True)
        kokotyo_osuus=1-osatyo_osuus

        return kokotyo_osuus,osatyo_osuus

    def comp_parttime_ratio_by_age(self,emp=None,grouped=False,g=0):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        nn=np.sum(emp,1)

        if self.minimal:
            kokotyo_osuus=(emp[:,1])/nn
            osatyo_osuus=(emp[:,3])/nn
        else:
            if grouped:
                for g in range(6):
                    kokotyo_osuus=(emp[:,1,g]+emp[:,9,g])/nn
                    osatyo_osuus=(emp[:,8,g]+emp[:,10,g])/nn
            else:
                kokotyo_osuus=(emp[:,1]+emp[:,9])/nn
                osatyo_osuus=(emp[:,8]+emp[:,10])/nn

        osatyo_osuus=np.reshape(osatyo_osuus,(osatyo_osuus.shape[0],1))
        kokotyo_osuus=np.reshape(kokotyo_osuus,(osatyo_osuus.shape[0],1))

        return kokotyo_osuus,osatyo_osuus

    def comp_employed_ratio(self,emp):
        tyoll_osuus,htv_osuus=self.comp_employed_ratio_by_age(emp)
        tyot_osuus=self.comp_unemployed_ratio_by_age(emp)
        kokotyo_osuus,osatyo_osuus=self.comp_parttime_ratio_by_age(emp)

        return tyoll_osuus,htv_osuus,tyot_osuus,kokotyo_osuus,osatyo_osuus

    def comp_unemployed_detailed(self,emp):
        if self.minimal:
            ansiosid_osuus=emp[:,0]/np.sum(emp,1)
            tm_osuus=ansiosid_osuus*0
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            ansiosid_osuus=(emp[:,0]+emp[:,4])/np.sum(emp,1)
            tm_osuus=(emp[:,13])/np.sum(emp,1)

        return ansiosid_osuus,tm_osuus

    def comp_children(self,scale_time=True):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        start=self.min_age
        end=self.max_age
        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+2

        scalex=demog2[min_cage:max_cage]/self.n_pop*scale
        scalex2=demog2[min_cage:max_cage]*scale
        
        c3=np.sum(self.infostats_children_under3[min_cage:max_cage,:]*scalex,axis=1)
        c7=np.sum(self.infostats_children_under7[min_cage:max_cage,:]*scalex,axis=1)
        c18=np.sum(self.infostats_children_under18[min_cage:max_cage,:]*scalex,axis=1)
        
        return c3,c7,c18

    def comp_tyollisyys_stats(self,emp,scale_time=True,start=19,end=68,full=False,tyot_stats=False,agg=False,shapes=False,only_groups=False,g=0,agegroups=False):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1

        scalex=demog2[min_cage:max_cage]/self.n_pop*scale
        scalex2=demog2[min_cage:max_cage]*scale

        if only_groups:
            tyollosuus,htvosuus,tyot_osuus,kokotyo_osuus,osatyo_osuus=self.comp_employed_ratio(emp)
        else:
            tyollosuus,htvosuus,tyot_osuus,kokotyo_osuus,osatyo_osuus=self.comp_employed_ratio(emp)

        htv=np.sum(scalex2*htvosuus[min_cage:max_cage])
        tyollvaikutus=np.sum(scalex2*tyollosuus[min_cage:max_cage])
        tyottomat=np.sum(scalex2*tyot_osuus[min_cage:max_cage])
        osatyollvaikutus=np.sum(scalex2*osatyo_osuus[min_cage:max_cage])
        kokotyollvaikutus=np.sum(scalex2*kokotyo_osuus[min_cage:max_cage])
        haj=np.mean(np.std(tyollosuus[min_cage:max_cage]))

        tyollaste=tyollvaikutus/(np.sum(scalex)*self.n_pop)
        osatyollaste=osatyollvaikutus/(kokotyollvaikutus+osatyollvaikutus)
        kokotyollaste=kokotyollvaikutus/(kokotyollvaikutus+osatyollvaikutus)

        if tyot_stats:
            if agg:
                #d2=np.squeeze(demog2)
                tyolliset_osuus=np.squeeze(tyollosuus)
                tyottomat_osuus=np.squeeze(tyot_osuus)

                return tyolliset_ika,tyottomat_ika,htv_ika,tyolliset_osuus,tyottomat_osuus
            else:
                d2=np.squeeze(demog2)
                tyolliset_ika=np.squeeze(scale*d2*np.squeeze(htvosuus))
                tyottomat_ika=np.squeeze(scale*d2*np.squeeze(tyot_osuus))
                htv_ika=np.squeeze(scale*d2*np.squeeze(htvosuus))
                tyolliset_osuus=np.squeeze(tyollosuus)
                tyottomat_osuus=np.squeeze(tyot_osuus)

                return tyolliset_ika,tyottomat_ika,htv_ika,tyolliset_osuus,tyottomat_osuus
        elif full:
            return htv,tyollvaikutus,haj,tyollaste,tyollosuus,osatyollvaikutus,kokotyollvaikutus,osatyollaste,kokotyollaste
        elif agegroups:
            tyot_osuus=self.comp_unemployed_aggregate(start=start,end=end)
            return htv,tyollvaikutus,tyollaste,tyot_osuus,tyottomat,osatyollaste
        else:
            return htv,tyollvaikutus,haj,tyollaste,tyollosuus

    def comp_employment_stats(self,scale_time=True,returns=False):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(self.min_age)
        max_cage=self.map_age(self.max_age)+1

        scalex=np.squeeze(demog2/self.n_pop*self.timestep)

        d=np.squeeze(demog2[min_cage:max_cage])

        self.ratiostates=self.empstate/self.alive
        self.demogstates=(self.empstate.T*scalex).T
        if self.minimal>0:
            self.stats_employed=self.demogstates[:,0]+self.demogstates[:,3]
            self.stats_parttime=self.demogstates[:,3]
            self.stats_unemployed=self.demogstates[:,0]
            self.stats_all=np.sum(self.demogstates,1)
        else:
            self.stats_employed=self.demogstates[:,0]+self.demogstates[:,10]+self.demogstates[:,8]+self.demogstates[:,9]
            self.stats_parttime=self.demogstates[:,10]+self.demogstates[:,8]
            self.stats_unemployed=self.demogstates[:,0]+self.demogstates[:,4]+self.demogstates[:,13]
            self.stats_all=np.sum(self.demogstates,1)

        if returns:
            return self.stats_employed,self.stats_parttime,self.stats_unemployed


    def comp_participants(self,scale=True,include_retwork=True,grouped=False,g=0,lkm=False):
        '''
        Laske henkilöiden lkm / htv

        scalex olettaa, että naisia & miehiä yhtä paljon. Tämän voisi tarkentaa.
        '''
        demog2=self.empstats.get_demog()

        scalex=np.squeeze(demog2/self.n_pop*self.timestep)
        if lkm:
            scalex_lkm=np.squeeze(demog2/self.n_pop*self.timestep)
            osa_aika_kerroin=1.0
        else:
            scalex_lkm=np.squeeze(demog2/self.n_pop*self.timestep)
            osa_aika_kerroin=0.5

        q={}
        if self.version in set([1,2,3,4,104]):
            if grouped:
                #print('group=',g)
                emp=np.squeeze(self.gempstate[:,:,g])
                q['yhteensä']=np.sum(np.sum(emp,axis=1)*scalex)
                if include_retwork:
                    q['palkansaajia']=np.sum((emp[:,1]+osa_aika_kerroin*emp[:,10]+osa_aika_kerroin*emp[:,8]+emp[:,9])*scalex)
                else:
                    q['palkansaajia']=np.sum((emp[:,1]+osa_aika_kerroin*emp[:,10])*scalex)

                q['työssä ja eläkkeellä']=osa_aika_kerroin*np.sum(osa_aika_kerroin*emp[:,8]*scalex_lkm)+np.sum(emp[:,9]*scalex_lkm)
                q['työssä yli 63v']=np.sum(np.sum(emp[self.map_age(63):,[1,9]],axis=1)*scalex_lkm[self.map_age(63):])+osa_aika_kerroin*np.sum(np.sum(emp[self.map_age(63):,[8,10]],axis=1)*scalex_lkm[self.map_age(63):])
                q['ansiosidonnaisella']=np.sum((emp[:,0]+emp[:,4])*scalex_lkm)
                q['tmtuella']=np.sum(emp[:,13]*scalex_lkm)
                q['isyysvapaalla']=np.sum(emp[:,6]*scalex_lkm)
                q['kotihoidontuella']=np.sum(emp[:,7]*scalex_lkm)
                q['vanhempainvapaalla']=np.sum(emp[:,5]*scalex_lkm)
                q['ovella']=np.sum(np.sum(self.infostats_ove,axis=1)*scalex)
            else:
                q['yhteensä']=np.sum(np.sum(self.empstate[:,:],axis=1)*scalex)
                if include_retwork:
                    q['palkansaajia']=np.sum((self.empstate[:,1]+osa_aika_kerroin*self.empstate[:,10]+osa_aika_kerroin*self.empstate[:,8]+self.empstate[:,9])*scalex)
                else:
                    q['palkansaajia']=np.sum((self.empstate[:,1]+osa_aika_kerroin*self.empstate[:,10])*scalex)

                q['työssä ja eläkkeellä']=osa_aika_kerroin*np.sum(self.empstate[:,8]*scalex_lkm)+np.sum(self.empstate[:,9]*scalex_lkm)
                q['työssä yli 63v']=np.sum(np.sum(self.empstate[self.map_age(63):,[1,9]],axis=1)*scalex_lkm[self.map_age(63):])+osa_aika_kerroin*np.sum(np.sum(self.empstate[self.map_age(63):,[8,10]],axis=1)*scalex_lkm[self.map_age(63):])
                q['ansiosidonnaisella']=np.sum((self.empstate[:,0]+self.empstate[:,4])*scalex_lkm)
                q['tmtuella']=np.sum(self.empstate[:,13]*scalex_lkm)
                q['isyysvapaalla']=np.sum(self.empstate[:,6]*scalex_lkm)
                q['kotihoidontuella']=np.sum(self.empstate[:,7]*scalex_lkm)
                q['vanhempainvapaalla']=np.sum(self.empstate[:,5]*scalex_lkm)
                q['ovella']=np.sum(np.sum(self.infostats_ove,axis=1)*scalex)
        else:
            q['yhteensä']=np.sum(np.sum(self.empstate[:,:],1)*scalex)
            q['palkansaajia']=np.sum((self.empstate[:,1])*scalex)
            q['ansiosidonnaisella']=np.sum((self.empstate[:,0])*scalex)
            q['tmtuella']=np.sum(self.empstate[:,1]*0)
            q['isyysvapaalla']=np.sum(self.empstate[:,1]*0)
            q['kotihoidontuella']=np.sum(self.empstate[:,1]*0)
            q['vanhempainvapaalla']=np.sum(self.empstate[:,1]*0)

        return q

    def comp_employment_groupstats(self,scale_time=True,g=0,include_retwork=True,grouped=True):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        #min_cage=self.map_age(self.min_age)
        #max_cage=self.map_age(self.max_age)+1

        scalex=np.squeeze(demog2/self.n_pop*scale)

        #d=np.squeeze(demog2[min_cage:max_cage])

        if grouped:
            ratiostates=np.squeeze(self.gempstate[:,:,g])/self.alive
            demogstates=np.squeeze(self.gempstate[:,:,g])
        else:
            ratiostates=self.empstate[:,:]/self.alive
            demogstates=self.empstate[:,:]

        if self.version in set([1,2,3,4,104]):
            if include_retwork:
                stats_employed=np.sum((demogstates[:,1]+demogstates[:,9])*scalex)
                stats_parttime=np.sum((demogstates[:,10]+demogstates[:,8])*scalex)
            else:
                stats_employed=np.sum((demogstates[:,1])*scalex)
                stats_parttime=np.sum((demogstates[:,10])*scalex)
            stats_unemployed=np.sum((demogstates[:,0]+demogstates[:,4]+demogstates[:,13])*scalex)
        else:
            stats_employed=np.sum((demogstates[:,0]+demogstates[:,3])*scalex)
            stats_parttime=np.sum((demogstates[:,3])*scalex)
            stats_unemployed=np.sum((demogstates[:,0])*scalex)
            #stats_all=np.sum(demogstates,1)

        return stats_employed,stats_parttime,stats_unemployed

    def comp_state_stats(self,state,scale_time=True,start=20,end=63.5,ratio=False):
        demog2=np.squeeze(self.empstats.get_demog())

        #if scale_time:
        #    scale=self.timestep
        #else:
        #    scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1

        #vaikutus=np.round(scale*np.sum(demog2[min_cage:max_cage]*state[min_cage:max_cage]))/np.sum(demog2[min_cage:max_cage])
        vaikutus=np.sum(demog2[min_cage:max_cage]*state[min_cage:max_cage])/np.sum(demog2[min_cage:max_cage])
        x=np.sum(demog2[min_cage:max_cage]*state[min_cage:max_cage])
        y=np.sum(demog2[min_cage:max_cage])
        #print(f'vaikutus {vaikutus} x {x} y {y}\n s {state[min_cage:max_cage]} mean {np.mean(state[min_cage:max_cage])}\n d {demog2[min_cage:max_cage]}')

        return vaikutus

    def get_vanhempainvapaat(self,skip=4,show=False,all=True,csv=None):
        '''
        Laskee vanhempainvapaalla olevien määrän outsider-mallia (Excel) varten, ilman työvoimassa olevia vanhempainvapailla olevia
        '''

        alive=np.zeros((self.galive.shape[0],1))
        alive[:,0]=np.sum(self.galive[:,0:3],1)
        ulkopuolella_m=np.sum(self.gempstate[:,7,0:3],axis=1)[:,None]/alive
        tyovoimassa_m=np.sum(self.gempstate[:,6,0:3],axis=1)[:,None]/alive

        alive[:,0]=np.sum(self.galive[:,3:6],1)
        nn=np.sum(self.gempstate[:,5,3:6]+self.gempstate[:,7,3:6],axis=1)[:,None]
        if not all:
            nn-=self.infostats_mother_in_workforce
        ulkopuolella_n=nn/alive
        tyovoimassa_n=self.infostats_mother_in_workforce/alive

        if show:
            m,n=ulkopuolella_m[::skip],ulkopuolella_n[::skip]
            print('ulkopuolella_m=[',end='')
            for k in range(1,42):
                print('{},'.format(m[k,0]),end='')
            print(']')
            print('ulkopuolella_n=[',end='')
            for k in range(1,42):
                print('{},'.format(n[k,0]),end='')
            print(']')
            m,n=tyovoimassa_m[::skip],tyovoimassa_n[::skip]
            print('tyovoimassa_m=[',end='')
            for k in range(1,42):
                print('{},'.format(m[k,0]),end='')
            print(']')
            print('tyovoimassa_n=[',end='')
            for k in range(1,42):
                print('{},'.format(n[k,0]),end='')
            print(']')
            
        if csv is not None:
            n=ulkopuolella_m[::skip].shape[0]
            x=np.linspace(self.min_age,self.min_age+n-1,n).reshape(-1,1)
            df = pd.DataFrame(np.hstack([x,ulkopuolella_m[::skip],ulkopuolella_n[::skip],tyovoimassa_m[::skip],tyovoimassa_n[::skip]]), 
                columns = ['ikä','ulkopuolella_m','ulkopuolella_n','tyovoimassa_m','tyovoimassa_n'])
            df.to_csv(csv, sep=";", decimal=",")
            
        
        return ulkopuolella_m[::skip],ulkopuolella_n[::skip],tyovoimassa_m,tyovoimassa_n

    def get_muut(self,skip=4,show=False,all=True,csv=None):
        '''
        Laskee vanhempainvapaalla olevien määrän outsider-mallia (Excel) varten, ilman työvoimassa olevia vanhempainvapailla olevia
        '''

        alive=np.zeros((self.galive.shape[0],1))
        alive[:,0]=np.sum(self.galive[:,0:3],1)
        muut_m=np.sum(self.gempstate[:,0,0:3]+self.gempstate[:,1,0:3]+self.gempstate[:,4,0:3]
            +self.gempstate[:,10,0:3]+self.gempstate[:,13,0:3],axis=1)[:,None]/alive
        opisk_m=np.sum(self.gempstate[:,12,0:3],axis=1)[:,None]/alive
        svpaivaraha_m=np.sum(self.gempstate[:,14,0:3],axis=1)[:,None]/alive

        alive[:,0]=np.sum(self.galive[:,3:6],1)
        muut_n=np.sum(self.gempstate[:,0,3:6]+self.gempstate[:,1,3:6]+self.gempstate[:,4,3:6]
            +self.gempstate[:,10,3:6]+self.gempstate[:,13,3:6],axis=1)[:,None]/alive
        opisk_n=np.sum(self.gempstate[:,12,3:6],axis=1)[:,None]/alive
        svpaivaraha_n=np.sum(self.gempstate[:,14,3:6],axis=1)[:,None]/alive

        if show:
            m,n=muut_m[::skip],muut_n[::skip]
            print('muut_m=[',end='')
            for k in range(1,53):
                print('{},'.format(m[k,0]),end='')
            print(']')
            print('muut_n=[',end='')
            for k in range(1,53):
                print('{},'.format(n[k,0]),end='')
            print(']')
            m,n=opisk_m[::skip],opisk_n[::skip]
            print('opisk_m=[',end='')
            for k in range(1,53):
                print('{},'.format(m[k,0]),end='')
            print(']')
            print('opisk_n=[',end='')
            for k in range(1,53):
                print('{},'.format(n[k,0]),end='')
            print(']')
        
        if csv is not None:
            n=muut_m[::skip].shape[0]
            x=np.linspace(self.min_age,self.min_age+n-1,n).reshape(-1,1)
            df = pd.DataFrame(np.hstack([x,muut_m[::skip],muut_n[::skip],opisk_m[::skip],opisk_n[::skip],svpaivaraha_m[::skip],svpaivaraha_n[::skip]]), 
                columns = ['ikä','muut_m','muut_n','opisk_m','opisk_n','svpaivaraha_m','svpaivaraha_n'])
            df.to_csv(csv, sep=";", decimal=",")
        
        return muut_m[::skip],muut_n[::skip]

#     def get_vanhempainvapaat_md(self,skip=4,show=False):
#         '''
#         Laskee vanhempainvapaalla olevien määrän outsider-mallia (Excel) varten, tila 7
#         
#         '''
# 
#         alive=np.zeros((self.galive.shape[0],1))
#         alive[:,0]=np.sum(self.galive[:,0:3],1)
#         vapaat=[5,6,7]
#         ulkopuolella_m=np.sum(self.gempstate[:,vapaat,0:3],axis=(1,2))[:,None]/alive
# 
#         alive[:,0]=np.sum(self.galive[:,3:6],1)
#         #nn=
#         nn=(np.sum(self.gempstate[:,vapaat,3:6],axis=(1,2))[:,None]-self.infostats_mother_in_workforce)/alive
#         ulkopuolella_n=nn/alive
# 
#         if show:
#             m,n=ulkopuolella_m[::skip],ulkopuolella_n[::skip]
#             print('m=[',end='')
#             for k in range(1,42):
#                 print('{},'.format(m[k,0]),end='')
#             print(']')
#             print('n=[',end='')
#             for k in range(1,42):
#                 print('{},'.format(n[k,0]),end='')
#             print(']')
#         else:
#             return ulkopuolella_m[::skip],ulkopuolella_n[::skip]

    def comp_L2error(self):

        tyollisyysaste_m,osatyoaste_m,tyottomyysaste_m,ka_tyottomyysaste=self.comp_gempratios(gender='men',unempratio=False)
        tyollisyysaste_w,osatyoaste_w,tyottomyysaste_w,ka_tyottomyysaste=self.comp_gempratios(gender='women',unempratio=False)
        emp_statsratio_m=self.empstats.emp_stats(g=1)[:-1]*100
        emp_statsratio_w=self.empstats.emp_stats(g=2)[:-1]*100
        unemp_statsratio_m=self.empstats.unemp_stats(g=1)[:-1]*100
        unemp_statsratio_w=self.empstats.unemp_stats(g=2)[:-1]*100

        w1=1.0
        w2=3.0

        L2= w1*np.sum(np.abs(emp_statsratio_m-tyollisyysaste_m[:-1])**2)+\
            w1*np.sum(np.abs(emp_statsratio_w-tyollisyysaste_w[:-1])**2)+\
            w2*np.sum(np.abs(unemp_statsratio_m-tyottomyysaste_m[:-1])**2)+\
            w2*np.sum(np.abs(unemp_statsratio_w-tyottomyysaste_w[:-1])**2)
        L2=L2/self.n_pop

        #print(L1,emp_statsratio_m,tyollisyysaste_m,tyollisyysaste_w,unemp_statsratio_m,tyottomyysaste_m,tyottomyysaste_w)

        print('L2 error {}'.format(L2))

        return L2

    def comp_budgetL2error(self,ref_muut,scale=1):

        q=self.comp_budget()
        muut=q['muut tulot']
        L2=-((ref_muut-muut)/scale)**2

        print(f'L2 error {L2} (muut {muut} muut_ref {ref_muut})')

        return L2

    def optimize_scale(self,target,averaged=scale_error):
        opt=scipy.optimize.least_squares(self.scale_error,0.20,bounds=(-1,1),kwargs={'target':target,'averaged':averaged})

        #print(opt)
        return opt['x']

    def optimize_logutil(self,target,source):
        '''
        analytical compensated consumption
        does not implement final reward, hence duration 110 y
        '''
        n_time=110
        gy=np.empty(n_time)
        g=1
        gx=np.empty(n_time)
        for t in range(0,n_time):
            gx[t]=g
            g*=self.gamma

        for t in range(1,n_time):
            gy[t]=np.sum(gx[0:t])

        gf=np.mean(gy[1:])/10
        lx=(target-source)
        opt=np.exp(lx/gf)-1.0

        print(opt)

    def min_max(self):
        min_wage=np.min(self.infostats_pop_wage)
        max_wage=np.max(self.infostats_pop_wage)
        max_pension=np.max(self.infostats_pop_pension)
        min_pension=np.min(self.infostats_pop_pension)
        print(f'min wage {min_wage} max wage {max_wage}')
        print(f'min pension {min_pension} max pension {max_pension}')

    def setup_labels(self):
        self.labels=self.lab.get_labels(self.language)

    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)

    def map_t_to_age(self,t):
        return self.min_age+t/self.inv_timestep

    def episodestats_exit(self):
        plt.close(self.episode_fig)

    def comp_annual_irr(self,npv,premium,pension,doprint=False):
        k=0
        max_npv=int(np.ceil(npv))
        cashflow=-premium+pension
        
        if np.abs(np.sum(premium))<1e-6 or np.abs(np.sum(pension))<1e-6:
            if np.abs(np.sum(premium))<1e-6 and np.abs(np.sum(pension))>1e-6:
                irri=1.0*100
            elif np.abs(np.sum(pension))<1e-6 and np.abs(np.sum(premium))>1e-6:
                irri=-1.0*100
            else:
                irri=np.nan
        else:
            x=np.zeros(cashflow.shape[0]+max_npv)
        
            eind=np.zeros(max_npv+1)

            el=1
            for k in range(max_npv+1):
                eind[k]=el
                el=el*self.elakeindeksi

            x[:cashflow.shape[0]]=cashflow
            if npv>0:
                x[cashflow.shape[0]-1:]=cashflow[-2]*eind[:max_npv+1]

            y=np.zeros(int(np.ceil(x.shape[0]/4)))
            for k in range(y.shape[0]):
                y[k]=np.sum(x[4*k:4*k+4])
            irri=npf.irr(y)*100

        #if np.isnan(irri):
        #    if np.sum(pension)<0.1 and np.sum(empstate[0:self.map_age(63)]==15)>0: # vain maksuja, joista ei saa tuottoja, joten tappio 100%
        #        irri=-100

        if doprint:
            if irri<0.01 and doprint:
                print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,pension,empstate))

            if irri>100 and doprint:
                print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,pension,empstate))

            if np.isnan(irri) and doprint:
                print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,np.sum(pension),empstate))

            if irri<-50 and doprint:
                print('---------\nirri {}\nnpv {}\nx {}\ny {}\nprem {}\npens {}\nemps {}\n---------\n'.format(irri,npv,x,y,premium,pension,empstate))

        return irri

    def comp_irr(self):
        '''
        Laskee sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        self.infostats_irr_tyel_full=np.zeros((self.n_pop,1))
        self.infostats_irr_tyel_reduced=np.zeros((self.n_pop,1))
        for k in range(self.n_pop):
            # tyel-eläke ilman vähennyksiä
            self.infostats_irr_tyel_full[k]=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(self.infostats_npv0[k,0],self.infostats_tyelpremium[:,k],self.infostats_pop_tyoelake[:,k])
            # tyel-eläke ml. muiden eläkkeiden vähenteisyys
            self.infostats_irr_tyel_reduced[k]=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(self.infostats_npv0[k,0],self.infostats_tyelpremium[:,k],self.infostats_paid_tyel_pension[:,k])

    def comp_aggirr(self,gender=None):
        '''
        Laskee aggregoidun sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        
        if gender is None:
            gendername='Kaikki'
            alive=self.alive[-1]
            maxnpv=np.sum(self.infostats_npv0)/alive # keskimääräinen npv ilman diskonttausta tarkasteluvälin lopussa

            agg_premium=np.sum(self.infostats_tyelpremium,axis=1)
            agg_pensions_reduced=np.sum(self.infostats_paid_tyel_pension,axis=1)
            agg_pensions_full=np.sum(self.infostats_pop_tyoelake,axis=1)
            agg_irr_tyel_full=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_full)
            agg_irr_tyel_reduced=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_reduced)
        else:
            if gender==1: # naiset
                gendermask = (self.infostats_group>2).astype(float).T
                gendername='Naiset'
                alive=np.sum(self.galive[-1,3:6])
            else: # miehet
                gendermask = (self.infostats_group<3).astype(float).T
                gendername='Miehet'
                alive=np.sum(self.galive[-1,0:3])
        
            maxnpv=np.sum(self.infostats_npv0.T*gendermask)/alive # keskimääräinen npv ilman diskonttausta tarkasteluvälin lopussa

            agg_premium=np.sum(self.infostats_tyelpremium*gendermask,axis=1)
            agg_pensions_reduced=np.sum(self.infostats_paid_tyel_pension*gendermask,axis=1)
            agg_pensions_full=np.sum(self.infostats_pop_tyoelake*gendermask,axis=1)
            agg_irr_tyel_full=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_full)
            agg_irr_tyel_reduced=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_reduced)

        print('{}: aggregate irr tyel reduced {:.4f} % reaalisesti'.format(gendername,agg_irr_tyel_reduced))
        print('{}: aggregate irr tyel full {:.4f} % reaalisesti'.format(gendername,agg_irr_tyel_full))
        
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('age')
        ax.set_ylabel('pension/premium')
        ax.plot(x[:-1],agg_pensions_full[:-1],label='työeläkemeno')
        ax.plot(x[:-1],agg_pensions_reduced[:-1],label='yhteensovitettu työeläkemeno')
        ax.plot(x[:-1],agg_premium[:-1],label='työeläkemaksu')
        plt.show()

    def comp_unemp_durations(self,popempstate=None,popunemprightused=None,putki=True,\
            tmtuki=False,laaja=False,outsider=False,ansiosid=True,tyott=False,kaikki=False,\
            return_q=True,max_age=100):
        '''
        Poikkileikkaushetken työttömyyskestot
        '''
        unempset=[]

        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
        if laaja:
            unempset=[0,4,11,13]
        if kaikki:
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14]

        unempset=set(unempset)

        if popempstate is None:
            popempstate=self.popempstate

        if popunemprightused is None:
            popunemprightused=self.popunemprightused

        keskikesto=np.zeros((5,5)) # 20-29, 30-39, 40-49, 50-59, 60-69, vastaa TYJin tilastoa
        n=np.zeros(5)

        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k] in unempset:
                        if age<29:
                            l=0
                        elif age<39:
                            l=1
                        elif age<49:
                            l=2
                        elif age<59:
                            l=3
                        else:
                            l=4

                        n[l]+=1
                        if self.popunemprightused[t,k]<=0.51:
                            keskikesto[l,0]+=1
                        elif self.popunemprightused[t,k]<=1.01:
                            keskikesto[l,1]+=1
                        elif self.popunemprightused[t,k]<=1.51:
                            keskikesto[l,2]+=1
                        elif self.popunemprightused[t,k]<=2.01:
                            keskikesto[l,3]+=1
                        else:
                            keskikesto[l,4]+=1

        for k in range(5):
            keskikesto[k,:] /= n[k]

        if return_q:
            return self.empdur_to_dict(keskikesto)
        else:
            return keskikesto

    def empdur_to_dict(self,empdur):
        q={}
        q['20-29']=empdur[0,:]
        q['30-39']=empdur[1,:]
        q['40-49']=empdur[2,:]
        q['50-59']=empdur[3,:]
        q['60-65']=empdur[4,:]
        return q

    def comp_unemp_durations_v2(self,popempstate=None,putki=True,tmtuki=False,laaja=False,\
            outsider=False,ansiosid=True,tyott=False,kaikki=False,\
            return_q=True,max_age=100):
        '''
        Poikkileikkaushetken työttömyyskestot
        Tässä lasketaan tulos tiladatasta, jolloin kyse on viimeisimmän jakson kestosta
        '''
        unempset=[]

        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
        if laaja:
            unempset=[0,4,11,13]
        if kaikki:
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14]

        unempset=set(unempset)

        if popempstate is None:
            popempstate=self.popempstate

        keskikesto=np.zeros((5,5)) # 20-29, 30-39, 40-49, 50-59, 60-69, vastaa TYJin tilastoa
        n=np.zeros(5)

        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] not in unempset:
                            prev_state=popempstate[t,k]
                            duration=(t-prev_trans)*self.timestep
                            prev_trans=t

                            if age<29:
                                l=0
                            elif age<39:
                                l=1
                            elif age<49:
                                l=2
                            elif age<59:
                                l=3
                            else:
                                l=4

                            n[l]+=1
                            if duration<=0.51:
                                keskikesto[l,0]+=1
                            elif duration<=1.01:
                                keskikesto[l,1]+=1
                            elif duration<=1.51:
                                keskikesto[l,2]+=1
                            elif duration<=2.01:
                                keskikesto[l,3]+=1
                            else:
                                keskikesto[l,4]+=1
                        elif prev_state not in unempset and popempstate[t,k] in unempset:
                            prev_trans=t
                            prev_state=popempstate[t,k]
                        else: # some other state
                            prev_state=popempstate[t,k]
                            prev_trans=t

        for k in range(5):
            keskikesto[k,:] /= n[k]

        if return_q:
            return self.empdur_to_dict(keskikesto)
        else:
            return keskikesto

    def comp_virrat(self,popempstate=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,kaikki=False,max_age=100):
        tyoll_virta=np.zeros((self.n_time,1))
        tyot_virta=np.zeros((self.n_time,1))
        unempset=[]
        empset=[]

        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]
        if laaja:
            unempset=[0,4,11,13]
        if kaikki:
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14]

        empset=set([1,10])
        unempset=set(unempset)

        if popempstate is None:
            popempstate=self.popempstate

        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] in empset:
                            tyoll_virta[t]+=1
                            prev_state=popempstate[t,k]
                        elif prev_state in empset and popempstate[t,k] in unempset:
                            tyot_virta[t]+=1
                            prev_state=popempstate[t,k]
                        else: # some other state
                            prev_state=popempstate[t,k]

        return tyoll_virta,tyot_virta

    def comp_tyollistymisdistribs(self,popempstate=None,popunemprightleft=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
        tyoll_distrib=[]
        tyoll_distrib_bu=[]
        unempset=[]

        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]

        if laaja:
            unempset=[0,4,11,13]

        empset=set([1,10])
        unempset=set(unempset)

        if popempstate is None or popunemprightleft is None:
            popempstate=self.popempstate
            popunemprightleft=self.popunemprightleft

        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] in empset:
                            tyoll_distrib.append((t-prev_trans)*self.timestep)
                            tyoll_distrib_bu.append(popunemprightleft[t,k])
                            prev_state=popempstate[t,k]
                            prev_trans=t
                        else: # some other state
                            prev_state=popempstate[t,k]
                            prev_trans=t

        return tyoll_distrib,tyoll_distrib_bu

    def comp_empdistribs(self,popempstate=None,popunemprightleft=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
        unemp_distrib=[]
        unemp_distrib_bu=[]
        emp_distrib=[]
        unempset=[]

        if tmtuki:
            unempset.append(13)
        if outsider:
            unempset.append(11)
        if putki:
            unempset.append(4)
        if ansiosid:
            unempset.append(0)
        if tyott:
            unempset=[0,4,13]

        if laaja:
            unempset=[0,4,11,13]

        if popempstate is None or popunemprightleft is None:
            popempstate=self.popempstate
            popunemprightleft=self.popunemprightleft

        empset=set([1,10])
        unempset=set(unempset)

        for k in range(self.n_pop):
            prev_state=popempstate[0,k]
            prev_trans=0
            for t in range(1,self.n_time):
                age=self.min_age+t*self.timestep
                if age<=max_age:
                    if self.popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] not in unempset:
                            unemp_distrib.append((t-prev_trans)*self.timestep)
                            unemp_distrib_bu.append(popunemprightleft[t,k])

                            prev_state=popempstate[t,k]
                            prev_trans=t
                        elif prev_state in empset and popempstate[t,k] not in unempset:
                            emp_distrib.append((t-prev_trans)*self.timestep)
                            prev_state=popempstate[t,k]
                            prev_trans=t
                        else: # some other state
                            prev_state=popempstate[t,k]
                            prev_trans=t

        return unemp_distrib,emp_distrib,unemp_distrib_bu

    def empdist_stat(self):
        ratio=np.array([1,0.287024901703801,0.115508955875928,0.0681083442551332,0.0339886413280909,0.0339886413280909,0.0114460463084316,0.0114460463084316,0.0114460463084316,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206])

        return ratio

    def comp_gempratios(self,unempratio=True,gender='men'):
        if gender=='men': # men
            gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
            alive=np.zeros((self.galive.shape[0],1))
            alive[:,0]=np.sum(self.galive[:,0:3],1)
            mother_in_workforce=0
        else: # women
            gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
            alive=np.zeros((self.galive.shape[0],1))
            alive[:,0]=np.sum(self.galive[:,3:6],1)
            mother_in_workforce=self.infostats_mother_in_workforce

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive,unempratio=unempratio)#,mother_in_workforce=mother_in_workforce)

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste

    def comp_workforce(self,emp,alive):
        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]

        if self.version in set([1,2,3,4,104]):
            disabled=emp[:,3]
            piped=emp[:,4]
            mother=emp[:,5]
            dad=emp[:,6]
            kotihoidontuki=emp[:,7]
            vetyo=emp[:,9]
            veosatyo=emp[:,8]
            osatyo=emp[:,10]
            outsider=emp[:,11]
            student=emp[:,12]
            tyomarkkinatuki=emp[:,13]
            workforce=100*(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki)/alive[:,0]
        elif self.version in set([0,101]):
            if False:
                osatyo=emp[:,3]
            else:
                osatyo=0
            workforce=100*(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki)/alive[:,0]

        return workforce

    def comp_empratios(self,emp,alive,unempratio=True,mother_in_workforce=0):
        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]

        if self.version in set([1,2,3,4,104]):
            disabled=emp[:,3]
            piped=emp[:,4]
            mother=emp[:,5]
            dad=emp[:,6]
            kotihoidontuki=emp[:,7]
            vetyo=emp[:,9]
            veosatyo=emp[:,8]
            osatyo=emp[:,10]
            outsider=emp[:,11]
            student=emp[:,12]
            tyomarkkinatuki=emp[:,13]
            tyollisyysaste=100*(employed+osatyo+veosatyo+vetyo+dad+mother_in_workforce)/alive[:,0]
            osatyoaste=100*(osatyo+veosatyo)/(employed+osatyo+veosatyo+vetyo)
            if unempratio:
                tyottomyysaste=100*(unemployed+piped+tyomarkkinatuki)/(tyomarkkinatuki+unemployed+employed+piped+osatyo+veosatyo+vetyo)
                ka_tyottomyysaste=100*np.sum(unemployed+tyomarkkinatuki+piped)/np.sum(tyomarkkinatuki+unemployed+employed+piped+osatyo+veosatyo+vetyo)
            else:
                tyottomyysaste=100*(unemployed+piped+tyomarkkinatuki)/alive[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed+tyomarkkinatuki+piped)/np.sum(alive[:,0])
        elif self.version in set([0,101]):
            if False:
                osatyo=emp[:,3]
            else:
                osatyo=0
            tyollisyysaste=100*(employed+osatyo)/alive[:,0]
            #osatyoaste=np.zeros(employed.shape)
            osatyoaste=100*(osatyo)/(employed+osatyo)
            if unempratio:
                tyottomyysaste=100*(unemployed)/(unemployed+employed+osatyo)
                ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(unemployed+employed+osatyo)
            else:
                tyottomyysaste=100*(unemployed)/alive[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(alive[:,0])

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste

    def plot_ratiostats(self,t):
        '''
        Tee kuvia tuloksista
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('palkat')
        ax.set_ylabel('freq')
        ax.hist(self.infostats_pop_wage[t,:])
        plt.show()
        fig,ax=plt.subplots()
        ax.set_xlabel('aika')
        ax.set_ylabel('palkat')
        meansal=np.mean(self.infostats_pop_wage,axis=1)
        stdsal=np.std(self.infostats_pop_wage,axis=1)
        ax.plot(x,meansal)
        ax.plot(x,meansal+stdsal)
        ax.plot(x,meansal-stdsal)
        plt.show()

    def plot_empdistribs(self,emp_distrib):
        fig,ax=plt.subplots()
        ax.set_xlabel('työsuhteen pituus [v]')
        ax.set_ylabel('freq')
        ax.set_yscale('log')
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled,x2=np.histogram(emp_distrib,x)
        scaled=scaled/np.sum(emp_distrib)
        #ax.hist(emp_distrib)
        ax.bar(x2[1:-1],scaled[1:],align='center')
        plt.show()
        
    def plot_alive(self):
        alive=self.alive/self.n_pop
        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Alive [%]')
        nn_time = int(np.ceil((self.max_age-self.min_age)*self.inv_timestep))+1
        x=np.linspace(self.min_age,self.max_age,nn_time)
        ax.plot(x[1:],alive[1:]*100)
        plt.show()
        
    def plot_pension_time(self):
        self.plot_y(self.infostats_tyoelake,label='työeläke',
            y2=self.infostats_kansanelake,label2='kansaneläke',
            ylabel='eläke [e/v]',
            start_from=60,end_at=70,show_legend=True)

        demog2=self.empstats.get_demog()
        scalex=demog2/self.n_pop
        tyoelake_meno=self.infostats_tyoelake*scalex
        kansanelake_meno=self.infostats_kansanelake*scalex
        print('työeläkemeno alle 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(tyoelake_meno[:self.map_age(63)]),1_807.1))
        print('työeläkemeno yli 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(tyoelake_meno[self.map_age(63):]),24_227.2))
        print('kansaneläkemeno alle 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(kansanelake_meno[:self.map_age(63)]),679.6))
        print('kansaneläkemeno yli 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(kansanelake_meno[self.map_age(63):]),1_419.9))
        
        self.plot_y(self.infostats_kansanelake,label='kansaneläke',
            ylabel='kansaneläke [e/v]',
            start_from=60,end_at=70,show_legend=True)

        self.plot_y(self.infostats_kansanelake/self.infostats_tyoelake*100,label='suhde',
            ylabel='kansaneläke/työeläke [%]',
            start_from=60,end_at=70,show_legend=True)

    def plot_pension_stats(self,pd,age,label,max_pen=60_000,mask=None,plot_ke=False):
        fig,ax=plt.subplots()
        if mask is None:
            pens_distrib=ma.array(pd[self.map_age(age),:])
        else:
            pens_distrib=ma.array(pd[self.map_age(age),:],mask=mask[self.map_age(age),:])
        
        ax.set_xlabel('eläke [e/v]')
        ax.set_ylabel('freq')
        #ax.set_yscale('log')
        x=np.linspace(0,max_pen,51)
        scaled,x2=np.histogram(pens_distrib.compressed(),x)
        
        scaled=scaled/np.sum(pens_distrib)
        #print(pens_distrib,scaled,x2)
        #ax.bar(x2[1:-1],scaled[1:],align='center')
        ax.plot(x2[1:],scaled)
        #ax.bar(x2[1:],scaled)
        axvcolor='gray'
        lstyle='--'
        ka=np.mean(pens_distrib)
        plt.axvline(x=ka,ls=lstyle,color=axvcolor)
        if plot_ke:
            arv=self.env.ben.laske_kansanelake(66,0/12,1,disability=True)*12
            plt.axvline(x=arv,ls=lstyle,color='red')
            plt.axvline(x=0.5*arv,ls=lstyle,color='pink')
            
        plt.title(f'{label} at age {age}, mean {ka:.0f}')
        plt.show()

    def plot_compare_empdistribs(self,emp_distrib,emp_distrib2,label2='vaihtoehto',label1=''):
        fig,ax=plt.subplots()
        ax.set_xlabel('työsuhteen pituus [v]')
        ax.set_ylabel(self.labels['probability'])
        ax.set_yscale('log')
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled,x2=np.histogram(emp_distrib,x)
        scaled=scaled/np.sum(emp_distrib)
        x=np.linspace(0,max_time,nn_time)
        scaled3,x3=np.histogram(emp_distrib2,x)
        scaled3=scaled3/np.sum(emp_distrib2)

        ax.plot(x3[:-1],scaled3,label=label1)
        ax.plot(x2[:-1],scaled,label=label2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_vlines_unemp(self,point=0):
        axvcolor='gray'
        lstyle='--'
        plt.axvline(x=300/(12*21.5),ls=lstyle,color=axvcolor)
        plt.text(310/(12*21.5),point,'300',rotation=90)
        plt.axvline(x=400/(12*21.5),ls=lstyle,color=axvcolor)
        plt.text(410/(12*21.5),point,'400',rotation=90)
        plt.axvline(x=500/(12*21.5),ls=lstyle,color=axvcolor)
        plt.text(510/(12*21.5),point,'500',rotation=90)

    def plot_tyolldistribs(self,emp_distrib,tyoll_distrib,tyollistyneet=True,max=10,figname=None):
        max_time=55
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        if not tyollistyneet:
            scaled=scaled0
            x2=x0
        else:
            scaled,x2=np.histogram(tyoll_distrib,x)
        jaljella=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien kumulatiivinen summa
        scaled=scaled/jaljella

        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        if tyollistyneet:
            ax.set_ylabel('työllistyneiden osuus')
            point=0.5
        else:
            ax.set_ylabel('pois siirtyneiden osuus')
            point=0.9
        self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:])
        #ax.bar(x2[1:-1],scaled[1:],align='center',width=self.timestep)
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'tyollistyneetdistrib.'+self.figformat, format=self.figformat)

        plt.show()

    def plot_tyolldistribs_both(self,emp_distrib,tyoll_distrib,max=10,figname=None):
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        scaled=scaled0
        scaled_tyoll,x2=np.histogram(tyoll_distrib,x)

        jaljella=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        scaled=scaled/jaljella
        jaljella_tyoll=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        scaled_tyoll=scaled_tyoll/jaljella_tyoll
        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        point=0.6
        self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:],label='pois siirtyneiden osuus')
        ax.plot(x2[1:-1],scaled_tyoll[1:],label='työllistyneiden osuus')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend()
        ax.set_ylabel('pois siirtyneiden osuus')

        plt.xlim(0,max)
        plt.ylim(0,0.8)
        if figname is not None:
            plt.savefig(figname+'tyolldistribs.'+self.figformat, format=self.figformat)
        plt.show()

    def plot_tyolldistribs_both_bu(self,emp_distrib,tyoll_distrib,max=2):
        max_time=4
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(-max_time,0,nn_time)
        scaled0,x0=np.histogram(emp_distrib,x)
        scaled=scaled0
        scaled_tyoll,x2=np.histogram(tyoll_distrib,x)

        jaljella=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        #jaljella=np.cumsum(scaled0)
        scaled=scaled/jaljella
        jaljella_tyoll=np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien summa
        #jaljella_tyoll=np.cumsum(scaled0)
        scaled_tyoll=scaled_tyoll/jaljella_tyoll
        fig,ax=plt.subplots()
        ax.set_xlabel('aika ennen ansiopäivärahaoikeuden loppua [v]')
        point=0.6
        #self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:],label='pois siirtyneiden osuus')
        ax.plot(x2[1:-1],scaled_tyoll[1:],label='työllistyneiden osuus')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_ylabel('pois siirtyneiden osuus')

        plt.xlim(-max,0)
        #plt.ylim(0,0.8)
        plt.show()

    def plot_compare_tyolldistribs(self,emp_distrib1,tyoll_distrib1,emp_distrib2,
                tyoll_distrib2,tyollistyneet=True,max=4,label1='perus',label2='vaihtoehto',
                figname=None):
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)

        # data1
        scaled01,x0=np.histogram(emp_distrib1,x)
        if not tyollistyneet:
            scaled1=scaled01
            x1=x0
        else:
            scaled1,x1=np.histogram(tyoll_distrib1,x)
        jaljella1=np.cumsum(scaled01[::-1])[::-1] # jäljellä olevien summa
        scaled1=scaled1/jaljella1

        # data2
        scaled02,x0=np.histogram(emp_distrib2,x)
        if not tyollistyneet:
            scaled2=scaled02
            x2=x0
        else:
            scaled2,x2=np.histogram(tyoll_distrib2,x)
        jaljella2=np.cumsum(scaled02[::-1])[::-1] # jäljellä olevien summa
        scaled2=scaled2/jaljella2

        fig,ax=plt.subplots()
        ax.set_xlabel('työttömyysjakson pituus [v]')
        if tyollistyneet:
            ax.set_ylabel('työllistyneiden osuus')
        else:
            ax.set_ylabel('pois siirtyneiden osuus')
        self.plot_vlines_unemp()
        ax.plot(x2[1:-1],scaled2[1:],label=label2)
        ax.plot(x1[1:-1],scaled1[1:],label=label1)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend()
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'comp_tyollistyneetdistrib.'+self.figformat, format=self.figformat)

        plt.show()

    def plot_unempdistribs(self,unemp_distrib,max=4,figname=None,miny=None,maxy=None):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(0,max_time,nn_time)
        scaled,x2=np.histogram(unemp_distrib,x)
        scaled=scaled/np.sum(unemp_distrib)
        fig,ax=plt.subplots()
        self.plot_vlines_unemp(0.6)
        ax.set_xlabel(self.labels['unemp duration'])
        ax.set_ylabel(self.labels['probability'])

        ax.plot(x[:-1],scaled)
        ax.set_yscale('log')
        plt.xlim(0,max)
        if miny is not None:
            plt.ylim(miny,maxy)
        if figname is not None:
            plt.savefig(figname+'unempdistribs.'+self.figformat, format=self.figformat)

        plt.show()
        
    def plot_saldist(self,t=0,sum=False,all=False,n=10,bins=30):
        if all:
            fig,ax=plt.subplots()
            for t in range(1,self.n_time-1,5):
                scaled,x=np.histogram(self.infostats_pop_wage[t,:],bins=bins)
                x2=0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label=t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x=np.histogram(np.sum(self.infostats_pop_wage,axis=0),bins=bins)
                x2=0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax=plt.subplots()
                for t1 in range(t,t+n,1):
                    scaled,x=np.histogram(self.infostats_pop_wage[t1,:],bins=bins)
                    x2=0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label=t1)
                plt.legend()
                plt.show()

    def test_salaries(self):

        def kuva(sal,ika,m,p,palkka):
            plt.hist(sal[:m],bins=50,density=True)
            ave=np.mean(sal[:m])/12
            palave=np.sum(palkka*p)/12/np.sum(palkka)
            plt.title('{}: ave {} vs {}'.format(ika,ave,palave))
            plt.plot(p,palkka/sum(palkka)/2000)
            plt.show()

        def kuva2(sal,ika,m):
            plt.hist(sal[:m],bins=50,density=True)
            ave=np.mean(sal[:m])/12
            plt.title('{}: ave {}'.format(ika,ave))
            plt.show()

        def cdf_kuva(sal,ika,m,p,palkka):
            pal=np.cumsum(palkka)/np.sum(palkka)
            m_x,m_y=empirical_cdf(sal[:m])
            plt.plot(m_x,m_y,label='malli')
            plt.plot(p,pal,label='havainto')
            plt.title('age {}'.format(ika))
            plt.legend()
            plt.show()
            plt.loglog(m_x,m_y,label='malli')
            plt.loglog(p,pal,label='havainto')
            plt.title('age {}'.format(ika))
            plt.legend()
            plt.show()

    
        n=self.n_pop
        
        palkat_ika_miehet=12.5*np.array([2039.15,2256.56,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
        palkat_ika_naiset=12.5*np.array([2058.55,2166.68,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])

        #palkat_ika_miehet=12.5*np.array([2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
        #palkat_ika_naiset=12.5*np.array([2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        #g_r=[0.77,1.0,1.23]
        data_range=np.arange(self.min_age,self.max_age)

        sal20=np.zeros((n,1))
        sal25=np.zeros((n,1))
        sal30=np.zeros((n,1))
        sal40=np.zeros((n,1))
        sal50=np.zeros((n,1))
        sal60=np.zeros((n,1))
        sal65=np.zeros((n,1))
        sal=np.zeros((n,self.max_age))

        p=np.arange(700,17500,100)*12.5
        palkka20=np.array([10.3,5.6,4.5,14.2,7.1,9.1,22.8,22.1,68.9,160.3,421.6,445.9,501.5,592.2,564.5,531.9,534.4,431.2,373.8,320.3,214.3,151.4,82.3,138.0,55.6,61.5,45.2,19.4,32.9,13.1,9.6,7.4,12.3,12.5,11.5,5.3,2.4,1.6,1.2,1.2,14.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        palkka25=np.array([12.4,11.3,30.2,4.3,28.5,20.3,22.5,23.7,83.3,193.0,407.9,535.0,926.5,1177.1,1540.9,1526.4,1670.2,1898.3,1538.8,1431.5,1267.9,1194.8,1096.3,872.6,701.3,619.0,557.2,465.8,284.3,291.4,197.1,194.4,145.0,116.7,88.7,114.0,56.9,57.3,55.0,25.2,24.4,20.1,25.2,37.3,41.4,22.6,14.1,9.4,6.3,7.5,8.1,9.0,4.0,3.4,5.4,4.1,5.2,1.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        palkka30=np.array([1.0,2.0,3.0,8.5,12.1,22.9,15.8,21.8,52.3,98.2,295.3,392.8,646.7,951.4,1240.5,1364.5,1486.1,1965.2,1908.9,1729.5,1584.8,1460.6,1391.6,1551.9,1287.6,1379.0,1205.6,1003.6,1051.6,769.9,680.5,601.2,552.0,548.3,404.5,371.0,332.7,250.0,278.2,202.2,204.4,149.8,176.7,149.0,119.6,76.8,71.4,56.3,75.9,76.8,58.2,50.2,46.8,48.9,30.1,32.2,28.8,31.1,45.5,41.2,36.5,18.1,11.6,8.5,10.2,4.3,13.5,12.3,4.9,13.9,5.4,5.9,7.4,14.1,9.6,8.4,11.5,0.0,3.3,9.0,5.2,5.0,3.1,7.4,2.0,4.0,4.1,14.0,2.0,3.0,1.0,0.0,6.2,2.0,1.2,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        palkka50=np.array([2.0,3.1,2.4,3.9,1.0,1.0,11.4,30.1,29.3,34.3,231.9,341.9,514.4,724.0,1076.8,1345.2,1703.0,1545.8,1704.0,1856.1,1805.4,1608.1,1450.0,1391.4,1338.5,1173.2,1186.3,1024.8,1105.6,963.0,953.0,893.7,899.8,879.5,857.0,681.5,650.5,579.2,676.8,498.0,477.5,444.3,409.1,429.0,340.5,297.2,243.1,322.5,297.5,254.1,213.1,249.3,212.1,212.8,164.4,149.3,158.6,157.4,154.1,112.7,93.4,108.4,87.3,86.7,82.0,115.9,66.9,84.2,61.4,43.7,58.1,40.9,73.9,50.0,51.6,25.7,43.2,48.2,43.0,32.6,21.6,22.4,36.3,28.3,19.4,21.1,21.9,21.5,19.2,15.8,22.6,9.3,14.0,22.4,14.0,13.0,11.9,18.7,7.3,21.6,9.5,11.2,12.0,18.2,12.9,2.2,10.7,6.1,11.7,7.6,1.0,4.7,8.5,6.4,3.3,4.6,1.2,3.7,5.8,1.0,1.0,1.0,1.0,3.2,1.2,3.1,2.2,2.3,2.1,1.1,2.0,2.1,2.2,4.6,2.2,1.0,1.0,1.0,0.0,3.0,1.2,0.0,8.2,3.0,1.0,1.0,2.1,1.2,3.2,1.0,5.2,1.1,5.2,1.0,1.2,2.3,1.0,3.1,1.0,1.0,1.1,1.6,1.1,1.1,1.0,1.0,1.0,1.0])

        m20=0
        m25=0
        m30=0
        m40=0
        m50=0
        m60=0
        m65=0
        salx=np.zeros((self.n_time+2,1))
        saln=np.zeros((self.n_time+2,1))
        salgx=np.zeros((self.n_time+2,self.n_groups))
        salgn=np.zeros((self.n_time+2,self.n_groups))
        salx_m=np.zeros((self.n_time+2,1))
        saln_m=np.zeros((self.n_time+2,1))
        salx_f=np.zeros((self.n_time+2,1))
        saln_f=np.zeros((self.n_time+2,1))
        for k in range(self.n_pop):
            for t in range(self.n_time-2):
                if self.popempstate[t,k] in set([1]): # 9,8,10
                    wage=self.infostats_pop_wage[t,k]
                    salx[t]=salx[t]+wage
                    saln[t]=saln[t]+1
                    salgx[t,self.infostats_group[k]]=salgx[t,self.infostats_group[k]]+wage
                    salgn[t,self.infostats_group[k]]=salgn[t,self.infostats_group[k]]+1
                    
            if self.popempstate[self.map_age(20),k] in set([1]):
                sal20[m20]=self.infostats_pop_wage[self.map_age(20),k]
                m20=m20+1
            if self.popempstate[self.map_age(25),k] in set([1]):
                sal25[m25]=self.infostats_pop_wage[self.map_age(25),k]
                m25=m25+1
            if self.popempstate[self.map_age(30),k] in set([1]):
                sal30[m30]=self.infostats_pop_wage[self.map_age(30),k]
                m30=m30+1
            if self.popempstate[self.map_age(40),k] in set([1]):
                sal40[m40]=self.infostats_pop_wage[self.map_age(40),k]
                m40=m40+1
            if self.popempstate[self.map_age(50),k] in set([1]):
                sal50[m50]=self.infostats_pop_wage[self.map_age(50),k]
                m50=m50+1
            if self.popempstate[self.map_age(60),k] in set([1]):
                sal60[m60]=self.infostats_pop_wage[self.map_age(60),k]
                m60=m60+1
            if self.popempstate[self.map_age(65),k] in set([1,9]):
                sal65[m65]=self.infostats_pop_wage[self.map_age(65),k]
                m65=m65+1

        salx_f[t]=np.sum(salgx[t,3:6])
        saln_f[t]=np.sum(salgn[t,3:6])
        salx_m[t]=np.sum(salgx[t,0:3])
        saln_m[t]=np.sum(salgn[t,0:3])

        salx=salx/np.maximum(1,saln)
        salgx=salgx/np.maximum(1,salgn)
        salx_f=salx_f/np.maximum(1,saln_f)
        salx_m=salx_m/np.maximum(1,saln_m)
        #print(sal25,self.infostats_pop_wage)
            
        cdf_kuva(sal20,20,m20,p,palkka20)
        cdf_kuva(sal25,25,m25,p,palkka25)
        cdf_kuva(sal30,30,m30,p,palkka30)
        cdf_kuva(sal50,50,m50,p,palkka50)

        kuva(sal20,20,m20,p,palkka20)
        kuva(sal25,25,m25,p,palkka25)
        kuva(sal30,30,m30,p,palkka30)
        kuva2(sal40,40,m40)
        kuva(sal50,50,m50,p,palkka50)
        kuva2(sal60,60,m60)
        kuva2(sal65,65,m65)

        data_range=np.arange(self.min_age,self.max_age+1)
        plt.plot(data_range,np.mean(self.infostats_pop_wage[::4],axis=1),label='malli kaikki')
        plt.plot(data_range,salx[::4],label='malli töissä')
        data_range=np.arange(self.min_age,self.max_age+2)
        plt.plot(data_range,0.5*palkat_ika_miehet+0.5*palkat_ika_naiset,label='data')
        plt.legend()
        plt.show()

        data_range=np.arange(self.min_age,self.max_age+1)
        plt.plot(data_range,salx_m[::4],label='malli töissä miehet')
        plt.plot(data_range,salx_f[::4],label='malli töissä naiset')
        data_range=np.arange(self.min_age,self.max_age+2)
        plt.plot(data_range,palkat_ika_miehet,label='data miehet')
        plt.plot(data_range,palkat_ika_naiset,label='data naiset')
        plt.legend()
        plt.show()
        
        fig,ax=plt.subplots()
        data_range=np.arange(self.min_age,self.max_age+1)
        for g in range(self.n_groups):
            ax.plot(data_range,salgx[::4,g],ls='--',label='malli '+str(g))
        data_range=np.arange(self.min_age,self.max_age+2)
        ax.plot(data_range,palkat_ika_miehet,label='data miehet')
        ax.plot(data_range,palkat_ika_naiset,label='data naiset')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        fig,ax=plt.subplots()
        data_range=np.arange(self.min_age,self.max_age+1)
        ax.plot(data_range,salgx[::4,0]/salgx[::4,1],ls='--',label='miehet low/mid ')
        ax.plot(data_range,salgx[::4,2]/salgx[::4,1],ls='--',label='miehet high/mid ')
        ax.plot(data_range,salgx[::4,3]/salgx[::4,4],ls='--',label='naiset low/mid ')
        ax.plot(data_range,salgx[::4,5]/salgx[::4,4],ls='--',label='naiset high/mid ')
        data_range=np.arange(self.min_age,self.max_age+2)
        x,m1,m2,w1,w2=self.empstats.stat_wageratio()
        ax.plot(x,m1,ls='-',label='data men low/mid ')
        ax.plot(x,m2,ls='-',label='data men high/mid ')
        ax.plot(x,w1,ls='-',label='data women low/mid ')
        ax.plot(x,w2,ls='-',label='data women high/mid ')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('suhteellinen kehitys')
        plt.show()

        fig,ax=plt.subplots()
        data_range=np.arange(self.min_age,self.max_age+1)
        for g in range(self.n_groups):
            ax.plot(data_range,salgx[::4,g]/salgx[1,g],ls='--',label='suhde '+str(g))
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('ikäkehitys')
        plt.show()

    def test_tail(self):
        from scipy.stats import pareto
        n=self.n_pop
        
        data_range=np.arange(self.min_age,self.max_age)
        minwage=78_400

        salx=np.zeros((self.n_pop*self.n_time,1))
        saln=0
        salgx=np.zeros((self.n_groups,self.n_pop*self.n_time))
        salgn=np.zeros(self.n_groups,dtype=int)
        for k in range(self.n_pop):
            for t in range(self.n_time-2):
                wage=self.infostats_pop_wage[t,k]
                if self.popempstate[t,k] in set([1]) and wage>minwage: # 9,8,10
                    salx[saln]=wage
                    saln+=1
                    g=int(self.infostats_group[k])
                    m=int(salgn[g])
                    salgx[g,m]=wage
                    salgn[g]+=1

        a=2.95
        mm=a*minwage/(a-1)
        fig,ax=plt.subplots()
        pscale=minwage**(-(a/(a+1)))
        count, bins, _ = ax.hist(salx[:saln], 200, density=True,log=True)
        px=np.linspace(minwage,1_000_000,200)
        rv=pareto(a)
#        vals = rv.pdf(px*pscale)
        vals = rv.pdf(px/minwage)
        ax.plot(px,vals, linewidth=1, color='r')
        med=(rv.median())*minwage
        med_data=np.median(salx[:saln])
        ax.set_xlim([0,1_000_000])
        plt.title(f'pareto {med:.0f} data {med_data:.0f}')
        plt.show()
        
        rv=pareto(a)
        fig,ax=plt.subplots()
        r = (rv.rvs(size=1_000))*minwage
        count, bins, _ = ax.hist(r, 200, density=True,log=True)
        px=np.linspace(minwage,1_000_000,200)
        vals = rv.pdf(px/minwage)
        ax.plot(px,vals, linewidth=1, color='r')
        ax.set_xlim([0,1_000_000])
        med=(rv.median())*minwage
        med_data=np.median(r)
        plt.title(f'pareto {med:.0f} data {med_data:.0f}')
        plt.show()
        
    def plot_rewdist(self,t=0,sum=False,all=False):
        if all:
            fig,ax=plt.subplots()
            for t in range(1,self.n_time-1,5):
                scaled,x=np.histogram(self.poprewstate[t,:])
                x2=0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label=t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x=np.histogram(np.sum(self.poprewstate,axis=0))
                x2=0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax=plt.subplots()
                for t in range(t,t+10,1):
                    scaled,x=np.histogram(self.poprewstate[t,:])
                    x2=0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label=t)
                plt.legend()
                plt.show()

    def plot_unempdistribs_bu(self,unemp_distrib,max=2):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(-max_time,0,nn_time)
        scaled,x2=np.histogram(unemp_distrib,x)
        scaled=scaled/np.abs(np.sum(unemp_distrib))
        fig,ax=plt.subplots()
        #self.plot_vlines_unemp(0.6)
        ax.set_xlabel(self.labels['unemp duration'])
        ax.set_ylabel(self.labels['probability'])
        #x3=np.flip(x[:-1])
        #ax.plot(x3,scaled)
        ax.plot(x[:-1],scaled)
        #ax.set_yscale('log')
        plt.xlim(-max,0)
        plt.show()

    def plot_compare_unempdistribs(self,unemp_distrib1,unemp_distrib2,max=4,
            label2='none',label1='none',logy=True,diff=False,figname=None):
        #fig,ax=plt.subplots()
        max_time=50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x=np.linspace(self.timestep,max_time,nn_time)
        scaled1,x1=np.histogram(unemp_distrib1,x)
        print('{} keskikesto {} v {} keskikesto {} v'.format(label1,np.mean(unemp_distrib1),label2,np.mean(unemp_distrib2)))
        print('Skaalaamaton {} lkm {} v {} lkm {} v'.format(label1,len(unemp_distrib1),label2,len(unemp_distrib2)))
        print('Skaalaamaton {} työtpäiviä yht {} v {} työtpäiviä yht {} v'.format(label1,np.sum(unemp_distrib1),label2,np.sum(unemp_distrib2)))
        #scaled=scaled/np.sum(unemp_distrib)
        scaled1=scaled1/np.sum(scaled1)

        scaled2,x1=np.histogram(unemp_distrib2,x)
        scaled2=scaled2/np.sum(scaled2)
        fig,ax=plt.subplots()
        if not diff:
            self.plot_vlines_unemp(0.5)
        ax.set_xlabel(self.labels['unemp duration'])
        ax.set_ylabel(self.labels['osuus'])
        if diff:
            ax.plot(x[:-1],scaled1-scaled2,label=label1+'-'+label2)
        else:
            ax.plot(x[:-1],scaled2,label=label2)
            ax.plot(x[:-1],scaled1,label=label1)
        if logy and not diff:
            ax.set_yscale('log')
        if not diff:
            plt.ylim(1e-4,1.0)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend()
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'comp_unempdistrib.'+self.figformat, format=self.figformat)

        plt.show()

    def plot_compare_virrat(self,virta1,virta2,min_time=25,max_time=65,label1='perus',label2='vaihtoehto',virta_label='työllisyys',ymin=None,ymax=None):
        x=np.linspace(self.min_age,self.max_age,self.n_time)

        demog2=self.empstats.get_demog()

        scaled1=virta1*demog2/self.n_pop #/self.alive
        scaled2=virta2*demog2/self.n_pop #/self.alive

        fig,ax=plt.subplots()
        plt.xlim(min_time,max_time)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(virta_label+'virta')
        ax.plot(x,scaled1,label=label1)
        ax.plot(x,scaled2,label=label2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin,ymax)

        plt.show()

    def plot_family(self,printtaa=True):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(self.empstate[:,5]+self.empstate[:,6]+self.empstate[:,7])/self.alive[:,0],label='vanhempainvapailla')
        #emp_statsratio=100*self.empstats.outsider_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*np.sum(self.gempstate[:,5,3:6]+self.gempstate[:,6,3:6]+self.gempstate[:,7,3:6],1,
            keepdims=True)/np.sum(self.galive[:,3:6],1,keepdims=True),label='vanhempainvapailla, naiset')
        ax.plot(x,100*np.sum(self.gempstate[:,5,0:3]+self.gempstate[:,6,0:3]+self.gempstate[:,7,0:3],1,
            keepdims=True)/np.sum(self.galive[:,0:3],1,keepdims=True),label='vanhempainvapailla, miehet')
#        emp_statsratio=100*self.empstats.outsider_stats(g=1)
#        ax.plot(x,emp_statsratio,label=self.labels['havainto, naiset'])
#        emp_statsratio=100*self.empstats.outsider_stats(g=2)
#        ax.plot(x,emp_statsratio,label=self.labels['havainto, miehet'])

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        
        if printtaa:
            nn=np.sum(self.galive[:,3:6],1,keepdims=True)
            n=np.sum(100*(self.gempstate[:,5,3:6]+self.gempstate[:,6,3:6]+self.gempstate[:,7,3:6]),1,keepdims=True)/nn
            mn=np.sum(self.galive[:,0:3],1,keepdims=True)
            m=np.sum(100*(self.gempstate[:,5,0:3]+self.gempstate[:,6,0:3]+self.gempstate[:,7,0:3]),1,keepdims=True)/mn
            
        ratio_label=self.labels['ratio']
        empstate_ratio=100*self.empstate/self.alive
        self.plot_states(empstate_ratio,ylabel=ratio_label,parent=True,stack=False)

    def plot_outsider(self,printtaa=True):
        '''
        plottaa työvoiman ulkopuolella olevat
        mukana ei isyysvapaat, opiskelijat, armeijassa olevat eikä alle 3 kk kestäneet äitiysvpaat
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(self.empstate[:,11]+self.empstate[:,5]+self.empstate[:,7]+self.empstate[:,14]
            -self.infostats_mother_in_workforce[:,0])/self.alive[:,0],
            label='työvoiman ulkopuolella, ei opiskelija, sis. vanh.vapaat')
        emp_statsratio=100*self.empstats.outsider_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(self.empstate[:,11])/self.alive[:,0],
            label='työvoiman ulkopuolella, ei opiskelija, ei vanh.vapaat')
        emp_statsratio=100*self.empstats.outsider_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(np.sum(self.gempstate[:,11,3:6]+self.gempstate[:,5,3:6]+self.gempstate[:,7,3:6]+self.gempstate[:,14,3:6],1,keepdims=True)
            -self.infostats_mother_in_workforce)/np.sum(self.galive[:,3:6],1,keepdims=True),
            label='työvoiman ulkopuolella, naiset')
        ax.plot(x,100*np.sum(self.gempstate[:,11,0:3]+self.gempstate[:,7,0:3]+self.gempstate[:,14,0:3],1,keepdims=True)/np.sum(self.galive[:,0:3],1,keepdims=True),
            label='työvoiman ulkopuolella, miehet')
        emp_statsratio=100*self.empstats.outsider_stats(g=1)
        ax.plot(x,emp_statsratio,label=self.labels['havainto, naiset'])
        emp_statsratio=100*self.empstats.outsider_stats(g=2)
        ax.plot(x,emp_statsratio,label=self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        if printtaa:
            #print('yht',100*(self.empstate[:,11]+self.empstate[:,5]+self.empstate[:,6]+self.empstate[:,7])/self.alive[:,0])
            nn=np.sum(self.galive[:,3:6],1,keepdims=True)
            n=np.sum(100*(self.gempstate[:,5,3:6]+self.gempstate[:,6,3:6]+self.gempstate[:,7,3:6]),1,keepdims=True)/nn
            mn=np.sum(self.galive[:,0:3],1,keepdims=True)
            m=np.sum(100*(self.gempstate[:,5,0:3]+self.gempstate[:,6,0:3]+self.gempstate[:,7,0:3]),1,keepdims=True)/mn
            #print('naiset vv',n[1::4,0])
            #print('miehet vv',m[1::4,0])
            
    def plot_tulot(self):
        '''
        plot net income per person
        '''
        x=np.linspace(self.min_age+self.timestep,self.max_age-1,self.n_time-2)
        fig,ax=plt.subplots()
        tulot=self.infostats_tulot_netto[1:-1,0]/self.timestep/self.alive[1:-1,0]
        ax.plot(x,tulot,label='tulot netto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Tulot netto [e/v]')
        ax.legend()
        plt.show()

    def plot_pinkslip(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*self.infostats_pinkslip[:,0]/self.empstate[:,0],label='ansiosidonnaisella')
        ax.plot(x,100*self.infostats_pinkslip[:,4]/self.empstate[:,4],label='putkessa')
        ax.plot(x,100*self.infostats_pinkslip[:,13]/self.empstate[:,13],label='työmarkkinatuella')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Irtisanottujen osuus tilassa [%]')
        ax.legend()
        plt.show()

    def plot_student(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x+self.timestep,100*self.empstate[:,12]/self.alive[:,0],label='opiskelija tai armeijassa')
        emp_statsratio=100*self.empstats.student_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        emp_statsratio=100*self.empstats.student_stats()
        diff=100.0*self.empstate[:,12]/self.alive[:,0]-emp_statsratio
        ax.plot(x+self.timestep,diff,label='opiskelija tai armeijassa, virhe')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

    def plot_kassanjasen(self):
    
        x2,vrt=self.empstats.get_kassanjasenyys_rate()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        jasenia=100.0*self.infostats_kassanjasen/self.alive
        ax.plot(x+self.timestep,jasenia,label='työttömyyskassan jäsenien osuus kaikista')
        ax.plot(x2,100.0*vrt,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        mini=np.nanmin(jasenia)
        maxi=np.nanmax(jasenia)
        print('Kassanjäseniä min {:1f} % max {:1f} %'.format(mini,maxi))

    def plot_group_student(self):
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Opiskelijat+Armeija Miehet'
                opiskelijat=np.sum(self.gempstate[:,12,0:3],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                leg='Opiskelijat+Armeija Naiset'
                opiskelijat=np.sum(self.gempstate[:,12,3:6],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)

            opiskelijat=np.reshape(opiskelijat,(self.galive.shape[0],1))
            osuus=100*opiskelijat/alive
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label=leg)

        emp_statsratio=100*self.empstats.student_stats(g=1)
        ax.plot(x,emp_statsratio,label=self.labels['havainto, naiset'])
        emp_statsratio=100*self.empstats.student_stats(g=2)
        ax.plot(x,emp_statsratio,label=self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_group_disab(self,xstart=None,xend=None):
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='TK Miehet'
                opiskelijat=np.sum(self.gempstate[:,3,0:3],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
            else:
                leg='TK Naiset'
                opiskelijat=np.sum(self.gempstate[:,3,3:6],axis=1)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)

            opiskelijat=np.reshape(opiskelijat,(self.galive.shape[0],1))
            osuus=100*opiskelijat/alive
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label=leg)

        ax.plot(x,100*self.empstats.disab_stat(g=1),label=self.labels['havainto, naiset'])
        ax.plot(x,100*self.empstats.disab_stat(g=2),label=self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if xstart is not None:
            ax.set_xlim([xstart,xend])
        plt.show()

    def plot_taxes(self,figname=None):
        valtionvero_ratio=100*self.infostats_valtionvero_distrib/np.reshape(np.sum(self.infostats_valtionvero_distrib,1),(-1,1))
        kunnallisvero_ratio=100*self.infostats_kunnallisvero_distrib/np.reshape(np.sum(self.infostats_kunnallisvero_distrib,1),(-1,1))
        vero_ratio=100*(self.infostats_kunnallisvero_distrib+self.infostats_valtionvero_distrib)/(np.reshape(np.sum(self.infostats_valtionvero_distrib,1),(-1,1))+np.reshape(np.sum(self.infostats_kunnallisvero_distrib,1),(-1,1)))

        if figname is not None:
            self.plot_states(vero_ratio,ylabel='Valtioneronmaksajien osuus tilassa [%]',stack=True,figname=figname+'_stack')
        else:
            self.plot_states(vero_ratio,ylabel='Valtioneronmaksajien osuus tilassa [%]',stack=True)

        if figname is not None:
            self.plot_states(valtionvero_ratio,ylabel='Veronmaksajien osuus tilassa [%]',stack=True,figname=figname+'_stack')
        else:
            self.plot_states(valtionvero_ratio,ylabel='Veronmaksajien osuus tilassa [%]',stack=True)

        if figname is not None:
            self.plot_states(kunnallisvero_ratio,ylabel='Kunnallisveron maksajien osuus tilassa [%]',stack=True,figname=figname+'_stack')
        else:
            self.plot_states(kunnallisvero_ratio,ylabel='Kunnallisveron maksajien osuus tilassa [%]',stack=True)

        valtionvero_osuus,kunnallisvero_osuus,vero_osuus=self.comp_taxratios()

        print('Valtionveron maksajien osuus\n{}'.format(self.v2_groupstates(valtionvero_osuus)))
        print('Kunnallisveron maksajien osuus\n{}'.format(self.v2_groupstates(kunnallisvero_osuus)))
        print('Veronmaksajien osuus\n{}'.format(self.v2_groupstates(vero_osuus)))

    def group_taxes(self,ratios):
        if len(ratios.shape)>1:
            vv_osuus=np.zeros((ratios.shape[0],5))
            vv_osuus[:,0]=ratios[:,0]+ratios[:,4]+ratios[:,5]+ratios[:,6]+\
                          ratios[:,7]+ratios[:,8]+ratios[:,9]+ratios[:,11]+\
                          ratios[:,12]+ratios[:,13]
            vv_osuus[:,1]=ratios[:,1]+ratios[:,10]
            vv_osuus[:,2]=ratios[:,2]+ratios[:,3]+ratios[:,8]+ratios[:,9]
            vv_osuus[:,3]=ratios[:,1]+ratios[:,10]+ratios[:,8]+ratios[:,9]
        else:
            vv_osuus=np.zeros((4))
            vv_osuus[0]=ratios[0]+ratios[4]+ratios[5]+ratios[6]+\
                          ratios[7]+ratios[8]+ratios[9]+ratios[11]+\
                          ratios[12]+ratios[13]
            vv_osuus[1]=ratios[1]+ratios[10]
            vv_osuus[2]=ratios[2]+ratios[3]+ratios[8]+ratios[9]
            vv_osuus[3]=ratios[1]+ratios[10]+ratios[8]+ratios[9]
        return vv_osuus

    def comp_taxratios(self,grouped=False):
        valtionvero_osuus=100*np.sum(self.infostats_valtionvero_distrib,0)/np.sum(self.infostats_valtionvero_distrib)
        kunnallisvero_osuus=100*np.sum(self.infostats_kunnallisvero_distrib,0)/np.sum(self.infostats_kunnallisvero_distrib)
        vero_osuus=100*(np.sum(self.infostats_kunnallisvero_distrib,0)+np.sum(self.infostats_valtionvero_distrib,0))/(np.sum(self.infostats_kunnallisvero_distrib)+np.sum(self.infostats_valtionvero_distrib))

        if grouped:
            vv_osuus=self.group_taxes(valtionvero_osuus)
            kv_osuus=self.group_taxes(kunnallisvero_osuus)
            v_osuus=self.group_taxes(vero_osuus)
        else:
            vv_osuus=valtionvero_osuus
            kv_osuus=kunnallisvero_osuus
            v_osuus=vero_osuus

        return vv_osuus,kv_osuus,v_osuus

    def comp_verokiila(self,include_retwork=True,debug=False):
        '''
        Computes the tax effect as in Lundberg 2017
        However, this applies the formulas for averages
        '''
        if debug:
            print('comp_verokiila')
        demog2=self.empstats.get_demog()
        scalex=demog2/self.n_pop

        valtionvero_osuus=np.sum(self.infostats_valtionvero_distrib*scalex,0)
        kunnallisvero_osuus=np.sum(self.infostats_kunnallisvero_distrib*scalex,0)
        taxes_distrib=np.sum(self.infostats_taxes_distrib*scalex,0)
        taxes=self.group_taxes(taxes_distrib)

        q=self.comp_budget()
        q2=self.comp_participants(scale=True,include_retwork=include_retwork)
        #htv=q2['palkansaajia']
        #muut_tulot=q['muut tulot']

        # kulutuksen verotus
        tC=0.24*max(0,q['tyotulosumma']-taxes[3])
        # (työssäolevien verot + ta-maksut + kulutusvero)/(työtulosumma + ta-maksut)
        kiila=(taxes[3]+q['ta_maksut']+tC)/(q['tyotulosumma']+q['ta_maksut'])
        qq={}
        qq['tI']=taxes[3]/q['tyotulosumma']
        qq['tC']=tC/q['tyotulosumma']
        qq['tP']=q['ta_maksut']/q['tyotulosumma']

        if debug:
            print('qq',qq,'kiila',kiila)

        return kiila,qq

    def comp_verokiila_kaikki_ansiot(self):
        demog2=self.empstats.get_demog()
        scalex=demog2/self.n_pop

        valtionvero_osuus=np.sum(self.infostats_valtionvero_distrib*scalex,0)
        kunnallisvero_osuus=np.sum(self.infostats_kunnallisvero_distrib*scalex,0)
        taxes_distrib=np.sum(self.infostats_taxes_distrib*scalex,0)
        taxes=self.group_taxes(taxes_distrib)

        q=self.comp_budget()
        q2=self.comp_participants(scale=True)
        htv=q2['palkansaajia']
        muut_tulot=q['muut tulot']
        # kulutuksen verotus
        tC=0.2*max(0,q['tyotulosumma']-taxes[3])
        # (työssäolevien verot + ta-maksut + kulutusvero)/(työtulosumma + ta-maksut)
        kiila=(taxes[0]+q['ta_maksut']+tC)/(q['tyotulosumma']+q['verotettava etuusmeno']+q['ta_maksut'])
        qq={}
        qq['tI']=taxes[0]/q['tyotulosumma']
        qq['tC']=tC/q['tyotulosumma']
        qq['tP']=q['ta_maksut']/q['tyotulosumma']

        #print(qq)

        return kiila,qq

    def v2_states(self,x):
        return 'Ansiosidonnaisella {:.2f}\nKokoaikatyössä {:.2f}\nVanhuuseläkeläiset {:.2f}\nTyökyvyttömyyseläkeläiset {:.2f}\n'.format(x[0],x[1],x[2],x[3])+\
          'Putkessa {:.2f}\nÄitiysvapaalla {:.2f}\nIsyysvapaalla {:.2f}\nKotihoidontuella {:.2f}\n'.format(x[4],x[5],x[6],x[7])+\
          'VE+OA {:.2f}\nVE+kokoaika {:.2f}\nOsa-aikatyö {:.2f}\nTyövoiman ulkopuolella {:.2f}\n'.format(x[8],x[9],x[10],x[11])+\
          'Opiskelija/Armeija {:.2f}\nTM-tuki {:.2f}\n'.format(x[12],x[13])

    def v2_groupstates(self,xx):
        x=self.group_taxes(xx)
        return 'Etuudella olevat {:.2f}\nTyössä {:.2f}\nEläkkeellä {:.2f}\n'.format(x[0],x[1],x[2])

    def plot_children(self,figname=None):
        c3,c7,c18=self.comp_children()
        
        self.plot_y(c3,label='Alle 3v lapset',
                    y2=c7,label2='Alle 7v lapset',
                    y3=c18,label3='Alle 18v lapset',
                    ylabel='Lapsia (lkm)',
                    show_legend=True)

    def plot_workforce(self,figname=None):

        workforce=self.comp_workforce(self.empstate,self.alive)

        age_label=self.labels['age']
        ratio_label=self.labels['osuus']

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,workforce,label=self.labels['malli'])
        emp_statsratio=100*self.empstats.workforce_stats()
        ax.plot(x,emp_statsratio,ls='--',label=self.labels['havainto'])
        ax.set_xlabel(age_label)
        ax.set_ylabel(self.labels['työvoima %'])
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'workforce.'+self.figformat, format=self.figformat)
        plt.show()

    def plot_emp(self,figname=None,tyovoimatutkimus=False):

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=\
            self.comp_empratios(self.empstate,self.alive,unempratio=False) #,mother_in_workforce=self.infostats_mother_in_workforce)
        #tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=False,mother_in_workforce=self.infostats_mother_in_workforce)

        age_label=self.labels['age']
        ratio_label=self.labels['osuus']

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,tyollisyysaste,label=self.labels['malli'])
        #ax.plot(x,tyottomyysaste,label=self.labels['tyottomien osuus'])
        if tyovoimatutkimus:
            emp_statsratio_tvt=100*self.empstats.emp_stats(tyossakayntitutkimus=False)
            ax.plot(x,emp_statsratio_tvt,ls='--',label=self.labels['havainto']+' työvoimatilasto')
        emp_statsratio_tkt=100*self.empstats.emp_stats(tyossakayntitutkimus=True)
        ax.plot(x,emp_statsratio_tkt,ls='--',label=self.labels['havainto']+' työssäkäyntitilasto')
        ax.set_xlabel(age_label)
        ax.set_ylabel(self.labels['tyollisyysaste %'])
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste.'+self.figformat, format=self.figformat)
        plt.show()

        #if self.version in set([1,2,3]):
        fig,ax=plt.subplots()
        ax.stackplot(x,osatyoaste,100-osatyoaste,
                    labels=('osatyössä','kokoaikaisessa työssä')) #, colors=pal) pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        ax.legend()
        plt.show()

    def plot_savings(self):
        savings_0=np.zeros((self.n_time,1))
        savings_1=np.zeros((self.n_time,1))
        savings_2=np.zeros((self.n_time,1))
        act_savings_0=np.zeros((self.n_time,1))
        act_savings_1=np.zeros((self.n_time,1))
        act_savings_2=np.zeros((self.n_time,1))

        for t in range(self.n_time):
            state_0=np.argwhere(self.popempstate[t,:]==0)
            savings_0[t]=np.mean(self.infostats_savings[t,state_0[:]])
            act_savings_0[t]=np.mean(self.sav_actions[t,state_0[:]])
            state_1=np.argwhere(self.popempstate[t,:]==1)
            savings_1[t]=np.mean(self.infostats_savings[t,state_1[:]])
            act_savings_1[t]=np.mean(self.sav_actions[t,state_1[:]])
            state_2=np.argwhere(self.popempstate[t,:]==2)
            savings_2[t]=np.mean(self.infostats_savings[t,state_2[:]])
            act_savings_2[t]=np.mean(self.sav_actions[t,state_2[:]])

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=np.mean(self.infostats_savings,axis=1)
        ax.plot(x,savings,label='savings all')
        ax.legend()
        plt.title('Savings all')
        plt.show()

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=np.mean(self.infostats_savings,axis=1)
        ax.plot(x,savings_0,label='unemp')
        ax.plot(x,savings_1,label='emp')
        ax.plot(x,savings_2,label='retired')
        plt.title('Savings by emp state')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=np.mean(self.sav_actions,axis=1)
        savings_plus=np.nanmean(np.where(self.sav_actions>0,self.sav_actions,np.nan),axis=1)[:,None]
        savings_minus=np.nanmean(np.where(self.sav_actions<0,self.sav_actions,np.nan),axis=1)[:,None]
        ax.plot(x[1:],savings[1:],label='savings action')
        ax.plot(x[1:],savings_plus[1:],label='+savings')
        ax.plot(x[1:],savings_minus[1:],label='-savings')
        ax.legend()
        plt.title('Saving action')
        plt.show()

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        pops=np.random.randint(self.n_pop,size=20)
        ax.plot(x,self.infostats_savings[:,pops],label='savings all')
        #ax.legend()
        plt.title('Savings all')
        plt.show()

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=self.sav_actions[:,pops]
        ax.plot(x[1:],savings[1:,:],label='savings action')
        #ax.legend()
        plt.title('Saving action')
        plt.show()

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=np.mean(self.infostats_savings,axis=1)
        ax.plot(x[1:],act_savings_0[1:],label='unemp')
        ax.plot(x[1:],act_savings_1[1:],label='emp')
        ax.plot(x[1:],act_savings_2[1:],label='retired')
        plt.title('Saving action by emp state')
        ax.legend()
        plt.show()

    def plot_emp_by_gender(self,figname=None):

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        for gender in range(2):
            if gender<1:
                empstate_ratio=100*np.sum(self.gempstate[:,:,0:3],axis=2)/(np.sum(self.galive[:,0:3],axis=1)[:,None])
                genderlabel='miehet'
            else:
                empstate_ratio=100*np.sum(self.gempstate[:,:,3:6],axis=2)/(np.sum(self.galive[:,3:6],axis=1)[:,None])
                genderlabel='naiset'
            if figname is not None:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),stack=True,figname=figname+'_stack')
            else:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),stack=True)

            if self.version in set([1,2,3,4,104]):
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),ylimit=20,stack=False)
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),parent=True,stack=False)
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),unemp=True,stack=False)

            if figname is not None:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),start_from=60,stack=True,figname=figname+'_stack60')
            else:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),start_from=60,stack=True)

    def plot_parents_in_work(self):
        empstate_ratio=100*self.empstate/self.alive
        ml=100*self.infostats_mother_in_workforce/self.alive
        self.plot_y(ml,label='mothers in workforce',show_legend=True,
            y2=empstate_ratio[:,6],ylabel=self.labels['ratio'],label2='isyysvapaa')

    def plot_spouse(self,figname=None,grayscale=False):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        puolisoita=np.sum(self.infostats_puoliso,axis=1)
            
        spouseratio=puolisoita/self.alive[:,0]
        if figname is not None:
            fname=figname+'spouses.'+self.figformat
        else:
            fname=None

        self.plot_y(spouseratio,ylabel=self.labels['spouses'],figname=fname)

    def plot_unemp(self,unempratio=True,figname=None,grayscale=False,tyovoimatutkimus=False):
        '''
        Plottaa työttömyysaste (unempratio=True) tai työttömien osuus väestöstö (False)
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        if unempratio:
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=True)
            unempratio_stat_tvt=100*self.empstats.unempratio_stats(g=0,tyossakayntitutkimus=False)
            unempratio_stat_tkt=100*self.empstats.unempratio_stats(g=0,tyossakayntitutkimus=True)
            if self.language=='Finnish':
                labeli='keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)
                ylabeli=self.labels['tyottomyysaste']
                labeli2='työttömyysaste'
            else:
                labeli='average unemployment rate '+str(ka_tyottomyysaste)
                ylabeli=self.labels['tyottomyysaste']
                labeli2='Unemployment rate'
        else:
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(self.empstate,self.alive,unempratio=False)
            unempratio_stat_tvt=100*self.empstats.unemp_stats(g=0,tyossakayntitutkimus=False)
            unempratio_stat_tkt=100*self.empstats.unemp_stats(g=0,tyossakayntitutkimus=True)
            if self.language=='Finnish':
                labeli='keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
                ylabeli='Työttömien osuus väestöstö [%]'
                labeli2='työttömien osuus väestöstö'
            else:
                labeli='proportion of unemployed'+str(ka_tyottomyysaste)
                ylabeli='Proportion of unemployed [%]'
                labeli2='proportion of unemployed'

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])

        ax.set_ylabel(ylabeli)
        ax.plot(x,tyottomyysaste,label=self.labels['malli'])
        if tyovoimatutkimus:
            ax.plot(x,unempratio_stat_tvt,ls='--',label=self.labels['havainto']+',työvoimatutkimus')
        ax.plot(x,unempratio_stat_tkt,ls='--',label=self.labels['havainto']+',työssäkäyntitutkimus')
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste.'+self.figformat, format=self.figformat)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if tyovoimatutkimus:
            ax.plot(x,unempratio_stat_tvt,label=self.labels['havainto']+',työvoimatutkimus')
        ax.plot(x,unempratio_stat_tkt,label=self.labels['havainto']+',työssäkäyntitutkimus')
        ax.legend()
        if grayscale:
            pal=sns.light_palette("black", 8, reverse=True)
        else:
            pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        ax.stackplot(x,tyottomyysaste,colors=pal) #,label=self.labels['malli'])
        #ax.plot(x,tyottomyysaste)
        plt.show()

        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg='Naiset'
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive,unempratio=unempratio)

            ax.plot(x,tyottomyysaste,color=color,label='{} {}'.format(labeli2,leg))

        if grayscale:
            lstyle='--'
        else:
            lstyle='--'

        if self.version in set([1,2,3,4,104]):
            if unempratio:
                ax.plot(x,100*self.empstats.unempratio_stats(g=1,tyossakayntitutkimus=True),ls=lstyle,label=self.labels['havainto, naiset'])
                ax.plot(x,100*self.empstats.unempratio_stats(g=2,tyossakayntitutkimus=True),ls=lstyle,label=self.labels['havainto, miehet'])
                labeli='keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)
                ylabeli=self.labels['tyottomyysaste']
            else:
                ax.plot(x,100*self.empstats.unemp_stats(g=1,tyossakayntitutkimus=True),ls=lstyle,label=self.labels['havainto, naiset'])
                ax.plot(x,100*self.empstats.unemp_stats(g=2,tyossakayntitutkimus=True),ls=lstyle,label=self.labels['havainto, miehet'])
                labeli='keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
                ylabeli=self.labels['tyottomien osuus']

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste_spk.'+self.figformat, format=self.figformat)
        plt.show()

    def plot_parttime_ratio(self,grayscale=True,figname=None):
        '''
        Plottaa osatyötä tekevien osuus väestöstö
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        labeli2='Osatyötä tekevien osuus'
        fig,ax=plt.subplots()
        for gender in range(2):
            if gender==0:
                leg='Miehet'
                g='men'
                pstyle='-'
            else:
                g='women'
                leg='Naiset'
                pstyle=''

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_gempratios(gender=g,unempratio=False)

            ax.plot(x,osatyoaste,'{}'.format(pstyle),label='{} {}'.format(labeli2,leg))


        o_x=np.array([20,30,40,50,60,70])
        f_osatyo=np.array([55,21,16,12,18,71])
        m_osatyo=np.array([32,8,5,4,9,65])
        if grayscale:
            ax.plot(o_x,f_osatyo,ls='--',label=self.labels['havainto, naiset'])
            ax.plot(o_x,m_osatyo,ls='--',label=self.labels['havainto, miehet'])
        else:
            ax.plot(o_x,f_osatyo,label=self.labels['havainto, naiset'])
            ax.plot(o_x,m_osatyo,label=self.labels['havainto, miehet'])
        labeli='osatyöaste '#+str(ka_tyottomyysaste)
        ylabeli='Osatyön osuus työnteosta [%]'

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'osatyoaste_spk.'+self.figformat, format=self.figformat)
        plt.show()


    def plot_unemp_shares(self):
        empstate_ratio=100*self.empstate/self.alive
        self.plot_states(empstate_ratio,ylabel='Osuus tilassa [%]',onlyunemp=True,stack=True)

    def plot_gender_emp(self,grayscale=False,figname=None):
        fig,ax=plt.subplots()
        if grayscale:
            lstyle='--'
        else:
            lstyle='--'

        for gender in range(2):
            if gender==0:
                leg=self.labels['Miehet']
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg=self.labels['Naiset']
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive)

            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,tyollisyysaste,color=color,label='{} {}'.format(self.labels['tyollisyysaste %'],leg))

        emp_statsratio=100*self.empstats.emp_stats(g=2)
        ax.plot(x,emp_statsratio,ls=lstyle,color='darkgray',label=self.labels['havainto, miehet'])
        emp_statsratio=100*self.empstats.emp_stats(g=1)
        ax.plot(x,emp_statsratio,ls=lstyle,color='black',label=self.labels['havainto, naiset'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        if False:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste_spk.'+self.figformat, format=self.figformat)

        plt.show()
        
    def plot_group_emp(self,grayscale=False,figname=None):
        fig,ax=plt.subplots()
        if grayscale:
            lstyle='--'
        else:
            lstyle='--'

        for gender in range(2):
            if gender==0:
                leg=self.labels['Miehet']
                gempstate=np.sum(self.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=np.sum(self.galive[:,3:6],1)
                leg=self.labels['Naiset']
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive)

            x=np.linspace(self.min_age,self.max_age,self.n_time)
            #ax.plot(x,tyollisyysaste,color=color,label='{} {}'.format(self.labels['tyollisyysaste %'],leg))

        for group in range(self.n_groups):
            if group<3:
                leg=self.labels['Miehet']+'group '+str(group)
                gempstate=self.gempstate[:,:,group]
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=self.galive[:,group]
                color='darkgray'
            else:
                gempstate=self.gempstate[:,:,group]
                alive=np.zeros((self.galive.shape[0],1))
                alive[:,0]=self.galive[:,group]
                leg=self.labels['Naiset']+'group '+str(group)
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.comp_empratios(gempstate,alive)

            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,tyollisyysaste,color=color,label='{} {}'.format(self.labels['tyollisyysaste %'],leg))

        emp_statsratio=100*self.empstats.emp_stats(g=2)
        ax.plot(x,emp_statsratio,ls=lstyle,color='darkgray',label=self.labels['havainto, miehet'])
        emp_statsratio=100*self.empstats.emp_stats(g=1)
        ax.plot(x,emp_statsratio,ls=lstyle,color='black',label=self.labels['havainto, naiset'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        if True:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste_spk.'+self.figformat, format=self.figformat)

        plt.show()
        
    def plot_pensions(self):
        if self.version in set([1,2,3,4,104]):
            self.plot_ratiostates(self.stat_pension,ylabel='Tuleva eläke [e/v]',stack=False)

    def plot_career(self):
        if self.version in set([1,2,3,4,104]):
            self.plot_ratiostates(self.stat_tyoura,ylabel='Työuran pituus [v]',stack=False)

    def plot_ratiostates(self,statistic,ylabel='',ylimit=None, show_legend=True, parent=False,work60=False,\
                         unemp=False,stack=False,no_ve=False,figname=None,emp=False,oa_unemp=False,start_from=None,end_at=None):
        self.plot_states(statistic/self.empstate[:statistic.shape[0],:statistic.shape[1]],ylabel=ylabel,ylimit=ylimit,no_ve=no_ve,\
                        show_legend=show_legend,parent=parent,unemp=unemp,\
                        stack=stack,figname=figname,emp=emp,oa_unemp=oa_unemp,work60=work60,start_from=start_from,end_at=end_at)

    def count_putki(self,emps=None):
        if emps is None:
            piped=np.reshape(self.empstate[:,4],(self.empstate[:,4].shape[0],1))
            demog2=self.empstats.get_demog()
            putkessa=self.timestep*np.nansum(piped[1:]/self.alive[1:]*demog2[1:])
            return putkessa
        else:
            piped=np.reshape(emps[:,4],(emps[:,4].shape[0],1))
            demog2=self.empstats.get_demog()
            alive=np.sum(emps,axis=1,keepdims=True)
            putkessa=self.timestep*np.nansum(piped[1:]/alive[1:]*demog2[1:])
            return putkessa

    def plot_y(self,y1,y2=None,y3=None,y4=None,label='',ylabel='',label2=None,label3=None,label4=None,
            ylimit=None,show_legend=False,start_from=None,end_at=None,figname=None,
            yminlim=None,ymaxlim=None,grayscale=False,title=None,reverse=False):
        

        fig,ax=plt.subplots()
        if start_from is None:
            x=np.linspace(self.min_age,self.max_age,self.n_time)
        else:
            if end_at is None:
                end_at=self.max_age
            x_n = end_at-start_from+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))#+2
            x=np.linspace(start_from,self.max_age,x_t)
            y1=y1[self.map_age(start_from):self.map_age(end_at)]
            if y2 is not None:
                y2=y2[self.map_age(start_from):self.map_age(end_at)]
            if y3 is not None:
                y3=y3[self.map_age(start_from):self.map_age(end_at)]
            if y4 is not None:
                y4=y4[self.map_age(start_from):self.map_age(end_at)]

        if grayscale:
            pal=sns.light_palette("black", 8, reverse=True)
        else:
            pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            
        if start_from is None:
            ax.set_xlim(self.min_age,self.max_age)
        else:
            ax.set_xlim(start_from,end_at)

        if title is None:
            plt.title(title)

        if ymaxlim is not None or yminlim is not None:
            ax.set_ylim(yminlim,ymaxlim)


        ax.plot(x,y1,label=label)
        if y2 is not None:
            ax.plot(x,y2,label=label2)
        if y3 is not None:
            ax.plot(x,y3,label=label3)
        if y4 is not None:
            ax.plot(x,y4,label=label4)

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabel)
        if show_legend:
            if not reverse:
                lgd=ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                handles, labels = ax.get_legend_handles_labels()
                lgd=ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
        #fig.tight_layout()
        if figname is not None:
            if show_legend:
                plt.savefig(figname,bbox_inches='tight',bbox_extra_artists=(lgd,), format=self.figformat)
            else:
                plt.savefig(figname,bbox_inches='tight', format=self.figformat)
        plt.show()

    def plot_groups(self,y1,y2=None,y3=None,y4=None,label='',ylabel='',label2=None,label3=None,label4=None,
            ylimit=None,show_legend=True,start_from=None,figname=None,
            yminlim=None,ymaxlim=None,grayscale=False,title=None,reverse=False):

        fig,ax=plt.subplots()
        if start_from is None:
            x=np.linspace(self.min_age,self.max_age,self.n_time)
        else:
            x_n = self.max_age-start_from+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))+1
            x=np.linspace(start_from,self.max_age,x_t)
            y1=y1[self.map_age(start_from):,:]
            if y2 is not None:
                y2=y2[self.map_age(start_from):,:]
            if y3 is not None:
                y3=y3[self.map_age(start_from):,:]
            if y4 is not None:
                y4=y4[self.map_age(start_from):,:]

        if grayscale:
            pal=sns.light_palette("black", 8, reverse=True)
        else:
            pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            
        if start_from is None:
            ax.set_xlim(self.min_age,self.max_age)
        else:
            ax.set_xlim(start_from,self.max_age)

        if title is None:
            plt.title(title)

        if ymaxlim is not None or yminlim is not None:
            ax.set_ylim(yminlim,ymaxlim)


        for g in range(6):
            ax.plot(x,y1[:,g],label='group '+str(g))
        if y2 is not None:
            ax.plot(x,y2,label=label2)
        if y3 is not None:
            ax.plot(x,y3,label=label3)
        if y4 is not None:
            ax.plot(x,y4,label=label4)

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabel)
        if show_legend:
            if not reverse:
                lgd=ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                handles, labels = ax.get_legend_handles_labels()
                lgd=ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
        #fig.tight_layout()
        if figname is not None:
            if show_legend:
                plt.savefig(figname,bbox_inches='tight',bbox_extra_artists=(lgd,), format=self.figformat)
            else:
                plt.savefig(figname,bbox_inches='tight', format=self.figformat)
        plt.show()

    def plot_states(self,statistic,ylabel='',ylimit=None,show_legend=True,parent=False,unemp=False,no_ve=False,
                    start_from=None,end_at=None,stack=True,figname=None,yminlim=None,ymaxlim=None,work60=False,
                    onlyunemp=False,reverse=False,grayscale=False,emp=False,oa_unemp=False):
        if start_from is None:
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            x=x[:statistic.shape[0]]
        else:
            x_n = self.max_age-start_from+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))+1
            x=np.linspace(start_from,self.max_age,x_t)
            #x=np.linspace(start_from,self.max_age,self.n_time)
            statistic=statistic[self.map_age(start_from):]
            x=x[:statistic.shape[0]]

        ura_emp=statistic[:,1]
        ura_ret=statistic[:,2]
        ura_unemp=statistic[:,0]
        if self.version in set([1,2,3,4,104]):
            ura_disab=statistic[:,3]
            ura_pipe=statistic[:,4]
            ura_mother=statistic[:,5]
            ura_dad=statistic[:,6]
            ura_kht=statistic[:,7]
            ura_vetyo=statistic[:,9]
            ura_veosatyo=statistic[:,8]
            ura_osatyo=statistic[:,10]
            ura_outsider=statistic[:,11]
            ura_student=statistic[:,12]
            ura_tyomarkkinatuki=statistic[:,13]
            ura_svpaiva=statistic[:,14]
        else:
            ura_osatyo=0 #statistic[:,3]

        if no_ve:
            ura_ret[-2:-1]=None

        fig,ax=plt.subplots()
        if stack:
            if grayscale:
                pal=sns.light_palette("black", 8, reverse=True)
            else:
                pal=sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            reverse=True

            if parent:
                if self.version in set([1,2,3,4,104]):
                    ax.stackplot(x,ura_mother,ura_dad,ura_kht,
                        labels=('äitiysvapaa','isyysvapaa','khtuki'), colors=pal)
            elif unemp:
                if self.version in set([1,2,3,4,104]):
                    ax.stackplot(x,ura_unemp,ura_pipe,ura_student,ura_outsider,ura_tyomarkkinatuki,ura_svpaiva,
                        labels=('tyött','putki','opiskelija','ulkona','tm-tuki','svpaivaraha'), colors=pal)
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            elif onlyunemp:
                if self.version in set([1,2,3,4,104]):
                    #urasum=np.nansum(statistic[:,[0,4,11,13]],axis=1)/100
                    urasum=np.nansum(statistic[:,[0,4,13]],axis=1)/100
                    osuus=(1.0-np.array([0.84,0.68,0.62,0.58,0.57,0.55,0.53,0.50,0.29]))*100
                    xx=np.array([22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5])
                    ax.stackplot(x,ura_unemp/urasum,ura_pipe/urasum,ura_tyomarkkinatuki/urasum,
                        labels=('ansiosidonnainen','lisäpäivät','tm-tuki'), colors=pal)
                    ax.plot(xx,osuus,color='k')
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            else:
                if self.version in set([1,2,3,4,104]):
                    ax.stackplot(x,ura_emp,ura_osatyo,ura_vetyo,ura_veosatyo,ura_unemp,ura_tyomarkkinatuki,ura_pipe,ura_ret,ura_disab,ura_mother,ura_dad,ura_kht,ura_student,ura_outsider,ura_svpaiva,
                        labels=('työssä','osatyö','ve+työ','ve+osatyö','työtön','tm-tuki','työttömyysputki','vanhuuseläke','tk-eläke','äitiysvapaa','isyysvapaa','kh-tuki','opiskelija','työvoiman ulkop.','svpaivaraha'),
                        colors=pal)
                else:
                    ax.stackplot(x,ura_emp,ura_unemp,ura_ret,
                        labels=('työssä','työtön','vanhuuseläke'), colors=pal)

            if ymaxlim is None:
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(yminlim,ymaxlim)
        else:
            if parent:
                if self.version in set([1,2,3,4,104]):
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
            elif unemp:
                ax.plot(x,ura_unemp,label='tyött')
                if self.version in set([1,2,3,4,104]):
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_pipe,label='putki')
            elif oa_unemp:
                ax.plot(x,ura_unemp,label='tyött')
                if self.version in set([1,2,3,4,104]):
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_pipe,label='putki')
                    ax.plot(x,ura_osatyo,label='osa-aika')
            elif emp:
                ax.plot(x,ura_emp,label='työssä')
                #if self.version in set([1,2,3,4]):
                ax.plot(x,ura_osatyo,label='osatyö')
            elif work60:
                ax.plot(x,ura_ret,label='eläke')
                ax.plot(x,ura_emp,label='työ')
                if self.version in set([1,2,3,4,104]):
                    ax.plot(x,ura_osatyo,label='osatyö')
                    ax.plot(x,ura_vetyo,label='ve+työ')
                    ax.plot(x,ura_veosatyo,label='ve+osatyö')
            else:
                ax.plot(x,ura_unemp,label='tyött')
                ax.plot(x,ura_ret,label='eläke')
                ax.plot(x,ura_emp,label='työ')
                if self.version in set([1,2,3,4,104]):
                    ax.plot(x,ura_osatyo,label='osatyö')
                    ax.plot(x,ura_disab,label='tk')
                    ax.plot(x,ura_pipe,label='putki')
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
                    ax.plot(x,ura_vetyo,label='ve+työ')
                    ax.plot(x,ura_veosatyo,label='ve+osatyö')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_svpaiva,label='svpaivaraha')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabel)
        if show_legend:
            if not reverse:
                lgd=ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                handles, labels = ax.get_legend_handles_labels()
                lgd=ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if start_from is None:
            if end_at is None:
                ax.set_xlim(self.min_age,self.max_age)
            else:
                ax.set_xlim(self.min_age,end_at)
        else:
            if end_at is None:
                ax.set_xlim(start_from,self.max_age)
            else:
                ax.set_xlim(start_from,end_at)
                
        if ylimit is not None:
            ax.set_ylim([0,ylimit])
        #fig.tight_layout()
        if figname is not None:
            if show_legend:
                plt.savefig(figname,bbox_inches='tight',bbox_extra_artists=(lgd,), format=self.figformat)
            else:
                plt.savefig(figname,bbox_inches='tight', format=self.figformat)
        plt.show()

    def plot_toe(self):
        if self.version in set([1,2,3,4,104]):
            self.plot_ratiostates(self.stat_toe,'työssäolo-ehdon pituus 28 kk aikana [v]',stack=False)

    def plot_sal(self):
        self.plot_ratiostates(self.salaries_emp,'Keskipalkka [e/v]',stack=False)

    def plot_moved(self):
        siirtyneet_ratio=self.siirtyneet/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        pysyneet_ratio=self.pysyneet/self.alive*100
        self.plot_states(pysyneet_ratio,ylabel='Pysyneet tilassa',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(pysyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,:,1]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet työhön tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,7,:]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet kotihoidontuelle',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,8,:]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet ve+oatyö tilaan',stack=True,start_from=60,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,9,:]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet ve+työ tilaan',stack=True,start_from=60,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,:,4]/self.alive*100
        self.plot_states(siirtyneet_ratio,start_from=58,ylabel='Siirtyneet putkeen tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,4,:]/self.alive*100
        self.plot_states(siirtyneet_ratio,start_from=58,ylabel='Siirtyneet putkesta tilaan',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,:,0]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet työttömäksi tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,:,13]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tm-tuelle tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.siirtyneet_det[:,:,10]/self.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet osa-aikatyöhön tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))

    def plot_ave_stay(self):
        self.plot_ratiostates(self.time_in_state,ylabel='Ka kesto tilassa',stack=False)
        self.plot_y(self.time_in_state[:,1]/self.empstate[:,1],ylabel='Ka kesto työssä')
        self.plot_y(self.time_in_state[:,0]/self.empstate[:,0],ylabel='Ka kesto työttömänä')

    def plot_ove(self):
        self.plot_ratiostates(self.infostats_ove,ylabel='Ove',stack=False,start_from=60)
        #self.plot_ratiostates(np.sum(self.infostats_ove,axis=1),ylabel='Ove',stack=False)
        #plt.plot(np.sum(cc1.episodestats.infostats_ove,axis=1)/cc1.episodestats.alive[:,0],label='Ove')
        self.plot_y(np.sum(self.infostats_ove,axis=1)/self.alive[:,0],ylabel='Oven ottaneet',start_from=60,end_at=70)
        self.plot_y((self.infostats_ove[:,1]+self.infostats_ove[:,10])/(self.empstate[:,1]+self.empstate[:,10]),label='Kaikista työllisistä',
            y2=(self.infostats_ove[:,1])/(self.empstate[:,1]),label2='Kokotyöllisistä',
            y3=(self.infostats_ove[:,10])/(self.empstate[:,10]),label3='Osatyöllisistä',
            ylabel='Oven ottaneet',show_legend=True,start_from=60,end_at=70)
        self.plot_y((self.infostats_ove[:,0]+self.infostats_ove[:,13]+self.infostats_ove[:,4])/(self.empstate[:,0]+self.empstate[:,13]+self.empstate[:,4]),label='Kaikista työttömistä',
            y2=(self.infostats_ove[:,0])/(self.empstate[:,0]),label2='Ansiosid.',
            y3=(self.infostats_ove[:,4])/(self.empstate[:,4]),label3='Putki',
            y4=(self.infostats_ove[:,13])/(self.empstate[:,13]),label4='TM-tuki',
            ylabel='Oven ottaneet',show_legend=True,start_from=60,end_at=70)
        self.plot_groups(np.sum(self.infostats_ove_g,axis=2)/self.galive,start_from=60,ylabel='Osuus')

    def plot_reward(self):
        self.plot_ratiostates(self.rewstate,ylabel='Keskireward tilassa',stack=False)
        self.plot_ratiostates(self.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,no_ve=True)
        self.plot_ratiostates(self.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,oa_unemp=True)
        self.plot_ratiostates(self.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,oa_unemp=True,start_from=60)
        self.plot_ratiostates(self.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,start_from=60)
        self.plot_ratiostates(self.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,start_from=60,work60=True)
        #self.plot_y(np.sum(self.rewstate,axis=1),label='Koko reward tilassa')

    def vector_to_array(self,x):
        return x[:,None]

    def comp_scaled_consumption(self,x0,averaged=False,t0=1,debug=False):
        '''
        Computes discounted actual reward at each time point
        with a given scaling x

        averaged determines whether the value is averaged over time or not
        '''
        x=x0[0]
        u=np.zeros((self.n_time,1))
        for k in range(self.n_pop):
            #g=self.infostats_group[k]
            for t in range(1,self.n_time-1):
                age=t+self.min_age
                income=self.infostats_poptulot_netto[t,k]
                employment_state=self.popempstate[t,k]
                v,_=self.env.log_utility((1+x)*income,employment_state,age)
                if not np.isfinite(v):
                    if debug:
                        print('NaN',v,income,employment_state,age)
                    v=0
                u[t]+=v
                #print(age,v)

            t=self.n_time-1
            age=t+self.min_age
            income=self.infostats_poptulot_netto[t,k]
            employment_state=self.popempstate[t,k]
            v0,_=self.env.log_utility(income,employment_state,age)
            if not np.isfinite(v0):
                factor=0
            else:
                factor=self.poprewstate[t,k]/v0 # life expectancy
            v,_=self.env.log_utility((1+x)*income,employment_state,age)
            if np.isnan(v) and debug:
                print('NaN',v,income,employment_state,age)
            if np.isnan(factor) and debug:
                print('NaN',factor,v0)

            u[t]+=v*factor
            if np.isnan(u[t]) and debug:
                print('NaN',age,v,v*factor,factor,u[t],income,employment_state)

        u=u/self.n_pop
        w=np.zeros((self.n_time,1))
        w[-1]=u[-1]
        for t in range(self.n_time-2,0,-1):
            w[t]=u[t]+self.gamma*w[t+1]

        if averaged:
            ret=np.mean(w[t0:])
        else:
            ret=w[t0]

        if not np.isfinite(ret):
            if debug:
                print('u',u,'\nw',w)
            u=np.zeros((self.n_time,1))
            for k in range(self.n_pop):
                #g=self.infostats_group[k]
                for t in range(1,self.n_time-1):
                    age=t+self.min_age
                    income=self.infostats_poptulot_netto[t,k]
                    employment_state=self.popempstate[t,k]
                    v,_=self.env.log_utility((1+x)*income,employment_state,age,debug=False) #,g=g,pinkslip=pinkslip)
                    if not np.isfinite(v):
                        v=0
                    #print(t,k,v,income)
                    u[t]+=v
                t=self.n_time-1
                age=t-1+self.min_age
                income=self.infostats_poptulot_netto[t,k]
                employment_state=self.popempstate[t,k]
                v0,_=self.env.log_utility(income,employment_state,age,debug=False) #,g=g,pinkslip=pinkslip)
                if not np.isfinite(v0):
                    v0=0
                    factor=0
                else:
                    factor=self.poprewstate[t,k]/v0 # life expectancy
                v,_=self.env.log_utility((1+x)*income,employment_state,age,debug=False) #,g=g,pinkslip=pinkslip)
                if not np.isfinite(v):
                    v=0
                #print(t,k,v,income)
                u[t]+=v*factor

        #print(x,ret)

        return ret

    def comp_presentvalue(self):
        '''
        Computes discounted actual reward at each time point
        with a given scaling x

        averaged determines whether the value is averaged over time or not
        '''
        u=np.zeros((self.n_time,self.n_pop))
        u[self.n_time-1,:]=self.poprewstate[self.n_time-1,:]
        for t in range(self.n_time-2,-1,-1):
            u[t,:]=self.poprewstate[t,:]+self.gamma*u[t+1,:]

        return u

    def comp_realoptimrew(self,discountfactor=None):
        '''
        Computes discounted actual reward at each time point
        '''
        if discountfactor is None:
            discountfactor=self.gamma

        realrew=np.zeros((self.n_time,1))
        for k in range(self.n_pop):
            prew=np.zeros((self.n_time,1))
            prew[-1]=self.poprewstate[-1,k]
            for t in range(self.n_time-2,0,-1):
                prew[t]=discountfactor*prew[t+1]+self.poprewstate[t,k]

            realrew+=prew

        realrew/=self.n_pop
        realrew=np.mean(realrew[1:])

        return realrew

    def get_reward(self,discounted=False):
        return self.comp_total_reward(output=False,discounted=discounted) #np.sum(self.rewstate)/self.n_pop

    def comp_total_reward(self,output=False,discounted=False,discountfactor=None):
        if not discounted:
            total_reward=np.sum(self.rewstate)
            rr=total_reward/self.n_pop
            disco='undiscounted'
        else:
            #discount=discountfactor**np.arange(0,self.n_time*self.timestep,self.timestep)[:,None]
            #total_reward=np.sum(self.poprewstate*discount)

            rr=self.comp_realoptimrew(discountfactor) #total_reward/self.n_pop
            disco='discounted'

        #print('total rew1 {} rew2 {}'.format(total_reward,np.sum(self.poprewstate)))
        #print('ave rew1 {} rew2 {}'.format(rr,np.mean(np.sum(self.poprewstate,axis=0))))
        #print('shape rew2 {} pop {} alive {}'.format(self.poprewstate.shape,self.n_pop,self.alive[0]))

        if output:
            print(f'Ave {disco} reward {rr}')

        return rr
        
    def comp_total_netincome(self,output=True):
        if True:
            alivemask=(self.popempstate==self.env.get_mortstate())
            alive=np.sum(self.alive)
        
            multiplier=np.sum(self.infostats_npv0)/alive
            rr=np.sum(self.infostats_tulot_netto)/(alive*multiplier)
            eq=np.sum(self.infostats_equivalent_income)/(alive*multiplier)
        elif False:
            multiplier=np.sum(self.infostats_npv0)/self.n_pop
            rr=np.sum(self.infostats_tulot_netto)/(self.n_pop*multiplier)
            eq=np.sum(self.infostats_equivalent_income)/(self.n_pop*multiplier)
        else:
            rr=np.sum(self.infostats_tulot_netto)/(self.n_pop*(self.n_time*self.timestep+21.0)) # 21 years approximates the time in pension
            eq=np.sum(self.infostats_equivalent_income)/(self.n_pop*(self.n_time*self.timestep+21.0)) # 21 years approximates the time in pension
        
        #print(self.infostats_tulot_netto,self.infostats_tulot_netto.shape,self.n_pop,(self.n_time*self.timestep+21.0))
        #print(self.infostats_equivalent_income,self.infostats_equivalent_income.shape,self.n_pop,(self.n_time*self.timestep+21.0))

        if output:
            print('Ave net income {} Ave equivalent net income {}'.format(rr,eq))

        return rr,eq

    def plot_wage_reduction(self):
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False)
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False,unemp=True)
        self.plot_ratiostates(self.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False,emp=True)
        #self.plot_ratiostates(np.log(1.0+self.stat_wage_reduction),ylabel='log 5wage-reduction tilassa',stack=False)
        self.plot_ratiostates(np.sum(self.stat_wage_reduction_g[:,:,0:3],axis=2),ylabel='wage-reduction tilassa naiset',stack=False)
        self.plot_ratiostates(np.sum(self.stat_wage_reduction_g[:,:,3:6],axis=2),ylabel='wage-reduction tilassa miehet',stack=False)
        self.plot_ratiostates(np.sum(self.stat_wage_reduction_g[:,:,0:3],axis=2),ylabel='wage-reduction tilassa, naiset',stack=False,unemp=True)
        self.plot_ratiostates(np.sum(self.stat_wage_reduction_g[:,:,3:6],axis=2),ylabel='wage-reduction tilassa, miehet',stack=False,unemp=True)
        self.plot_ratiostates(np.sum(self.stat_wage_reduction_g[:,:,0:3],axis=2),ylabel='wage-reduction tilassa, naiset',stack=False,emp=True)
        self.plot_ratiostates(np.sum(self.stat_wage_reduction_g[:,:,3:6],axis=2),ylabel='wage-reduction tilassa, miehet',stack=False,emp=True)

    def plot_distrib(self,label='',plot_emp=False,plot_bu=False,ansiosid=False,tmtuki=False,putki=False,outsider=False,max_age=500,laaja=False,max=4,figname=None):
        unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        if plot_emp:
            self.plot_empdistribs(emp_distrib)
        if plot_bu:
            self.plot_unempdistribs_bu(unemp_distrib_bu)
        else:
            self.plot_unempdistribs(unemp_distrib,figname=figname)
        #self.plot_tyolldistribs(unemp_distrib,tyoll_distrib,tyollistyneet=False)
        if plot_bu:
            self.plot_tyolldistribs_both_bu(unemp_distrib_bu,tyoll_distrib_bu,max=max)
        else:
            self.plot_tyolldistribs_both(unemp_distrib,tyoll_distrib,max=max,figname=figname)

    def plot_irr(self,figname=''):
        self.comp_aggirr()
        self.comp_aggirr(gender=1)
        self.comp_aggirr(gender=2)
        self.comp_irr()
        self.plot_irrdistrib(self.infostats_irr_tyel_reduced,figname=figname+'_reduced',reduced=True)
        self.plot_irrdistrib(self.infostats_irr_tyel_full,figname=figname+'_full')
        self.plot_irrdistrib(self.infostats_irr_tyel_full,figname=figname+'_full_naiset',gender=1)
        self.plot_irrdistrib(self.infostats_irr_tyel_full,figname=figname+'_full_miehet',gender=2)

    def plot_irrdistrib(self,irr_distrib,grayscale=True,figname='',reduced=False,gender=None):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        mortstate=self.env.get_mortstate()
        
        if gender is None:
            gendermask=np.ones_like(irr_distrib)
            gendername=''
        else:
            if gender==1: # naiset
                gendermask = self.infostats_group>2
                gendername=' (Naiset)'
            else: # miehet
                gendermask = self.infostats_group<3
                gendername=' (Miehet)'
        
        nanmask = (np.isnan(irr_distrib).astype(int)*gendermask.astype(int)).astype(bool)
        v2_irrdata=ma.array(irr_distrib,mask=nanmask)
        v2data=v2_irrdata.compressed() # nans dropped
        
        ika=65
        alivemask = self.popempstate[self.map_age(ika),:]!=mortstate
        alivemask = np.reshape(alivemask,(-1,1))
        alivemask = (alivemask.astype(int)*gendermask.astype(int)).astype(bool)
        
        #nan_alivemask=alivemask*nanmask[:,0]
        #nanalive_irrdata=ma.array(irr_distrib,mask=nan_alivemask)
        #nanalive_data=nanalive_irrdata.compressed() # nans dropped

        if reduced:
            print('\nTyel-irr huomioiden kansan- ja takuueläkkeen yhteensovitus'+gendername)
        else:
            print('\nTyel-irr ILMAN kansan- ja takuueläkkeen yhteensovitusta'+gendername)

        fig,ax=plt.subplots()
        ax.set_xlabel('Sisäinen tuottoaste [%]')
        lbl=ax.set_ylabel('Taajuus')
        x=np.linspace(-7,7,100)
        scaled,x2=np.histogram(irr_distrib,x)
        ax.bar(x2[1:-1],scaled[1:],align='center')
        if figname is not None:
            plt.savefig(figname+'irrdistrib.'+self.figformat, bbox_inches='tight', format=self.figformat)
        plt.show()

        fig,ax=plt.subplots()
        ax.hist(irr_distrib,bins=40,density=True)
        plt.show()

        irr_distrib_wout_nans=irr_distrib.copy()
        irr_distrib_wout_nans[np.isnan(irr_distrib_wout_nans)]=0 # mostly not defined of type 0/0, hence use 0
        
        print('Kaikki havainnot (NaN -> 0):\n- Keskimääräinen irr {:.3f} % reaalisesti'.format(np.mean(irr_distrib_wout_nans)))
        print('- Mediaani irr {:.3f} % reaalisesti'.format(np.median(irr_distrib_wout_nans)))
        
        # v2
        print('v2 Kaikki havainnot:\n- Keskimääräinen irr {:.3f} % reaalisesti'.format(np.mean(v2data)))
        print('- Mediaani irr {:.3f} % reaalisesti'.format(np.median(v2data)))

        print('Ilman Nan:eja\n- Keskimääräinen irr {:.3f} % reaalisesti'.format(np.nanmean(irr_distrib)))
        print('- Mediaani irr {:.3f} % reaalisesti'.format(np.nanmedian(irr_distrib)))
        print('- Nans {} %'.format(100*np.sum(np.isnan(irr_distrib))/irr_distrib.shape[0]))

        # kaikki havainnot
        percent_m50 = 100*np.true_divide((irr_distrib <=-50).sum(axis=0),irr_distrib.shape[0])[0]
        percent_m10 = 100*np.true_divide((irr_distrib <=-10).sum(axis=0),irr_distrib.shape[0])[0]
        percent_0 =   100*np.true_divide((irr_distrib <=0).sum(axis=0),irr_distrib.shape[0])[0]
        percent_p10 = 100*np.true_divide((irr_distrib >=10).sum(axis=0),irr_distrib.shape[0])[0]
        print(f'Kaikki havainnot\nOsuudet\n- irr < -50% {percent_m50:.2f} %:lla\n- irr < -10% {percent_m10:.2f} %')
        print(f'- irr < 0% {percent_0:.2f} %:lla\n- irr > 10% {percent_p10:.2f} %\n')
        
        # v2
        percent_m50 = 100*((v2data <=-50).sum(axis=0)/v2data.shape[0])
        percent_m10 = 100*((v2data <=-10).sum(axis=0)/v2data.shape[0])
        percent_0 =   100*((v2data <=0).sum(axis=0)/v2data.shape[0])
        percent_p10 = 100*((v2data >=10).sum(axis=0)/v2data.shape[0])
        print(f'Ilman NaNeja\nOsuudet\n- irr < -50% {percent_m50:.2f} %:lla\n- irr < -10% {percent_m10:.2f} %')
        print(f'- irr < 0% {percent_0:.2f} %:lla\n- irr > 10% {percent_p10:.2f} %\n')

        count = (np.sum(self.stat_pop_paidpension,axis=0)<0.1).sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus mikään eläke ei lainkaan maksussa {:.2f} %'.format(100*percent))
        
        no_pension = (np.sum(self.infostats_pop_tyoelake,axis=0)<0.1)
        count = no_pension.sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläkettä ei lainkaan maksussa {:.2f} %'.format(100*percent))
        
        nopremium = (np.sum(self.infostats_tyelpremium,axis=0)<0.1)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei lainkaan maksua maksettu {:.2f} %'.format(100*percent))
        
        count = (np.sum(self.infostats_paid_tyel_pension,axis=0)<0.1).sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei vastinetta maksulle {:.2f} %'.format(100*percent))
        
        nopp = no_pension*nopremium
        count=nopp.sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei lainkaan maksua eikä maksettua eläkettä {:.2f} %'.format(100*percent))

        print('\nOsuudet\n')
        arri=ma.sum(ma.array(self.stat_pop_paidpension[self.map_age(ika),:],mask=alivemask))<0.1
        percent = np.true_divide(ma.sum(arri),self.alive[self.map_age(ika),0])
        print('{}v osuus eläke ei maksussa, ei kuollut {:.2f} %'.format(ika,100*percent))

        alivemask = self.popempstate[self.map_age(ika),:]!=mortstate
        count = (np.sum(self.infostats_pop_tyoelake,axis=0)<0.1).sum(axis=0)
        percent = np.true_divide(count,self.alive[self.map_age(ika),0])
        print('{}v osuus ei työeläkekarttumaa, ei kuollut {:.2f} %'.format(ika,100*percent))

        count = 1-self.alive[self.map_age(ika),0]/self.n_pop
        print('{}v osuus kuolleet {:.2f} % '.format(ika,100*count))

        count = 1-self.alive[-1,0]/self.n_pop
        print('Lopussa osuus kuolleet {:.2f} % '.format(100*count))

    def get_initial_reward(self,startage=None):
        real=self.comp_presentvalue()
        if startage is None:
            startage=self.min_age
        age=max(1,startage-self.min_age)
        realage=max(self.min_age+1,startage)
        print('Initial discounted reward at age {}: {}'.format(realage,np.mean(real[age,:])))
        return np.mean(real[age,:])


    def plot_img(self,img,xlabel="eläke",ylabel="Palkka",title="Employed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        heatmap = plt.pcolor(img)
        plt.colorbar(heatmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()


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

    def plot_Vk(self,k,getV=None):
        obsV=np.zeros(((self.n_time,1)))

        #obsV[-1]=self.poprewstate[-1,k]
        for t in range(self.n_time-2,-1,-1):
            obsV[t]=self.poprewstate[t+1,k]+self.gamma*obsV[t+1]
            rewerr=self.poprewstate[t+1,k]-self.pop_predrew[t+1,k]
            delta=obsV[t]-self.aveV[t+1,k]
            wage=self.infostats_pop_wage[t,k]
            old_wage=self.infostats_pop_wage[max(0,t-1),k]
            age=self.map_t_to_age(t)
            old_age=int(self.map_t_to_age(max(0,t-1)))
            emp=self.popempstate[t,k]
            predemp=self.popempstate[max(0,t-1),k]
            pen=self.infostats_pop_pension[t,k]
            predwage=self.env.wage_process_mean(old_wage,old_age,state=predemp)
            print(f'{age}: {obsV[t]:.4f} vs {self.aveV[t+1,k]:.4f} d {delta:.4f} re {rewerr:.6f} in state {emp} ({k},{wage:.2f},{pen:.2f}) ({predwage:.2f}) ({self.poprewstate[t+1,k]:.5f},{self.pop_predrew[t+1,k]:.5f})')


        err=obsV-self.aveV[t,k]
        obsV_error=np.abs(obsV-self.aveV[t,k])


    def plot_Vstats(self):
        obsV=np.zeros((self.n_time,self.n_pop))
        obsVemp=np.zeros((self.n_time,3))
        obsVemp_error=np.zeros((self.n_time,3))
        obsVemp_abserror=np.zeros((self.n_time,3))

        obsV[self.n_time-1,:]=self.poprewstate[self.n_time-1,:]
        for t in range(self.n_time-2,0,-1):
            obsV[t,:]=self.poprewstate[t,:]+self.gamma*obsV[t+1,:]
            delta=obsV[t,:]-self.aveV[t,:]
            for k in range(self.n_pop):
                if np.abs(delta[k])>0.2:
                    wage=self.infostats_pop_wage[t,k]
                    pen=self.infostats_pop_pension[t,k]
                    #print(f't {t}: {obsV[t,k]} vs {self.aveV[t+1,k]} d {delta} in state {self.popempstate[t,k]} ({k},{wage:.2f},{pen:.2f})')

            for state in range(2):
                s=np.asarray(self.popempstate[t,:]==state).nonzero()
                obsVemp[t,state]=np.mean(obsV[t,s])
                obsVemp_error[t,state]=np.mean(obsV[t,s]-self.aveV[t,s])
                obsVemp_abserror[t,state]=np.mean(np.abs(obsV[t,s]-self.aveV[t,s]))

        err=obsV[:self.n_time]-self.aveV
        obsV_error=np.abs(err)

        mean_obsV=np.mean(obsV,axis=1)
        mean_predV=np.mean(self.aveV,axis=1)
        mean_abs_errorV=np.mean(np.abs(err),axis=1)
        mean_errorV=np.mean(err,axis=1)
        fig,ax=plt.subplots()
        ax.plot(mean_abs_errorV[1:],label='abs. error')
        ax.plot(mean_errorV[1:],label='error')
        ax.plot(np.max(err,axis=1),label='max')
        ax.plot(np.min(err,axis=1),label='min')
        ax.set_xlabel('time')
        ax.set_ylabel('error (pred-obs)')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(mean_obsV[1:],label='observed')
        ax.plot(mean_predV[1:],label='predicted')
        ax.set_xlabel('time')
        ax.set_ylabel('V')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(obsVemp[1:,0],label='state 0')
        ax.plot(obsVemp[1:,1],label='state 1')
        ax.set_xlabel('time')
        ax.set_ylabel('V')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(obsVemp_error[1:,0],label='state 0')
        ax.plot(obsVemp_error[1:,1],label='state 1')
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(obsVemp_abserror[1:,0],label='state 0')
        ax.plot(obsVemp_abserror[1:,1],label='state 1')
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        plt.legend()
        plt.show()

        #self.scatter_density(self.infostats_pop_wage,obsV_error,label1='t',label2='emp obsV error')


    def render(self,load=None,figname=None,grayscale=False):
        if load is not None:
            self.load_sim(load)

        self.plot_results(figname=figname,grayscale=grayscale)


    def comp_budget(self,scale=True,debug=False):
        demog2=self.empstats.get_demog()

        scalex=demog2/self.n_pop

        q={}
        q['tyotulosumma']=np.sum(self.infostats_palkkatulo*scalex)
        q['tyotulosumma eielakkeella']=np.sum(self.infostats_palkkatulo_eielakkeella*scalex) #np.sum(self.comp_ps_norw()*scalex)*self.timestep

        q['etuusmeno']=np.sum(self.infostats_etuustulo*scalex)
        #q['etuusmeno_v2']=0
        q['verot+maksut']=np.sum(self.infostats_taxes*scalex)
        #q['verot+maksut_v2']=0
        q['verot+maksut+alv']=0
        q['palkkaverot+maksut']=np.sum(self.infostats_wagetaxes*scalex)
        q['muut tulot']=0
        q['valtionvero']=np.sum(self.infostats_valtionvero*scalex)
        q['kunnallisvero']=np.sum(self.infostats_kunnallisvero*scalex)
        q['ptel']=np.sum(self.infostats_ptel*scalex)
        q['tyottomyysvakuutusmaksu']=np.sum(self.infostats_tyotvakmaksu*scalex)
        q['ansiopvraha']=np.sum(self.infostats_ansiopvraha*scalex)
        q['peruspvraha']=0*np.sum(self.infostats_ansiopvraha*scalex)
        q['tyottomyyspvraha']=np.sum(self.infostats_ansiopvraha*scalex)
        q['asumistuki']=np.sum(self.infostats_asumistuki*scalex)
        q['tyoelakemeno']=np.sum(self.infostats_tyoelake*scalex)
        q['kansanelakemeno']=np.sum(self.infostats_kansanelake*scalex)
        q['kokoelakemeno']=np.sum(self.infostats_kokoelake*scalex)
        q['takuuelakemeno']=q['kokoelakemeno']-q['tyoelakemeno']-q['kansanelakemeno']
        #q['takuuelakemeno_v2']=np.sum(self.infostats_takuuelake*scalex)
        q['elatustuki']=np.sum(self.infostats_elatustuki*scalex)
        q['lapsilisa']=np.sum(self.infostats_lapsilisa*scalex)
        q['tyoelakemaksu']=np.sum(self.infostats_tyelpremium*scalex)
        #q['tyoelake_maksettu']=np.sum(self.infostats_paid_tyel_pension*scalex)
        q['opintotuki']=np.sum(self.infostats_opintotuki*scalex)
        q['isyyspaivaraha']=np.sum(self.infostats_isyyspaivaraha*scalex)
        q['aitiyspaivaraha']=np.sum(self.infostats_aitiyspaivaraha*scalex)
        q['kotihoidontuki']=np.sum(self.infostats_kotihoidontuki*scalex)
        q['sairauspaivaraha']=np.sum(self.infostats_sairauspaivaraha*scalex)
        q['toimeentulotuki']=np.sum(self.infostats_toimeentulotuki*scalex)
        pt=np.sum(self.infostats_perustulo*scalex)
        if pt>0.0:
            q['perustulo']=pt
        q['sairausvakuutusmaksu']=np.sum(self.infostats_sairausvakuutus*scalex)
        q['pvhoitomaksu']=np.sum(self.infostats_pvhoitomaksu*scalex)
        q['ylevero']=np.sum(self.infostats_ylevero*scalex)

        q['tyottomyyspvraha']=q['ansiopvraha']+q['peruspvraha']
        q['ta_maksut']=q['tyoelakemaksu']-q['ptel']+(0.2057-0.1695)*q['tyotulosumma'] # karkea
        q['verotettava etuusmeno']=q['kokoelakemeno']+q['tyottomyyspvraha']+q['aitiyspaivaraha']+q['isyyspaivaraha']
        q['alv']=np.sum(self.infostats_alv*scalex)
        q['verot+maksut+alv']=q['verot+maksut']+q['alv']
        q['muut tulot']=q['etuusmeno']-q['verot+maksut+alv']

        q['nettotulot']=np.sum(self.infostats_tulot_netto*scalex)
        if debug:
            q['tulot_netto_v2']=q['tyotulosumma']+q['etuusmeno']-q['verot+maksut+alv']-q['pvhoitomaksu']
            q['etuusmeno_v2']=q['tyottomyyspvraha']+q['kokoelake']+q['opintotuki']+q['isyyspaivaraha']+\
                q['aitiyspaivaraha']+q['sairauspaivaraha']+q['toimeentulotuki']+q['perustulo']+\
                q['asumistuki']+q['kotihoidontuki']+q['elatustuki']+q['lapsilisa']
            q['verot+maksut_v2']=q['valtionvero']+q['kunnallisvero']+q['ptel']+q['tyotvakmaksu']+\
                q['ylevero']+q['sairausvakuutusmaksu']
        
        return q

    def comp_ps_norw(self):
        #print(self.salaries_emp[:,1]+self.salaries_emp[:,10])
        return self.salaries_emp[:,1]+self.salaries_emp[:,10]




    def comp_cumurewstate(self):
        return np.cumsum(np.mean(self.poprewstate[:,:],axis=1))



    def compare_with(self,cc2,label1='perus',label2='vaihtoehto',grayscale=True,figname=None,dash=False):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        diff_emp=self.empstate/self.n_pop-cc2.empstate/cc2.n_pop
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        real1=self.comp_presentvalue()
        real2=cc2.comp_presentvalue()
        mean_real1=np.mean(real1,axis=1)
        mean_real2=np.mean(real2,axis=1)
        initial1=np.mean(real1[1,:])
        initial2=np.mean(real2[1,:])

        rew1=self.comp_total_reward(output=False,discounted=True)
        rew2=cc2.comp_total_reward(output=False,discounted=True)
        net1,eqnet1=self.comp_total_netincome(output=False)
        net2,eqnet2=cc2.comp_total_netincome(output=False)

        print(f'{label1} reward {rew1} netincome {net1:.2f} eq {eqnet1:.3f} initial {initial1}')
        print(f'{label2} reward {rew2} netincome {net2:.2f} eq {eqnet2:.3f} initial {initial2}')

        if self.minimal>0:
            s=20
            e=70
        else:
            s=21
            e=60 #63.5

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.comp_employed_ratio(self.empstate)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.comp_employed_ratio(cc2.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1,osatyolliset1,kokotyolliset1,osata1,kokota1=self.comp_tyollisyys_stats(self.empstate/self.n_pop,scale_time=True,start=s,end=e,full=True)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2,osatyolliset2,kokotyolliset2,osata2,kokota2=self.comp_tyollisyys_stats(cc2.empstate/cc2.n_pop,scale_time=True,start=s,end=e,full=True)
        ansiosid_osuus1,tm_osuus1=self.comp_unemployed_detailed(self.empstate)
        ansiosid_osuus2,tm_osuus2=self.comp_unemployed_detailed(cc2.empstate)
        #khh_osuus1=self.comp_kht(self.empstate)
        #khh_osuus2=self.comp_kht(cc2.empstate)

        self.comp_employment_stats()
        cc2.comp_employment_stats()

        self.compare_against(cc=cc2,cctext=label2)

#         q1=self.comp_budget(scale=True)
#         q2=cc2.comp_budget(scale=True)
#         
#         df1 = pd.DataFrame.from_dict(q1,orient='index',columns=[label1])
#         df2 = pd.DataFrame.from_dict(q2,orient='index',columns=['one'])
#         df=df1.copy()
#         df[label2]=df2['one']
#         df['ero']=df1[label1]-df2['one']

        fig,ax=plt.subplots()
        ax.plot(x[1:self.n_time],mean_real1[1:self.n_time]-mean_real2[1:self.n_time],label=label1+'-'+label2)
        ax.legend()
        ax.set_xlabel('age')
        ax.set_ylabel('real diff')
        plt.show()

        fig,ax=plt.subplots()
        c1=self.comp_cumurewstate()
        c2=cc2.comp_cumurewstate()
        ax.plot(x,c1,label=label1)
        ax.plot(x,c2,label=label2)
        ax.legend()
        ax.set_xlabel('rev age')
        ax.set_ylabel('rew')
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(x,c1-c2,label=label1+'-'+label2)
        ax.legend()
        ax.set_xlabel('rev age')
        ax.set_ylabel('rew diff')
        plt.show()

#         if self.version in set([1,2,3,4]):
#             print('Rahavirrat skaalattuna väestötasolle')
#             print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

        if dash:
            ls='--'
        else:
            ls=None

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyollisyysaste %'])
        ax.plot(x,100*tyolliset1,label=label1)
        ax.plot(x,100*tyolliset2,ls=ls,label=label2)
        ax.set_ylim([0,100])
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'emp.'+self.figformat, format=self.figformat)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ero osuuksissa'])
        diff_emp=diff_emp*100
        ax.plot(x,100*(tyot_osuus1-tyot_osuus2),label='unemployment')
        ax.plot(x,100*(kokotyo_osuus1-kokotyo_osuus2),label='fulltime work')
        if self.version in set([1,2,3,4,104]):
            ax.plot(x,100*(osatyo_osuus1-osatyo_osuus2),label='osa-aikatyö')
            ax.plot(x,100*(tyolliset1-tyolliset2),label='työ yhteensä')
            ax.plot(x,100*(htv_osuus1-htv_osuus2),label='htv yhteensä')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomyysaste'])
        diff_emp=diff_emp*100
        ax.plot(x,100*tyot_osuus1,label=label1)
        ax.plot(x,100*tyot_osuus2,ls=ls,label=label2)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'unemp.'+self.figformat, format=self.figformat)
        plt.show()

        if self.minimal<0:
            fig,ax=plt.subplots()
            ax.set_xlabel(self.labels['age'])
            ax.set_ylabel('Kotihoidontuki [%]')
            #ax.plot(x,100*khh_osuus1,label=label1)
            #ax.plot(x,100*khh_osuus2,ls=ls,label=label2)
            ax.set_ylim([0,100])
            ax.legend()
            if figname is not None:
                plt.savefig(figname+'kht.'+self.figformat, format=self.figformat)
            plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Osatyö [%]')
        ax.plot(x,100*osatyo_osuus1,label=label1)
        ax.plot(x,100*osatyo_osuus2,ls=ls,label=label2)
        ax.set_ylim([0,100])
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'osatyo_osuus.'+self.figformat, format=self.figformat)
        plt.show()


        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ero osuuksissa'])
        diff_emp=diff_emp*100
        ax.plot(x,100*ansiosid_osuus2,ls=ls,label='ansiosid. työttömyys, '+label2)
        ax.plot(x,100*ansiosid_osuus1,label='ansiosid. työttömyys, '+label1)
        ax.plot(x,100*tm_osuus2,ls=ls,label='tm-tuki, '+label2)
        ax.plot(x,100*tm_osuus1,label='tm-tuki, '+label1)
        ax.legend()
        plt.show()

        if self.language=='English':
            print('Influence on employment of {:.0f}-{:.0f} years old approx. {:.0f} man-years and {:.0f} employed'.format(s,e,htv1-htv2,tyoll1-tyoll2))
            print('- full-time {:.0f}-{:.0f} y olds {:.0f} employed ({:.0f} vs {:.0f})'.format(s,e,(kokotyolliset1-kokotyolliset2),kokotyolliset1,kokotyolliset2))
            print('- part-time {:.0f}-{:.0f} y olds {:.0f} employed ({:.0f} vs {:.0f})'.format(s,e,(osatyolliset1-osatyolliset2),osatyolliset1,osatyolliset2))
            print('Employed {:.0f} vs {:.0f} man-years'.format(htv1,htv2))
            print('Influence on employment rate for {:.0f}-{:.0f} y olds {:.2f} % ({:.2f} vs {:.2f})'.format(s,e,(tyollaste1-tyollaste2)*100,tyollaste1*100,tyollaste2*100))
            print('- full-time {:.0f}-{:.0f} y olds {:.2f} % ({:.2f} vs {:.2f})'.format(s,e,(kokota1-kokota2)*100,kokota1*100,kokota2*100))
            print('- part-time {:.0f}-{:.0f} y olds {:.2f} % ({:.2f} vs {:.2f})'.format(s,e,(osata1-osata2)*100,osata1*100,osata2*100))
        else:
            print('Työllisyysvaikutus {:.0f}-{:.0f}-vuotiaisiin noin {:.0f} htv ja {:.0f} työllistä'.format(s,e,htv1-htv2,tyoll1-tyoll2))
            print('- kokoaikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(kokotyolliset1-kokotyolliset2),kokotyolliset1,kokotyolliset2))
            print('- osa-aikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(osatyolliset1-osatyolliset2),osatyolliset1,osatyolliset2))
            print('Työllisiä {:.0f} vs {:.0f} htv'.format(htv1,htv2))
            print('Työllisyysastevaikutus {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(tyollaste1-tyollaste2)*100,tyollaste1*100,tyollaste2*100))
            print('- kokoaikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(kokota1-kokota2)*100,kokota1*100,kokota2*100))
            print('- osa-aikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(osata1-osata2)*100,osata1*100,osata2*100))

        if self.minimal>0:
            unemp_htv1=np.nansum(self.demogstates[:,0])
            unemp_htv2=np.nansum(cc2.demogstates[:,0])
            e_unemp_htv1=np.nansum(self.demogstates[:,0])
            e_unemp_htv2=np.nansum(cc2.demogstates[:,0])
            tm_unemp_htv1=np.nansum(self.demogstates[:,0])*0
            tm_unemp_htv2=np.nansum(cc2.demogstates[:,0])*0
            f_unemp_htv1=np.nansum(self.demogstates[:,0])*0
            f_unemp_htv2=np.nansum(cc2.demogstates[:,0])*0
        else:
            unemp_htv1=np.nansum(self.demogstates[:,0]+self.demogstates[:,4]+self.demogstates[:,13])
            unemp_htv2=np.nansum(cc2.demogstates[:,0]+cc2.demogstates[:,4]+cc2.demogstates[:,13])
            e_unemp_htv1=np.nansum(self.demogstates[:,0])
            e_unemp_htv2=np.nansum(cc2.demogstates[:,0])
            tm_unemp_htv1=np.nansum(self.demogstates[:,13])
            tm_unemp_htv2=np.nansum(cc2.demogstates[:,13])
            f_unemp_htv1=np.nansum(self.demogstates[:,4])
            f_unemp_htv2=np.nansum(cc2.demogstates[:,4])

        # epävarmuus
        delta=1.96*1.0/np.sqrt(self.n_pop)

        if self.language=='English':
            print('Työttömyysvaikutus {:.0f}-{:.0f}-vuotiaisiin noin {:.0f} htv'.format(s,e,unemp_htv1-unemp_htv2))
            print('- ansiosidonnaiseen {:.0f}-{:.0f}-vuotiailla noin {:.0f} htv ({:.0f} vs {:.0f})'.format(s,e,(e_unemp_htv1-e_unemp_htv2),e_unemp_htv1,e_unemp_htv2))
            print('- tm-tukeen {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(tm_unemp_htv1-tm_unemp_htv2),tm_unemp_htv1,tm_unemp_htv2))
            print('- putkeen {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(f_unemp_htv1-f_unemp_htv2),f_unemp_htv1,f_unemp_htv2))
            print('Uncertainty in employment rates {:.4f}, std {:.4f}'.format(delta,haj1))
        else:
            print('Työttömyysvaikutus {:.0f}-{:.0f}-vuotiaisiin noin {:.0f} htv'.format(s,e,unemp_htv1-unemp_htv2))
            print('- ansiosidonnaiseen {:.0f}-{:.0f}-vuotiailla noin {:.0f} htv ({:.0f} vs {:.0f})'.format(s,e,(e_unemp_htv1-e_unemp_htv2),e_unemp_htv1,e_unemp_htv2))
            print('- tm-tukeen {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(tm_unemp_htv1-tm_unemp_htv2),tm_unemp_htv1,tm_unemp_htv2))
            print('- putkeen {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(f_unemp_htv1-f_unemp_htv2),f_unemp_htv1,f_unemp_htv2))
            print('epävarmuus työllisyysasteissa {:.4f}, hajonta {:.4f}'.format(delta,haj1))

        if True:
            unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            tyoll_distrib2,tyoll_distrib_bu2=cc2.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1=label1,label2=label2)
            if self.language=='English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2,logy=False)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2,logy=False,diff=True)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2)

            unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            tyoll_distrib2,tyoll_distrib_bu2=cc2.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1=label1,label2=label2)
            if self.language=='English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2)

        print(label2)
        keskikesto=self.comp_unemp_durations(return_q=False)
        self.plot_unemp_durdistribs(keskikesto)

        print(label1)
        keskikesto=cc2.comp_unemp_durations(return_q=False)
        self.plot_unemp_durdistribs(keskikesto)

        tyoll_virta,tyot_virta=self.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_compare_virrat(tyoll_virta,tyoll_virta2,virta_label='Työllisyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='Työttömyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='Työttömyys',label1=label1,label2=label2)

        tyoll_virta,tyot_virta=self.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='ei-tm-Työttömyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='ei-tm-Työttömyys',label1=label1,label2=label2)

        tyoll_virta,tyot_virta=self.comp_virrat(ansiosid=False,tmtuki=True,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.comp_virrat(ansiosid=False,tmtuki=True,putki=True,outsider=False)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='tm-Työttömyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='tm-Työttömyys',label1=label1,label2=label2)


    def comp_gini_old(self):
        '''
        Laske Gini-kerroin summatuloille, ei populaatiolle
        '''
        income=np.sort(self.infostats_tulot_netto,axis=None)
        
        n=len(income)
        L=ma.arange(n,0,-1)
        A=ma.sum(L*income)/ma.sum(income)
        G=(n+1-2*A)/2

        return G


    def comp_gini(self):
        '''
        Laske Gini-kerroin populaatiolle
        '''
        if False:
            income=np.sort(self.infostats_tulot_netto,axis=None)
        else:
            alivemask=(self.popempstate[:-1,:]==self.env.get_mortstate())
            netto=ma.array(self.infostats_poptulot_netto[:-1,:],mask=alivemask)
            income=ma.sort(netto,axis=None).compressed()

        n=len(income)
        #L=np.arange(n+1,1,-1)
        #A=np.sum(L*income)/np.sum(income)
        #G=(n+1-2*A)/n
        
        # v2
        L = np.arange(1,n+1,1)
        A = np.sum(L*income)/np.sum(income)
        G = 2*A/n-(n+1)/n
        G *= 100

        return G

    def comp_pienituloisuus(self,level=1_000):
        alivemask=(self.popempstate==self.env.get_mortstate())
        netto=ma.array(self.infostats_poptulot_netto/self.timestep,mask=alivemask) # vuositasolle
        alive=np.sum(self.alive)
        #level=level/self.timestep
        median=ma.median(netto)
        mean=ma.mean(netto)
        med60=0.60*median
        med50=0.50*median
        #print(f'mean {mean} median {median} med50 {med50} med60 {med60}')
        ratio_med50=ma.sum(netto<med50)/alive
        ratio_med60=ma.sum(netto<med60)/alive
        pt_levl=ma.sum(netto<level)/alive
        #print(f'ratio_med50 {ratio_med50} ratio_med60 {ratio_med60} pt_levl {pt_levl}')
        
        return ratio_med50,ratio_med60,pt_levl
                    