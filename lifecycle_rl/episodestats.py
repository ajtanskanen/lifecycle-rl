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
from .utils import empirical_cdf,print_html,modify_offsettext,NpEncoder
from datetime import datetime


#locale.setlocale(locale.LC_ALL, 'fi_FI')

class EpisodeStats():
    def __init__(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,
                year=2018,version=9,params=None,gamma=0.92,silent=False,rapid=False,save_pop=False,
                include_perustulo=False,parttime_actions=None,lang='English'):
        self.version=version
        self.gamma=gamma
        self.params=params
        self.params['n_time'] = n_time
        self.params['n_emps'] = n_emps
        self.lab=Labels()
        self.silent=silent
        self.rapid=rapid
        self.n_states = 16
        self.n_emps = n_emps
        self.save_pop = save_pop
        self.include_perustulo = include_perustulo
        self.parttime_actions = parttime_actions
        self.labels = self.lab.get_output_labels(lang)

        self.complexmodels = set([4,5,6,7,8,9,104])
        self.recentmodels = set([6,7,8,9])

        print('save_pop',self.save_pop)
        #print('episodestats, lang',lang)

        if self.parttime_actions is not None:
            self.fullhtv = True
        else:
            self.fullhtv = False

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
        elif self.version==5:
            self.add=self.add_v5
        elif self.version==6:
            self.add=self.add_v6
        elif self.version==7:
            self.n_states = 17
            self.add=self.add_v7
        elif self.version==8:
            self.n_states = 17
            self.add=self.add_v8
        elif self.version==9:
            self.n_states = 17
            self.add=self.add_v9
        elif self.version==101:
            self.add=self.add_v101
        elif self.version==104:
            self.add=self.add_v104
        else:
            print('Unknown version ',self.version)

        self.reset(timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,params=params) #,lang=lang)

    def sum_q_to_a(self,myarray,axis=0):
        x = round(1/self.timestep)
        M = myarray.shape[axis]
        L = round(M/x)+1
        r = np.zeros(L)
        for k in range(L):
            r[k] = sum(myarray[x*k:min(M,x*k+x)])
        return r

    def reset(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,version=None,params=None,dynprog=False): # lang=None,
        self.min_age=min_age
        self.max_age=max_age
        self.min_retirementage=min_retirementage
        self.minimal=minimal

        if params is not None:
            self.params=params

        #if lang is None:
        #    self.language='English'
        #else:
        #    self.language=lang

        if version is not None:
            self.version=version

        #self.setup_labels()

        self.figformat='pdf'

        self.n_employment = n_emps
        self.n_time = n_time
        self.timestep = timestep # 0.25 = 3kk askel
        self.inv_timestep = round(1/self.timestep) 
        self.n_pop = n_pop
        self.year = year
        self.env = env
        self.reaalinen_palkkojenkasvu = 0.016
        self.palkkakerroin = (0.8*1+0.2*1.0/(1+self.reaalinen_palkkojenkasvu))**self.timestep
        self.elakeindeksi = (0.2*1+0.8*1.0/(1+self.reaalinen_palkkojenkasvu))**self.timestep
        self.dynprog = dynprog
        self.gini_coef = None

        if self.version in set([0,101]):
            self.n_groups = 1
        else:
            self.n_groups = 6
            
        self.empstats = Empstats(year=self.year,max_age=self.max_age,n_groups=self.n_groups,timestep=self.timestep,n_time=self.n_time,
                                min_age=self.min_age)
        self.init_variables()
        self.set_datetime()

    def init_variables(self):
        n_emps=self.n_states
        self.empstate = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.emp_htv = np.zeros((self.n_time,self.n_groups,n_emps),dtype = float)
        self.gempstate = np.zeros((self.n_time,n_emps,self.n_groups),dtype = np.int64)
        self.deceiced = np.zeros((self.n_time,1),dtype = np.int64)
        self.alive = np.zeros((self.n_time,1),dtype = np.int64)
        self.galive = np.zeros((self.n_time,self.n_groups),dtype = np.int64)
        self.rewstate = np.zeros((self.n_time,n_emps))
        self.salaries_emp = np.zeros((self.n_time,n_emps))

        self.siirtyneet = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.siirtyneet_det = np.zeros((self.n_time,n_emps,n_emps),dtype = np.int64)
        self.pysyneet = np.zeros((self.n_time,n_emps),dtype = np.int64)
        if self.minimal:
            self.aveV = np.zeros((self.n_time,self.n_pop),dtype=float)
        self.time_in_state = np.zeros((self.n_time,n_emps))
        self.stat_tyoura = np.zeros((self.n_time,n_emps))
        self.stat_toe = np.zeros((self.n_time,n_emps))
        self.out_of_work = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.stat_wage_reduction = np.zeros((self.n_time,n_emps))
        self.stat_wage_reduction_g = np.zeros((self.n_time,n_emps,self.n_groups))
        self.infostats_taxes = np.zeros((self.n_time,1))
        self.infostats_wagetaxes = np.zeros((self.n_time,1))
        self.infostats_taxes_distrib = np.zeros((self.n_time,n_emps))
        self.infostats_etuustulo = np.zeros((self.n_time,1))
        self.infostats_etuustulo_group = np.zeros((self.n_time,self.n_groups))
        self.infostats_perustulo = np.zeros((self.n_time,1))
        self.infostats_palkkatulo = np.zeros((self.n_time,1))
        self.infostats_palkkatulo_emp = np.zeros((self.n_time,n_emps))
        self.infostats_palkkatulo_eielakkeella = np.zeros((self.n_time,1))
        self.infostats_palkkatulo_group = np.zeros((self.n_time,self.n_groups))
        self.infostats_palkkatulo_eielakkeella_group = np.zeros((self.n_time,1))
        self.infostats_ansiopvraha = np.zeros((self.n_time,1))
        self.infostats_tyotpvraha = np.zeros((self.n_time,1))
        self.infostats_peruspvraha = np.zeros((self.n_time,1))
        self.infostats_ansiopvraha_group = np.zeros((self.n_time,self.n_groups))
        self.infostats_asumistuki = np.zeros((self.n_time,1))
        self.infostats_asumistuki_group = np.zeros((self.n_time,self.n_groups))
        self.infostats_valtionvero = np.zeros((self.n_time,1))
        self.infostats_valtionvero_group = np.zeros((self.n_time,self.n_groups))
        self.infostats_kunnallisvero = np.zeros((self.n_time,1))
        self.infostats_kunnallisvero_group = np.zeros((self.n_time,self.n_groups))
        self.infostats_valtionvero_distrib = np.zeros((self.n_time,n_emps))
        self.infostats_kunnallisvero_distrib = np.zeros((self.n_time,n_emps))
        self.infostats_ptel = np.zeros((self.n_time,1))
        self.infostats_tyotvakmaksu = np.zeros((self.n_time,1))
        self.infostats_tyoelake = np.zeros((self.n_time,1))
        self.infostats_kansanelake = np.zeros((self.n_time,1))
        self.infostats_kokoelake = np.zeros((self.n_time,1))
        #self.infostats_takuuelake = np.zeros((self.n_time,1))
        self.infostats_lapsilisa = np.zeros((self.n_time,1))
        self.infostats_elatustuki = np.zeros((self.n_time,1))
        self.infostats_opintotuki = np.zeros((self.n_time,1))
        self.infostats_isyyspaivaraha = np.zeros((self.n_time,1))
        self.infostats_aitiyspaivaraha = np.zeros((self.n_time,1))
        self.infostats_kotihoidontuki = np.zeros((self.n_time,1))
        self.infostats_sairauspaivaraha = np.zeros((self.n_time,1))
        self.infostats_toimeentulotuki = np.zeros((self.n_time,1))
        self.infostats_tulot_netto = np.zeros((self.n_time,1))
        self.infostats_tulot_netto_emp = np.zeros((self.n_time,n_emps))
        self.infostats_pinkslip = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.infostats_tyelpremium = np.zeros((self.n_time,1),dtype=float)
        self.stat_paidpension = np.zeros((self.n_time,n_emps),dtype=float)
        self.stat_pension = np.zeros((self.n_time,n_emps),dtype=float)
        self.stat_unemp_after_ra = np.zeros((self.n_time,n_emps),dtype=float)
        self.poprewstate = np.zeros((self.n_time,self.n_pop),dtype=float)
        self.popempstate = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
        self.infostats_poptulot_netto = np.zeros((self.n_time,self.n_pop),dtype=float)

        if self.save_pop:
            self.tyoll_distrib_bu = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.unemp_distrib_bu = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_group = np.zeros((self.n_pop,1),dtype = np.int8)
            self.popunemprightleft = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.popunemprightused = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.stat_unemp_len = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_pinkslip = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.infostats_pop_wage_reduction = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_tyelpremium = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_tyoelake = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_kansanelake = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_takuuelake = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_paidpension = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_pension = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_paid_tyel_pension = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_toe = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_wage = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_unempwagebasis = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_unempwagebasis_acc = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_irr_tyel_reduced = np.zeros((self.n_pop,1),dtype=float)
            self.infostats_irr_tyel_full = np.zeros((self.n_pop,1),dtype=float)
            self.infostats_pop_emtr = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_tva = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_pt_act = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.infostats_pop_potential_wage = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.pop_actions = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.infostats_pop_puoliso = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.pop_predrew = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.infostats_pop_children_under3 = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.infostats_pop_children_under7 = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.infostats_pop_children_under18 = np.zeros((self.n_time,self.n_pop),dtype = np.int8)
            self.stat_pop_diswage5y = np.zeros((self.n_pop,1),dtype=float)
            self.infostats_npv0 = np.zeros((self.n_pop,1),dtype=float)
            self.infostats_lleft = np.zeros((self.n_time,self.n_pop),dtype=float)
            
        self.infostats_pt_act = np.zeros((self.n_time,n_emps,self.n_groups,3),dtype = np.int64)
        self.infostats_puoliso = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.infostats_sairausvakuutus = np.zeros((self.n_time,1))
        self.infostats_pvhoitomaksu = np.zeros((self.n_time,1),dtype=float)
        self.infostats_ylevero = np.zeros((self.n_time,1))
        self.infostats_ylevero_distrib = np.zeros((self.n_time,n_emps))
        self.infostats_children_under3 = np.zeros((self.n_time,n_emps),dtype = np.int64)
        #self.infostats_children_born = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.infostats_children_under7 = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.infostats_children_under18 = np.zeros((self.n_time,n_emps),dtype = np.int64)
        self.infostats_toe = np.zeros((self.n_time,1),dtype=float)
        self.infostats_ove = np.zeros((self.n_time,n_emps))
        self.infostats_ove_g = np.zeros((self.n_time,n_emps,self.n_groups))
        self.infostats_kassanjasen = np.zeros((self.n_time,1),dtype = np.int64)
        self.infostats_equivalent_income = np.zeros((self.n_time,1))
        self.infostats_alv = np.zeros((self.n_time,1))
        self.initial_reward=0
        self.average_discounted_reward=None
        self.average_undiscounted_reward=None
        self.rewards_computed=False
        if self.version in set([101,102,104]):
            self.infostats_savings = np.zeros((self.n_time,self.n_pop),dtype=float)
            self.sav_actions = np.zeros((self.n_time,self.n_pop),dtype=float)

        if not self.version in self.complexmodels:
            self.actions = np.zeros((self.n_time,n_emps),dtype = np.int64)
        else:
            self.actions = np.zeros((self.n_time,n_emps,5),dtype = np.int64)

    def add_taxes(self,t,q,newemp,n,g,person=''):
        scale=self.timestep*12
        self.infostats_taxes[t] += q[person+'verot']*scale
        self.infostats_wagetaxes[t] += q[person+'verot_ilman_etuuksia']*scale
        self.infostats_taxes_distrib[t,newemp] += q[person+'verot']*scale
        self.infostats_etuustulo[t] += q[person+'etuustulo_brutto']*scale
        self.infostats_etuustulo_group[t,g] += q[person+'etuustulo_brutto']*scale
        self.infostats_perustulo[t] += q[person+'perustulo']*scale
        self.infostats_palkkatulo[t] += q[person+'palkkatulot']*scale
        self.infostats_palkkatulo_emp[t,newemp] += q[person+'palkkatulot']*scale
        self.infostats_palkkatulo_eielakkeella[t] += q[person+'palkkatulot_eielakkeella']*scale
        self.infostats_ansiopvraha[t] += q[person+'ansiopvraha']*scale
        self.infostats_peruspvraha[t] += q[person+'peruspvraha']*scale 
        self.infostats_tyotpvraha[t] += q[person+'tyotpvraha']*scale
        self.infostats_asumistuki[t] += q[person+'asumistuki']*scale
        self.infostats_valtionvero[t] += q[person+'valtionvero']*scale
        self.infostats_valtionvero_distrib[t,newemp] += q[person+'valtionvero']*scale
        self.infostats_kunnallisvero[t] += q[person+'kunnallisvero']*scale
        self.infostats_kunnallisvero_distrib[t,newemp] += q[person+'kunnallisvero']*scale
        self.infostats_ptel[t] += q[person+'ptel']*scale
        self.infostats_tyotvakmaksu[t] += q[person+'tyotvakmaksu']*scale
        self.infostats_kokoelake[t] += q[person+'kokoelake']*scale
        self.infostats_lapsilisa[t] += q[person+'lapsilisa']*scale # tämä taidetaan skaalata väärin FIXME
        self.infostats_elatustuki[t] += q[person+'elatustuki']*scale
        self.infostats_opintotuki[t] += q[person+'opintotuki']*scale
        self.infostats_isyyspaivaraha[t] += q[person+'isyyspaivaraha']*scale
        self.infostats_aitiyspaivaraha[t] += q[person+'aitiyspaivaraha']*scale
        self.infostats_kotihoidontuki[t] += q[person+'kotihoidontuki']*scale
        self.infostats_sairauspaivaraha[t] += q[person+'sairauspaivaraha']*scale
        self.infostats_toimeentulotuki[t] += q[person+'toimeentulotuki']*scale
        self.infostats_pvhoitomaksu[t] += q[person+'pvhoito']*scale
        
        if self.version in self.complexmodels:
            self.infostats_tulot_netto[t] += q[person+'netto']*scale # vuositasolla, huomioi alv:n
            self.infostats_tulot_netto_emp[t,newemp] += q[person+'netto']*scale # vuositasolla, huomioi alv:n
            self.infostats_tyoelake[t] += q[person+'tyoelake']*scale
            self.infostats_kansanelake[t] += q[person+'kansanelake']*scale
            self.infostats_poptulot_netto[t,n] = q[person+'netto']*scale
            if self.save_pop:
                self.infostats_pop_paidpension[t,n] = q[person+'kokoelake']*scale
                self.infostats_pop_tyoelake[t,n] = q[person+'tyoelake']*scale
                self.infostats_pop_kansanelake[t,n] = q[person+'kansanelake']*scale
                self.infostats_pop_takuuelake[t,n] = q[person+'takuuelake']*scale
                self.infostats_paid_tyel_pension[t,n] = q[person+'puhdas_tyoelake']*scale
                self.infostats_pop_tyelpremium[t,n] = q[person+'tyel_kokomaksu']*scale
                self.infostats_pop_wage[t,n] = q[person+'palkkatulot']*scale # at annual level
                self.infostats_pop_emtr[t,n] = q[person+'emtr']
                self.infostats_pop_tva[t,n] = q[person+'tva']
                self.infostats_pop_potential_wage[t,n] = q[person+'potential_wage']*self.timestep # ei skaalausta kuukausilla, koska on jo vuositasossa *scale
                if self.stat_pop_diswage5y[n]<1.0:
                    self.stat_pop_diswage5y[n] += q[person+'dis_wage5y']
                self.infostats_npv0[n] = q[person+'multiplier']
            #self.infostats_takuuelake[t] += q[person+'takuuelake']*scale
        else:
            self.infostats_tulot_netto[t] += q['kateen']*scale # legacy for v1-3
            self.infostats_tulot_netto_emp[t,newemp] += q[person+'netto']*scale # vuositasolla, huomioi alv:n
            self.infostats_poptulot_netto[t,n] = q['kateen']*scale
            self.infostats_tyoelake[t] += q[person+'elake_maksussa']*scale

            
        self.infostats_tyelpremium[t] += q[person+'tyel_kokomaksu']*scale
        self.infostats_sairausvakuutus[t] += q[person+'sairausvakuutusmaksu']*scale
        self.infostats_ylevero[t] += q[person+'ylevero']*scale
        self.infostats_ylevero_distrib[t,newemp] = q[person+'ylevero']*scale
        self.infostats_equivalent_income[t] += q[person+'eq']*self.timestep # already at annual level
        if 'alv' in q:
            self.infostats_alv[t] += q[person+'alv']*scale          
            
    def append_episodestat(self,cc):
        n_emps1=self.n_employment
        n_emps2=cc.n_employment
        
        self.empstate += cc.empstate
        self.emp_htv += cc.emp_htv
        self.gempstate += cc.gempstate
        self.deceiced += cc.deceiced
        self.alive += cc.alive
        self.galive += cc.galive
        self.rewstate += cc.rewstate
        self.poprewstate = np.append(self.poprewstate,cc.poprewstate,axis=1)
        self.salaries_emp += cc.salaries_emp
        self.siirtyneet += cc.siirtyneet
        self.siirtyneet_det += cc.siirtyneet_det
        self.pysyneet += cc.pysyneet
        self.siirtyneet_det += cc.siirtyneet_det
        if self.minimal:
            self.aveV = np.append(self.aveV[:,:self.n_pop],cc.aveV[:,:cc.n_pop],axis=1)
        self.time_in_state += cc.time_in_state
        self.stat_tyoura += cc.stat_tyoura
        self.stat_toe += cc.stat_toe
        self.out_of_work += cc.out_of_work
        self.stat_wage_reduction += cc.stat_wage_reduction
        self.stat_wage_reduction_g += cc.stat_wage_reduction_g
        self.infostats_taxes += cc.infostats_taxes
        self.infostats_wagetaxes += cc.infostats_wagetaxes
        self.infostats_taxes_distrib += cc.infostats_taxes_distrib
        self.infostats_etuustulo += cc.infostats_etuustulo
        self.infostats_etuustulo_group += cc.infostats_etuustulo_group
        self.infostats_perustulo += cc.infostats_perustulo
        self.infostats_palkkatulo += cc.infostats_palkkatulo
        self.infostats_palkkatulo_emp += cc.infostats_palkkatulo_emp
        self.infostats_palkkatulo_eielakkeella += cc.infostats_palkkatulo_eielakkeella
        self.infostats_palkkatulo_group += cc.infostats_palkkatulo_group
        self.infostats_palkkatulo_eielakkeella_group += cc.infostats_palkkatulo_eielakkeella_group
        self.infostats_ansiopvraha += cc.infostats_ansiopvraha
        self.infostats_tyotpvraha += cc.infostats_tyotpvraha
        self.infostats_peruspvraha += cc.infostats_peruspvraha
        self.infostats_ansiopvraha_group += cc.infostats_ansiopvraha_group
        self.infostats_asumistuki += cc.infostats_asumistuki
        self.infostats_asumistuki_group += cc.infostats_asumistuki_group
        self.infostats_valtionvero += cc.infostats_valtionvero
        self.infostats_valtionvero_group += cc.infostats_valtionvero_group
        self.infostats_kunnallisvero += cc.infostats_kunnallisvero
        self.infostats_kunnallisvero_group += cc.infostats_kunnallisvero_group
        self.infostats_valtionvero_distrib += cc.infostats_valtionvero_distrib
        self.infostats_kunnallisvero_distrib += cc.infostats_kunnallisvero_distrib
        self.infostats_valtionvero_distrib += cc.infostats_valtionvero_distrib
        self.infostats_ptel += cc.infostats_ptel
        self.infostats_tyotvakmaksu += cc.infostats_tyotvakmaksu
        self.infostats_tyoelake += cc.infostats_tyoelake
        self.infostats_kansanelake += cc.infostats_kansanelake
        #self.infostats_takuuelake = self.infostats_takuuelake+cc.infostats_takuuelake
        self.infostats_kokoelake += cc.infostats_kokoelake
        self.infostats_lapsilisa += cc.infostats_lapsilisa
        self.infostats_elatustuki += cc.infostats_elatustuki
        self.infostats_opintotuki += cc.infostats_opintotuki
        self.infostats_isyyspaivaraha += cc.infostats_isyyspaivaraha
        self.infostats_aitiyspaivaraha += cc.infostats_aitiyspaivaraha
        self.infostats_kotihoidontuki += cc.infostats_kotihoidontuki
        self.infostats_sairauspaivaraha += cc.infostats_sairauspaivaraha
        self.infostats_toimeentulotuki += cc.infostats_toimeentulotuki
        self.infostats_tulot_netto += cc.infostats_tulot_netto
        self.infostats_tulot_netto_emp += cc.infostats_tulot_netto_emp
        self.infostats_pinkslip += cc.infostats_pinkslip
        self.infostats_tyelpremium += cc.infostats_tyelpremium
        self.stat_paidpension += cc.stat_paidpension
        self.stat_pension += cc.stat_pension
        self.stat_unemp_after_ra += cc.stat_unemp_after_ra
        self.popempstate = np.append(self.popempstate,cc.popempstate,axis=1)

        if self.save_pop:
            self.infostats_group = np.append(self.infostats_group[:,:self.n_pop],cc.infostats_group,axis=0)
            self.tyoll_distrib_bu = np.append(self.tyoll_distrib_bu[:,:self.n_pop],cc.tyoll_distrib_bu[:,:cc.n_pop],axis=1)
            self.unemp_distrib_bu = np.append(self.unemp_distrib_bu[:,:self.n_pop],cc.unemp_distrib_bu[:,:cc.n_pop],axis=1)
            self.popunemprightleft = np.append(self.popunemprightleft[:,:self.n_pop],cc.popunemprightleft[:,:cc.n_pop],axis=1)
            self.popunemprightused = np.append(self.popunemprightused[:,:self.n_pop],cc.popunemprightused[:,:cc.n_pop],axis=1)
            self.stat_unemp_len = np.append(self.stat_unemp_len[:,:self.n_pop],cc.stat_unemp_len[:,:cc.n_pop],axis=1)
            self.infostats_pop_pinkslip = np.append(self.infostats_pop_pinkslip[:,:self.n_pop],cc.infostats_pop_pinkslip[:,:cc.n_pop],axis=1)
            self.infostats_pop_wage_reduction = np.append(self.infostats_pop_wage_reduction[:,:self.n_pop],cc.infostats_pop_wage_reduction[:,:cc.n_pop],axis=1)
            self.infostats_pop_tyelpremium = np.append(self.infostats_pop_tyelpremium[:,:self.n_pop],cc.infostats_pop_tyelpremium[:,:cc.n_pop],axis=1)
            self.infostats_pop_paidpension = np.append(self.infostats_pop_paidpension[:,:self.n_pop],cc.infostats_pop_paidpension[:,:cc.n_pop],axis=1)
            self.infostats_pop_pension = np.append(self.infostats_pop_pension[:,:self.n_pop],cc.infostats_pop_pension[:,:cc.n_pop],axis=1)
            self.infostats_paid_tyel_pension = np.append(self.infostats_paid_tyel_pension[:,:self.n_pop],cc.infostats_paid_tyel_pension[:,:cc.n_pop],axis=1)
            self.infostats_pop_tyoelake = np.append(self.infostats_pop_tyoelake[:,:self.n_pop],cc.infostats_pop_tyoelake[:,:cc.n_pop],axis=1)
            self.infostats_pop_kansanelake = np.append(self.infostats_pop_kansanelake[:,:self.n_pop],cc.infostats_pop_kansanelake[:,:cc.n_pop],axis=1)
            self.infostats_pop_takuuelake = np.append(self.infostats_pop_takuuelake[:,:self.n_pop],cc.infostats_pop_takuuelake[:,:cc.n_pop],axis=1)
            self.infostats_unempwagebasis = np.append(self.infostats_unempwagebasis[:,:self.n_pop],cc.infostats_unempwagebasis[:,:cc.n_pop],axis=1)
            self.infostats_unempwagebasis_acc = np.append(self.infostats_unempwagebasis_acc[:,:self.n_pop],cc.infostats_unempwagebasis_acc[:,:cc.n_pop],axis=1)
            self.infostats_pop_toe = np.append(self.infostats_pop_toe[:,:self.n_pop],cc.infostats_pop_toe[:,:cc.n_pop],axis=1)
            self.infostats_pop_wage = np.append(self.infostats_pop_wage[:,:self.n_pop],cc.infostats_pop_wage[:,:cc.n_pop],axis=1)
            self.infostats_irr_tyel_reduced = np.append(self.infostats_irr_tyel_reduced[:,:self.n_pop],cc.infostats_irr_tyel_reduced[:,:cc.n_pop],axis=0)
            self.infostats_irr_tyel_full = np.append(self.infostats_irr_tyel_full[:,:self.n_pop],cc.infostats_irr_tyel_full[:,:cc.n_pop],axis=0)
            self.infostats_npv0 = np.append(self.infostats_npv0[:,:self.n_pop],cc.infostats_npv0[:,:cc.n_pop],axis=0)
            self.infostats_pop_emtr = np.append(self.infostats_pop_emtr[:,:self.n_pop],cc.infostats_pop_emtr[:,:cc.n_pop],axis=1)
            self.infostats_pop_tva = np.append(self.infostats_pop_tva[:,:self.n_pop],cc.infostats_pop_tva[:,:cc.n_pop],axis=1)
            self.infostats_pop_pt_act = np.append(self.infostats_pop_pt_act[:,:self.n_pop],cc.infostats_pop_pt_act[:,:cc.n_pop],axis=1)
            self.infostats_pop_potential_wage = np.append(self.infostats_pop_potential_wage[:,:self.n_pop],cc.infostats_pop_potential_wage[:,:cc.n_pop],axis=1)
            self.infostats_pop_children_under3 = np.append(self.infostats_pop_children_under3[:,:self.n_pop],cc.infostats_pop_children_under3[:,:cc.n_pop],axis=1)
            self.infostats_pop_children_under7 = np.append(self.infostats_pop_children_under7[:,:self.n_pop],cc.infostats_pop_children_under7[:,:cc.n_pop],axis=1)
            self.infostats_pop_children_under18 = np.append(self.infostats_pop_children_under18[:,:self.n_pop],cc.infostats_pop_children_under18[:,:cc.n_pop],axis=1)
            self.pop_actions = np.append(self.pop_actions,cc.pop_actions[:,:cc.n_pop],axis=1)
            self.pop_predrew = np.append(self.pop_predrew[:,:self.n_pop],cc.pop_predrew[:,:cc.n_pop],axis=1)
            self.stat_pop_diswage5y = np.append(self.stat_pop_diswage5y[:self.n_pop],cc.stat_pop_diswage5y[:cc.n_pop],axis=0)
            self.infostats_pop_puoliso = np.append(self.infostats_pop_puoliso[:,:self.n_pop],cc.infostats_pop_puoliso[:,:cc.n_pop],axis=1)
            if self.version in self.recentmodels:
                self.infostats_lleft = np.append(self.infostats_lleft[:,:self.n_pop],cc.infostats_lleft[:,:cc.n_pop],axis=1)

        self.infostats_poptulot_netto = np.append(self.infostats_poptulot_netto[:,:self.n_pop],cc.infostats_poptulot_netto[:,:cc.n_pop],axis=1)
        self.infostats_sairausvakuutus += cc.infostats_sairausvakuutus
        self.infostats_pvhoitomaksu += cc.infostats_pvhoitomaksu
        self.infostats_ylevero += cc.infostats_ylevero
        self.infostats_ylevero_distrib += cc.infostats_ylevero_distrib
        self.infostats_children_under3 += cc.infostats_children_under3
        #self.infostats_children_born += cc.infostats_children_born
        self.infostats_children_under7 += cc.infostats_children_under7
        self.infostats_children_under18 += cc.infostats_children_under18
        self.infostats_toe += cc.infostats_toe
        self.infostats_ove += cc.infostats_ove
        self.infostats_ove_g += cc.infostats_ove_g
        self.infostats_kassanjasen += cc.infostats_kassanjasen
        self.infostats_equivalent_income += cc.infostats_equivalent_income
        self.infostats_alv += cc.infostats_alv
        if self.version in set([101,102,104]):
            self.infostats_savings = np.append(self.infostats_savings[:,:self.n_pop],cc.infostats_savings[:,:cc.n_pop],axis=1)
            self.sav_actions = np.append(self.sav_actions[:,:self.n_pop],cc.sav_actions[:,:cc.n_pop],axis=1)

        if self.version in self.complexmodels:
            self.actions += cc.actions
            self.infostats_puoliso += cc.infostats_puoliso
        else:
            self.actions += cc.actions

        self.n_pop += cc.n_pop

    def add_v0(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,a,_,_=self.env.state_decode(state) # current employment state
        newemp,newpen,newsal,a2,tis,next_wage=self.env.state_decode(newstate)
        g=0
        bu=0
        ove=0
        jasen=0
        puoliso=0

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            self.empstate[t,newemp] += 1
            self.alive[t] += 1
            self.rewstate[t,newemp] += r
            self.infostats_tulot_netto[t] += q['netto'] # already at annual level
            self.infostats_poptulot_netto[t,n] = q['netto']

            self.poprewstate[t,n] = r
            self.popempstate[t,n] = newemp
            self.salaries_emp[t,newemp] += newsal
            self.time_in_state[t,newemp] += tis
            self.infostats_equivalent_income[t] += q['eq']
            self.infostats_pop_wage[t,n] = newsal
            if self.save_pop:
                self.infostats_pop_pension[t,n] = newpen
                self.pop_actions[t,n] = act
            if self.dynprog and pred_r is not None:
                self.pop_predrew[t,n] = pred_r

            self.actions[t,newemp,act] += 1

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
        elif newemp<0:
            self.deceiced[t] += 1
            
    def add_v1(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):
        emp,_,_,_,a,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,oof,bu,wr,p=self.env.state_decode(newstate)
        ove=0
        jasen=0
        puoliso=0

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2
            self.empstate[t,newemp] += 1
            self.alive[t] += 1
            self.rewstate[t,newemp] += r
            self.poprewstate[t,n] = r
            
            self.actions[t,newemp,act] += 1
            self.popempstate[t,n] = newemp
            #self.salaries[t,n] = newsal
            self.salaries_emp[t,newemp] += newsal
            self.time_in_state[t,newemp] += tis
            self.infostats_pinkslip[t,newemp] += pink
            self.gempstate[t,newemp,g] += 1
            self.stat_wage_reduction[t,newemp] += wr
            self.stat_wage_reduction_g[t,newemp,g] += wr
            self.galive[t,g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_toe[t,newemp] += toe
            self.stat_pension[t,newemp] += newpen
            self.stat_paidpension[t,newemp] += paidpens
            self.infostats_toe[t] += toe
            self.infostats_ove[t,newemp] += ove
            self.infostats_kassanjasen[t] += jasen
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.pop_actions[t,n] = act
                self.stat_unemp_len[t,n] = tis
                self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                self.popunemprightused[t,n] = bu
                self.infostats_pop_wage[t,n] = newsal
                self.infostats_unempwagebasis[t,n] = uw
                self.infostats_unempwagebasis_acc[t,n] = uwr
                self.infostats_pop_toe[t,n] = toe
                self.infostats_pop_pinkslip[t,n] = pink
                self.infostats_pop_pension[t,n] = newpen
                self.infostats_pop_children_under3[t,n] = c3
                self.infostats_pop_children_under7[t,n] = c7
                self.infostats_pop_children_under18[t,n] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18

            if q is not None:
                self.add_taxes(t,q,newemp,n,g)
                    
            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
        elif newemp<0:
            self.deceiced[t] += 1
            
    def add_v2(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):
        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58=self.env.state_decode(newstate)
        ove=0
        jasen=0
        puoliso=0

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2
            self.empstate[t,newemp] += 1
            self.alive[t] += 1
            self.rewstate[t,newemp] += r
            self.poprewstate[t,n] = r
            
            self.actions[t,newemp,act] += 1
            self.popempstate[t,n] = newemp
            self.salaries_emp[t,newemp] += newsal
            self.time_in_state[t,newemp] += tis
            self.infostats_pinkslip[t,newemp] += pink
            self.gempstate[t,newemp,g] += 1
            self.stat_wage_reduction[t,newemp] += wr
            self.stat_wage_reduction_g[t,newemp,g] += wr
            self.galive[t,g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_toe[t,newemp] += toe
            self.stat_pension[t,newemp] += newpen
            self.stat_paidpension[t,newemp] += paidpens
            self.infostats_toe[t] += toe
            self.infostats_ove[t,newemp] += ove
            self.infostats_kassanjasen[t] += jasen
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.pop_actions[t,n] = act
                self.stat_unemp_len[t,n] = tis
                self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                self.popunemprightused[t,n] = bu
                self.infostats_unempwagebasis[t,n] = uw
                self.infostats_unempwagebasis_acc[t,n] = uwr
                self.infostats_pop_wage[t,n] = newsal
                self.infostats_pop_toe[t,n] = toe
                self.infostats_pop_pinkslip[t,n] = pink
                self.infostats_pop_pension[t,n] = newpen
                self.infostats_pop_children_under3[t,n] = c3
                self.infostats_pop_children_under7[t,n] = c7
                self.infostats_pop_children_under18[t,n] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18

            if q is not None:
                self.add_taxes(t,q,newemp,n,g)

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
        elif newemp<0:
            self.deceiced[t] += 1
            
    def add_v3(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58,ove,jasen=self.env.state_decode(newstate)
        puoliso=0

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            if a2>self.min_retirementage and newemp==3:
                newemp=2

            self.empstate[t,newemp] += 1
            self.alive[t] += 1
            self.rewstate[t,newemp] += r
            self.poprewstate[t,n] = r
            
            self.actions[t,newemp,act] += 1
            self.popempstate[t,n] = newemp
            #self.salaries[t,n] = newsal
            self.salaries_emp[t,newemp] += newsal
            self.time_in_state[t,newemp] += tis
            self.infostats_pinkslip[t,newemp] += pink
            self.gempstate[t,newemp,g] += 1
            self.stat_wage_reduction[t,newemp] += wr
            self.stat_wage_reduction_g[t,newemp,g] += wr
            self.galive[t,g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_toe[t,newemp] += toe
            self.stat_pension[t,newemp] += newpen
            self.stat_paidpension[t,newemp] += paidpens
            self.infostats_toe[t] += toe
            self.infostats_ove[t,newemp] += ove
            self.infostats_kassanjasen[t] += jasen
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.pop_actions[t,n] = act
                self.stat_unemp_len[t,n] = tis
                self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                self.popunemprightused[t,n] = bu
                self.infostats_unempwagebasis[t,n] = uw
                self.infostats_unempwagebasis_acc[t,n] = uwr
                self.infostats_pop_wage[t,n] = newsal
                self.infostats_pop_toe[t,n] = toe
                self.infostats_pop_pinkslip[t,n] = pink
                self.infostats_pop_pension[t,n] = newpen
                self.infostats_pop_children_under3[t,n] = c3
                self.infostats_pop_children_under7[t,n] = c7
                self.infostats_pop_children_under18[t,n] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18

            if q is not None:
                self.add_taxes(t,q,newemp,n,g)

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
        elif newemp<0:
            self.deceiced[t] += 1                                    

    def add_v4(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,p_tila_vanha,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,ura,bu,wr,upr,uw,uwr,pr,\
            c3,c7,c18,unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_g,p_w,\
            p_newpen,p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,\
            p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,\
            kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,nw\
            =self.env.state_decode(newstate)

        t=np.round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) # FIXME: tässä newemp>0
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2
                    
            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp<15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] += toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                    self.stat_wage_reduction[t,newemp] += wr
                    self.stat_wage_reduction_g[t,newemp,g] += wr
                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                self.popunemprightused[t,n] = bu
                self.infostats_ove[t,newemp] += ove
                self.infostats_ove_g[t,newemp,g] += ove
                self.infostats_puoliso[t,newemp] += puoliso
                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila<15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                    self.stat_wage_reduction[t,p_tila] += p_wr
                    self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
                self.popunemprightused[t,n+1] = p_bu
                self.infostats_ove[t,p_tila] += p_ove
                self.infostats_ove_g[t,p_tila,p_g] += p_ove
                self.infostats_puoliso[t,p_tila] += puoliso
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r
            
            # spouse is a first-class citizen
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.infostats_group[n+1,0] = int(p_g)
                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
                
        elif newemp<0:
            self.deceiced[t] += 1                                    
            
    def add_v5(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,p_tila_vanha,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_\
            =self.env.states.state_decode(state) # current employment state
            
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,\
            ura,bu,wr,upr,uw,uwr,pr,c3,c7,c18,\
            unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_g,p_w,p_newpen,\
            p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,\
            p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,\
            nw,old_pw,s_old_pw,pt_act,s_pt_act,wbasis,s_wbasis\
            =self.env.states.state_decode(newstate)

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) # FIXME: tässä newemp>0
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2
                    
            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp<15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] = toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_pop_wage_reduction[t,n] = wr
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                    self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                    self.popunemprightused[t,n] = bu
                    self.infostats_pop_pt_act[t,n] = pt_act
                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.stat_wage_reduction[t,newemp] += wr
                self.stat_wage_reduction_g[t,newemp,g] += wr
                self.infostats_pt_act[t,newemp,g,pt_act] += 1
                self.infostats_ove[t,newemp] += ove
                self.infostats_ove_g[t,newemp,g] += ove
                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila<15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_pop_wage_reduction[t,n+1] = wr
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                    self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
                    self.popunemprightused[t,n+1] = p_bu
                    self.infostats_pop_pt_act[t,n+1] = s_pt_act
                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.stat_wage_reduction[t,p_tila] += p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                self.infostats_pt_act[t,p_tila,p_g,s_pt_act] += 1
                self.infostats_ove[t,p_tila] += p_ove
                self.infostats_ove_g[t,p_tila,p_g] += p_ove
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r
            
            # spouse is a first-class citizen
            self.infostats_puoliso[t,n] = puoliso
            self.infostats_puoliso[t,n+1] = puoliso
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.infostats_group[n+1,0] = int(p_g)
                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
                
        elif newemp<0:
            self.deceiced[t] += 1                                                                  
                        
    def add_v6(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,a,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,p_tila_vanha,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_\
            =self.env.states.state_decode(state) # current employment state
            
        newemp,g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,\
            ura,bu,wr,upr,uw,uwr,pr,c3,c7,c18,\
            unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_g,p_w,p_newpen,\
            p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,\
            p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,\
            nw,old_pw,s_old_pw,pt_act,s_pt_act,wbasis,s_wbasis,m_lleft,s_lleft\
            =self.env.states.state_decode(newstate)

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) # FIXME: tässä newemp>0
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2
                    
            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp<15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] += toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_pop_wage_reduction[t,n] = wr
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                    self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                    self.popunemprightused[t,n] = bu
                    self.infostats_pop_pt_act[t,n] = pt_act
                    self.infostats_lleft[t,n] = m_lleft
                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.stat_wage_reduction[t,newemp] += wr
                self.stat_wage_reduction_g[t,newemp,g] += wr
                self.infostats_ove[t,newemp] += ove
                self.infostats_pt_act[t,newemp,g,pt_act] += 1
                self.infostats_ove_g[t,newemp,g] += ove
                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila<15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_pop_wage_reduction[t,n+1] = wr
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                    self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
                    self.popunemprightused[t,n+1] = p_bu
                    self.infostats_pop_pt_act[t,n+1] = s_pt_act
                    self.infostats_lleft[t,n+1] = s_lleft
                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.stat_wage_reduction[t,p_tila] += p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                self.infostats_pt_act[t,p_tila,p_g,s_pt_act] += 1
                self.infostats_ove[t,p_tila] += p_ove
                self.infostats_ove_g[t,p_tila,p_g] += p_ove
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r
            
            # spouse is a first-class citizen
            self.infostats_puoliso[t,n] = puoliso
            self.infostats_puoliso[t,n+1] = puoliso
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.infostats_group[n+1,0] = int(p_g)
                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
                
        elif newemp<0:
            self.deceiced[t] += 1     

    def add_v7(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,_,a,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,p_tila_vanha,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,\
            _,_,_,_\
            =self.env.states.state_decode(state) # current employment state
            
        newemp,g,p_g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,\
            ura,bu,wr,upr,uw,uwr,pr,c3,c7,c18,\
            unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_w,p_newpen,\
            p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,\
            p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,\
            nw,old_pw,s_old_pw,pt_act,s_pt_act,wbasis,s_wbasis,m_lleft,s_lleft,m_ud,s_ud,\
            time_to_marriage,time_to_divorce,until_birth,\
            main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
            =self.env.states.state_decode(newstate)

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) 
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2

            if newemp==12 and act[0]>0: # töissä oleva opiskelija
                newemp=16
            if p_tila==12 and act[1]>0: # töissä oleva opiskelija
                p_tila=16

            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp!= 15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] += toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_pop_wage_reduction[t,n] = wr
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                    self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                    self.popunemprightused[t,n] = bu
                    self.infostats_pop_pt_act[t,n] = pt_act
                    self.infostats_pop_puoliso[t,n] = puoliso
                    self.infostats_lleft[t,n] = m_lleft

                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.stat_wage_reduction[t,newemp] += wr
                self.stat_wage_reduction_g[t,newemp,g] += wr
                self.infostats_pt_act[t,newemp,g,pt_act] += 1
                self.infostats_ove[t,newemp] += ove
                self.infostats_ove_g[t,newemp,g] += ove
                self.infostats_puoliso[t,newemp] += puoliso

                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila != 15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,p_tila,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_pop_wage_reduction[t,n+1] = wr
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                    self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
                    self.popunemprightused[t,n+1] = p_bu
                    self.infostats_pop_pt_act[t,n+1] = s_pt_act
                    self.infostats_pop_puoliso[t,n+1] = puoliso
                    self.infostats_lleft[t,n+1] = s_lleft

                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.stat_wage_reduction[t,p_tila] += p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                self.infostats_pt_act[t,p_tila,p_g,s_pt_act] += 1
                self.infostats_ove[t,p_tila] += p_ove
                self.infostats_ove_g[t,p_tila,p_g] += p_ove
                self.infostats_puoliso[t,p_tila] += puoliso
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r

            self.emp_htv[t,g,newemp] += self.parttime_actions[newemp,pt_act]
            self.emp_htv[t,p_g,p_tila] += self.parttime_actions[p_tila,s_pt_act]

            # spouse is a first-class citizen
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.infostats_group[n+1,0] = int(p_g)
                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            # for both parents
            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18
            self.infostats_children_under3[t,p_tila] += c3
            self.infostats_children_under7[t,p_tila] += c7
            self.infostats_children_under18[t,p_tila] += c18

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
                
        elif newemp<0:
            self.deceiced[t] += 1        


    def add_v8(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,_,a,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,p_tila_vanha,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,\
            _,_,_,_\
            =self.env.states.state_decode(state) # current employment state
            
        newemp,g,p_g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,\
            ura,bu,wr,upr,uw,uwr,pr,c3,c7,c18,\
            unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_w,p_newpen,\
            p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,\
            p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,\
            nw,old_pw,s_old_pw,pt_act,s_pt_act,wbasis,s_wbasis,m_lleft,s_lleft,m_ud,s_ud,\
            time_to_marriage,time_to_divorce,until_birth,\
            main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
            =self.env.states.state_decode(newstate)

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) 
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2

            if newemp==12 and act[0]>0: # töissä oleva opiskelija
                newemp=16
            if p_tila==12 and act[1]>0: # töissä oleva opiskelija
                p_tila=16

            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp!= 15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] += toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_pop_wage_reduction[t,n] = wr
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                    self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                    self.popunemprightused[t,n] = bu
                    self.infostats_pop_pt_act[t,n] = pt_act
                    self.infostats_pop_puoliso[t,n] = puoliso
                    self.infostats_lleft[t,n] = m_lleft

                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.stat_wage_reduction[t,newemp] += wr
                self.stat_wage_reduction_g[t,newemp,g] += wr
                self.infostats_pt_act[t,newemp,g,pt_act] += 1
                self.infostats_ove[t,newemp] += ove
                self.infostats_ove_g[t,newemp,g] += ove
                self.infostats_puoliso[t,newemp] += puoliso

                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila != 15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,p_tila,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_pop_wage_reduction[t,n+1] = wr
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                    self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
                    self.popunemprightused[t,n+1] = p_bu
                    self.infostats_pop_pt_act[t,n+1] = s_pt_act
                    self.infostats_pop_puoliso[t,n+1] = puoliso
                    self.infostats_lleft[t,n+1] = s_lleft

                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.stat_wage_reduction[t,p_tila] += p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                self.infostats_pt_act[t,p_tila,p_g,s_pt_act] += 1
                self.infostats_ove[t,p_tila] += p_ove
                self.infostats_ove_g[t,p_tila,p_g] += p_ove
                self.infostats_puoliso[t,p_tila] += puoliso
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.emp_htv[t,g,newemp] += self.parttime_actions[newemp,pt_act]
            self.emp_htv[t,p_g,p_tila] += self.parttime_actions[p_tila,s_pt_act]

            # spouse is a first-class citizen
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.infostats_group[n+1,0] = int(p_g)

                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            # for both parents
            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18
            self.infostats_children_under3[t,p_tila] += c3
            self.infostats_children_under7[t,p_tila] += c7
            self.infostats_children_under18[t,p_tila] += c18

            # print(f'c3 {c3} c7 {c7} c18 {c18}')

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
                
        elif newemp<0:
            self.deceiced[t] += 1     



    def add_v9(self,n,act,r,state,newstate,q=None,debug=False,plot=False,aveV=None,pred_r=None):

        emp,_,_,_,_,a,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,p_tila_vanha,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,\
            _,_,_,_\
            =self.env.states.state_decode(state) # current employment state
            
        newemp,g,p_g,newpen,newsal,a2,tis,paidpens,pink,toe,toek,\
            ura,bu,wr,upr,uw,uwr,pr,c3,c7,c18,\
            unemp_left,aa,toe58,ove,jasen,puoliso,p_tila,p_w,p_newpen,\
            p_wr,p_paidpens,p_nw,p_bu,p_unemp_benefit_left,p_unemp_after_ra,p_uw,p_uwr,p_aa,p_toe58,\
            p_toe,p_toekesto,p_ura,p_tis,p_pink,p_ove,kansanelake,p_kansanelake,te_maksussa,p_te_maksussa,\
            nw,old_pw,s_old_pw,pt_act,s_pt_act,wbasis,s_wbasis,m_lleft,s_lleft,m_ud,s_ud,\
            time_to_marriage,time_to_divorce,until_birth,\
            main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
            =self.env.states.state_decode(newstate)

        t=round((a2-self.min_age)*self.inv_timestep)#-1
        if a2>a and newemp>=0: # new state is not reset (age2>age) 
            if a2>self.min_retirementage:
                if newemp==3:
                    newemp=2
                if p_tila==3:
                    p_tila=2

            if newemp==12 and act[0]>0: # töissä oleva opiskelija
                newemp=16
            if p_tila==12 and act[1]>0: # töissä oleva opiskelija
                p_tila=16

            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp!= 15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] += toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_pop_wage_reduction[t,n] = wr
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                    self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
                    self.popunemprightused[t,n] = bu
                    self.infostats_pop_pt_act[t,n] = pt_act
                    self.infostats_pop_puoliso[t,n] = puoliso
                    self.infostats_lleft[t,n] = m_lleft

                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.stat_wage_reduction[t,newemp] += wr
                self.stat_wage_reduction_g[t,newemp,g] += wr
                self.infostats_pt_act[t,newemp,g,pt_act] += 1
                self.infostats_ove[t,newemp] += ove
                self.infostats_ove_g[t,newemp,g] += ove
                self.infostats_puoliso[t,newemp] += puoliso

                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                
            if p_tila != 15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,p_tila,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_pop_wage_reduction[t,n+1] = wr
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                    self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
                    self.popunemprightused[t,n+1] = p_bu
                    self.infostats_pop_pt_act[t,n+1] = s_pt_act
                    self.infostats_pop_puoliso[t,n+1] = puoliso
                    self.infostats_lleft[t,n+1] = s_lleft

                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.stat_wage_reduction[t,p_tila] += p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                self.infostats_pt_act[t,p_tila,p_g,s_pt_act] += 1
                self.infostats_ove[t,p_tila] += p_ove
                self.infostats_ove_g[t,p_tila,p_g] += p_ove
                self.infostats_puoliso[t,p_tila] += puoliso
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.emp_htv[t,g,newemp] += self.parttime_actions[newemp,pt_act]
            self.emp_htv[t,p_g,p_tila] += self.parttime_actions[p_tila,s_pt_act]

            # spouse is a first-class citizen
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            if self.save_pop:
                self.infostats_group[n,0] = int(g)
                self.infostats_group[n+1,0] = int(p_g)

                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            # for both parents
            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18
            self.infostats_children_under3[t,p_tila] += c3
            self.infostats_children_under7[t,p_tila] += c7
            self.infostats_children_under18[t,p_tila] += c18

            # print(f'c3 {c3} c7 {c7} c18 {c18}')

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
                
        elif newemp<0:
            self.deceiced[t] += 1                                                                                
            
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

        t=round((a2-self.min_age)*self.inv_timestep)
        if a2>a and newemp>=0: # new state is not reset (age2>age)
            self.empstate[t,newemp] += 1
            self.alive[t] += 1
            self.rewstate[t,newemp] += r
            self.infostats_tulot_netto[t] += q['netto'] # already at annual level
            self.infostats_poptulot_netto[t,n] = q['netto']

            self.poprewstate[t,n] = r
            self.popempstate[t,n] = newemp
            #self.salaries[t,n] = newsal
            self.salaries_emp[t,newemp] += newsal
            #self.time_in_state[t,newemp] += tis
            self.infostats_equivalent_income[t] += q['eq']
            if self.save_pop:
                self.infostats_pop_wage[t,n] = newsal
                self.infostats_pop_pension[t,n] = newpen
                self.pop_actions[t,n] = act[1]
            if self.dynprog and pred_r is not None:
                self.pop_predrew[t,n] = pred_r

            self.infostats_savings[t,n] = savings*self.timestep # scale it correctly
            self.actions[t,newemp,act[0]] += 1
            self.sav_actions[t,n] = mod_sav_action

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
        elif newemp<0:
            self.deceiced[t] += 1        
            

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

        t=round((a2-self.min_age)*self.inv_timestep)#-1
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
                    
            self.empstate[t,newemp] += 1
            self.empstate[t,p_tila] += 1
            if newemp<15:
                self.alive[t] += 1
                self.galive[t,g] += 1
                #self.pop_alive[t,n] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[0]] += 1
                self.salaries_emp[t,newemp] += newsal
                self.infostats_pinkslip[t,newemp] += pink
                self.stat_toe[t,newemp] += toe
                self.infostats_toe[t] += toe
                if self.save_pop:
                    self.pop_actions[t,n] = act[0]
                    self.infostats_pop_toe[t,n] = toe
                    self.infostats_pop_pinkslip[t,n] = pink
                    self.infostats_pop_pension[t,n] = newpen
                    self.infostats_unempwagebasis[t,n] = uw
                    self.infostats_unempwagebasis_acc[t,n] = uwr
                    self.stat_unemp_len[t,n] = tis
                self.stat_pension[t,newemp] += newpen
                self.stat_paidpension[t,newemp] += paidpens
                self.stat_unemp_after_ra[t,newemp] += upr
                self.stat_wage_reduction[t,newemp] += wr
                self.stat_wage_reduction_g[t,newemp,g] += wr
                if q is not None:
                    self.add_taxes(t,q,newemp,n,g,person='omat_')
                self.infostats_savings[t,n] = savings
                self.sav_actions[t,n] = mod_sav_action
                
            if p_tila<15:
                self.alive[t] += 1
                self.galive[t,p_g] += 1
                #self.pop_alive[t,n+1] = 1
                self.infostats_kassanjasen[t] += jasen
                self.actions[t,newemp,act[1]] += 1
                self.salaries_emp[t,p_tila] += p_w
                self.infostats_pinkslip[t,p_tila] += p_pink
                self.stat_toe[t,p_tila] += p_toe
                self.infostats_toe[t] += p_toe
                if self.save_pop:
                    self.pop_actions[t,n+1] = act[1]
                    self.infostats_pop_toe[t,n+1] = p_toe
                    self.infostats_pop_pinkslip[t,n+1] = p_pink
                    self.infostats_pop_pension[t,n+1] = p_newpen
                    self.infostats_unempwagebasis[t,n+1] = p_uw
                    self.infostats_unempwagebasis_acc[t,n+1] = p_uwr
                    self.stat_unemp_len[t,n+1] = p_tis
                self.stat_pension[t,p_tila] += p_newpen
                self.stat_paidpension[t,p_tila] += p_paidpens
                self.stat_unemp_after_ra[t,p_tila] += p_unemp_after_ra
                self.stat_wage_reduction[t,p_tila] += p_wr
                self.stat_wage_reduction_g[t,p_tila,p_g] += p_wr
                if q is not None:
                    self.add_taxes(t,q,p_tila,n+1,p_g,person='puoliso_')
                self.infostats_savings[t,n+1] = p_savings
                self.sav_actions[t,n+1] = mod_p_sav_action
                
            self.rewstate[t,newemp] += r
            self.rewstate[t,p_tila] += r
            
            self.poprewstate[t,n] = r
            self.poprewstate[t,n+1] = r
            
            # spouse is a first-class citizen
            self.infostats_puoliso[t,n] = puoliso
            self.infostats_puoliso[t,n+1] = puoliso
            self.popempstate[t,n] = newemp
            self.popempstate[t,n+1] = p_tila
            self.time_in_state[t,newemp] += tis
            self.time_in_state[t,p_tila] += p_tis
            self.gempstate[t,newemp,g] += 1
            self.gempstate[t,p_tila,p_g] += 1
            self.stat_tyoura[t,newemp] += ura
            self.stat_tyoura[t,p_tila] += p_ura
            self.popunemprightleft[t,n] = -self.env.unempright_left(newemp,tis,bu,a2,ura)
            self.popunemprightleft[t,n+1] = -self.env.unempright_left(p_tila,p_tis,p_bu,a2,p_ura)
            self.popunemprightused[t,n] = bu
            self.popunemprightused[t,n+1] = p_bu
            self.infostats_group[n,0] = int(g)
            self.infostats_group[n+1,0] = int(p_g)
            self.infostats_ove[t,newemp] += ove
            self.infostats_ove[t,p_tila] += p_ove
            self.infostats_ove_g[t,newemp,g,] += ove
            self.infostats_ove_g[t,p_tila,p_g] += p_ove
            #self.infostats_pop_wage[t,n] = newsal
            #self.infostats_pop_wage[t,n+1] = p_w
            if self.save_pop:
                self.infostats_pop_children_under3[t,[n,n+1]] = c3
                self.infostats_pop_children_under7[t,[n,n+1]] = c7
                self.infostats_pop_children_under18[t,[n,n+1]] = c18

            self.infostats_children_under3[t,newemp] += c3
            self.infostats_children_under7[t,newemp] += c7
            self.infostats_children_under18[t,newemp] += c18
            #if puoliso:
            #    self.infostats_children_under3[t,n+1] = c3
            #    self.infostats_children_under7[t,n+1] = c7
            #    self.infostats_children_under18[t,n+1] = c18

            # fixme
            #self.infostats_tyoelake[t] += q[person+'elake_maksussa']*scale+q[person+'elake_maksussa']*scale

            if aveV is not None:
                self.aveV[t,n] = aveV

            if not emp==newemp:
                self.siirtyneet[t,emp] += 1
                self.siirtyneet_det[t,emp,newemp] += 1
            else:
                self.pysyneet[t,emp] += 1
                
            if not p_tila_vanha==p_tila:
                self.siirtyneet[t,p_tila_vanha] += 1
                self.siirtyneet_det[t,p_tila_vanha,p_tila] += 1
            else:
                self.pysyneet[t,p_tila_vanha] += 1
        elif newemp<0:
            self.deceiced[t] += 1 
      
    def scale_sim(self):
        '''
        Scale results as needed after single_sim
        '''
        self.stat_tyoura=self.stat_tyoura/np.maximum(1,self.empstate)
        self.time_in_state=self.time_in_state/np.maximum(1,self.empstate)
        self.stat_wage_reduction=self.stat_wage_reduction/np.maximum(1,self.empstate)
        #self.infostats_ove=self.infostats_ove/np.maximum(1,self.empstate)
        self.stat_unemp_after_ra=self.stat_unemp_after_ra/np.maximum(1,self.empstate)
        self.stat_paidpension=self.stat_paidpension/np.maximum(1,self.empstate)
        #self.infostats_pinkslip=self.infostats_pinkslip
        self.stat_pension=self.stat_pension/np.maximum(1,self.empstate)
        self.salaries_emp=self.salaries_emp/np.maximum(1,self.empstate)
        
        self.stat_wage_reduction_g=self.stat_wage_reduction_g/np.maximum(1,self.gempstate)
        self.infostats_ove_g=self.infostats_ove_g/np.maximum(1,self.gempstate)

        self.update_reward_stats()

    def rescale_sim_with_procs(self,procs):
        '''
        Scale results with multiple processes
        '''
        self.stat_tyoura=self.stat_tyoura/np.maximum(1,procs)
        self.time_in_state=self.time_in_state/np.maximum(1,procs)
        self.stat_wage_reduction=self.stat_wage_reduction/np.maximum(1,procs)
        #self.infostats_ove=self.infostats_ove/np.maximum(1,procs)
        self.stat_unemp_after_ra=self.stat_unemp_after_ra/np.maximum(1,procs)
        self.stat_paidpension=self.stat_paidpension/np.maximum(1,procs)
        #self.infostats_pinkslip=self.infostats_pinkslip
        self.stat_pension=self.stat_pension/np.maximum(1,procs)
        self.salaries_emp=self.salaries_emp/np.maximum(1,procs)
        
        self.stat_wage_reduction_g=self.stat_wage_reduction_g/np.maximum(1,procs)
        self.infostats_ove_g=self.infostats_ove_g/np.maximum(1,procs)

        self.gini_coef = self.comp_gini()
        self.update_reward_stats()

    def update_reward_stats(self):
        r=self.comp_presentvalue()
        self.initial_reward = np.mean(r[0,:])
        self.average_discounted_reward = self.comp_total_reward(discounted=True)
        self.average_undiscounted_reward = self.comp_total_reward(discounted=False)
        #print('disco reward: {} undiscounted reward: {}'.format(self.average_discounted_reward,self.average_undiscounted_reward))
        self.rewards_computed = True

    # def get_initial_reward_old(self,startage=None):
    #     real=self.comp_presentvalue()
    #     if startage is None:
    #         startage=self.min_age
    #     age=max(1,startage-self.min_age)
    #     realage=max(self.min_age+1,startage)
    #     print('Initial discounted reward at age {}: {}'.format(realage,np.mean(real[age,:])))
    #     return np.mean(real[age,:])
        
    def get_initial_reward(self,recomp=False):
        if recomp or not self.rewards_computed:
            self.update_reward_stats()

        return self.initial_reward

    def get_reward(self,recomp=False):
        if recomp or not self.rewards_computed or self.average_discounted_reward is None or self.average_undiscounted_reward is None:
            self.update_reward_stats()

        return self.average_discounted_reward,self.average_undiscounted_reward

    def get_reward_trajectory(self):
        '''
        Palauta keskimääräinen rewardin kehitys ajan mukana
        '''
        r=self.comp_presentvalue()
        return np.mean(r,axis=1)

    def set_datetime(self):
        now = datetime.now() # current date and time

        self.date_time = now.strftime("%Y-%m-%d, %H:%M:%S")

    def save_sim(self,filename):
        '''
        Save simulation results
        '''
        
        f = h5py.File(filename, 'w')
        ftype='float64'
        _ = f.create_dataset('version', data=self.version, dtype = np.int64)
        dt = h5py.special_dtype(vlen=str)
        _ = f.create_dataset('date_time', data=self.date_time, dtype = dt)
        _ = f.create_dataset('save_pop', data=self.save_pop, dtype = bool)
        
        #_ = f.create_dataset('date', data=self.date, dtype=dtype.str)
        _ = f.create_dataset('n_pop', data=self.n_pop, dtype = np.int64)
        _ = f.create_dataset('empstate', data=self.empstate, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('emp_htv', data=self.emp_htv, dtype = float,compression="gzip", compression_opts=9)
        _ = f.create_dataset('gempstate', data=self.gempstate, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('deceiced', data=self.deceiced, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('rewstate', data=self.rewstate, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('salaries_emp', data=self.salaries_emp, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('alive', data=self.alive, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('galive', data=self.galive, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('siirtyneet', data=self.siirtyneet, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('siirtyneet_det', data=self.siirtyneet_det, dtype = np.int64,compression="gzip", compression_opts=9)
        _ = f.create_dataset('pysyneet', data=self.pysyneet, dtype = np.int64,compression="gzip", compression_opts=9)
        if self.dynprog:
            _ = f.create_dataset('aveV', data=self.aveV, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('pop_predrew', data=self.pop_predrew, dtype=ftype,compression="gzip", compression_opts=9)

        _ = f.create_dataset('time_in_state', data=self.time_in_state, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_tyoura', data=self.stat_tyoura, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_toe', data=self.stat_toe, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_pension', data=self.stat_pension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_paidpension', data=self.stat_paidpension, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('popempstate', data=self.popempstate, dtype = np.int8,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_wage_reduction', data=self.stat_wage_reduction, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_wage_reduction_g', data=self.stat_wage_reduction_g, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_taxes', data=self.infostats_taxes, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_wagetaxes', data=self.infostats_wagetaxes, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_taxes_distrib', data=self.infostats_taxes_distrib, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_etuustulo', data=self.infostats_etuustulo, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_etuustulo_group', data=self.infostats_etuustulo_group, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_perustulo', data=self.infostats_perustulo, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_palkkatulo', data=self.infostats_palkkatulo, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_palkkatulo_emp', data=self.infostats_palkkatulo_emp, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_palkkatulo_eielakkeella', data=self.infostats_palkkatulo_eielakkeella, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_tyotpvraha', data=self.infostats_tyotpvraha, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_ansiopvraha', data=self.infostats_ansiopvraha, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_peruspvraha', data=self.infostats_peruspvraha, dtype=ftype,compression="gzip", compression_opts=9)
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
        _ = f.create_dataset('infostats_tulot_netto_emp', data=self.infostats_tulot_netto_emp, dtype=ftype)
        _ = f.create_dataset('poprewstate', data=self.poprewstate, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pinkslip', data=self.infostats_pinkslip, dtype=int)
        _ = f.create_dataset('infostats_tyelpremium', data=self.infostats_tyelpremium, dtype=ftype,compression="gzip", compression_opts=9)

        if self.save_pop:
            _ = f.create_dataset('infostats_pop_wage_reduction', data=self.infostats_pop_wage_reduction, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_pinkslip', data=self.infostats_pop_pinkslip, dtype = np.int8)
            _ = f.create_dataset('infostats_pop_pension', data=self.infostats_pop_pension, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_paid_tyel_pension', data=self.infostats_paid_tyel_pension, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_paidpension', data=self.infostats_pop_paidpension, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_kansanelake', data=self.infostats_pop_kansanelake, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_takuuelake', data=self.infostats_pop_takuuelake, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_tyoelake', data=self.infostats_pop_tyoelake, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_tyelpremium', data=self.infostats_pop_tyelpremium, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_toe', data=self.infostats_pop_toe, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_wage', data=self.infostats_pop_wage, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_unempwagebasis', data=self.infostats_unempwagebasis, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_unempwagebasis_acc', data=self.infostats_unempwagebasis_acc, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('stat_unemp_len', data=self.stat_unemp_len, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('popunemprightleft', data=self.popunemprightleft, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('popunemprightused', data=self.popunemprightused, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_irr_tyel_full', data=self.infostats_irr_tyel_full, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_irr_tyel_reduced', data=self.infostats_irr_tyel_reduced, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_emtr', data=self.infostats_pop_emtr, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_tva', data=self.infostats_pop_tva, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_potential_wage', data=self.infostats_pop_potential_wage, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_children_under3', data=self.infostats_pop_children_under3, dtype = np.int8,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_children_under7', data=self.infostats_pop_children_under7, dtype = np.int8,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_children_under18', data=self.infostats_pop_children_under18, dtype = np.int8,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_pt_act', data=self.infostats_pop_pt_act,  dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('pop_actions', data=self.pop_actions, dtype = np.int8,compression="gzip", compression_opts=9)
            _ = f.create_dataset('stat_pop_diswage5y', data=self.stat_pop_diswage5y,  dtype=ftype,compression="gzip", compression_opts=9)
            if self.version in self.recentmodels:
                _ = f.create_dataset('infostats_lleft', data=self.infostats_lleft,  dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_npv0', data=self.infostats_npv0, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_pop_puoliso', data=self.infostats_pop_puoliso, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('infostats_group', data=self.infostats_group, dtype = np.int8,compression="gzip", compression_opts=9)

        _ = f.create_dataset('actions', data=self.actions, dtype = int,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_sairausvakuutus', data=self.infostats_sairausvakuutus, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_pvhoitomaksu', data=self.infostats_pvhoitomaksu, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_ylevero', data=self.infostats_ylevero, dtype=ftype)
        _ = f.create_dataset('infostats_ylevero_distrib', data=self.infostats_ylevero_distrib, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_children_under3', data=self.infostats_children_under3, dtype = ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_children_under7', data=self.infostats_children_under7, dtype = ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_children_under18', data=self.infostats_children_under18, dtype = ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_toe', data=self.infostats_toe, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_ove', data=self.infostats_ove, dtype=ftype)
        _ = f.create_dataset('infostats_ove_g', data=self.infostats_ove_g, dtype=ftype)
        _ = f.create_dataset('infostats_kassanjasen', data=self.infostats_kassanjasen, dtype = np.int64)
        _ = f.create_dataset('infostats_poptulot_netto', data=self.infostats_poptulot_netto, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('infostats_equivalent_income', data=self.infostats_equivalent_income, dtype=ftype)
        _ = f.create_dataset('infostats_alv', data=self.infostats_alv, dtype=ftype,compression="gzip", compression_opts=9)
        _ = f.create_dataset('stat_unemp_after_ra', data=self.stat_unemp_after_ra, dtype=ftype,compression="gzip", compression_opts=9)
        
        if self.version in set([101,104]):
            _ = f.create_dataset('infostats_savings', data=self.infostats_savings, dtype=ftype,compression="gzip", compression_opts=9)
            _ = f.create_dataset('sav_actions', data=self.sav_actions, dtype=ftype,compression="gzip", compression_opts=9)

        if self.version in self.complexmodels:
            _ = f.create_dataset('infostats_puoliso', data=self.infostats_puoliso,  dtype=ftype,compression="gzip", compression_opts=9)

        if self.version in set([5,6,7,8,9]):
            _ = f.create_dataset('infostats_pt_act', data=self.infostats_pt_act,  dtype=int,compression="gzip", compression_opts=9)
        
        _ = f.create_dataset('initial_reward', data=self.initial_reward,  dtype=ftype)
        _ = f.create_dataset('average_discounted_reward', data=self.average_discounted_reward,  dtype=ftype)
        _ = f.create_dataset('average_undiscounted_reward', data=self.average_undiscounted_reward,  dtype=ftype)
        _ = f.create_dataset('rewards_computed', data=self.rewards_computed,  dtype=ftype)
        _ = f.create_dataset('gini_coef', data=self.gini_coef,  dtype=ftype)

        _ = f.create_dataset('params', data=json.dumps(self.params,cls = NpEncoder))

        f.close()

    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()

    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f[nimi][()]
        f.close()
        return val

    def load_sim(self,filename,n_pop=None,print_pop=False,silent=False):
        f = h5py.File(filename, 'r')

        if 'version' in f.keys():
            self.version=int(f['version'][()])
        else:
            self.version=1

        if 'date_time' in f.keys():
            self.date_time = f['date_time'][()]
        else:
            self.set_datetime()

        if 'date' in f.keys():
            self.date=f['date'][()]
        else:
            self.date=None
            
        if not silent and not self.silent:
            print(f'Loading results from {filename} version {self.version}')

        if 'max_ge' in f.keys():
            self.max_age=f['max_age'][()]
            self.min_age=f['min_age'][()]

        if 'save_pop' in f.keys():
            self.save_pop = f['save_pop'][()]
        else:
            self.save_pop = False

        if 'emp_htv' in f.keys():
            self.emp_htv = f['emp_htv'][()]

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

        if 'n_pop' in f.keys():
            self.n_pop=int(f['n_pop'][()])
        else:
            self.n_pop = np.sum(self.empstate[0,:])

        self.alive=f['alive'][()]
        if np.isnan(self.alive[0]):
            self.alive[0]=self.n_pop
        elif self.alive[0]<1:
            self.alive[0]=self.n_pop

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
        if 'infostats_pop_paidpension' in f.keys():
            self.infostats_pop_paidpension=f['infostats_pop_paidpension'][()]

        self.popempstate=f['popempstate'][()]
        self.stat_wage_reduction=f['stat_wage_reduction'][()]

        if 'stat_unemp_len' in f.keys():
            self.stat_unemp_len=f['stat_unemp_len'][()]

        if 'popunemprightleft' in f.keys():
            self.popunemprightleft=f['popunemprightleft'][()]
            self.popunemprightused=f['popunemprightused'][()]

        if 'infostats_wagetaxes' in f.keys():
            self.infostats_wagetaxes=f['infostats_wagetaxes'][()]

        if 'infostats_tyotpvraha'  in f.keys():
            self.infostats_tyotpvraha=f['infostats_tyotpvraha'][()]

        if 'infostats_palkkatulo_emp' in f.keys():
            self.infostats_palkkatulo_emp=f['infostats_palkkatulo_emp'][()]

        if 'infostats_taxes' in f.keys():
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
            
        if 'infostats_tulot_netto_emp' in f.keys():
            self.infostats_tulot_netto_emp=f['infostats_tulot_netto_emp'][()]

        if 'infostats_peruspvraha' in f.keys():
            self.infostats_peruspvraha=f['infostats_peruspvraha'][()]

        if 'infostats_kansanelake' in f.keys():
            self.infostats_kansanelake=f['infostats_kansanelake'][()]

        if 'infostats_pt_act' in f.keys():
            self.infostats_pt_act=f['infostats_pt_act'][()]

        #if 'infostats_takuuelake' in f.keys():
        #    self.infostats_takuuelake = f['infostats_takuuelake'][()]

        if 'infostats_lapsilisa' in f.keys():
            self.infostats_lapsilisa=f['infostats_lapsilisa'][()]
            self.infostats_elatustuki=f['infostats_elatustuki'][()]

        if 'spouseactions' in f.keys():
            self.spouseactions=f['spouseactions'][()]
        if 'infostats_etuustulo_group' in f.keys():
            self.infostats_etuustulo_group=f['infostats_etuustulo_group'][()]

        if 'infostats_valtionvero_distrib' in f.keys():
            self.infostats_valtionvero_distrib=f['infostats_valtionvero_distrib'][()]
            self.infostats_kunnallisvero_distrib=f['infostats_kunnallisvero_distrib'][()]

        if 'infostats_taxes_distrib' in f.keys():
            self.infostats_taxes_distrib=f['infostats_taxes_distrib'][()]
            self.infostats_ylevero_distrib=f['infostats_ylevero_distrib'][()]

        if 'infostats_pinkslip' in f.keys():
            self.infostats_pinkslip=f['infostats_pinkslip'][()]

        if 'infostats_pop_pinkslip' in f.keys():
            self.infostats_pop_pinkslip=f['infostats_pop_pinkslip'][()]

        if 'infostats_paid_tyel_pension' in f.keys():
            self.infostats_paid_tyel_pension=f['infostats_paid_tyel_pension'][()]

        if 'infostats_tyelpremium' in f.keys():
            self.infostats_tyelpremium=f['infostats_tyelpremium'][()]

        if 'infostats_pop_children_under3' in f.keys():
            self.infostats_pop_children_under3=f['infostats_pop_children_under3'][()]
            self.infostats_pop_children_under7=f['infostats_pop_children_under7'][()]
            self.infostats_pop_children_under18=f['infostats_pop_children_under18'][()]

        if 'infostats_pop_pt_act' in f.keys():
            self.infostats_pop_pt_act=f['infostats_pop_pt_act'][()]

        if 'infostats_pop_tyelpremium' in f.keys():
            self.infostats_pop_tyelpremium=f['infostats_pop_tyelpremium'][()]

        if 'infostats_pop_tyoelake' in f.keys():
            self.infostats_pop_tyoelake=f['infostats_pop_tyoelake'][()]
            self.infostats_pop_kansanelake=f['infostats_pop_kansanelake'][()]

        if 'infostats_pop_takuuelake' in f.keys():
            self.infostats_pop_takuuelake=f['infostats_pop_takuuelake'][()]

        if 'infostats_npv0' in f.keys():
            self.infostats_npv0=f['infostats_npv0'][()]

        if 'infostats_irr_tyel_full' in f.keys():
            self.infostats_irr=f['infostats_irr_tyel_full'][()]
            self.infostats_irr=f['infostats_irr_tyel_reduced'][()]

        #if 'infostats_chilren7' in f.keys():
        #    self.infostats_chilren7=f['infostats_chilren7'][()]
        #if 'infostats_chilren18' in f.keys():
        #    self.infostats_chilren18=f['infostats_chilren18'][()]

        if 'infostats_group' in f.keys():
            self.infostats_group=f['infostats_group'][()]

        if 'infostats_sairausvakuutus' in f.keys():
            self.infostats_sairausvakuutus=f['infostats_sairausvakuutus'][()]
            self.infostats_pvhoitomaksu=f['infostats_pvhoitomaksu'][()]
            self.infostats_ylevero=f['infostats_ylevero'][()]

        if 'infostats_pop_puoliso' in f.keys():
            self.infostats_pop_puoliso=f['infostats_pop_puoliso'][()]

        if 'stat_wage_reduction_g' in f.keys():
            self.stat_wage_reduction_g=f['stat_wage_reduction_g'][()]

        if 'siirtyneet_det' in f.keys():
            self.siirtyneet_det=f['siirtyneet_det'][()]

        if 'infostats_kassanjasen' in f.keys():
            self.infostats_kassanjasen=f['infostats_kassanjasen'][()]
            
        if 'infostats_pop_wage_reduction' in f.keys():
            self.infostats_pop_wage_reduction=f['infostats_pop_wage_reduction'][()]

        if 'infostats_puoliso' in f.keys():
            self.infostats_puoliso=f['infostats_puoliso'][()]

        if 'infostats_ove' in f.keys():
            self.infostats_ove=f['infostats_ove'][()]
        if 'infostats_ove_g' in f.keys():
            self.infostats_ove_g=f['infostats_ove_g'][()]

        if 'infostats_children_under3' in f.keys():
            self.infostats_children_under3=f['infostats_children_under3'][()]
            self.infostats_children_under7=f['infostats_children_under7'][()]
            self.infostats_children_under18=f['infostats_children_under18'][()]

        if 'infostats_unempwagebasis' in f.keys():
            self.infostats_unempwagebasis=f['infostats_unempwagebasis'][()]
            self.infostats_unempwagebasis_acc=f['infostats_unempwagebasis_acc'][()]
            self.infostats_toe=f['infostats_toe'][()]

        if 'infostats_pop_toe' in f.keys():
            self.infostats_pop_toe=f['infostats_pop_toe'][()]

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

        if 'infostats_pop_pension' in f.keys():
            self.infostats_pop_pension=f['infostats_pop_pension'][()]

        if 'infostats_savings' in f.keys():
            self.infostats_savings=f['infostats_savings'][()]
            self.sav_actions=f['sav_actions'][()]

        if 'infostats_alv' in f.keys():
            self.infostats_alv=f['infostats_alv'][()]
            
        if 'stat_unemp_after_ra' in f.keys():
            self.stat_unemp_after_ra=f['stat_unemp_after_ra'][()]
            
        if 'infostats_pop_emtr' in f.keys():
            self.infostats_pop_emtr=f['infostats_pop_emtr'][()]
            self.infostats_pop_tva=f['infostats_pop_tva'][()]
            
        if 'infostats_pop_potential_wage' in f.keys():
            self.infostats_pop_potential_wage=f['infostats_pop_potential_wage'][()]
            
        if 'initial_reward' in f.keys():
            self.initial_reward=f['initial_reward'][()]
            self.average_discounted_reward=f['average_discounted_reward'][()]
            self.rewards_computed=f['rewards_computed'][()]

        if 'average_undiscounted_reward' in f.keys():
            self.average_undiscounted_reward=f['average_undiscounted_reward'][()]
        else:
            self.average_undiscounted_reward=None

        if 'stat_pop_diswage5y' in f.keys():
            self.stat_pop_diswage5y=f['stat_pop_diswage5y'][()]

        if 'gini_coef' in f.keys():
            self.gini_coef=f['gini_coef'][()]
        else:
            gini_coef=None

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

    def comp_total_reward(self,discounted=True,discountfactor=None):
        if not discounted:
            total_reward = np.sum(self.rewstate)
            rr=total_reward/self.n_pop
        else:
            rr=self.comp_realoptimrew(discountfactor) #total_reward/self.n_pop

        return rr
        
    def comp_total_netincome(self):
        if True:
            #alivemask=(self.popempstate==self.env.get_mortstate())
            alivemask = self.get_alivemask()
            alive = np.sum(self.alive)
        
            multiplier = np.sum(self.infostats_npv0)/alive
            rr = np.sum(self.infostats_tulot_netto)/(alive*multiplier)
            eq = np.sum(self.infostats_equivalent_income)/(alive*multiplier)
        elif False:
            multiplier = np.sum(self.infostats_npv0)/self.n_pop
            rr = np.sum(self.infostats_tulot_netto)/(self.n_pop*multiplier)
            eq = np.sum(self.infostats_equivalent_income)/(self.n_pop*multiplier)
        else:
            rr = np.sum(self.infostats_tulot_netto)/(self.n_pop*(self.n_time*self.timestep+21.0)) # 21 years approximates the time in pension
            eq = np.sum(self.infostats_equivalent_income)/(self.n_pop*(self.n_time*self.timestep+21.0)) # 21 years approximates the time in pension
        return rr,eq
        

    def comp_budget(self,scale=True,debug=True):
        demog2=self.empstats.get_demog()
        scalex=demog2/self.alive

        q={}
        q[self.labels['tyotulosumma']] = np.sum(self.infostats_palkkatulo*scalex)
        q[self.labels['tyotulosumma eielakkeella']] = np.sum(self.infostats_palkkatulo_eielakkeella*scalex)
        q[self.labels['tyotulosumma opiskelijat']] = np.sum(self.infostats_palkkatulo_emp[:,16]*scalex[:,0])
        q[self.labels['tyotulosumma osa-aika']] = np.sum((self.infostats_palkkatulo_emp[:,10]+self.infostats_palkkatulo_emp[:,8])*scalex[:,0])
        q[self.labels['tyotulosumma kokoaika']] = np.sum((self.infostats_palkkatulo_emp[:,1]+self.infostats_palkkatulo_emp[:,9])*scalex[:,0])

        q[self.labels['etuusmeno']] = np.sum(self.infostats_etuustulo*scalex)
        #q[self.labels['etuusmeno_v2']] = 0
        q[self.labels['verot+maksut']] = np.sum(self.infostats_taxes*scalex) # sisältää verot ja sosiaalivakuutusmaksut
        #q[self.labels['verot+maksut_v2']] = 0
        #q[self.labels['verot+maksut+alv']] = 0
        q[self.labels['palkkaverot+maksut']] = np.sum(self.infostats_wagetaxes*scalex)
        #q[self.labels['muut tulot']] = 0
        q[self.labels['valtionvero']] = np.sum(self.infostats_valtionvero*scalex)
        q[self.labels['kunnallisvero']] = np.sum(self.infostats_kunnallisvero*scalex)
        q[self.labels['ptel']] = np.sum(self.infostats_ptel*scalex)
        q[self.labels['tyottomyysvakuutusmaksu']] = np.sum(self.infostats_tyotvakmaksu*scalex)
        q[self.labels['tyottomyyspvraha']] = np.sum(self.infostats_tyotpvraha*scalex)
        q[self.labels['ansiopvraha']] = np.sum(self.infostats_ansiopvraha*scalex)
        q[self.labels['peruspvraha']] = np.sum(self.infostats_peruspvraha*scalex)
        q[self.labels['asumistuki']] = np.sum(self.infostats_asumistuki*scalex)
        if self.include_perustulo:
            q[self.labels['perustulo']] = np.sum(self.infostats_perustulo*scalex)
        q[self.labels['tyoelakemeno']] = np.sum(self.infostats_tyoelake*scalex)
        q[self.labels['kansanelakemeno']] = np.sum(self.infostats_kansanelake*scalex)
        q[self.labels['kokoelakemeno']] = np.sum(self.infostats_kokoelake*scalex)
        q[self.labels['takuuelakemeno']] = q[self.labels['kokoelakemeno']]-q[self.labels['tyoelakemeno']]-q[self.labels['kansanelakemeno']]
        #q[self.labels['takuuelakemeno_v2']] = np.sum(self.infostats_takuuelake*scalex)
        q[self.labels['elatustuki']] = np.sum(self.infostats_elatustuki*scalex)
        q[self.labels['lapsilisa']] = np.sum(self.infostats_lapsilisa*scalex)
        q[self.labels['tyoelakemaksu']] = np.sum(self.infostats_tyelpremium*scalex)
        #q[self.labels['tyoelake_maksettu']] = np.sum(self.infostats_paid_tyel_pension*scalex)
        q[self.labels['opintotuki']] = np.sum(self.infostats_opintotuki*scalex)
        q[self.labels['isyyspaivaraha']] = np.sum(self.infostats_isyyspaivaraha*scalex)
        q[self.labels['aitiyspaivaraha']] = np.sum(self.infostats_aitiyspaivaraha*scalex)
        q[self.labels['kotihoidontuki']] = np.sum(self.infostats_kotihoidontuki*scalex)
        q[self.labels['sairauspaivaraha']] = np.sum(self.infostats_sairauspaivaraha*scalex)
        q[self.labels['toimeentulotuki']] = np.sum(self.infostats_toimeentulotuki*scalex)
        pt = np.sum(self.infostats_perustulo*scalex)
        if pt>0.0:
            q[self.labels['perustulo']] = pt
        q[self.labels['sairausvakuutusmaksu']] = np.sum(self.infostats_sairausvakuutus*scalex)
        q[self.labels['pvhoitomaksu']] = np.sum(self.infostats_pvhoitomaksu*scalex)
        q[self.labels['ylevero']] = np.sum(self.infostats_ylevero*scalex)

        q[self.labels['ta_maksut_elake']] = q[self.labels['tyoelakemaksu']]-q[self.labels['ptel']]
        q[self.labels['ta_maksut_muut']] = (0.2057-0.1695)*q[self.labels['tyotulosumma']] # karkea
        q[self.labels['ta_maksut']] = q[self.labels['ta_maksut_elake']]+q[self.labels['ta_maksut_muut']]
        if self.include_perustulo:
            q[self.labels['verotettava etuusmeno']] = q[self.labels['kokoelakemeno']]+q[self.labels['tyottomyyspvraha']]+q[self.labels['aitiyspaivaraha']]+q[self.labels['isyyspaivaraha']]+q[self.labels['perustulo']]+q[self.labels['sairauspaivaraha']]+q[self.labels['kotihoidontuki']]+q[self.labels['opintotuki']]
        else:
            q[self.labels['verotettava etuusmeno']] = q[self.labels['kokoelakemeno']]+q[self.labels['tyottomyyspvraha']]+q[self.labels['aitiyspaivaraha']]+q[self.labels['isyyspaivaraha']]+q[self.labels['sairauspaivaraha']]+q[self.labels['kotihoidontuki']]+q[self.labels['opintotuki']]
        q[self.labels['alv']] = np.sum(self.infostats_alv*scalex)
        q[self.labels['verot+maksut+alv']] = q[self.labels['verot+maksut']]+q[self.labels['alv']]
        q[self.labels['verot+maksut+alv+ta']] = q[self.labels['verot+maksut+alv']]+q[self.labels['ta_maksut_elake']] + q[self.labels['ta_maksut_muut']]
        q[self.labels['julkinen talous, netto']] = q[self.labels['verot+maksut+alv+ta']] - q[self.labels['etuusmeno']]

        q[self.labels['nettotulot']] = np.sum(self.infostats_tulot_netto*scalex)
        if debug:
            q[self.labels['tulot_netto_v2']] = q[self.labels['tyotulosumma']]+q[self.labels['etuusmeno']]-q[self.labels['verot+maksut+alv']]-q[self.labels['pvhoitomaksu']]
            q[self.labels['etuusmeno_v2']] = q[self.labels['tyottomyyspvraha']]+q[self.labels['kansanelakemeno']]+q[self.labels['takuuelakemeno']]+q[self.labels['opintotuki']]+q[self.labels['isyyspaivaraha']]+\
                q[self.labels['aitiyspaivaraha']]+q[self.labels['sairauspaivaraha']]+q[self.labels['toimeentulotuki']]+\
                q[self.labels['asumistuki']]+q[self.labels['kotihoidontuki']]+q[self.labels['elatustuki']]+q[self.labels['lapsilisa']] + q[self.labels['kokoelakemeno']]
            q[self.labels['verot+maksut_v2']] = q[self.labels['valtionvero']]+q[self.labels['kunnallisvero']]+q[self.labels['tyottomyysvakuutusmaksu']]+\
                q[self.labels['ylevero']]+q[self.labels['sairausvakuutusmaksu']]+q[self.labels['ptel']]
        
        return q

    def comp_ps_norw(self):
        #print(self.salaries_emp[:,1]+self.salaries_emp[:,10])
        return self.salaries_emp[:,1]+self.salaries_emp[:,10]
        
    def comp_family_matrix(self):
        spouse_g = np.zeros((3,3))
        if self.infostats_pop_puoliso.shape[1]<10:
            print('infostats_pop_puoliso.shape',self.infostats_pop_puoliso.shape,'incorrect')
            return
        for t in range(self.n_time):
            for k in range(0,self.n_pop,2):
                if self.infostats_pop_puoliso[t,k]>0 and self.popempstate[t,k]!=15:
                    m_g=self.infostats_group[k]
                    p_g=self.infostats_group[k+1]-3
                    spouse_g[m_g,p_g] += 1

        spouse_g /= np.sum(spouse_g)

        q = {}
        q['male basic'] = spouse_g[0,:]
        q['male secondary'] = spouse_g[1,:]
        q['male higher'] = spouse_g[2,:]

        df = pd.DataFrame.from_dict(q,orient='index',columns=['female basic','female secondary','female higher'])
        #df.['basic']=spouse_g[:,0]
        #df['secondary']=spouse_g[:,1]
        #df['higher']=spouse_g[:,2]

        print('cross-sections',df)


    def comp_gini(self):
        '''
        Laske Gini-kerroin populaatiolle
        '''
        if False:
            income = np.sort(self.infostats_tulot_netto,axis=None)
        else:
            alivemask = self.get_alivemask()
            #alivemask=(self.popempstate[:-1,:] ==self.env.get_mortstate())
            netto=ma.array(self.infostats_poptulot_netto[:-1,:],mask=alivemask[:-1,:])
            income=ma.sort(netto,axis=None).compressed()

        n=len(income)
        #L = np.arange(n+1,1,-1)
        #A = np.sum(L*income)/np.sum(income)
        #G=(n+1-2*A)/n
        
        # v2
        L = np.arange(1,n+1,1)
        A = np.sum(L*income)/np.sum(income)
        G = 2*A/n-(n+1)/n
        G *= 100

        return G

    def comp_pienituloisuus(self,level=1_000):
        #alivemask=(self.popempstate==self.env.get_mortstate())
        alivemask = self.get_alivemask()
        netto=ma.array(self.infostats_poptulot_netto/self.timestep,mask=alivemask) # vuositasolle
        alive = np.sum(self.alive)
        #level=level/self.timestep
        median=ma.median(netto)
        mean=ma.mean(netto)
        med60=0.60*median
        med50=0.50*median
        ratio_med50=ma.sum(netto<med50)/alive
        ratio_med60=ma.sum(netto<med60)/alive
        pt_levl=ma.sum(netto<level)/alive
        
        return ratio_med50,ratio_med60,pt_levl
                    
    def comp_employed_ratio_by_age(self,emp=None,grouped=False,g=0,muut=None,emp_htv=None):
        '''
        palauttaa ikärakenteen työllisyysasteesta
        '''
        if emp is None:
            if grouped:
                emp = np.squeeze(self.gempstate[:,:,g])
            else:
                emp=self.empstate

        if self.minimal:
            nn = np.sum(emp,1)
            tyoll_osuus=(emp[:,1]+emp[:,3])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,3])/nn
            tyoll_osuus = np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            htv_osuus = np.reshape(htv_osuus,(htv_osuus.shape[0],1))
        elif self.fullhtv:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jätetty pois, vaikka vapaa kestää alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,5]+emp[:,6]+emp[:,8]+emp[:,9]+emp[:,10]+emp[:,14])
            htv_osuus=(emp_htv[:,1]+emp_htv[:,8]+emp_htv[:,9]+emp_htv[:,10])

            if muut is not None:
                tyoll_osuus += muut

            nn=self.alive[:,0]
            tyoll_osuus = tyoll_osuus / nn
            htv_osuus = htv_osuus / nn

            tyoll_osuus = np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            htv_osuus = np.reshape(htv_osuus,(htv_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jätetty pois, vaikka vapaa kestää alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,5]+emp[:,6]+emp[:,8]+emp[:,9]+emp[:,10]+emp[:,14])
            htv_osuus=(emp[:,1]+0.5*emp[:,8]+emp[:,9]+0.5*emp[:,10])

            if muut is not None:
                tyoll_osuus += muut

            nn=self.alive[:,0]
            tyoll_osuus = tyoll_osuus / nn
            htv_osuus = htv_osuus / nn

            tyoll_osuus = np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            htv_osuus = np.reshape(htv_osuus,(htv_osuus.shape[0],1))

        return tyoll_osuus,htv_osuus

    def comp_popemployed_ratio_by_age(self,mask=None,maxage=None):
        '''
        lasketaan työllisyysaste maskatuista alkioista
        '''

        if self.minimal:
            print('comp_popemployed_ratio_by_age not implemented')
            return 
            nn = np.sum(emp,1)
            tyoll_osuus=(emp[:,1]+emp[:,3])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,3])/nn
            tyoll_osuus = np.reshape(tyoll_osuus,(tyoll_osuus.shape[0],1))
            htv_osuus = np.reshape(htv_osuus,(htv_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jätetty pois, vaikka vapaa kestää alle 3kk

            if maxage is not None:
                n_time = self.map_age(maxage)
            else:
                n_time = self.n_time
            tyoll_osuus = np.zeros(n_time)
            htv_osuus = np.zeros(n_time)
            nn = np.zeros(n_time)
            for t in range(n_time):
                for k in range(self.n_pop):
                    if not mask[t,k]:
                        emp = self.popempstate[t,k]
                        if emp not in set([2,8,9,15]):
                            nn[t] += 1
                        if emp in set([1,10]):
                            tyoll_osuus[t] += 1
                            pt_act = int(self.infostats_pop_pt_act[t,k])
                            htv_osuus[t] += self.parttime_actions[emp,pt_act]

            #nn=np.sum(mask,axis=1)
            tyoll_osuus = tyoll_osuus / nn
            htv_osuus = htv_osuus / nn

        return tyoll_osuus,htv_osuus       

    def comp_popemployed_ratio(self,mask=None,maxage=None):
        '''
        lasketaan työllisyysaste maskatuista alkioista
        '''
        scalex = self.empstats.get_demog(1/self.alive)
        n_time = self.map_age(maxage)
        tyoll_osuus,htv_osuus = self.comp_popemployed_ratio_by_age(mask=mask,maxage=maxage)
        tyoll_osuus = np.nansum(tyoll_osuus[:n_time] * scalex[:n_time])/np.sum(scalex[:n_time])
        htv_osuus = np.nansum(htv_osuus[:n_time] * scalex[:n_time])/np.sum(scalex[:n_time])

        return tyoll_osuus,htv_osuus         

    def comp_employed_aggregate(self,emp=None,start=20,end=63.5,grouped=False,g=0,muut=None):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        if self.minimal:
            nn = np.sum(emp,1)
            tyoll_osuus=(emp[:,1]+emp[:,3])/nn
            htv_osuus=(emp[:,1]+0.5*emp[:,3])/nn
        elif self.fullhtv:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            tyoll_osuus=(emp[:,1]+emp[:,5]+emp[:,6]+emp[:,8]+emp[:,9]+emp[:,10]+emp[:,14])
            htv_osuus=(emp_htv[:,1]+emp_htv[:,8]+emp_htv[:,9]+emp_htv[:,10])
            nn=self.alive

            if muut is not None:
                tyoll_osuus += muut

            tyoll_osuus /= nn
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            nn=self.alive
            tyoll_osuus=(emp[:,1]+emp[:,8]+emp[:,9]+emp[:,10]+emp[:,5]+emp[:,6]+emp[:,14])
            htv_osuus=(emp[:,1]+0.5*emp[:,8]+emp[:,9]+0.5*emp[:,10])/nn

            if muut is not None:
                tyoll_osuus += muut

            tyoll_osuus /= nn

        htv_osuus=self.comp_state_stats(htv_osuus,start=start,end=end)
        tyoll_osuus=self.comp_state_stats(tyoll_osuus,start=start,end=end)

        return tyoll_osuus,htv_osuus

    def comp_group_ps(self):
        return self.comp_palkkasumma(grouped=True)

    def comp_palkkasumma(self,start=18,end=68,grouped=False,scale_time=True):
        '''
        Computes the sum of actually paid wages either by groups or as an aggregate
        '''
        if not self.save_pop:
            print('test_salaries: not enough data (save_pop=False)')
            return

        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1

        if grouped:
            scalex=demog2/self.alive
            ps = np.zeros((self.n_time,6))
            ps_norw = np.zeros((self.n_time,6))
            a_ps = np.zeros(6)
            a_ps_norw = np.zeros(6)
            for k in range(self.n_pop):
                g=int(self.infostats_group[k,0])
                for t in range(min_cage,max_cage):
                    e=int(self.popempstate[t,k])
                    if e in set([1,10,16]):
                        ps[t,g] += self.infostats_pop_wage[t,k]
                        ps_norw[t,g] += self.infostats_pop_wage[t,k]
                    elif e in set([8,9]):
                        ps[t,g] += self.infostats_pop_wage[t,k]
            for g in range(6):
                a_ps[g] = np.sum(scalex[min_cage:max_cage,0]*ps[min_cage:max_cage,g])
                a_ps_norw[g] = np.sum(scalex[min_cage:max_cage,0]*ps_norw[min_cage:max_cage,g])
        else:
            scalex=demog2/self.alive
            ps = np.zeros((self.n_time,1))
            ps_norw = np.zeros((self.n_time,1))

            for k in range(self.n_pop):
                for t in range(min_cage,max_cage):
                    e=int(self.popempstate[t,k])
                    if e in set([1,10,16]):
                        ps[t,0] += self.infostats_pop_wage[t,k]
                        ps_norw[t,0] += self.infostats_pop_wage[t,k]
                    elif e in set([8,9]):
                        ps[t,0] += self.infostats_pop_wage[t,k]

            a_ps = np.sum(scalex[min_cage:max_cage,0]*ps[min_cage:max_cage,0])
            a_ps_norw = np.sum(scalex[min_cage:max_cage,0]*ps_norw[min_cage:max_cage,0])

        return a_ps,a_ps_norw
        
    def comp_potential_palkkasumma(self,start=18,end=70,grouped=False,scale_time=True,full=False,include_kela=False):
        '''
        Laskee menetetyn palkkasumman joko tiloittain tai aggregaattina
        '''
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1

        if grouped:
            scalex=demog2/self.alive
            ps = np.zeros((self.n_time,self.n_states))
            a_ps = np.zeros(self.n_states)
            for k in range(self.n_pop):
                for t in range(min_cage,max_cage):
                    e=int(self.popempstate[t,k])
                    if e in set([1,8,9,10,16]):
                        ps[t,e] += self.infostats_pop_wage[t,k]
                    elif e in set([0,2,3,4,5,6,7,10,11,12,13,14,16]):
                        if include_kela:
                            ps[t,e] += self.infostats_pop_potential_wage[t,k]*(1-self.infostats_pop_wage_reduction[t,k])
                        else:
                            if self.infostats_paid_tyel_pension[t,k]>0:
                                ps[t,e] += self.infostats_pop_potential_wage[t,k]*(1-self.infostats_pop_wage_reduction[t,k])
            for g in range(self.n_states):
                a_ps[g] = np.sum(scalex[min_cage:max_cage,0]*ps[min_cage:max_cage,g])
        else:
            scalex=demog2/self.alive
            ps = np.zeros((self.n_time,1))
            ps_norw = np.zeros((self.n_time,1))

            for k in range(self.n_pop):
                for t in range(min_cage,max_cage):
                    e=int(self.popempstate[t,k])
                    if e in set([3,14]):
                        ps[t,0] += self.infostats_pop_potential_wage[t,k]*(1-self.infostats_pop_wage_reduction[t,k])

            a_ps = np.sum(scalex[min_cage:max_cage,0]*ps[min_cage:max_cage,0])

        if full:
            return a_ps,ps
        else:
            return a_ps

    def comp_potential_disab(self,start=18,end=70,scale_time=True,include_kela=False):
        '''
        Laskee menetetyn palkkasumman joko tiloittain tai aggregaattina
        '''
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1

        scalex=demog2/self.alive 
        ps = np.zeros((self.n_time))
        for k in range(self.n_pop):
            for t in range(min_cage,max_cage):
                if int(self.popempstate[t,k])==3:
                    if include_kela:
                        ps[t] += self.stat_pop_diswage5y[k]
                    else:
                        if self.infostats_paid_tyel_pension[t,k]>0:
                            ps[t] += self.stat_pop_diswage5y[k]

        ps *= scale
        a_ps = np.sum(scalex[min_cage:max_cage,0]*ps[min_cage:max_cage])

        return a_ps,ps

    def comp_tkstats(self):
        '''
        Laskee työkyvyttömyyseläkkeisiin menetetty työpanos
        '''
        return self.comp_potempstats(3)

    def comp_potempstats(self,e):
        '''
        Laskee tilassa e menetetyn työpanoksen
        '''
        menetetty_palkka_reduced = np.zeros((self.n_time,1))
        tk = np.zeros((self.n_time,1))
        
        retage=self.map_age(self.min_retirementage)
        #pot=self.comp_potential_palkkasumma()
        allowed = set([e])

        for k in range(self.n_pop):
            for t in range(retage):
                if self.popempstate[t,k] in allowed:
                    menetetty_palkka_reduced[t] += self.infostats_pop_potential_wage[t,k]*(1-self.infostats_pop_wage_reduction[t,k])
                    tk[t] += 1

        demog2=self.empstats.get_demog()
        scalex=demog2/self.alive

        sum1 = np.sum(menetetty_palkka_reduced*scalex)
        sum2 = sum1*2.13 # työpanos
        n_tk = np.sum(tk*scalex)

        return sum1,sum2,n_tk
        
    def comp_stats_agegroup(self,border=[19,35,50],doprint=True):
        n_groups=len(border)
        low=border.copy()
        high=border.copy()
        high[0:n_groups-1] = border[1:n_groups]
        high[-1] = 65
        employed = np.zeros(n_groups)
        unemployed = np.zeros(n_groups)
        ahtv = np.zeros(n_groups)
        parttimeratio = np.zeros(n_groups)
        unempratio = np.zeros(n_groups)
        empratio = np.zeros(n_groups)
        i_ps = np.zeros(n_groups)
        i_ps_norw = np.zeros(n_groups)
        for n in range(n_groups):
            l=low[n]
            h=high[n]
            emp_htv2 = np.sum(self.episodestats.emp_htv,axis=1)
            htv,tyollvaikutus,tyollaste,tyotosuus,tyottomat,osatyollaste=\
                self.comp_unemp_simstats_agegroups(self.empstate,scale_time=True,start=l,end=h,agegroups=True,emp_htv=emp_htv2)
            ps,ps_norw=self.comp_palkkasumma(start=l,end=h)

            if doprint:
                print(f'l {l} h {h}\nhtv {htv}\ntyollaste {tyollaste}\ntyotosuus {tyotosuus}\ntyottomat {tyottomat}\nosatyollaste {osatyollaste}\nps {ps}')

            employed[n] = tyollvaikutus
            ahtv[n] = htv
            unemployed[n] = tyottomat
            unempratio[n] = tyotosuus
            empratio[n] = tyollaste
            parttimeratio[n] = osatyollaste
            i_ps[n] = ps
            i_ps_norw[n] = ps_norw

        return employed,ahtv,unemployed,parttimeratio,i_ps,i_ps_norw,unempratio,empratio


    def comp_unemployed_ratio_by_age(self,emp=None,grouped=False,g=0,per_pop=True):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        if self.minimal:
            tyot_osuus = emp[:,0]/np.sum(emp,1)
            tyot_osuus = np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # suhteessa väestöön
            if per_pop:
                nn = self.alive[:,0][:,None]
            else:
                nn = self.comp_workforce(emp)

            tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])[:,None]/np.maximum(1,nn)

        return tyot_osuus

    def comp_unemployed_aggregate(self,emp=None,start=20,end=63.5,scale_time=True,grouped=False,g=0,per_pop=True):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        if self.minimal:
            tyot_osuus=emp[:,0]/np.sum(emp,1)
        else:
            if per_pop:
                nn = self.alive[:,0]
            else:
                nn = self.comp_workforce(emp)

            if len(nn.shape)>1:
                tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])/nn[:,0]
            else:
                tyot_osuus=(emp[:,0]+emp[:,4]+emp[:,13])/nn

        unemp=self.comp_state_stats(tyot_osuus,start=start,end=end)

        return unemp

    def comp_employed_ratio(self,emp,emp_htv=None):
        tyoll_osuus,htv_osuus = self.comp_employed_ratio_by_age(emp,emp_htv=emp_htv)
        tyot_osuus = self.comp_unemployed_ratio_by_age(emp)
        kokotyo_osuus,osatyo_osuus = self.comp_parttime_ratio_by_age(emp)

        return tyoll_osuus,htv_osuus,tyot_osuus,kokotyo_osuus,osatyo_osuus

    def comp_unemployed_detailed(self,emp,per_pop=False):
        if self.minimal:
            ansiosid_osuus=emp[:,0]/np.sum(emp,1)
            tm_osuus=ansiosid_osuus*0
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # isyysvapaalla olevat jötetty pois, vaikka vapaa kestöö alle 3kk
            if per_pop:
                nn = self.alive[:,0][:,None]
            else:
                nn = self.comp_workforce(emp)

            ansiosid_osuus=(emp[:,0]+emp[:,4])[:,None]/nn
            tm_osuus=(emp[:,13])[:,None]/nn

        return ansiosid_osuus,tm_osuus    

    def comp_unemp_stats_by_group(self,emp,scale_time=True,start=19,end=68,per_pop=True):
        '''
        Laskee työttömien osuuden ryhmittäin
        '''
        if emp is None:
            emp=self.gempstate

        if self.minimal:
            tyot_osuus = emp[:,0]/np.sum(emp,1)
            tyot_osuus = np.reshape(tyot_osuus,(tyot_osuus.shape[0],1))
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # suhteessa väestöön
            if per_pop:
                nn = self.galive[:,0,:]
            else:
                nn = self.comp_workforce_group(emp)

            tyot_osuus=(emp[:,0,:]+emp[:,4,:]+emp[:,13,:])/nn

        return tyot_osuus

    def comp_unemp_stats_agg(self,emp=None,scale_time=True,start=19,end=68,per_pop=True):
        '''
        Laskee työttömien osuuden yhteensä
        '''
        if emp is None:
            emp=self.empstate

        if self.minimal:
            tyot_osuus = np.sum(emp[:,0,:])/np.sum(emp,1)
        else:
            # työllisiksi lasketaan kokoaikatyössä olevat, osa-aikaiset, ve+työ, ve+osatyö
            # suhteessa väestöön
            if per_pop:
                nn = self.alive[:,0]
            else:
                nn = self.comp_workforce(emp,ratio=False)

            tyot_osuus=np.sum(emp[:,0]+emp[:,4]+emp[:,13])/np.sum(nn)

        return tyot_osuus                    

    def comp_parttime_aggregate(self,emp=None,start=20,end=63.5,scale_time=True,grouped=False,g=0):
        '''
        Lukumäärätiedot (EI HTV!)
        '''

        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        if not self.minimal:
            nn = self.alive[:,0][:,None]
            tyossa=(emp[:,1]+emp[:,10]+emp[:,8]+emp[:,9])/nn
            osatyossa=(emp[:,10]+emp[:,8])/nn
        else:
            nn = np.sum(emp,1)
            tyossa=emp[:,1]/nn
            osatyossa=0*tyossa

        osatyo_osuus=self.comp_state_stats(osatyo_osuus,start=start,end=end)
        kokotyo_osuus=1-osatyo_osuus

        return kokotyo_osuus,osatyo_osuus

    def comp_parttime_ratio_by_age(self,emp=None,grouped=False,g=0):
        if emp is None:
            if grouped:
                emp=self.gempstate[:,:,g]
            else:
                emp=self.empstate

        if self.minimal:
            nn = np.sum(emp,1)
            kokotyo_osuus=(emp[:,1])/nn
            osatyo_osuus=(emp[:,3])/nn
        else:
            nn = self.alive[:,0]
            if grouped:
                for g in range(6):
                    kokotyo_osuus=(emp[:,1,g]+emp[:,9,g])/nn
                    osatyo_osuus=(emp[:,8,g]+emp[:,10,g])/nn
            else:
                kokotyo_osuus=(emp[:,1]+emp[:,9])/nn
                osatyo_osuus=(emp[:,8]+emp[:,10])/nn

        osatyo_osuus = np.reshape(osatyo_osuus,(osatyo_osuus.shape[0],1))
        kokotyo_osuus = np.reshape(kokotyo_osuus,(osatyo_osuus.shape[0],1))

        return kokotyo_osuus,osatyo_osuus

    def comp_genders(self,scale_time=True):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        start=self.min_age
        end=self.max_age
        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+2

        scalex=demog2[min_cage:max_cage]/self.alive*scale
        scalex2=demog2[min_cage:max_cage]*scale
        
        np.zeros((self.n_time,self.n_groups),dtype = np.int64)
        women = np.sum(np.sum(self.galive[min_cage:max_cage,3:6],axis=1,keepdims=True)*scalex)
        men = np.sum(np.sum(self.galive[min_cage:max_cage,0:3],axis=1,keepdims=True)*scalex)

        return women,men

    def comp_children(self,scale_time=True):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        start = self.min_age
        end = self.max_age
        min_cage = self.map_age(start)
        max_cage = self.map_age(end)+2

        scalex = demog2[min_cage:max_cage]/self.alive[min_cage:max_cage]

        c3 = np.sum(self.infostats_children_under3[min_cage:max_cage,:],axis=1)[:,None] * scalex * scale * 0.5
        c7 = np.sum(self.infostats_children_under7[min_cage:max_cage,:],axis=1)[:,None] * scalex * scale * 0.5
        c18 = np.sum(self.infostats_children_under18[min_cage:max_cage,:],axis=1)[:,None] * scalex * scale * 0.5
        
        w = np.sum(self.galive[min_cage:max_cage,3:6],axis=1,keepdims=True)*scalex # eri ikäisiä naisia kvartaalilla
        naiset_in = np.array([0.005727417359024218,0.01059128848986463,0.01890046459242574,0.02426763737216155,0.03430309811874956,0.04005336001334001,0.04828193258133925,0.0616650502143103,0.06816135422294255,0.08110212638514526,0.08670537010159651,0.09630583689479538,0.1017242873234555,0.1049780682386882,0.09842716448625606,0.0926950575069941,0.08550675779214809,0.07716610533218086,0.06553524804177546,0.05361265498076101,0.04137449445721435,0.03453863465866466,0.02571906863491097,0.01716422485134555,0.01061106141460255,0.007058537179019107,0.00411962160512664,0.002444757874941232,0.001108792068875555,0.000507964152244113,0.0002458210422812193,0.0001691474966170501,0.0001316872427983539,3.270538984824699e-05,0.0,3.007066606525334e-05,6.002400960384154e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        c_vrt = w[min_cage:max_cage:4] * naiset_in[:,None]

        return c3,c7,c18,c_vrt

    def comp_children_ages(self,scale_time=True):
        '''
        Palauta lasten määrä ikäryhmittäin
        '''
        min_cage = self.min_age
        max_cage = self.max_age+self.timestep
        x = np.arange(self.min_age,self.max_age+self.timestep,1.0)
        c3,c7,c18,c_vrt = self.comp_children()
        c3 = self.sum_q_to_a(c3) # annualize
        c7 = self.sum_q_to_a(c7) # annualize
        c18 = self.sum_q_to_a(c18) # annualize
        
        return x,c3,c7,c18,c_vrt

    def comp_children_hist(self):
        demog2=self.empstats.get_demog()

        start=self.min_age
        end=self.max_age
        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+2
        scalex=np.sum(demog2[min_cage:max_cage])/np.sum(self.alive[min_cage:max_cage])
        
        data = np.ravel(self.infostats_pop_children_under18[min_cage:max_cage,:])
        h, edges = np.histogram(data)

        return h*scalex, edges

    def comp_tyollisyys_stats(self,emp,scale_time=True,start=19,end=68,only_groups=False,agegroups=False,emp_htv=None):
        if agegroups:
            return self.comp_unemp_simstats_agegroups(emp,scale_time=scale_time,start=start,end=end,only_groups=only_groups,emp_htv=emp_htv)
        else:
            return self.comp_unemp_simstats_aggregate(emp,scale_time=scale_time,start=start,end=end,only_groups=only_groups,emp_htv=emp_htv)

    def comp_unemp_simstats_agegroups(self,emp,scale_time=True,start=19,end=68,only_groups=False,emp_htv=None):
        demog2 = self.empstats.get_demog()

        if scale_time:
            scale = self.timestep
        else:
            scale = 1.0

        min_cage = max(0,self.map_age(start))
        max_cage = self.map_age(end)+1

        scalex = demog2[min_cage:max_cage]*scale
        tyollosuus,htvosuus,tyott_osuus,kokotyo_osuus,osatyo_osuus = self.comp_employed_ratio(emp,emp_htv=emp_htv)

        htv_ika = scalex*htvosuus[min_cage:max_cage]
        tyolliset_ika = scalex*tyollosuus[min_cage:max_cage]
        tyottomat_ika = scalex*tyott_osuus[min_cage:max_cage]
        tyot_osuus = self.comp_unemployed_aggregate(emp,start=start,end=end,per_pop=True)
        tyot_aste = self.comp_unemployed_aggregate(emp,start=start,end=end,per_pop=False)
        osatyolliset_ika = scalex*osatyo_osuus[min_cage:max_cage]
        kokotyollvaikutus_ika = scalex*kokotyo_osuus[min_cage:max_cage]
        haj = np.mean(np.std(tyollosuus[min_cage:max_cage]))
        tyoll_osuus = tyollosuus[min_cage:max_cage]
        osatyoll_osuus = osatyo_osuus[min_cage:max_cage]
        kokotyoll_osuus = kokotyo_osuus[min_cage:max_cage]
        tyottomien_osuus = tyott_osuus[min_cage:max_cage]
        tyot_aste_age = self.comp_unemployed_ratio_by_age(emp,per_pop=False)
        tyottomyys_aste = tyot_aste_age[min_cage:max_cage]

        return htv_ika,tyolliset_ika,tyottomat_ika,osatyolliset_ika,kokotyollvaikutus_ika,tyoll_osuus,osatyoll_osuus,kokotyoll_osuus,tyottomien_osuus,tyottomyys_aste

    def comp_unemp_simstats_aggregate(self,emp,scale_time=True,start=19,end=68,only_groups=False,emp_htv=None):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale = self.timestep
        else:
            scale = 1.0

        min_cage = max(0,self.map_age(start))
        max_cage = self.map_age(end)+1

        scalex = demog2[min_cage:max_cage]*scale
        tyollosuus,htvosuus,tyot_osuus,kokotyo_osuus,osatyo_osuus = self.comp_employed_ratio(emp,emp_htv=emp_htv)

        htv = np.sum(scalex*htvosuus[min_cage:max_cage])
        tyolliset = np.sum(scalex*tyollosuus[min_cage:max_cage])
        tyottomat = np.sum(scalex*tyot_osuus[min_cage:max_cage])
        osatyolliset = np.sum(scalex*osatyo_osuus[min_cage:max_cage])
        kokotyollvaikutus = np.sum(scalex*kokotyo_osuus[min_cage:max_cage])
        tyot_aste = self.comp_unemployed_aggregate(emp,start=start,end=end,per_pop=False)
        haj = np.mean(np.std(tyollosuus[min_cage:max_cage]))
        elossa = np.sum(scalex)

        tyollaste = tyolliset/elossa
        osatyollaste = osatyolliset/elossa
        kokotyollaste = kokotyollvaikutus/elossa
        tyot_aste = tyottomat/(tyottomat+tyolliset)

        return htv,tyolliset,tyottomat,osatyolliset,kokotyollvaikutus,tyollaste,osatyollaste,kokotyollaste,tyot_aste

    def comp_uncertainty(self,emp,emp_htv=None,start=19,end=68):
        min_cage=max(0,self.map_age(start))
        max_cage=self.map_age(end)+1

        tyollosuus,htvosuus,tyot_osuus,kokotyo_osuus,osatyo_osuus=self.comp_employed_ratio(emp,emp_htv=emp_htv)
        haj = np.mean(np.std(tyollosuus[min_cage:max_cage]))
        return haj

    def comp_employment_stats(self,scale_time=True,returns=False):
        demog2=self.empstats.get_demog()

        if scale_time:
            scale=self.timestep
        else:
            scale=1.0

        min_cage=self.map_age(self.min_age)
        max_cage=self.map_age(self.max_age)+1

        scalex = np.squeeze(demog2/self.n_pop*self.timestep)

        d = np.squeeze(demog2[min_cage:max_cage])

        self.ratiostates=self.empstate/self.alive
        self.demogstates=(self.empstate.T*scalex).T
        if self.minimal:
            self.stats_employed=self.demogstates[:,0]+self.demogstates[:,3]
            self.stats_parttime=self.demogstates[:,3]
            self.stats_unemployed=self.demogstates[:,0]
            self.stats_all = np.sum(self.demogstates,1)
        else:
            self.stats_employed=self.demogstates[:,0]+self.demogstates[:,10]+self.demogstates[:,8]+self.demogstates[:,9]+self.demogstates[:,5]+self.demogstates[:,6]
            self.stats_parttime=self.demogstates[:,10]+self.demogstates[:,8]
            self.stats_unemployed=self.demogstates[:,0]+self.demogstates[:,4]+self.demogstates[:,13]
            self.stats_all = np.sum(self.demogstates,1)

        if returns:
            return self.stats_employed,self.stats_parttime,self.stats_unemployed


    def comp_participants(self,scale=True,include_retwork=True,grouped=False,g=0,lkm=False,emp_htv=None):
        '''
        Laske henkilöiden lkm / htv

        scalex olettaa, että naisia & miehiä yhtä paljon. Tämän voisi tarkentaa.
        '''
        demog2=self.empstats.get_demog()

        scalex = np.squeeze(demog2/self.alive*self.timestep)
        scalex_lkm = np.squeeze(demog2/self.alive*self.timestep)
        scalex_lapset = np.squeeze(demog2/self.alive*self.timestep)

        q={}
        if self.version in set([1,2,3,4,5,6,104]):
            print('version 6')
            if lkm:
                osa_aika_kerroin=1.0
            else:
                osa_aika_kerroin=0.5
                
            retage=self.map_age(self.min_retirementage)
            if grouped:
                emp = np.squeeze(self.gempstate[:,:,g])
            else:
                emp=self.empstate

            aikuisia = np.sum(np.sum(emp,axis=1)*scalex)
            työikäisiä = np.sum(np.sum(emp[self.map_age(18):self.map_age(63)+1,:],axis=1)*scalex[self.map_age(18):self.map_age(63)+1])
            eläkkeellä = np.sum(np.sum(emp[:,2],axis=1)*scalex)
            # tässä scalex on timestep*aikuisten lkm. infostats_children_under18 sisältää lasten määrän per naiset ja miehet
            lapsia = np.sum(np.sum(self.infostats_children_under18,axis=1)*scalex_lapset)*0.5 # lapset lasketaan sekä isälle että äidille
            if lkm or True:
                q[self.labels['yhteensä']] = aikuisia + lapsia
                q[self.labels['aikuisia']] = aikuisia
                q[self.labels['lapsia']] = lapsia
            q[self.labels['työikäisiä 18-62']] = työikäisiä
            q[self.labels['eläkkeellä']] = eläkkeellä + np.sum(emp[:retage,3]*scalex_lkm[:retage])
            q[self.labels['vanhuuseläkkeellä']] = eläkkeellä
            q[self.labels['työssä ja eläkkeellä']] = osa_aika_kerroin*np.sum(osa_aika_kerroin*emp[:,8]*scalex_lkm)+np.sum(emp[:,9]*scalex_lkm)
            q[self.labels['työssä yli 63v']] = np.sum(np.sum(emp[self.map_age(63):,[1,9]],axis=1)*scalex_lkm[self.map_age(63):])+osa_aika_kerroin*np.sum(np.sum(emp[self.map_age(63):,[8,10]],axis=1)*scalex_lkm[self.map_age(63):])
            if include_retwork:
                q[self.labels['palkansaajia']] = np.sum((emp[:,1]+osa_aika_kerroin*emp[:,10]+osa_aika_kerroin*emp[:,8]+emp[:,9])*scalex)
            else:
                q[self.labels['palkansaajia']] = np.sum((emp[:,1]+osa_aika_kerroin*emp[:,10])*scalex)
            q[self.labels['osaaikatyössä']] = osa_aika_kerroin*np.sum((emp[:,8]+emp[:,10])*scalex_lkm)
            q[self.labels['kokoaikatyössä']] = np.sum((emp[:,1]+emp[:,9])*scalex_lkm)
            if lkm:
                q[self.labels['ansiosidonnaisella']] = np.sum((emp[:,0]+emp[:,4])*scalex_lkm)
                q[self.labels['tmtuella']] = np.sum(emp[:,13]*scalex_lkm)
                q[self.labels['isyysvapaalla']] = np.sum(emp[:,6]*scalex_lkm)
                q[self.labels['kotihoidontuella']] = np.sum(emp[:,7]*scalex_lkm)
                q[self.labels['työkyvyttömyyseläke']] = np.sum(emp[:retage,3]*scalex_lkm[:retage])
                q[self.labels['svpäiväraha']] = osa_aika_kerroin*np.sum(emp[:,14]*scalex_lkm)
                q[self.labels['vanhempainvapaalla']] = np.sum(emp[:,5]*scalex_lkm)
                q[self.labels['opiskelijoita']] = np.sum((emp[:,12]+emp[:,16])*scalex_lkm)
                q[self.labels['ovella']] = np.sum(np.sum(self.infostats_ove,axis=1)*scalex)
                q[self.labels['pareja']] = np.sum(np.sum(self.infostats_puoliso,axis=1)*scalex)/2
        elif self.version in set([7,8,9]):
            #print('version 7/8/9')
            retage=self.map_age(self.min_retirementage)
            if lkm:
                if grouped:
                    emp = np.squeeze(self.gempstate[:,:,g])
                else:
                    emp = self.empstate
            else:
                if grouped:
                    emp = np.squeeze(self.emp_htv[:,g,:])
                else:
                    emp = np.sum(self.emp_htv[:,:,:],axis=1)

            aikuisia = np.sum(np.sum(emp,axis=1)*scalex)
            työikäisiä = np.sum(np.sum(emp[self.map_age(18):self.map_age(63)+1,:],axis=1)*scalex[self.map_age(18):self.map_age(63)+1])
            lapsia = np.sum(np.sum(self.infostats_children_under18,axis=1)*scalex_lapset)*0.5 # lasketaan kerran kummallekin vanhemlla, siksi kerroin 0,5
            eläkkeellä = np.sum(emp[:,2]*scalex)
            if lkm:
                q[self.labels['yhteensä']] = aikuisia + lapsia
                q[self.labels['aikuisia']] = aikuisia
                q[self.labels['lapsia']] = lapsia
            q[self.labels['työikäisiä 18-62']] = työikäisiä
            q[self.labels['työllisiä']] = np.sum((emp[:,1]+emp[:,10]+emp[:,8]+emp[:,9])*scalex)
            q[self.labels['työssä 63+']] = np.sum(np.sum(emp[self.map_age(63):,[1,9]],axis=1)*scalex_lkm[self.map_age(63):])+np.sum(np.sum(emp[self.map_age(63):,[8,10]],axis=1)*scalex_lkm[self.map_age(63):])
            q[self.labels['työssä ja eläkkeellä']] = np.sum(emp[:,8]*scalex_lkm)+np.sum(emp[:,9]*scalex_lkm)
            q[self.labels['eläkkeellä']] = eläkkeellä + np.sum(emp[:retage,3]*scalex_lkm[:retage])
            q[self.labels['vanhuuseläkkeellä']] = eläkkeellä

            if include_retwork:
                q[self.labels['palkansaajia']] = np.sum((emp[:,1]+emp[:,10]+emp[:,8]+emp[:,9])*scalex)
            else:
                q[self.labels['palkansaajia']] = np.sum((emp[:,1]+emp[:,10])*scalex)

            q[self.labels['osaaikatyössä']] = np.sum((emp[:,8]+emp[:,10])*scalex_lkm)
            q[self.labels['kokoaikatyössä']] = np.sum((emp[:,1]+emp[:,9])*scalex_lkm)
            q[self.labels['ansiosidonnaisella']] = np.sum((emp[:,0]+emp[:,4])*scalex_lkm)
            q[self.labels['tmtuella']] = np.sum(emp[:,13]*scalex_lkm)
            q[self.labels['isyysvapaalla']] = np.sum(emp[:,6]*scalex_lkm)
            q[self.labels['kotihoidontuella']] = np.sum(emp[:,7]*scalex_lkm)
            q[self.labels['työkyvyttömyyseläke']] = np.sum(emp[:retage,3]*scalex_lkm[:retage])
            q[self.labels['svpäiväraha']] = np.sum(emp[:,14]*scalex_lkm)
            q[self.labels['vanhempainvapaalla']] = np.sum(emp[:,5]*scalex_lkm)
            q[self.labels['opiskelijoita']] = np.sum((emp[:,12]+emp[:,16])*scalex_lkm)
            q[self.labels['ovella']] = np.sum(np.sum(self.infostats_ove,axis=1)*scalex)
            q[self.labels['pareja']] = np.sum(np.sum(self.infostats_puoliso,axis=1)*scalex)/2            
        else:
            q[self.labels['yhteensä']] = np.sum(np.sum(self.empstate[:,:],1)*scalex)
            q[self.labels['palkansaajia']] = np.sum((self.empstate[:,1])*scalex)
            q[self.labels['ansiosidonnaisella']] = np.sum((self.empstate[:,0])*scalex)
            q[self.labels['tmtuella']] = np.sum(self.empstate[:,1]*0)
            q[self.labels['isyysvapaalla']] = np.sum(self.empstate[:,1]*0)
            q[self.labels['kotihoidontuella']] = np.sum(self.empstate[:,1]*0)
            q[self.labels['vanhempainvapaalla']] = np.sum(self.empstate[:,1]*0)

        return q

    # def comp_employment_groupstats(self,scale_time=True,g=0,include_retwork=True,grouped=True):
    #     demog2=self.empstats.get_demog()

    #     if scale_time:
    #         scale=self.timestep
    #     else:
    #         scale=1.0

    #     #min_cage=self.map_age(self.min_age)
    #     #max_cage=self.map_age(self.max_age)+1

    #     scalex = np.squeeze(demog2/self.alive*scale) # self.n_pop

    #     #d = np.squeeze(demog2[min_cage:max_cage])

    #     if grouped:
    #         ratiostates = np.squeeze(self.gempstate[:,:,g])/self.alive
    #         demogstates = np.squeeze(self.gempstate[:,:,g])
    #     else:
    #         ratiostates=self.empstate[:,:]/self.alive
    #         demogstates=self.empstate[:,:]

    #     if self.version in set([1,2,3,4,5,6,7,104]):
    #         if include_retwork:
    #             stats_employed = np.sum((demogstates[:,1]+demogstates[:,9])*scalex)
    #             stats_parttime = np.sum((demogstates[:,10]+demogstates[:,8])*scalex)
    #         else:
    #             stats_employed = np.sum((demogstates[:,1])*scalex)
    #             stats_parttime = np.sum((demogstates[:,10])*scalex)
    #         stats_unemployed = np.sum((demogstates[:,0]+demogstates[:,4]+demogstates[:,13])*scalex)
    #     else:
    #         stats_employed = np.sum((demogstates[:,0]+demogstates[:,3])*scalex)
    #         stats_parttime = np.sum((demogstates[:,3])*scalex)
    #         stats_unemployed = np.sum((demogstates[:,0])*scalex)
    #         #stats_all = np.sum(demogstates,1)

    #     return stats_employed,stats_parttime,stats_unemployed

    def comp_state_stats(self,state,start=20,end=63.5):

        min_cage=self.map_age(start)
        max_cage=self.map_age(end)+1
        demog2 = np.squeeze(self.empstats.get_demog())
        vaikutus = np.sum(demog2[min_cage:max_cage]*state[min_cage:max_cage])/np.sum(demog2[min_cage:max_cage])

        return vaikutus

    def get_vanhempainvapaat(self,skip=4,show=False,all=True,csv=None):
        '''
        Laskee vanhempainvapaalla olevien määrän outsider-mallia (Excel) varten, ilman työvoimassa olevia vanhempainvapailla olevia
        '''

        alive = np.zeros((self.galive.shape[0],1))
        alive[:,0] = np.sum(self.galive[:,0:3],1)
        ulkopuolella_m = np.sum(self.gempstate[:,7,0:3],axis=1)[:,None]/alive
        tyovoimassa_m = np.sum(self.gempstate[:,6,0:3],axis=1)[:,None]/alive

        alive[:,0] = np.sum(self.galive[:,3:6],1)
        ulkopuolella_n=np.sum(self.gempstate[:,7,3:6],axis=1)[:,None]/alive
        tyovoimassa_n= np.sum(self.gempstate[:,5,0:3],axis=1)[:,None]/alive

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
            x = np.linspace(self.min_age,self.min_age+n-1,n).reshape(-1,1)
            df = pd.DataFrame(np.hstack([x,ulkopuolella_m[::skip],ulkopuolella_n[::skip],tyovoimassa_m[::skip],tyovoimassa_n[::skip]]), 
                columns = ['ikä','ulkopuolella_m','ulkopuolella_n','tyovoimassa_m','tyovoimassa_n'])
            df.to_csv(csv, sep=";", decimal=",")
            
        
        return ulkopuolella_m[::skip],ulkopuolella_n[::skip],tyovoimassa_m,tyovoimassa_n

    def get_muut(self,skip=4,show=False,csv=None):
        '''
        Laskee vanhempainvapaalla olevien määrän outsider-mallia (Excel) varten, ilman työvoimassa olevia vanhempainvapailla olevia
        '''

        alive = np.zeros((self.galive.shape[0],1))
        alive[:,0] = np.sum(self.galive[:,0:3],1)
        muut_m = np.sum(self.gempstate[:,0,0:3]+self.gempstate[:,1,0:3]+self.gempstate[:,4,0:3]
            +self.gempstate[:,10,0:3]+self.gempstate[:,13,0:3],axis=1)[:,None]/alive
        opisk_m = np.sum(self.gempstate[:,12,0:3]+self.gempstate[:,16,0:3],axis=1)[:,None]/alive
        svpaivaraha_m = np.sum(self.gempstate[:,14,0:3],axis=1)[:,None]/alive

        alive[:,0] = np.sum(self.galive[:,3:6],1)
        muut_n = np.sum(self.gempstate[:,0,3:6]+self.gempstate[:,1,3:6]+self.gempstate[:,4,3:6]
            +self.gempstate[:,10,3:6]+self.gempstate[:,13,3:6],axis=1)[:,None]/alive
        opisk_n = np.sum(self.gempstate[:,12,3:6]+self.gempstate[:,16,3:6],axis=1)[:,None]/alive
        svpaivaraha_n = np.sum(self.gempstate[:,14,3:6],axis=1)[:,None]/alive

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

        alive = np.zeros((self.galive.shape[0],1))
        alive[:,0] = np.sum(self.galive[:,0:3],1)
        ulkopuolella_m = np.sum(self.gempstate[:,7,0:3],axis=1)[:,None]/alive
        tyovoimassa_m = np.sum(self.gempstate[:,6,0:3],axis=1)[:,None]/alive

        alive[:,0] = np.sum(self.galive[:,3:6],1)
        ulkopuolella_n=(np.sum(self.gempstate[:,7,3:6],axis=1)[:,None] )/alive
        tyovoimassa_n = np.sum(self.gempstate[:,5,3:6],axis=1)[:,None]/alive

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
            n=muut_m[::skip].shape[0]
            x = np.linspace(self.min_age,self.min_age+n-1,n).reshape(-1,1)
            df = pd.DataFrame(np.hstack([x,muut_m[::skip],muut_n[::skip],opisk_m[::skip],opisk_n[::skip],svpaivaraha_m[::skip],svpaivaraha_n[::skip],
                ulkopuolella_m[::skip],ulkopuolella_n[::skip],tyovoimassa_m[::skip],tyovoimassa_n[::skip],svpaivaraha_m[::skip],svpaivaraha_n[::skip]]), 
                columns = ['ikä','muut_m','muut_n','opisk_m','opisk_n','svpaivaraha_m','svpaivaraha_n','vv_ulkopuolella_m','vv_ulkopuolella_n','vv_tyovoimassa_m','vv_tyovoimassa_n','sv_tyovoimassa_m','sv_tyovoimassa_n'])
            df.to_csv(csv, sep=";", decimal=",",index=False)
        
        return muut_m[::skip],muut_n[::skip]

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
        gy = np.empty(n_time)
        g=1
        gx = np.empty(n_time)
        for t in range(0,n_time):
            gx[t] = g
            g*=self.gamma

        for t in range(1,n_time):
            gy[t] = np.sum(gx[0:t])

        gf = np.mean(gy[1:])/10
        lx=(target-source)
        opt = np.exp(lx/gf)-1.0

        print(opt)

    def min_max(self):
        min_wage = np.min(self.infostats_pop_wage)
        max_wage = np.max(self.infostats_pop_wage)
        max_pension = np.max(self.infostats_pop_pension)
        min_pension = np.min(self.infostats_pop_pension)
        print(f'min wage {min_wage} max wage {max_wage}')
        print(f'min pension {min_pension} max pension {max_pension}')

    def map_age(self,age,start_zero=False):
        if start_zero:
            return round((age)*self.inv_timestep)
        else:
            return round((age-self.min_age)*self.inv_timestep)

    def map_t_to_age(self,t):
        return self.min_age+t/self.inv_timestep

    def episodestats_exit(self):
        plt.close(self.episode_fig)

    def comp_pop_takuuelake(self):
        takuuelake = np.sum(self.infostats_pop_takuuelake,axis=0)
        kansanelake = np.sum(self.infostats_pop_kansanelake,axis=0)
        suhde_plus = 0
        n_plus = 0
        n_minus = 0
        for k in range(self.n_pop):
            for t in range(self.n_time):
                if self.infostats_pop_takuuelake[t,k]>0 and self.infostats_pop_kansanelake[t,k]>0:
                    suhde_plus += self.infostats_pop_takuuelake[t,k]/np.maximum(1.0,self.infostats_pop_kansanelake[t,k])
                    n_plus += 1
                elif self.infostats_pop_takuuelake[t,k]>0:
                    n_minus += 1

        suhde_plus = suhde_plus/max(1.0,n_plus)

        lkm_takuuelake = np.sum(self.infostats_pop_takuuelake>0)
        lkm_kansanelake = np.sum(self.infostats_pop_kansanelake>0)
        return suhde_plus,n_minus,lkm_takuuelake,lkm_kansanelake

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
                irri = np.nan
        else:
            x = np.zeros(cashflow.shape[0]+max_npv)
        
            eind = np.zeros(max_npv+1)

            el=1
            for k in range(max_npv+1):
                eind[k] = el
                el=el*self.elakeindeksi

            x[:cashflow.shape[0]] = cashflow
            if npv>0:
                x[cashflow.shape[0]-1:] = cashflow[-2]*eind[:max_npv+1]

            y = np.zeros(int(np.ceil(x.shape[0]/4)))
            for k in range(y.shape[0]):
                y[k] = np.sum(x[4*k:4*k+4])
            irri = npf.irr(y)*100

        #if np.isnan(irri):
        #    if np.sum(pension)<0.1 and np.sum(empstate[0:self.map_age(63)] ==15)>0: # vain maksuja, joista ei saa tuottoja, joten tappio 100%
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
        self.infostats_irr_tyel_full = np.zeros((self.n_pop,1))
        self.infostats_irr_tyel_reduced = np.zeros((self.n_pop,1))
        for k in range(self.n_pop):
            # tyel-eläke ilman vähennyksiä
            self.infostats_irr_tyel_full[k] = self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(self.infostats_npv0[k,0],self.infostats_pop_tyelpremium[:,k],self.infostats_pop_tyoelake[:,k])
            # tyel-eläke ml. muiden eläkkeiden vähenteisyys
            self.infostats_irr_tyel_reduced[k] = self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(self.infostats_npv0[k,0],self.infostats_pop_tyelpremium[:,k],self.infostats_paid_tyel_pension[:,k])

    def get_alivemask(self):
        return self.popempstate==self.env.get_mortstate()

    def get_empstatemask(self,state):
        return self.popempstate != state

    def get_gendermask(self,gender,time=True):
        if gender is None:
            gmask = np.zeros_like(self.infostats_group) # mask no-one
        else:
            if gender == 1: # naiset
                gmask = self.infostats_group<3 # maskaa miehet
            elif gender == -1: # kaikki
                gmask = self.infostats_group>9 # maskaa ei ketään
            else: # miehet
                gmask = self.infostats_group>2 # maskaa naiset

        if time:
            gendermask = np.zeros((self.n_time,self.n_pop))
            gendermask[:self.n_time,:] = gmask[:,0]
        else:
            gendermask = gmask

        return gendermask

    def get_childmask(self,childg=0,g=None):
        if childg==0: # alle 3v
            gendermask = self.infostats_pop_children_under3<1
        elif childg==1: # alle 7v
            gendermask1 = self.infostats_pop_children_under7<1 
            gendermask2 = self.infostats_pop_children_under3>0
            gendermask = ma.mask_or(gendermask1,gendermask2)
        elif childg==2: # alle 18v
            gendermask1 = self.infostats_pop_children_under18<1 
            gendermask2 = self.infostats_pop_children_under7>0
            gendermask = ma.mask_or(gendermask1,gendermask2)
        else: # ei lapsia
            gendermask = self.infostats_pop_children_under18>0

        if g is not None:
            gendermask = gendermask * self.get_gendermask(g,time=True)

        return gendermask

    def comp_aggirr(self,gender=None,full=False):
        '''
        Laskee aggregoidun sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        
        if gender is None:
            alive=self.alive[-1]
            maxnpv = np.sum(self.infostats_npv0)/alive # keskimääräinen npv ilman diskonttausta tarkasteluvälin lopussa

            agg_premium = np.sum(self.infostats_pop_tyelpremium,axis=1)
            agg_pensions_reduced = np.sum(self.infostats_paid_tyel_pension,axis=1)
            agg_pensions_full = np.sum(self.infostats_pop_tyoelake,axis=1)
            agg_irr_tyel_full=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_full)
            agg_irr_tyel_reduced=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_reduced)
        else:
            if gender==1: # naiset
                gendermask = self.get_gendermask(1).astype(float).T
                #gendermask = (self.infostats_group>2).astype(float).T
                alive = np.sum(self.galive[-1,3:6])
            else: # miehet
                gendermask = self.get_gendermask(2).astype(float).T
                #gendermask = (self.infostats_group<3).astype(float).T
                alive = np.sum(self.galive[-1,0:3])
        
            maxnpv = np.sum(self.infostats_npv0.T*gendermask)/alive # keskimääräinen npv ilman diskonttausta tarkasteluvälin lopussa

            agg_premium = np.sum(self.infostats_pop_tyelpremium*gendermask,axis=1)
            agg_pensions_reduced = np.sum(self.infostats_paid_tyel_pension*gendermask,axis=1)
            agg_pensions_full = np.sum(self.infostats_pop_tyoelake*gendermask,axis=1)
            agg_irr_tyel_full=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_full)
            agg_irr_tyel_reduced=self.reaalinen_palkkojenkasvu*100+self.comp_annual_irr(maxnpv,agg_premium,agg_pensions_reduced)

        if full:
            return agg_irr_tyel_full,agg_irr_tyel_reduced,agg_premium,agg_pensions_reduced,agg_pensions_full,maxnpv
        else:
            return agg_irr_tyel_full,agg_irr_tyel_reduced

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
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14,16]

        unempset=set(unempset)

        if popempstate is None:
            popempstate=self.popempstate

        if popunemprightused is None:
            popunemprightused=self.popunemprightused

        keskikesto = np.zeros((5,5)) # 20-29, 30-39, 40-49, 50-59, 60-69, vastaa TYJin tilastoa
        n = np.zeros(5)

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

                        n[l] += 1
                        if self.popunemprightused[t,k]<=0.51:
                            keskikesto[l,0] += 1
                        elif self.popunemprightused[t,k]<=1.01:
                            keskikesto[l,1] += 1
                        elif self.popunemprightused[t,k]<=1.51:
                            keskikesto[l,2] += 1
                        elif self.popunemprightused[t,k]<=2.01:
                            keskikesto[l,3] += 1
                        else:
                            keskikesto[l,4] += 1

        for k in range(5):
            keskikesto[k,:] /= n[k]

        if return_q:
            return self.empdur_to_dict(keskikesto)
        else:
            return keskikesto

    def empdur_to_dict(self,empdur):
        q={}
        q['20-29'] = empdur[0,:]
        q['30-39'] = empdur[1,:]
        q['40-49'] = empdur[2,:]
        q['50-59'] = empdur[3,:]
        q['60-65'] = empdur[4,:]
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
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14,16]

        unempset=set(unempset)

        if popempstate is None:
            popempstate=self.popempstate

        keskikesto = np.zeros((5,5)) # 20-29, 30-39, 40-49, 50-59, 60-69, vastaa TYJin tilastoa
        n = np.zeros(5)

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

                            n[l] += 1
                            if duration<=0.51:
                                keskikesto[l,0] += 1
                            elif duration<=1.01:
                                keskikesto[l,1] += 1
                            elif duration<=1.51:
                                keskikesto[l,2] += 1
                            elif duration<=2.01:
                                keskikesto[l,3] += 1
                            else:
                                keskikesto[l,4] += 1
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
        tyoll_virta = np.zeros((self.n_time,1))
        tyot_virta = np.zeros((self.n_time,1))
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
            unempset=[0,2,3,4,5,6,7,8,9,11,12,13,14,16]

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
                            tyoll_virta[t] += 1
                            prev_state=popempstate[t,k]
                        elif prev_state in empset and popempstate[t,k] in unempset:
                            tyot_virta[t] += 1
                            prev_state=popempstate[t,k]
                        else: # some other state
                            prev_state=popempstate[t,k]

        return tyoll_virta,tyot_virta

    def comp_tyollistymisdistribs(self,popempstate=None,popunemprightleft=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100):
        '''
        Laskee työllistyneiden osuuden eri aikoina
        '''
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
        '''
        computes the disribution of employment and unemployment spells
        '''
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

    def comp_empdistribs_v2(self,popempstate=None,popunemprightleft=None,putki=True,tmtuki=True,laaja=False,outsider=False,ansiosid=True,tyott=False,max_age=100,filter=400):
        '''
        Työttömien siirtyminen pois työttömyydestä ensimmäistä kertaa
        jakauma
        '''
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
                passed=False
                if age<=max_age:
                    if popunemprightleft[t,k]>=filter:
                        passed=True
                    
                    if self.popempstate[t,k]!=prev_state:
                        if prev_state in unempset and popempstate[t,k] not in unempset:
                            if passed:
                                unemp_distrib.append((t-prev_trans)*self.timestep)
                                unemp_distrib_bu.append(popunemprightleft[t,k])
                                passed=False
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
        ratio = np.array([1,0.287024901703801,0.115508955875928,0.0681083442551332,0.0339886413280909,0.0339886413280909,0.0114460463084316,0.0114460463084316,0.0114460463084316,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00419397116644823,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00166011358671909,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206,0.00104849279161206])

        return ratio

    def comp_empratios_gender(self,gempstate=None,galive=None,unempratio=True,gender='men'):
        if gempstate is None:
            gempstate = self.gempstate
        if galive is None:
            galive = self.galive
        if gender=='men': # men
            gempstate2 = np.sum(gempstate[:,:,0:3],axis=2)
            alive = np.zeros((galive.shape[0],1))
            alive[:,0] = np.sum(galive[:,0:3],1)
        else: # women
            gempstate2 = np.sum(gempstate[:,:,3:6],axis=2)
            alive = np.zeros((galive.shape[0],1))
            alive[:,0] = np.sum(galive[:,3:6],1)

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = \
            self.comp_empratios(gempstate2,alive,unempratio=unempratio)

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste

    def comp_empratios_groups(self,unempratio=True,g=None):
        if g is None:
            x = self.galive.shape[0]
            tyollisyysaste = np.zeros((x,6))
            osatyoaste = np.zeros((x,6))
            tyottomyysaste = np.zeros((x,6))
            ka_tyottomyysaste = np.zeros(6)

            for g in range(6):
                gempstate = self.gempstate[:,:,g]#,axis=2)
                alive = self.galive[:,g][:,None] # np.sum(self.galive[:,g],1)

                ta,ota,tta,ka_tta = \
                    self.comp_empratios(gempstate,alive,unempratio=unempratio)
                tyollisyysaste[:,g]=ta
                osatyoaste[:,g]=ota
                tyottomyysaste[:,g]=tta
                ka_tyottomyysaste[g]=ka_tta
        else:
            gempstate = self.gempstate[:,:,g]#,axis=2)
            alive = self.galive[:,g][:,None] # np.sum(self.galive[:,g],1)

            #print('comp_empratios_groups',gempstate.shape,alive.shape)

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = \
                self.comp_empratios(gempstate,alive,unempratio=unempratio)

        #print('comp_empratios_groups',tyollisyysaste.shape,osatyoaste.shape,tyottomyysaste.shape,ka_tyottomyysaste.shape)

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste        

    def comp_workforce_group(self,by_age=True,ratio=False):
        if self.version in set([1,2,3,4,5,6,7,8,9,104]):
            employed=self.gempstate[:,1,:]
            retired=self.gempstate[:,2,:]
            unemployed=self.gempstate[:,0,:]
            disabled=self.gempstate[:,1,:]
            piped=self.gempstate[:,4,:]
            mother=self.gempstate[:,5,:]
            dad=self.gempstate[:,6,:]
            kotihoidontuki=self.gempstate[:,7,:]
            vetyo=self.gempstate[:,9,:]
            veosatyo=self.gempstate[:,8,:]
            osatyo=self.gempstate[:,10,:]
            outsider=self.gempstate[:,11,:]
            student=self.gempstate[:,12,:]+self.gempstate[:,16,:]
            tyomarkkinatuki=self.gempstate[:,13,:]
            sv=self.gempstate[:,14,:]
            if ratio:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki+dad+mother+sv)/self.galive[:,:]
            else:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki+dad+mother+sv)
        elif self.version in set([0,101]):
            if False:
                osatyo=emp[:,3]
            else:
                osatyo=0

            if ratio:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki)/alive[:,0]
            else:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki)

        if not by_age:
            workforce=np.sum(workforce,1)

        return workforce        

    def comp_workforce(self,emp=None,alive=None,by_age=True,ratio=False):
        if emp is None:
            emp=self.empstate
        if alive is None:
            alive=self.alive

        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]

        if self.version in set([1,2,3,4,5,6,7,8,9,104]):
            disabled=emp[:,3]
            piped=emp[:,4]
            mother=emp[:,5]
            dad=emp[:,6]
            kotihoidontuki=emp[:,7]
            vetyo=emp[:,9]
            veosatyo=emp[:,8]
            osatyo=emp[:,10]
            outsider=emp[:,11]
            student=emp[:,12]+emp[:,16]
            tyomarkkinatuki=emp[:,13]
            sv=emp[:,14]
            if ratio:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki+mother+dad+sv)/alive[:,0]
            else:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki+mother+dad+sv)
        elif self.version in set([0,101]):
            if False:
                osatyo=emp[:,3]
            else:
                osatyo=0

            if ratio:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki)/alive[:,0]
            else:
                workforce=(employed+osatyo+veosatyo+vetyo+unemployed+piped+tyomarkkinatuki)

        if not by_age:
            workforce=np.sum(workforce,1)

        return workforce[:,None]

    def comp_empratios(self,emp,alive,unempratio=True):

        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]

        if self.version in set([1,2,3,4,5,6,7,8,9,104]):
            disabled=emp[:,3]
            piped=emp[:,4]
            mother=emp[:,5]
            dad=emp[:,6]
            kotihoidontuki=emp[:,7]
            vetyo=emp[:,9]
            veosatyo=emp[:,8]
            osatyo=emp[:,10]
            outsider=emp[:,11]
            student=emp[:,12]+emp[:,6]
            tyomarkkinatuki=emp[:,13]
            sv=emp[:,14]
            tyollisyysaste=100*(employed+osatyo+veosatyo+vetyo+dad+mother+sv)/alive[:,0]
            if False:
                student_at_work=emp[:,16]
                osatyoaste=100*(osatyo+veosatyo+student_at_work)/(employed+osatyo+veosatyo+vetyo+student_at_work)
            else:
                osatyoaste=100*(osatyo+veosatyo)/(employed+osatyo+veosatyo+vetyo)

            if unempratio:
                workforce = self.comp_workforce(emp,alive)
                tyottomyysaste=100*(unemployed+piped+tyomarkkinatuki)/workforce[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed+tyomarkkinatuki+piped)/np.sum(workforce)
            else:
                tyottomyysaste=100*(unemployed+piped+tyomarkkinatuki)/alive[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed+tyomarkkinatuki+piped)/np.sum(alive[:,0])
        elif self.version in set([0,101]):
            if False:
                osatyo=emp[:,3]
            else:
                osatyo=0
            tyollisyysaste=100*(employed+osatyo)/alive[:,0]
            #osatyoaste = np.zeros(employed.shape)
            osatyoaste=100*(osatyo)/(employed+osatyo)
            if unempratio:
                tyottomyysaste=100*(unemployed)/(unemployed+employed+osatyo)
                ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(unemployed+employed+osatyo)
            else:
                tyottomyysaste=100*(unemployed)/alive[:,0]
                ka_tyottomyysaste=100*np.sum(unemployed)/np.sum(alive[:,0])

        return tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste

    def comp_decisionratios(self,emp,alive):
        '''
        Päätöspohjaisesti valitut tilat
        '''
        employed=emp[:,1]
        retired=emp[:,2]
        unemployed=emp[:,0]

        disabled=emp[:,3]
        piped=emp[:,4]
        mother=emp[:,5]
        dad=emp[:,6]
        kotihoidontuki=emp[:,7]
        vetyo=emp[:,9]
        veosatyo=emp[:,8]
        osatyo=emp[:,10]
        outsider=emp[:,11]
        student=emp[:,12]+emp[:,16]
        tyomarkkinatuki=emp[:,13]
        sv=emp[:,14]
        tyolliset=(employed+osatyo+veosatyo+vetyo+dad+mother+sv)/alive[:,0]
        tyottomat=(unemployed+piped+tyomarkkinatuki)/alive[:,0]
        vella=(retired+disabled)/alive[:,0]
        yhteensa=vella+tyolliset+tyottomat
        tyolliset_osuus=tyolliset/yhteensa
        tyottomat_osuus=tyottomat/yhteensa
        vella_osuus=vella/yhteensa

        return tyolliset_osuus,tyottomat_osuus,vella_osuus
        
    def comp_ptproportions(self):
        ptsuhde = np.empty((self.n_groups,self.n_time,3))
        ftsuhde = np.empty((self.n_groups,self.n_time,3))
        combisuhde = np.empty((self.n_groups,self.n_time,6))
        veptsuhde = np.empty((self.n_groups,self.n_time,3))
        veftsuhde = np.empty((self.n_time,self.n_time,3))
        agg_ptsuhde = np.empty((3,3))
        agg_ftsuhde = np.empty((3,3))
        agg_combisuhde = np.empty((3,6))

        for g in range(3):
            if g==0:
                gset=[0,1,2,3,4,5]
            elif g==1:
                gset=[0,1,2]
            else:
                gset=[3,4,5]
            agg_ptsuhde[g,:] = np.sum(self.infostats_pt_act[:,10,gset,:],axis=(0,1)) / np.sum(self.infostats_pt_act[:,10,gset,:],axis=(0,1,2))
            agg_ftsuhde[g,:] = np.sum(self.infostats_pt_act[:,1,gset,:],axis=(0,1)) / np.sum(self.infostats_pt_act[:,1,gset,:],axis=(0,1,2))
            s_pt = np.sum(self.infostats_pt_act[:,10,gset,:],axis=1)
            s_ft = np.sum(self.infostats_pt_act[:,1,gset,:],axis=1)
            agg_combisuhde[g,0:3] = np.sum(s_pt,axis=0) / (np.sum(self.infostats_pt_act[:,1,gset,:],axis=(0,1,2))+np.sum(self.infostats_pt_act[:,1,gset,:],axis=(0,1,2)))
            agg_combisuhde[g,3:6] = np.sum(s_ft,axis=0) / (np.sum(self.infostats_pt_act[:,1,gset,:],axis=(0,1,2))+np.sum(self.infostats_pt_act[:,1,gset,:],axis=(0,1,2)))
            ptsuhde[g,:,:] = s_pt / np.sum(self.infostats_pt_act[:,10,gset,:],axis=(1,2))[:,None]
            ftsuhde[g,:,:] = s_ft / np.sum(self.infostats_pt_act[:,1,gset,:],axis=(1,2))[:,None]
            combisuhde[g,:,0:3] = s_pt/np.sum(s_pt+s_ft)
            combisuhde[g,:,3:6] = s_ft/np.sum(s_pt+s_ft)
            veptsuhde[g,:,:] = np.sum(self.infostats_pt_act[:,8,gset,:],axis=1) / np.sum(self.infostats_pt_act[:,8,gset,:],axis=(1,2))[:,None]
            veftsuhde[g,:,:] = np.sum(self.infostats_pt_act[:,9,gset,:],axis=1) / np.sum(self.infostats_pt_act[:,9,gset,:],axis=(1,2))[:,None]

        return agg_ptsuhde,agg_ftsuhde,ptsuhde,ftsuhde,veptsuhde,veftsuhde,combisuhde,agg_combisuhde
        
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
        scalex=demog2/self.alive

        valtionvero_osuus = np.sum(self.infostats_valtionvero_distrib*scalex,0)
        kunnallisvero_osuus = np.sum(self.infostats_kunnallisvero_distrib*scalex,0)
        taxes_distrib = np.sum(self.infostats_taxes_distrib*scalex,0)
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
        qq['tI'] = taxes[3]/q['tyotulosumma']
        qq['tC'] = tC/q['tyotulosumma']
        qq['tP'] = q['ta_maksut']/q['tyotulosumma']

        if debug:
            print('qq',qq,'kiila',kiila)

        return kiila,qq

    def comp_verokiila_kaikki_ansiot(self):
        demog2=self.empstats.get_demog()
        scalex=demog2/self.alive

        valtionvero_osuus = np.sum(self.infostats_valtionvero_distrib*scalex,0)
        kunnallisvero_osuus = np.sum(self.infostats_kunnallisvero_distrib*scalex,0)
        taxes_distrib = np.sum(self.infostats_taxes_distrib*scalex,0)
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
        qq['tI'] = taxes[0]/q['tyotulosumma']
        qq['tC'] = tC/q['tyotulosumma']
        qq['tP'] = q['ta_maksut']/q['tyotulosumma']

        return kiila,qq

    def v2_states(self,x):
        return 'Ansiosidonnaisella {:.2f}\nKokoaikatyössä {:.2f}\nVanhuuseläkeläiset {:.2f}\nTyökyvyttömyyseläkeläiset {:.2f}\n'.format(x[0],x[1],x[2],x[3])+\
          'Putkessa {:.2f}\nÄitiysvapaalla {:.2f}\nIsyysvapaalla {:.2f}\nKotihoidontuella {:.2f}\n'.format(x[4],x[5],x[6],x[7])+\
          'VE+OA {:.2f}\nVE+kokoaika {:.2f}\nOsa-aikatyö {:.2f}\nTyövoiman ulkopuolella {:.2f}\n'.format(x[8],x[9],x[10],x[11])+\
          'Opiskelija/Armeija {:.2f}\nTM-tuki {:.2f}\n'.format(x[12],x[13])

    def v2_groupstates(self,xx):
        x=self.group_taxes(xx)
        return 'Etuudella olevat {:.2f}\nTyössä {:.2f}\nEläkkeellä {:.2f}\n'.format(x[0],x[1],x[2])

    def count_putki(self,emps=None):
        if emps is None:
            piped = np.reshape(self.empstate[:,4],(self.empstate[:,4].shape[0],1))
            demog2=self.empstats.get_demog()
            putkessa=self.timestep*np.nansum(piped[1:]/self.alive[1:]*demog2[1:])
            return putkessa
        else:
            piped = np.reshape(emps[:,4],(emps[:,4].shape[0],1))
            demog2=self.empstats.get_demog()
            alive = np.sum(emps,axis=1,keepdims=True)
            putkessa=self.timestep*np.nansum(piped[1:]/alive[1:]*demog2[1:])
            return putkessa

    def vector_to_array(self,x):
        return x[:,None]

    def comp_scaled_consumption(self,x0,averaged=False,t0=1,debug=False):
        '''
        Computes discounted actual reward at each time point
        with a given scaling x

        averaged determines whether the value is averaged over time or not
        '''
        x=x0[0]
        u = np.zeros((self.n_time,1))
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
                u[t] += v

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

            u[t] += v*factor
            if np.isnan(u[t]) and debug:
                print('NaN',age,v,v*factor,factor,u[t],income,employment_state)

        u=u/self.n_pop
        w = np.zeros((self.n_time,1))
        w[-1] = u[-1]
        for t in range(self.n_time-2,0,-1):
            w[t] = u[t]+self.gamma*w[t+1]

        if averaged:
            ret = np.mean(w[t0:])
        else:
            ret=w[t0]

        if not np.isfinite(ret):
            if debug:
                print('u',u,'\nw',w)
            u = np.zeros((self.n_time,1))
            for k in range(self.n_pop):
                #g=self.infostats_group[k]
                for t in range(1,self.n_time-1):
                    age=t+self.min_age
                    income=self.infostats_poptulot_netto[t,k]
                    employment_state=self.popempstate[t,k]
                    v,_=self.env.log_utility((1+x)*income,employment_state,age,debug=False) #,g=g,pinkslip=pinkslip)
                    if not np.isfinite(v):
                        v=0
                    u[t] += v
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
                u[t] += v*factor

        return ret

    def comp_presentvalue(self):
        '''
        Computes discounted actual reward at each time point
        with a given scaling x

        averaged determines whether the value is averaged over time or not
        '''
        u = np.zeros((self.n_time,self.n_pop))
        u[self.n_time-1,:] = self.poprewstate[self.n_time-1,:]
        for t in range(self.n_time-2,-1,-1):
            u[t,:] = self.poprewstate[t,:]+self.gamma*u[t+1,:]

        return u

    def comp_initial_npv(self):
        '''
        Computes discounted actual reward at each time point
        with a given scaling x

        averaged determines whether the value is averaged over time or not
        '''
        real=self.comp_presentvalue()
        u = np.mean(real[1,:])
        
        return u

    def comp_palkkatulo_emp(self):
        demog2=self.empstats.get_demog()
        scalex=demog2/self.alive

        ps = np.empty(self.n_emps)
        for k in range(self.n_emps):
            ps[k] = np.sum(self.infostats_palkkatulo_emp[:,k]*scalex[:,0])

        return ps
        
    def comp_realoptimrew(self,discountfactor=None):
        '''
        Computes discounted actual reward at each time point
        '''
        if discountfactor is None:
            discountfactor=self.gamma

        realrew = np.zeros((self.n_time,1))
        for k in range(self.n_pop):
            prew = np.zeros((self.n_time,1))
            prew[-1] = self.poprewstate[-1,k]
            for t in range(self.n_time-2,0,-1):
                prew[t] = discountfactor*prew[t+1]+self.poprewstate[t,k]

            realrew+=prew

        realrew/=self.n_pop
        realrew = np.mean(realrew[1:])

        return realrew

    def comp_cumurewstate(self):
        return np.cumsum(np.mean(self.poprewstate[:,:],axis=1))

    def comp_emp_by_children(self):
        '''
        Laske työttömien osuus lasten lukumäärän mukaan
        '''
        emp_by_children = np.zeros((4,3))
        htv_by_children = np.zeros((4,3))
        for g in range(3):
            if g==0:
                gm = -1
            else:
                gm = g
            mask3 = self.get_childmask(0,gm)
            mask7 = self.get_childmask(1,gm)
            mask18 = self.get_childmask(2,gm)
            mask_none = self.get_childmask(-1,gm)
            emp_by_children[0,g],htv_by_children[0,g] = self.comp_popemployed_ratio(mask3,maxage=50)
            emp_by_children[1,g],htv_by_children[1,g] = self.comp_popemployed_ratio(mask7,maxage=50)
            emp_by_children[2,g],htv_by_children[2,g] = self.comp_popemployed_ratio(mask18,maxage=50)
            emp_by_children[3,g],htv_by_children[3,g] = self.comp_popemployed_ratio(mask_none,maxage=50)

        return emp_by_children,htv_by_children
