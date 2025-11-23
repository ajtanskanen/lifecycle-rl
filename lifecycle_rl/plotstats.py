'''

    plotstats.py

    plots statistics that are used in producing employment statistics for the
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
import fin_benefits
from fin_benefits import Labels
from scipy.stats import gaussian_kde
from .utils import empirical_cdf,print_html,modify_offsettext,add_source,setup_EK_fonts,lineplot,add_label
import timeit


#locale.setlocale(locale.LC_ALL, 'fi_FI')

class PlotStats():
    def __init__(self,stats,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year = 2018,version = 10,
        params = None,gamma = 0.92,lang = 'English'):
        self.version = version
        self.gamma = gamma
        self.params = params
        self.params['n_time'] = n_time
        self.params['n_emps'] = n_emps
        self.episodestats = stats

        self.complex_models = {1,2,3,4,5,6,7,8,9,10,11,104}
        self.recent_models = set([5,6,7,8,9,10,11])
        self.no_groups_models = {0,101}
        self.savings_models = {101,102,103,104}
        self.minimalmodels = set([0,101])
        self.ptmodels = set([5,6,7,8,9,10,11])

        self.lab = Labels()
        self.reset(timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,params = params,lang = lang)

    def set_episodestats(self,stats):
        self.episodestat = stats

    def reset(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,version = None,params = None,lang = None,dynprog = False):
        self.min_age = min_age
        self.max_age = max_age
        self.min_retirementage = min_retirementage
        self.minimal = minimal

        if lang is None:
            self.language = 'English'
        else:
            self.language = lang

        #print('lang',self.language)

        if version is not None:
            self.version = version

        self.setup_labels()

        self.figformat = 'pdf'

        self.n_employment = n_emps
        self.n_time = n_time
        self.timestep = timestep # 0.25 = 3kk askel
        self.inv_timestep = int(np.round(1/self.timestep)) # pitää olla kokonaisluku
        #self.episodestats.n_pop = n_pop
        self.year = year
        self.env = env
        self.reaalinen_palkkojenkasvu = 0.016
        self.palkkakerroin = (0.8*1+0.2*1.0/(1+self.reaalinen_palkkojenkasvu))**self.timestep
        self.elakeindeksi = (0.2*1+0.8*1.0/(1+self.reaalinen_palkkojenkasvu))**self.timestep
        self.dynprog = dynprog

        if self.version in self.no_groups_models:
            self.n_groups = 1
        else:
            self.n_groups = 6
            
        self.empstats = Empstats(year = self.year,max_age = self.max_age,n_groups = self.n_groups,timestep = self.timestep,n_time = self.n_time,
                                min_age = self.min_age,lang=lang)
        
    def plot_various_groups(self,empstate=None,alive=None,figname = None,ax = None):
        if empstate is not None and alive is not None:
            empstate_ratio = 100*empstate/alive
        else:
            empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive

        ratio_label = self.labels['osuus']
        if figname is not None:
            self.plot_states(empstate_ratio,ylabel = ratio_label,stack = True,figname = figname+'_stack',ax=ax)
        else:
            self.plot_states(empstate_ratio,ylabel = ratio_label,stack = True,ax=ax)

        if figname is not None:
            self.plot_states(empstate_ratio,ylabel = ratio_label,start_from = 60,stack = True,figname = figname+'_stack60',ax=ax)
            self.plot_states(empstate_ratio,ylabel = ratio_label,start_from = 57,stack = True,figname = figname+'_stack60',ax=ax)
        else:
            self.plot_states(empstate_ratio,ylabel = ratio_label,start_from = 60,stack = True,ax=ax)
            self.plot_states(empstate_ratio,ylabel = ratio_label,start_from = 57,stack = True,ax=ax)

        if self.version in self.complex_models:
            self.plot_states(empstate_ratio,ylabel = ratio_label,ylimit = 20,stack = False,ax=ax)
            self.plot_states(empstate_ratio,ylabel = ratio_label,unemp = True,stack = False,ax=ax)
            
    def compare_df(self,df1,df2,cctext1 = 'e/v',cctext1_new = None,cctext2 = 'toteuma',cctext2_new = None):
        if cctext1_new is None:
            cctext1_new = cctext1
        if cctext2_new is None:
            cctext2_new = cctext2

        df = df1.copy()
        #df[cctext2_new] = df2[cctext2]
        df[cctext2_new] = df2[cctext2]
        if cctext1_new is not None:
            df.rename(columns = {cctext1: cctext1_new},inplace = True)
        
        df['diff'] = df[cctext1_new]-df[cctext2_new]
        return df

    def compare_against(self,cc = None,cctext = 'toteuma',selftext = ''):
        if self.version in self.complex_models:
            q = self.episodestats.comp_budget(scale = True)
            if cc is None:
                q_stat = self.empstats.stat_budget()
            else:
                q_stat = cc.episodestats.comp_budget(scale = True)

            df1 = pd.DataFrame.from_dict(q,orient = 'index',columns = ['e/y'])
            df2 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = [cctext])
            df = self.compare_df(df1,df2,cctext1 = selftext+'e/y',cctext2 = cctext,cctext2_new = cctext)
            #df = df1.copy()
            #df[cctext] = df2[cctext]
            #df['diff'] = df1['e/v']-df2[cctext]

            print('Rahavirrat skaalattuna väestötasolle')
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.2f"))

            q = self.episodestats.comp_participants(scale = True,lkm = False)
            q_lkm = self.episodestats.comp_participants(scale = True,lkm = True)

            if cc is None:
                q_stat = self.empstats.stat_participants(scale = True,lkm = True)
                q_days = self.empstats.stat_days()
            else:
                q_stat = cc.episodestats.comp_participants(scale = True,lkm = True)
                q_days = cc.episodestats.comp_participants(scale = True,lkm = False)

            df1 = pd.DataFrame.from_dict(q,orient = 'index',columns = ['estimate (py)'])
            df2 = pd.DataFrame.from_dict(q_days,orient = 'index',columns = [cctext+' (py)'])
            df4 = pd.DataFrame.from_dict(q_lkm,orient = 'index',columns = ['estimate (#)'])
            df5 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = [cctext+' (#)'])

            df = df1.copy()
            df[cctext+' (py)'] = df2[cctext+' (py)']
            df['diff (py)'] = df['estimate (py)']-df[cctext+' (py)']

            print('Henkilövuosia töissä väestötasolla')
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))

            df = df4.copy()
            df[cctext+' (#)'] = df5[cctext+' (#)']
            df['diff (#)'] = df['estimate (#)']-df[cctext+' (#)']
            print('Henkilöiden lkm tiloissa väestötasolla')
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
        else:
            q = self.episodestats.comp_participants(scale = True)
            q_stat = self.empstats.stat_participants()
            q_days = self.empstats.stat_days()
            df1 = pd.DataFrame.from_dict(q,orient = 'index',columns = ['estimate (py)'])
            df2 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = ['toteuma'])
            df3 = pd.DataFrame.from_dict(q_days,orient = 'index',columns = ['htv_tot'])

            df = df1.copy()
            df[self.output_labels['toteuma (#)']] = df2['toteuma']
            df[self.output_labels['toteuma (py)']] = df3['htv_tot']
            df[self.output_labels['diff (py)']] = df['estimate (py)']-df[self.output_labels['toteuma (py)']]

            print('Henkilöitä tiloissa skaalattuna väestötasolle')
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))    

    def compare_disab(self,cc = None, xstart = None,xend = None, label1 = 'toteuma',label2='self',figname=None,ax=None):
        # fig,ax = plt.subplots()
        # leg = 'TK Miehet'
        tk1 = np.sum(self.episodestats.gempstate[:,3,:],axis = 1)
        tk2 = np.sum(cc.episodestats.gempstate[:,3,:],axis = 1)
        alive1 = np.zeros((self.episodestats.galive.shape[0],1))
        alive2 = np.zeros((cc.episodestats.galive.shape[0],1))
        alive1[:,0] = np.sum(self.episodestats.galive[:,:],1)
        alive2[:,0] = np.sum(cc.episodestats.galive[:,:],1)

        tk1 = np.reshape(tk1,(self.episodestats.galive.shape[0],1))
        tk2 = np.reshape(tk2,(cc.episodestats.galive.shape[0],1))
        osuus1 = 100*tk1/alive1
        osuus2 = 100*tk2/alive2
        x = np.linspace(self.min_age,self.max_age,self.n_time)

        # ax.plot(x,osuus1,label = label1)
        # ax.plot(x,osuus2,label = label2)
        # ax.set_xlabel(self.labels['age'])
        # ax.set_ylabel(self.labels['ratio'])
        # ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        # if xstart is not None:
        #     ax.set_xlim([xstart,xend])
        # if figname is not None:
        #     plt.savefig(figname+'disab.'+self.figformat, format = self.figformat)

        # plt.show()

        if figname is not None:
            fname = figname+'disab2.'+self.figformat
        else:
            fname = None
        if ax is not None:
            show = False
        else:
            show = True

        lineplot(x,osuus1,y2 = osuus2,xlim = [20,75],#ylim = [0,100],
                 label = label1,label2 = label2,xlabel = self.labels['age'],ylabel = self.labels['ratio'],
                 selite = True,figname = fname,legend_loc='upper left',ax = ax,show = show)

    def render_dist(self,grayscale = False,figname = None,palette_EK = True):

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' 

        if palette_EK:
            csfont,pal = setup_EK_fonts()
        else:
            csfont = {}

        print_html('<h1>Unemp</h1>')
        #self.plot_unemp(unempratio = True,figname = figname)
        #self.plot_unemp(unempratio = False)
        #self.plot_unemp_group()

        #if self.version in self.complex_models:
        #    self.plot_unemp_shares()
        #    self.plot_kassanjasen()
        #    self.plot_pinkslip()

        if self.episodestats.save_pop:
            #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki+putki, no max age',ansiosid = True,tmtuki = True,putki = True,outsider = False)
            self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää',ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 50,figname = figname)

            if self.version in self.complex_models:
                #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu = True,ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 50)
                self.plot_distrib(label = 'Jakauma ansiosidonnainen+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu = False,ansiosid = True,tmtuki = False,putki = True,outsider = False,max_age = 50)
                #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki ilman putkea',ansiosid = True,tmtuki = True,putki = False,outsider = False)
                #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki ilman putkea, max Ikä 50v',ansiosid = True,tmtuki = True,putki = False,outsider = False,max_age = 50)
                self.plot_distrib(label = 'Jakauma tmtuki',ansiosid = False,tmtuki = True,putki = False,outsider = False)
                #self.plot_distrib(label = 'Jakauma työvoiman ulkopuoliset',ansiosid = False,tmtuki = False,putki = False,outsider = True)
                #self.plot_distrib(label = 'Jakauma laaja (ansiosidonnainen+tmtuki+putki+ulkopuoliset)',laaja = True)
            


    def plot_results(self,grayscale = False,figname = None,palette_EK = True):

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' 

        if palette_EK:
            csfont,pal = setup_EK_fonts()
        else:
            csfont = {}

        print_html('<h1>Statistics</h1>')
        
        if self.episodestats.save_pop:
            net1,eqnet1 = self.episodestats.comp_total_netincome()
            print(f'netincome {net1:.2f} eq {eqnet1:.3f}')

        if self.version in self.complex_models:
            self.compare_against()
        else:
            q = self.episodestats.comp_participants(scale = True)
            q_stat = self.empstats.stat_participants(lkm = False)
            q_days = self.empstats.stat_days()
            df1 = pd.DataFrame.from_dict(q,orient = 'index',columns = ['estimate (py)'])
            df2 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = ['toteuma'])
            df3 = pd.DataFrame.from_dict(q_days,orient = 'index',columns = ['htv_tot'])

            df = df1.copy()
            df[self.output_labels['toteuma (#)']] = df2['toteuma']
            df[self.output_labels['toteuma (py)']] = df3['htv_tot']
            df[self.output_labels['diff (py)']] = df['estimate (py)']-df[self.output_labels['toteuma (py)']]

            print('Henkilöitä tiloissa skaalattuna väestötasolle')
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
            #print(tabulate(df, headers = 'keys', tablefmt = 'latex', floatfmt = ",.0f")) # FIXME

        print_html('<h2>Simulation stats</h2>')
        print('Simulated individuals',self.episodestats.n_pop)
        print('Simulated on',self.episodestats.date_time)

        print_html('<h2>Tilastot</h2>')

        #tic = timeit.default_timer()
        emp_htv2 = np.sum(self.episodestats.emp_htv,axis = 1)

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1 = \
            self.episodestats.comp_employed_ratio(self.episodestats.empstate,emp_htv = emp_htv2)
        htv1,tyolliset1,tyottomat1,osatyolliset1,kokotyolliset1,tyollaste1,osata1,kokota1,tyot_aste1 = \
            self.episodestats.comp_unemp_simstats_aggregate(self.episodestats.empstate,scale_time = True,start = 20,end = 64,emp_htv = emp_htv2)

        tyollaste = tyollaste1*100
        tyotaste = self.episodestats.comp_unemp_stats_agg(per_pop = False)*100
        tyovoimatutk_tyollaste = self.empstats.get_tyollisyysaste_tyovoimatutkimus(self.year)
        tyovoimatutk_tytaste = self.empstats.get_tyottomyysaste_tyovoimatutkimus(self.year)
        print('\nSic! Työllisyysaste vastaa työvoimatilaston laskutapaa!')
        print(f'Työllisyysaste 20-64: {tyollaste:.2f}% (työvoimatutkimus {tyovoimatutk_tyollaste:.2f}%)')
        print(f'Työttömyysaste 20-64: {tyotaste:.2f}% (työvoimatutkimus {tyovoimatutk_tytaste:.2f}%)')
        
        gini = self.empstats.get_gini(self.year)
        if self.episodestats.gini_coef is None:
            self.episodestats.gini_coef = self.episodestats.comp_gini()

        print('Gini coefficient is {:.3f} (havainto {:.3f})'.format(self.episodestats.gini_coef,gini))

        print('\nSic! pienituloisuus lasketaan vain aikuisväestöstä!')
        abs_pienituloisuus = 12000
        p50,p60,pt = self.episodestats.comp_pienituloisuus(level = abs_pienituloisuus)
        print('Pienituloisuus 50%: {:.2f}%; 60%: {:.2f}%; abs 1000 e/kk {:.2f}%'.format(100*p50,100*p60,100*pt))

        print_html('<h2>Sovite</h2>')

        discounted_reward,undiscounted_reward = self.episodestats.get_reward(recomp=False)
        
        print('Real discounted reward {}'.format(discounted_reward))
        print('Initial discounted reward {}'.format(self.episodestats.get_initial_reward()))

        print_html('<h2>Työssä</h2>')
        self.plot_emp(figname = figname,palette_EK = palette_EK)
        if self.version in self.complex_models:
            self.plot_gender_emp(figname = figname)
            self.plot_group_emp()
            self.plot_emp_vs_workforce()
            self.plot_workforce()
            if self.episodestats.save_pop:
                print_html('<h2>Tekemätön työ</h2>')
                self.plot_tekematon_tyo()

        print_html('<h2>Osa-aika</h2>')
        if self.version in self.recent_models:
            self.plot_pt_act()
            
        if self.version in self.complex_models:
            self.plot_parttime_ratio(figname = figname)
            
        if self.version in self.complex_models:
            print_html('<h2>Ryhmät</h2>')
            self.plot_outsider()        
            self.plot_various_groups(figname = figname)
            self.plot_group_student()
            ps = self.episodestats.comp_palkkatulo_emp()
            for k in range(self.n_employment):
                print(f'Tilassa {k} palkkasumma on {ps[k]:.2f} e')

        if self.version in self.complex_models:
            print_html('<h2>Lapset ja puolisot</h2>')
            self.plot_spouse()
            self.plot_children()
            self.plot_family()
            self.plot_parents_in_work()
            
        if self.version in self.savings_models:
            print_html('<h2>Säästöt</h2>')
            self.plot_savings()

        print_html('<h2>Tulot</h2>')
        if self.version in self.recent_models:
            self.plot_tulot()
            
        self.plot_sal()

        print_html('<h2>Työttömyys</h2>')
        self.plot_toe()

        if self.episodestats.save_pop:
            print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
            keskikesto = self.episodestats.comp_unemp_durations()
            df = pd.DataFrame.from_dict(keskikesto,orient = 'index',columns = ['0-6 m','6-12 m','12-18 m','18-24 m','yli 24 m'])
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.2f"))

            print('Keskikestot viimeisimmän työttömyysjakson mukaan')
            keskikesto = self.episodestats.comp_unemp_durations_v2()
            df = pd.DataFrame.from_dict(keskikesto,orient = 'index',columns = ['0-6 m','6-12 m','12-18 m','18-24 m','yli 24 m'])
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.2f"))

        self.plot_unemp_after_ra()

        if self.version in self.complex_models:
            print('Lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki()))

        self.plot_unemp(unempratio = True,figname = figname)
        self.plot_unemp(unempratio = False)
        self.plot_unemp_group()

        if self.version in self.complex_models:
            self.plot_unemp_shares()
            self.plot_kassanjasen()
            self.plot_pinkslip()

        if self.episodestats.save_pop:
            #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki+putki, no max age',ansiosid = True,tmtuki = True,putki = True,outsider = False)
            self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää',ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 50,figname = figname)

            if self.version in self.complex_models:
                #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu = True,ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 50)
                self.plot_distrib(label = 'Jakauma ansiosidonnainen+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu = False,ansiosid = True,tmtuki = False,putki = True,outsider = False,max_age = 50)
                #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki ilman putkea',ansiosid = True,tmtuki = True,putki = False,outsider = False)
                #self.plot_distrib(label = 'Jakauma ansiosidonnainen+tmtuki ilman putkea, max Ikä 50v',ansiosid = True,tmtuki = True,putki = False,outsider = False,max_age = 50)
                self.plot_distrib(label = 'Jakauma tmtuki',ansiosid = False,tmtuki = True,putki = False,outsider = False)
                #self.plot_distrib(label = 'Jakauma työvoiman ulkopuoliset',ansiosid = False,tmtuki = False,putki = False,outsider = True)
                #self.plot_distrib(label = 'Jakauma laaja (ansiosidonnainen+tmtuki+putki+ulkopuoliset)',laaja = True)
            
        if self.episodestats.save_pop:
            print_html('<h2>Eläkkeet</h2>')
            self.plot_all_pensions()
            
        print_html('<h2>Työkyvyttömyyseläke</h2>')
        if self.episodestats.save_pop:
            self.plot_disab_lost()
        
        self.plot_disab()

        print_html('<h2>Kuolleet</h2>')
        self.plot_mort()

        if self.version in self.complex_models:
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
            if self.episodestats.save_pop:
                if figname is not None:
                    self.plot_emtr(figname = figname+'_emtr')
                else:
                    self.plot_emtr()
            
            
    def plot_pt_act(self):
        agg_pt,agg_ft,pt,ft,vept,veft,cpt,agg_combi = self.episodestats.comp_ptproportions()

        x = np.linspace(0,2,3)
        plt.bar(x,agg_pt[0,:])
        plt.title('Osa-aika, pt-tila: ave {}'.format(ma.mean(agg_pt[0,:])))
        plt.show()

        plt.bar(x,agg_ft[0,:])
        plt.title('Kokoaika, pt-tila')
        plt.show()

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        plt.stackplot(x,pt[0,:,:].T)
        plt.legend(labels = ['25%','50%','75%'])
        plt.title('Osa-aika, pt-tila')
        plt.show()
        plt.stackplot(x,ft[0,:,:].T)
        plt.legend(labels = ['100%','125%','150%'])
        plt.title('Kokoaika, pt-tila')
        plt.show()
        plt.stackplot(x,vept[0,:,:].T)
        plt.legend(labels = ['25%','50%','75%'])
        plt.title('Ve+Osa-aika, pt-tila')
        plt.show()
        plt.stackplot(x,veft[0,:,:].T)
        plt.legend(labels = ['100%','125%','150%'])
        plt.title('Ve+Kokoaika, pt-tila')
        plt.show()

        x = np.linspace(0,2,3)
        plt.bar(x,agg_pt[2,:])
        plt.title('Osa-aika naiset, pt-tila: ave {}'.format(ma.mean(agg_pt[2,:])))
        plt.show()
        print('osa-aika','miehet',agg_pt[2,:])
        print('naiset',agg_pt[1,:])
        print('yht',agg_pt[0,:])

        plt.bar(x,agg_pt[1,:])
        plt.title('Osa-aika miehet, pt-tila: ave {}'.format(ma.mean(agg_pt[1,:])))
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(8,48,6)
        plt.bar(x,100*agg_combi[0,:],width=7)
        ax.set_xticks([8,16,24,32,40,48])
        ax.set_ylabel(self.labels['osuus'])
        ax.set_xlabel(self.labels['Työaika [h]'])
        plt.show()

        x = np.linspace(0,2,3)
        plt.bar(x,agg_ft[1,:])
        plt.title('Kokoaika miehet, pt-tila: ave {}'.format(ma.mean(agg_ft[1,:])))
        plt.show()

        x = np.linspace(0,2,3)
        plt.bar(x,agg_ft[2,:])
        plt.title('Kokoaika naiset, pt-tila: ave {}'.format(ma.mean(agg_ft[2,:])))
        plt.show()
        print('kokoaika','miehet',agg_ft[2,:])
        print('naiset',agg_ft[1,:])
        print('yht',agg_ft[0,:])

    def plot_unemp_after_ra(self):
        self.plot_states(self.episodestats.stat_unemp_after_ra,ylabel = 'Unemp after ret.age',stack = False,start_from = 60,end_at = 70)

    def plot_disab_lost(self):
        w1,w2,n_tk = self.episodestats.comp_tkstats()
        print(f'Työkyvyttömyyseläkkeisiin menetetty palkkasumma {w1:,.2f} ja työpanoksen arvo {w2:,.2f}')

    def plot_disab(self):
        self.plot_group_disab()
        self.plot_group_disab(xstart = 60,xend = 67)

    def plot_mort(self):
        #w1,w2,n_tk = self.episodestats.comp_tkstats()
        self.plot_group_mort()

    def plot_compare_disab_menetys(self):
        w1,wplt = self.episodestats.comp_potential_palkkasumma(grouped = True,full = True,include_kela = False)
        k = 3
        w2 = 2.1*w1[k] # kerroin 2,1 muuttaa palkan työpanoksen arvoksi
        print(f'Tilaan {k} menetetty palkkasumma {w1[k]:,.2f} ja työpanoksen arvo {w2:,.2f}')
        print(np.mean(wplt[:,k]))

        dw1,dwplt = self.episodestats.comp_potential_disab(include_kela = False)
        w2 = 2.1*dw1 # kerroin 2,1 muuttaa palkan työpanoksen arvoksi
        print(f'Tilaan Disab menetetty tulevan ajan palkkasumma {dw1:,.2f} ja työpanoksen arvo {w2:,.2f}')
        print(np.mean(dwplt))

        plt.plot(dwplt[:],label = '5y')
        plt.plot(wplt[:,k],label = 'pot')
        plt.show()

    def plot_tekematon_tyo(self):
        self.plot_compare_disab_menetys()
        w1,wplt = self.episodestats.comp_potential_palkkasumma(grouped = True,full = True,include_kela = True)
        wplt2 = wplt.copy()
        for k in range(15):
            w2 = 2.1*w1[k] # kerroin 2,1 muuttaa palkan työpanoksen arvoksi
            if k in [0,2,3,4,5,6,7,11,12,13,14,16]:
                print(f'Tilaan {k} menetetty palkkasumma {w1[k]:,.2f} ja työpanoksen arvo {w2:,.2f}')
            else:
                print(f'Tilan {k} palkkasumma {w1[k]:,.2f} ja työpanoksen arvo {w2:,.2f}')

        dw1,dwplt = self.episodestats.comp_potential_disab(include_kela = False)
        w2 = 2.1*dw1 # kerroin 2,1 muuttaa palkan työpanoksen arvoksi
        print(f'Tilaan Disab menetetty tulevan ajan palkkasumma {dw1:,.2f} ja työpanoksen arvo {w2:,.2f}')

        ps = np.sum(w1[[1,8,9,10]])
        tpa = 2.1*ps
        nops = np.sum(w1[[0,2,3,4,5,6,7,11,12,13,14,16]])
        notpa = 2.1*nops
        print(f'Palkkasumma {ps:,.2f} ja työpanoksen arvo {tpa:,.2f}')
        print(f'Menetetty palkkasumma {nops:,.2f} ja työpanoksen arvo {notpa:,.2f}')
        wplt[:,[1,8,9,10]] = 0
        self.plot_states(wplt,ylabel = self.labels['Menetetty palkkasumma'],stack = True,ymaxlim = np.max(np.sum(wplt,axis = 1)))
        wplt = wplt/np.sum(wplt,axis = 1,keepdims = True)*100
        self.plot_states(wplt,ylabel = self.labels['Menetetty palkkasumma %'],stack = True)
        wplt[:,[0,2,4,5,6,7,11,12,13,16]] = 0
        self.plot_states(wplt,ylabel = self.labels['Menetetty palkkasumma %'],stack = True,ymaxlim = np.max(np.nansum(wplt,axis = 1)))
        wplt = wplt/np.sum(wplt,axis = 1,keepdims = True)*100
        self.plot_states(wplt,ylabel = self.labels['Menetetty palkkasumma %'],stack = True)
        wplt2[:,[1,2,3,5,6,7,8,9,10,11,12,14,16]] = 0
        self.plot_states(wplt2,ylabel = self.labels['Menetetty palkkasumma'],stack = True,ymaxlim = np.max(np.nansum(wplt2,axis = 1)))
        wplt2 = wplt2/np.sum(wplt2,axis = 1,keepdims = True)*100
        self.plot_states(wplt2,ylabel = self.labels['Menetetty palkkasumma %'],stack = True)

    def plot_all_pensions(self):
        #self.compare_takuu_kansanelake()

        #alivemask = (self.episodestats.popempstate== self.env.get_mortstate()) # pois kuolleet
        alivemask = self.episodestats.get_alivemask()
        kemask = (self.episodestats.infostats_pop_kansanelake<0.1)
        kemask = ma.mask_or(kemask,alivemask)
        temask = (self.episodestats.infostats_pop_kansanelake>0.1) # pois kansaneläkkeen saajat
        temask = ma.mask_or(temask,alivemask) 
        notemask = (self.episodestats.infostats_pop_tyoelake>10.0) # pois kansaneläkkeen saajat
        notemask = ma.mask_or(notemask,alivemask)
        self.plot_pensions()
        self.plot_pension_stats(self.episodestats.infostats_pop_paidpension/self.timestep,65,'kokoeläke ilman kuolleita',mask = alivemask)
        self.plot_pension_stats(self.episodestats.infostats_pop_tyoelake/self.timestep,65,'työeläke')
        self.plot_pension_stats(self.episodestats.infostats_paid_tyel_pension/self.timestep,65,'työeläkemaksun vastine')
        self.plot_pension_stats(self.episodestats.infostats_paid_tyel_pension/self.timestep,65,'työeläkemaksun vastine, vain työeläke',mask = temask)
        self.plot_pension_stats(self.episodestats.infostats_pop_tyoelake/self.timestep,65,'vain työeläke',mask = temask)
        self.plot_pension_stats(self.episodestats.infostats_pop_kansanelake/self.timestep,65,'kansanelake kaikki',max_pen = 10_000,plot_ke = True)
        self.plot_pension_stats(self.episodestats.infostats_pop_kansanelake/self.timestep,65,'kansanelake>0',max_pen = 10_000,mask = kemask,plot_ke = True)
        self.plot_pension_stats(self.episodestats.infostats_pop_kansanelake/self.timestep,65,'kansaneläke, ei työeläkettä',max_pen = 10_000,mask = notemask,plot_ke = True)
        self.plot_pension_stats(self.episodestats.infostats_pop_tyoelake/self.timestep,65,'työeläke, jos kansanelake>0',max_pen = 20_000,mask = kemask)
        if self.episodestats.save_pop:
            self.plot_pension_stats(self.episodestats.infostats_pop_pension,60,'tulevat eläkkeet')
            self.plot_pension_stats(self.episodestats.infostats_pop_pension,60,'tulevat eläkkeet, vain elossa',mask = alivemask)
        self.plot_pension_time()

    def setup_labels(self):
        self.labels = self.lab.get_labels(self.language)
        self.output_labels = self.lab.get_output_labels(self.language)

    def map_age(self,age,start_zero = False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)

    def map_t_to_age(self,t):
        return self.min_age+t/self.inv_timestep

    def episodestats_exit(self):
        plt.close(self.episode_fig)

#     def plot_ratiostats(self,t):
#         '''
#         Tee kuvia tuloksista
#         '''
#         x = np.linspace(self.min_age,self.max_age,self.n_time)
#         fig,ax = plt.subplots()
#         ax.set_xlabel('palkat')
#         ax.set_ylabel('freq')
#         ax.hist(self.episodestats.infostats_pop_wage[t,:])
#         plt.show()
#         fig,ax = plt.subplots()
#         ax.set_xlabel('aika')
#         ax.set_ylabel('palkat')
#         meansal = np.mean(self.episodestats.infostats_pop_wage,axis = 1)
#         stdsal = np.std(self.episodestats.infostats_pop_wage,axis = 1)
#         ax.plot(x,meansal)
#         ax.plot(x,meansal+stdsal)
#         ax.plot(x,meansal-stdsal)
#         plt.show()

    def plot_empdistribs(self,emp_distrib):
        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['työsuhteen pituus [v]'])
        ax.set_ylabel('freq')
        #ax.set_yscale('log')
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(0,max_time,nn_time)
        scaled,x2 = np.histogram(emp_distrib,x)
        scaled = scaled/np.sum(emp_distrib)
        #ax.hist(emp_distrib)
        ax.bar(x2[1:-1],scaled[1:],align = 'center')
        plt.show()
        
    def plot_alive(self):
        alive = self.episodestats.alive/self.episodestats.n_pop
        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Alive [%]')
        nn_time = int(np.ceil((self.max_age-self.min_age)*self.inv_timestep))+1
        x = np.linspace(self.min_age,self.max_age,nn_time)
        ax.plot(x[1:],alive[1:]*100)
        plt.show()
        
    def plot_pension_time(self):
        self.plot_y(self.episodestats.infostats_tyoelake,label = 'työeläke',
            y2 = self.episodestats.infostats_kansanelake,label2 = 'kansaneläke',
            ylabel = 'eläke [e/v]',
            start_from = 60,end_at = 70,show_legend = True)

        demog2 = self.empstats.get_demog()
        scalex = demog2/self.episodestats.alive #n_pop
        tyoelake_meno = self.episodestats.infostats_tyoelake*scalex
        kansanelake_meno = self.episodestats.infostats_kansanelake*scalex
        print('työeläkemeno alle 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(tyoelake_meno[:self.map_age(63)]),1_807.1))
        print('työeläkemeno yli 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(tyoelake_meno[self.map_age(63):]),24_227.2))
        print('kansaneläkemeno alle 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(kansanelake_meno[:self.map_age(63)]),679.6))
        print('kansaneläkemeno yli 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(kansanelake_meno[self.map_age(63):]),1_419.9))
        
        self.plot_y(self.episodestats.infostats_kansanelake,label = 'kansaneläke',
            ylabel = 'kansaneläke [e/v]',
            start_from = 60,end_at = 70,show_legend = True)

        self.plot_y(self.episodestats.infostats_kansanelake/self.episodestats.infostats_tyoelake*100,label = 'suhde',
            ylabel = 'kansaneläke/työeläke [%]',
            start_from = 60,end_at = 70,show_legend = True)

    def plot_pension_stats(self,pd,age,label,max_pen = 60_000,mask = None,plot_ke = False):
        fig,ax = plt.subplots()
        if mask is None:
            pens_distrib = ma.array(pd[self.map_age(age),:])
        else:
            pens_distrib = ma.array(pd[self.map_age(age),:],mask = mask[self.map_age(age),:])
        
        ax.set_xlabel('eläke [e/v]')
        ax.set_ylabel('freq')
        #ax.set_yscale('log')
        x = np.linspace(0,max_pen,51)
        scaled,x2 = np.histogram(pens_distrib.compressed(),x)
        
        scaled = scaled/np.sum(pens_distrib)
        ax.plot(x2[1:],scaled)
        axvcolor = 'gray'
        lstyle = '--'
        ka = np.mean(pens_distrib)
        plt.axvline(x = ka,ls = lstyle,color = axvcolor)
        if plot_ke:
            arv = self.env.ben.laske_kansanelake(66,0/12,1,disability = True)*12
            plt.axvline(x = arv,ls = lstyle,color = 'red')
            plt.axvline(x = 0.5*arv,ls = lstyle,color = 'pink')
            
        plt.title(f'{label} at age {age}, mean {ka:.0f}')
        plt.show()

    def plot_compare_empdistribs(self,emp_distrib,emp_distrib2,label2 = 'vaihtoehto',label1 = ''):
        fig,ax = plt.subplots()
        ax.set_xlabel('työsuhteen pituus [v]')
        ax.set_ylabel(self.labels['probability'])
        ax.set_yscale('log')
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(0,max_time,nn_time)
        scaled,x2 = np.histogram(emp_distrib,x)
        scaled = scaled/np.sum(emp_distrib)
        x = np.linspace(0,max_time,nn_time)
        scaled3,x3 = np.histogram(emp_distrib2,x)
        scaled3 = scaled3/np.sum(emp_distrib2)

        ax.plot(x3[1:-1],scaled3[1:],label = label1)
        ax.plot(x2[1:-1],scaled[1:],label = label2)
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.show()

    def plot_vlines_unemp(self,point = 0):
        axvcolor = 'gray'
        lstyle = '--'
        plt.axvline(x = 300/(12*21.5),ls = lstyle,color = axvcolor)
        plt.text(310/(12*21.5),point,'300',rotation = 90)
        plt.axvline(x = 400/(12*21.5),ls = lstyle,color = axvcolor)
        plt.text(410/(12*21.5),point,'400',rotation = 90)
        plt.axvline(x = 500/(12*21.5),ls = lstyle,color = axvcolor)
        plt.text(510/(12*21.5),point,'500',rotation = 90)

    def plot_tyolldistribs(self,emp_distrib,tyoll_distrib,tyollistyneet = True,max = 10,figname = None):
        max_time = 55
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(0,max_time,nn_time)
        scaled0,x0 = np.histogram(emp_distrib,x)
        if not tyollistyneet:
            scaled = scaled0
            x2 = x0
        else:
            scaled,x2 = np.histogram(tyoll_distrib,x)
        jaljella = np.cumsum(scaled0[::-1])[::-1] # jäljellä olevien kumulatiivinen summa
        scaled = scaled/jaljella

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['työttömyysjakson pituus [v]'])
        if tyollistyneet:
            ax.set_ylabel(self.labels['työllistyneiden osuus'])
            point = 0.5
        else:
            ax.set_ylabel(self.labels['pois siirtyneiden osuus'])
            point = 0.9
        self.plot_vlines_unemp(point)
        ax.plot(x2[1:-1],scaled[1:])
        #ax.bar(x2[1:-1],scaled[1:],align = 'center',width = self.timestep)
        plt.xlim(0,max)
        if figname is not None:
            plt.savefig(figname+'tyollistyneetdistrib.'+self.figformat, format = self.figformat)

        plt.show()

    def plot_unempbasis_distrib(self,basis_distrib1,figname = None,basis_distrib2 = None,label1 = 'LCM 2024',label2 = ' ',ax=None,fig=None,vrt=False):
        '''
        Tulostaa työllistymisaikajakauman
        '''
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.array([0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,10000])*12
        scaled1,x1 = np.histogram(basis_distrib1,x)
        scaled1 = scaled1 / np.sum(scaled1)

        y = np.empty((2027,17))
        y[2021] = np.array([0.0,0.0100,0.0800,0.2000,0.2400,0.1900,0.1200,0.0600,0.0400,0.0200,0.0100,0.0100,0.0100,0.0,0.0,0.0,0.0100])
        y[2022] = np.array([0.0,0.0100,0.0800,0.2100,0.2400,0.1800,0.1100,0.0600,0.0400,0.0200,0.0100,0.0100,0.0100,0.0,0.0,0.0,0.0100])
        y[2023] = np.array([0.0,0.0,0.0600,0.1600,0.2200,0.1900,0.1300,0.0800,0.0500,0.0300,0.0200,0.0100,0.0100,0.0100,0.0,0.0,0.0100])
        y[2024] = np.array([0.0,0.0,0.0500,0.1300,0.2100,0.2000,0.1500,0.0900,0.0500,0.0400,0.0200,0.0100,0.0100,0.0100,0.0,0.0,0.0100])

        if basis_distrib2 is not None:
            scaled2,x2 = np.histogram(basis_distrib2,x)

        if fig is None:
            fig2,ax = plt.subplots()

        #ax.set_xlabel(self.labels['palkka [v]'])
        w = 12 * 250
        ax.set_xlabel('Salary [e/y]')
        #point = 0.6
        #self.plot_vlines_unemp(point)
        #ax.plot(x1[1:-1],scaled1[1:],label = label1)
        ax.bar(x1[:-1]+0.5*w, scaled1[:],label=label1,width=w)
        #if self.year in y:
        #ax.plot(x[1:],y[2024],label='TYJ')
        ax.bar(x[:-1]-0.5*w, y[2024],label='Data',width=w)
        if basis_distrib2 is not None:
            ax.bar(x2[:-1], scaled2[:],label=label2,width=w)
            #ax.plot(x2[1:-1],scaled2[1:],label = label2)
            ax.legend(frameon=False)
        ax.legend() #bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        
        ax.set_ylabel(self.labels['osuus'])

        #plt.xlim(0,max)
        #plt.ylim(0,0.8)
        if figname is not None:
            plt.savefig(figname+'tyolldistribs_v1.'+self.figformat, format = self.figformat)
        if fig is None:
            plt.show()


    def plot_tyolldistribs_both(self,emp_distrib1,tyoll_distrib1,max = 10,figname = None,emp_distrib2 = None,
                                tyoll_distrib2 = None,label1 = '',label2 = ' ',ax=None,fig=None,kyyra=False,
                                kuva1=True,kuva2=True,kuva3=True,kuva4=False):
        '''
        Tulostaa työllistymisaikajakauman
        '''
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(0,max_time,nn_time)
        scaled0_1,x0_1 = np.histogram(emp_distrib1,x)
        scaled1 = scaled0_1
        scaled_tyoll1,x2_1 = np.histogram(tyoll_distrib1,x)

        jaljella = np.cumsum(scaled0_1[::-1])[::-1] # jäljellä olevien summa
        scaled1 = scaled1/jaljella
        jaljella_tyoll1 = np.cumsum(scaled0_1[::-1])[::-1] # jäljellä olevien summa
        scaled_tyoll1 = scaled_tyoll1/jaljella_tyoll1

        # exit rate kyyrä ja pesola (2019)
        x_kyyra = (np.array([ 13.,  26.,  39.,  52.,  65.,  78.,  91., 104., 117., 130.]))*5/21.5/12
        y_kyyra =np.array([0.4839454 , 0.38065915, 0.30236521, 0.28661407, 0.25400715, 0.22479255, 0.23189427, 0.27568799, 0.15755385, 0.11960807])

        # exit rate Korplea
        x_korpela = (np.array([ 13.,  26.,  39.,  52.,  65.,  78.,  91., 104., 117., 130.]))*5/21.5/12
        y_korpela =np.array([0.49958705, 0.41401094, 0.27543074, 0.17505567, 0.20015082,0.26644122, 0.31987234, 0.16057378, 0.18221992, 0.21903784])

        # job rate Korplea
        x_korpela2 = (np.array([ 13.,  26.,  39.,  52.,  65.,  78.,  91., 104., 117., 130.]))*5/21.5/12
        y_korpela2 =np.array([0.31687424, 0.25103138, 0.21882637, 0.12202941, 0.10988931, 0.09760329, 0.10500084, 0.11543548, 0.11946684, 0.0853002 ])

        if emp_distrib2 is not None:
            scaled0_2,x0_2 = np.histogram(emp_distrib2,x)
            scaled2 = scaled0_2
            scaled_tyoll2,x2_2 = np.histogram(tyoll_distrib2,x)
            jaljella2 = np.cumsum(scaled0_2[::-1])[::-1] # jäljellä olevien summa
            scaled2 = scaled2/jaljella
            jaljella_tyoll2 = np.cumsum(scaled0_2[::-1])[::-1] # jäljellä olevien summa
            scaled_tyoll2 = scaled_tyoll2/jaljella_tyoll2
        
        if kuva1:
            if fig is None:
                fig2,ax = plt.subplots()
            ax.set_xlabel(self.labels['työttömyysjakson pituus [v]'])
            point = 0.6
            self.plot_vlines_unemp(point)
            ax.plot(x2_1[1:-1],scaled_tyoll1[1:],label = label1)
            if emp_distrib2 is not None:
                ax.plot(x2_2[1:-1],scaled_tyoll2[1:],label = label2)
                ax.legend(frameon=False)
            #ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            
            ax.set_ylabel(self.labels['työllistyneiden osuus'])

            plt.xlim(0,max)
            plt.ylim(0,0.8)
            if figname is not None:
                plt.savefig(figname+'tyolldistribs_v1.'+self.figformat, format = self.figformat)
            if fig is None:
                plt.show()

        if kuva2:
            if fig is None:
                fig2,ax = plt.subplots()
            ax.set_xlabel(self.labels['työttömyysjakson pituus [v]'])
            point = 0.6
            self.plot_vlines_unemp(point)
            ax.plot(x2_1[1:-1],scaled1[1:],label = label1)
            if emp_distrib2 is not None:
                ax.plot(x2_2[1:-1],scaled2[1:],label = label2)
                ax.legend(frameon=False)
            #ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            ax.set_ylabel(self.labels['pois siirtyneiden osuus'])

            plt.xlim(0,max)
            plt.ylim(0,0.8)
            if figname is not None:
                plt.savefig(figname+'tyolldistribs_v2.'+self.figformat, format = self.figformat)
            if fig is None:
                plt.show()

        if kuva3:
            if fig is None:
                fig2,ax = plt.subplots()
            ax.set_xlabel(self.labels['työttömyysjakson pituus [v]'])
            point = 0.6
            self.plot_vlines_unemp(point)
            #ax.plot(x2_1[1:-1],scaled1[1:],label = label1)
            ax.plot(x2_1[1:-1],scaled_tyoll1[1:],label = label1)

            ax.legend(frameon=False)
            #ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            ax.set_ylabel(self.labels['työllistyneiden osuus'])

            if kyyra:
                ax.plot(x_kyyra,y_kyyra,'--',label='Kyyrä & Pesola')
                #ax.plot(x_korpela,y_korpela,'-.',label='Korpela')

            if emp_distrib2 is not None:
                ax.plot(x2_2[1:-1],scaled_tyoll2[1:],label = label2)
                ax.legend(frameon=False)

            plt.xlim(0,max)
            plt.ylim(0,0.8)
            if figname is not None:
                plt.savefig(figname+'tyolldistribs_v3.'+self.figformat, format = self.figformat)
            if fig is None:
                plt.show()

        if kuva4:
            if fig is None:
                fig2,ax = plt.subplots()
            ax.set_xlabel(self.labels['työttömyysjakson pituus [v]'])
            point = 0.6
            self.plot_vlines_unemp(point)
            ax.plot(x2_1[1:-1],scaled_tyoll1[1:],label = label1)            
            ax.plot(x2_1[1:-1],scaled1[1:],label = label2)
            ax.set_ylabel(self.labels['osuus'])

            if kyyra:
                ax.plot(x_kyyra,y_kyyra,'--',label='Kyyrä & Pesola')
                #ax.plot(x_korpela,y_korpela,'-.',label='Korpela exit')
                ax.plot(x_korpela2,y_korpela2,'-.',label='Korpela job')

            ax.legend(frameon=False,loc='lower left')
            plt.xlim(0,max)
            plt.ylim(0,0.8)
            if figname is not None:
                plt.savefig(figname+'tyolldistribs_v4.'+self.figformat, format = self.figformat)
            if fig is None:
                plt.show()      


    def plot_tyolldistribs_both_bu(self,emp_distrib,tyoll_distrib,max = 2,emp_distrib2 = None,tyoll_distrib2 = None,label1 = '',label2 = ' 2'):
        '''
        plots proportion of those moved out of unemployment
        '''
        max_time = 4
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(-max_time,0,nn_time)
        scaled0_1,x0_1 = np.histogram(emp_distrib1,x)
        scaled1 = scaled0_1
        scaled_tyoll1,x2_1 = np.histogram(tyoll_distrib1,x)

        jaljella1 = np.cumsum(scaled0_1[::-1])[::-1] # jäljellä olevien summa
        scaled1 = scaled1/jaljella1
        jaljella_tyoll1 = np.cumsum(scaled0_1[::-1])[::-1] # jäljellä olevien summa
        scaled_tyoll1 = scaled_tyoll1/jaljella_tyoll1

        if emp_distrib2 is not None:
            scaled0_2,x0_2 = np.histogram(emp_distrib2,x)
            scaled2 = scaled0_2
            scaled_tyoll,x2_2 = np.histogram(tyoll_distrib,x)

            jaljella2 = np.cumsum(scaled0_2[::-1])[::-1] # jäljellä olevien summa
            scaled2 = scaled2/jaljella2
            jaljella_tyoll2 = np.cumsum(scaled0_2[::-1])[::-1] # jäljellä olevien summa
            scaled_tyoll2 = scaled_tyoll2/jaljella_tyoll2

        fig,ax = plt.subplots()
        ax.set_xlabel('aika ennen ansiopäivärahaoikeuden loppua [v]')
        point = 0.6
        #self.plot_vlines_unemp(point)
        ax.plot(x2_1[1:-1],scaled1[1:],label = 'pois siirtyneiden osuus'+label1)
        ax.plot(x2_1[1:-1],scaled_tyoll1[1:],label = 'työllistyneiden osuus'+label1)

        if emp_distrib2 is not None:
            ax.plot(x2_2[1:-1],scaled2[1:],label = 'pois siirtyneiden osuus'+label2)
            ax.plot(x2_2[1:-1],scaled_tyoll2[1:],label = 'työllistyneiden osuus'+label2)

        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        ax.set_ylabel('pois siirtyneiden osuus')

        plt.xlim(-max,0)
        #plt.ylim(0,0.8)
        plt.show()

    def plot_compare_tyolldistribs(self,emp_distrib1,tyoll_distrib1,emp_distrib2,
                tyoll_distrib2,tyollistyneet = True,max = 3,label1 = 'perus',label2 = 'vaihtoehto',
                figname = None):
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(0,max_time,nn_time)

        # data1
        scaled01,x0 = np.histogram(emp_distrib1,x)
        if not tyollistyneet:
            scaled1 = scaled01
            x1 = x0
        else:
            scaled1,x1 = np.histogram(tyoll_distrib1,x)
        jaljella1 = np.cumsum(scaled01[::-1])[::-1] # jäljellä olevien summa
        scaled1 = scaled1/jaljella1

        # data2
        scaled02,x0 = np.histogram(emp_distrib2,x)
        if not tyollistyneet:
            scaled2 = scaled02
            x2 = x0
        else:
            scaled2,x2 = np.histogram(tyoll_distrib2,x)

        jaljella2 = np.cumsum(scaled02[::-1])[::-1] # jäljellä olevien summa
        scaled2 = scaled2/jaljella2

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['työttömyysjakson pituus [v]'])
        if tyollistyneet:
            ax.set_ylabel(self.labels['työllistyneiden osuus'])
        else:
            ax.set_ylabel(self.labels['pois siirtyneiden osuus'])
        self.plot_vlines_unemp()
        ax.plot(x2[1:-1],scaled2[1:],label = label2)
        ax.plot(x1[1:-1],scaled1[1:],label = label1)
        #ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        ax.legend()
        plt.xlim(0,max)
        plt.gca().set_ylim(bottom = 0)
        if figname is not None:
            plt.savefig(figname+'comp_tyollistyneetdistrib.'+self.figformat, format = self.figformat)

        plt.show()

    def plot_unempdistribs(self,unemp_distrib,max = 2.5,figname = None,miny = None,maxy = None,unemp_distrib2 = None,label1 = '',label2 = '',fig=None,ax=None):
        '''
        plots distribution of unemployment durations
        '''
        if fig is None:
            fig2,ax = plt.subplots()

        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(0,max_time,nn_time)

        scaled1,xv1 = np.histogram(unemp_distrib,bins = x,density = True)
        density1 = scaled1 * (x[1]-x[0])
        #scaled = scaled/np.sum(unemp_distrib)

        if unemp_distrib2 is not None:
            scaled2,xv2 = np.histogram(unemp_distrib2,bins = x,density = True)
            density2 = scaled2 * (x[1]-x[0])
            #scaled2 = scaled2/np.sum(unemp_distrib2)
        else:
            scaled2 = None

        self.plot_vlines_unemp(0.6)
        ax.set_xlabel(self.labels['unemp duration'])
        ax.set_ylabel(self.labels['probability'])

        ax.plot(xv1[1:-1],density1[1:])
        if unemp_distrib2 is not None:
            ax.plot(xv2[1:-1],density2[1:])

        ax.set_yscale('log')
        plt.xlim(0,max)
        plt.gca().set_ylim(bottom = 0)
        if miny is not None:
            plt.ylim(miny,maxy)
        if figname is not None:
            plt.savefig(figname+'unempdistribs.'+self.figformat, format = self.figformat)

        if fig is None:
            plt.show()
        
    def plot_saldist(self,t = 0,sum = False,all = False,n = 10,bins = 30):
        if all:
            fig,ax = plt.subplots()
            for t in range(1,self.n_time-1,5):
                scaled,x = np.histogram(self.episodestats.infostats_pop_wage[t,:]/self.timestep,bins = bins)
                x2 = 0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label = t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x = np.histogram(np.sum(self.episodestats.infostats_pop_wage/self.timestep,axis = 0),bins = bins)
                x2 = 0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax = plt.subplots()
                for t1 in range(t,t+n,1):
                    scaled,x = np.histogram(self.episodestats.infostats_pop_wage[t1,:]/self.timestep,bins = bins)
                    x2 = 0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label = t1)
                plt.legend()
                plt.show()

    def test_salaries(self):

        def kuva(sal,ika,m,p,palkka):
            plt.hist(sal[:m],bins = 50,density = True)
            ave = np.mean(sal[:m])/12
            palave = np.sum(palkka*p)/12/np.sum(palkka)
            plt.title('{}: ave {:,.2f} vs {:,.2f}'.format(ika,ave,palave))
            plt.plot(p,palkka/sum(palkka)/2000)
            plt.show()

        def kuva2(sal,ika,m):
            plt.hist(sal[:m],bins = 50,density = True)
            ave = np.mean(sal[:m])/12
            plt.title('{}: ave {}'.format(ika,ave))
            plt.show()

        def cdf_kuva(sal,ika,m,p,palkka):
            pal = np.cumsum(palkka)/np.sum(palkka)
            m_x,m_y = empirical_cdf(sal[:m])
            plt.plot(m_x,m_y,label = 'malli')
            plt.plot(p,pal,label = 'havainto')
            plt.title('age {}'.format(ika))
            plt.legend()
            plt.show()
            plt.loglog(m_x,m_y,label = 'malli')
            plt.loglog(p,pal,label = 'havainto')
            plt.title('age {}'.format(ika))
            plt.legend()
            plt.show()

        if not self.episodestats.save_pop:
            print('test_salaries: not enough data (save_pop = False)')
            return

        n = self.episodestats.n_pop
        
        # 2018
        palkat_ika_miehet = 12.5*np.array([2039.15,2256.56,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
        palkat_ika_naiset = 12.5*np.array([2058.55,2166.68,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])

        #palkat_ika_miehet = 12.5*np.array([2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
        #palkat_ika_naiset = 12.5*np.array([2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        #g_r = [0.77,1.0,1.23]
        data_range = np.arange(self.min_age,self.max_age)

        sal20 = np.zeros((n,1))
        sal25 = np.zeros((n,1))
        sal30 = np.zeros((n,1))
        sal40 = np.zeros((n,1))
        sal50 = np.zeros((n,1))
        sal60 = np.zeros((n,1))
        sal65 = np.zeros((n,1))
        sal = np.zeros((n,self.max_age))

        p = np.arange(700,17500,100)*12.5
        palkka20 = np.array([10.3,5.6,4.5,14.2,7.1,9.1,22.8,22.1,68.9,160.3,421.6,445.9,501.5,592.2,564.5,531.9,534.4,431.2,373.8,320.3,214.3,151.4,82.3,138.0,55.6,61.5,45.2,19.4,32.9,13.1,9.6,7.4,12.3,12.5,11.5,5.3,2.4,1.6,1.2,1.2,14.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        palkka25 = np.array([12.4,11.3,30.2,4.3,28.5,20.3,22.5,23.7,83.3,193.0,407.9,535.0,926.5,1177.1,1540.9,1526.4,1670.2,1898.3,1538.8,1431.5,1267.9,1194.8,1096.3,872.6,701.3,619.0,557.2,465.8,284.3,291.4,197.1,194.4,145.0,116.7,88.7,114.0,56.9,57.3,55.0,25.2,24.4,20.1,25.2,37.3,41.4,22.6,14.1,9.4,6.3,7.5,8.1,9.0,4.0,3.4,5.4,4.1,5.2,1.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        palkka30 = np.array([1.0,2.0,3.0,8.5,12.1,22.9,15.8,21.8,52.3,98.2,295.3,392.8,646.7,951.4,1240.5,1364.5,1486.1,1965.2,1908.9,1729.5,1584.8,1460.6,1391.6,1551.9,1287.6,1379.0,1205.6,1003.6,1051.6,769.9,680.5,601.2,552.0,548.3,404.5,371.0,332.7,250.0,278.2,202.2,204.4,149.8,176.7,149.0,119.6,76.8,71.4,56.3,75.9,76.8,58.2,50.2,46.8,48.9,30.1,32.2,28.8,31.1,45.5,41.2,36.5,18.1,11.6,8.5,10.2,4.3,13.5,12.3,4.9,13.9,5.4,5.9,7.4,14.1,9.6,8.4,11.5,0.0,3.3,9.0,5.2,5.0,3.1,7.4,2.0,4.0,4.1,14.0,2.0,3.0,1.0,0.0,6.2,2.0,1.2,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        palkka50 = np.array([2.0,3.1,2.4,3.9,1.0,1.0,11.4,30.1,29.3,34.3,231.9,341.9,514.4,724.0,1076.8,1345.2,1703.0,1545.8,1704.0,1856.1,1805.4,1608.1,1450.0,1391.4,1338.5,1173.2,1186.3,1024.8,1105.6,963.0,953.0,893.7,899.8,879.5,857.0,681.5,650.5,579.2,676.8,498.0,477.5,444.3,409.1,429.0,340.5,297.2,243.1,322.5,297.5,254.1,213.1,249.3,212.1,212.8,164.4,149.3,158.6,157.4,154.1,112.7,93.4,108.4,87.3,86.7,82.0,115.9,66.9,84.2,61.4,43.7,58.1,40.9,73.9,50.0,51.6,25.7,43.2,48.2,43.0,32.6,21.6,22.4,36.3,28.3,19.4,21.1,21.9,21.5,19.2,15.8,22.6,9.3,14.0,22.4,14.0,13.0,11.9,18.7,7.3,21.6,9.5,11.2,12.0,18.2,12.9,2.2,10.7,6.1,11.7,7.6,1.0,4.7,8.5,6.4,3.3,4.6,1.2,3.7,5.8,1.0,1.0,1.0,1.0,3.2,1.2,3.1,2.2,2.3,2.1,1.1,2.0,2.1,2.2,4.6,2.2,1.0,1.0,1.0,0.0,3.0,1.2,0.0,8.2,3.0,1.0,1.0,2.1,1.2,3.2,1.0,5.2,1.1,5.2,1.0,1.2,2.3,1.0,3.1,1.0,1.0,1.1,1.6,1.1,1.1,1.0,1.0,1.0,1.0])

        m20 = 0
        m25 = 0
        m30 = 0
        m40 = 0
        m50 = 0
        m60 = 0
        m65 = 0
        salx = np.zeros((self.n_time+2,1))
        saln = np.zeros((self.n_time+2,1))
        salgx = np.zeros((self.n_time+2,self.n_groups))
        salgn = np.zeros((self.n_time+2,self.n_groups))
        salx_m = np.zeros((self.n_time+2,1))
        saln_m = np.zeros((self.n_time+2,1))
        salx_f = np.zeros((self.n_time+2,1))
        saln_f = np.zeros((self.n_time+2,1))
        for k in range(self.episodestats.n_pop):
            g = int(self.episodestats.infostats_group[k])
            for t in range(self.n_time-2):
                if self.episodestats.popempstate[t,k] in {1}: # 9,8,10
                    wage = self.episodestats.infostats_pop_wage[t,k]/self.timestep
                    salx[t] = salx[t]+wage
                    saln[t] = saln[t]+1
                    salgx[t,g] = salgx[t,g]+wage
                    salgn[t,g] = salgn[t,g]+1
                    
            if self.episodestats.popempstate[self.map_age(20),k] in {1}:
                sal20[m20] = self.episodestats.infostats_pop_wage[self.map_age(20),k]/self.timestep
                m20 = m20+1
            if self.episodestats.popempstate[self.map_age(25),k] in {1}:
                sal25[m25] = self.episodestats.infostats_pop_wage[self.map_age(25),k]/self.timestep
                m25 = m25+1
            if self.episodestats.popempstate[self.map_age(30),k] in {1}:
                sal30[m30] = self.episodestats.infostats_pop_wage[self.map_age(30),k]/self.timestep
                m30 = m30+1
            if self.episodestats.popempstate[self.map_age(40),k] in {1}:
                sal40[m40] = self.episodestats.infostats_pop_wage[self.map_age(40),k]/self.timestep
                m40 = m40+1
            if self.episodestats.popempstate[self.map_age(50),k] in {1}:
                sal50[m50] = self.episodestats.infostats_pop_wage[self.map_age(50),k]/self.timestep
                m50 = m50+1
            if self.episodestats.popempstate[self.map_age(60),k] in {1}:
                sal60[m60] = self.episodestats.infostats_pop_wage[self.map_age(60),k]/self.timestep
                m60 = m60+1
            if self.episodestats.popempstate[self.map_age(65),k] in set([1,9]):
                sal65[m65] = self.episodestats.infostats_pop_wage[self.map_age(65),k]/self.timestep
                m65 = m65+1

        salx_f = np.sum(salgx[:,3:6],axis = 1)
        saln_f = np.sum(salgn[:,3:6],axis = 1)
        salx_m = np.sum(salgx[:,0:3],axis = 1)
        saln_m = np.sum(salgn[:,0:3],axis = 1)

        salx = salx/np.maximum(1,saln)
        salgx = salgx/np.maximum(1,salgn)
        salx_f = salx_f/np.maximum(1,saln_f)
        salx_m = salx_m/np.maximum(1,saln_m)
        
        alivemask = self.episodestats.get_alivemask() #(self.episodestats.popempstate== 15).astype(bool)
        wdata = ma.array(self.episodestats.infostats_pop_wage,mask = alivemask)#.compressed()

        workmask = (self.episodestats.popempstate== 2) # ei vanhuuseläkkeellä
        workmask = ma.mask_or(workmask,self.episodestats.popempstate== 15) # ja ei vanhuuseläkkeellä
        workmask = ma.mask_or(workmask,self.episodestats.popempstate== 3) # ja ei tk
        wdata2 = ma.array(self.episodestats.infostats_pop_wage,mask = workmask)#.compressed()
            
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

        data_range = np.arange(self.min_age,self.max_age+1)
        plt.plot(data_range,ma.mean(wdata[::4,:],axis = 1),label = 'malli alive')
        plt.plot(data_range,ma.mean(wdata2[::4,:],axis = 1),label = 'malli not ret')
        plt.plot(data_range,salx[::4],label = 'malli töissä')
        data_range_72 = np.arange(self.min_age,72)
        plt.plot(data_range_72,0.5*palkat_ika_miehet+0.5*palkat_ika_naiset,label = 'data')
        plt.legend()
        plt.show()

        plt.plot(data_range,salx_m[::4],label = 'malli töissä miehet')
        plt.plot(data_range,salx_f[::4],label = 'malli töissä naiset')
        plt.plot(data_range_72,palkat_ika_miehet,label = 'data miehet')
        plt.plot(data_range_72,palkat_ika_naiset,label = 'data naiset')
        plt.legend()
        plt.show()
        
        fig,ax = plt.subplots()
        data_range = np.arange(self.min_age,self.max_age+1)
        for g in range(self.n_groups):
            ax.plot(data_range,salgx[::4,g],ls = '--',label = 'malli '+str(g))
        ax.plot(data_range_72,palkat_ika_miehet,label = 'data miehet')
        ax.plot(data_range_72,palkat_ika_naiset,label = 'data naiset')
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.show()

        fig,ax = plt.subplots()
        data_range = np.arange(self.min_age,self.max_age+1)
        ax.plot(data_range,salgx[::4,0]/salgx[::4,1],ls = '--',label = 'miehet low/mid ')
        ax.plot(data_range,salgx[::4,2]/salgx[::4,1],ls = '--',label = 'miehet high/mid ')
        ax.plot(data_range,salgx[::4,3]/salgx[::4,4],ls = '--',label = 'naiset low/mid ')
        ax.plot(data_range,salgx[::4,5]/salgx[::4,4],ls = '--',label = 'naiset high/mid ')
        x,m1,m2,w1,w2 = self.empstats.stat_wageratio()
        ax.plot(x,m1,ls = '-',label = 'data men low/mid ')
        ax.plot(x,m2,ls = '-',label = 'data men high/mid ')
        ax.plot(x,w1,ls = '-',label = 'data women low/mid ')
        ax.plot(x,w2,ls = '-',label = 'data women high/mid ')
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.title('suhteellinen kehitys')
        plt.show()

        fig,ax = plt.subplots()
        data_range = np.arange(self.min_age,self.max_age+1)
        for g in range(self.n_groups):
            ax.plot(data_range,salgx[::4,g]/salgx[1,g],ls = '--',label = 'suhde '+str(g))
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.title('ikäkehitys')
        plt.show()

    def test_tail(self):
        from scipy.stats import pareto
        n = self.episodestats.n_pop
        
        data_range = np.arange(self.min_age,self.max_age)
        minwage = 78_400

        salx = np.zeros((self.episodestats.n_pop*self.n_time,1))
        saln = 0
        salgx = np.zeros((self.n_groups,self.episodestats.n_pop*self.n_time))
        salgn = np.zeros(self.n_groups,dtype = int)
        for k in range(self.episodestats.n_pop):
            for t in range(self.n_time-2):
                wage = self.episodestats.infostats_pop_wage[t,k]
                if self.episodestats.popempstate[t,k] in {1} and wage>minwage: # 9,8,10
                    salx[saln] = wage
                    saln += 1
                    g = int(self.episodestats.infostats_group[k])
                    m = int(salgn[g])
                    salgx[g,m] = wage
                    salgn[g] += 1

        a = 2.95
        mm = a*minwage/(a-1)
        fig,ax = plt.subplots()
        pscale = minwage**(-(a/(a+1)))
        count, bins, _ = ax.hist(salx[:saln], 200, density = True,log = True)
        px = np.linspace(minwage,1_000_000,200)
        rv = pareto(a)
#        vals = rv.pdf(px*pscale)
        vals = rv.pdf(px/minwage)
        ax.plot(px,vals, linewidth = 1, color = 'r')
        med = (rv.median())*minwage
        med_data = np.median(salx[:saln])
        ax.set_xlim([0,1_000_000])
        plt.title(f'pareto {med:.0f} data {med_data:.0f}')
        plt.show()
        
        rv = pareto(a)
        fig,ax = plt.subplots()
        r = (rv.rvs(size = 1_000))*minwage
        count, bins, _ = ax.hist(r, 200, density = True,log = True)
        px = np.linspace(minwage,1_000_000,200)
        vals = rv.pdf(px/minwage)
        ax.plot(px,vals, linewidth = 1, color = 'r')
        ax.set_xlim([0,1_000_000])
        med = (rv.median())*minwage
        med_data = np.median(r)
        plt.title(f'pareto {med:.0f} data {med_data:.0f}')
        plt.show()
        
    def plot_rewdist(self,t = 0,sum = False,all = False):
        if all:
            fig,ax = plt.subplots()
            for t in range(1,self.n_time-1,5):
                scaled,x = np.histogram(self.poprewstate[t,:])
                x2 = 0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label = t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x = np.histogram(np.sum(self.poprewstate,axis = 0))
                x2 = 0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax = plt.subplots()
                for t in range(t,t+10,1):
                    scaled,x = np.histogram(self.poprewstate[t,:])
                    x2 = 0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label = t)
                plt.legend()
                plt.show()

    def plot_unempdistribs_bu(self,unemp_distrib,max = 2.5,unemp_distrib2 = None,label1 = '',label2 = ''):
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(-max_time,0,nn_time)

        scaled1,xv1 = np.histogram(unemp_distrib,bins = x,density = True)
        density1 = scaled1 * (x[1]-x[0])

        if unemp_distrib2 is not None:
            scaled2,xv2 = np.histogram(unemp_distrib2,bins = x,density = True)
            density2 = scaled2 * (x[1]-x[0])
            
        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['unemp duration'])
        ax.set_ylabel(self.labels['probability'])

        ax.plot(xv1[1:-1],scaled[1:])
        if unemp_distrib2 is not None:
            ax.plot(xv2[1:-1],scaled2[1:])
        plt.xlim(-max,0)
        plt.gca().set_ylim(bottom = 0)
        plt.show()

    def plot_compare_unempdistribs(self,unemp_distrib1,unemp_distrib2,max = 3,
            label2 = 'none',label1 = 'none',logy = False,diff = False,figname = None):
        #fig,ax = plt.subplots()
        max_time = 50
        nn_time = int(np.round((max_time)*self.inv_timestep))+1
        x = np.linspace(self.timestep,max_time,nn_time)
        
        print('{} keskikesto {} v {} keskikesto {} v'.format(label1,np.mean(unemp_distrib1),label2,np.mean(unemp_distrib2)))
        print('Skaalaamaton {} lkm {} v {} lkm {} v'.format(label1,len(unemp_distrib1),label2,len(unemp_distrib2)))
        print('Skaalaamaton {} työtpäiviä yht {} v {} työtpäiviä yht {} v'.format(label1,np.sum(unemp_distrib1),label2,np.sum(unemp_distrib2)))
        #scaled = scaled/np.sum(unemp_distrib)
        scaled1,x1 = np.histogram(unemp_distrib1,x)
        scaled1 = scaled1/np.sum(scaled1)

        scaled2,x1 = np.histogram(unemp_distrib2,x)
        scaled2 = scaled2/np.sum(scaled2)
        fig,ax = plt.subplots()
        if not diff:
            self.plot_vlines_unemp(0.5)
        ax.set_xlabel(self.labels['unemp duration'])
        ax.set_ylabel(self.labels['osuus'])
        if diff:
            ax.plot(x[:-1],scaled1-scaled2,label = label1+'-'+label2)
        else:
            ax.plot(x[:-1],scaled2,label = label2)
            ax.plot(x[:-1],scaled1,label = label1)
        if logy and not diff:
            ax.set_yscale('log')
        if not diff:
            plt.ylim(1e-4,1.0)
        #ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        ax.legend()
        plt.xlim(0,max)
        plt.gca().set_ylim(bottom = 0)
        if figname is not None:
            plt.savefig(figname+'comp_unempdistrib.'+self.figformat, format = self.figformat)

        plt.show()

    def plot_compare_virrat(self,virta1,virta2,min_time = 25,max_time = 65,label1 = 'perus',label2 = 'vaihtoehto',virta_label = 'työllisyys',ymin = None,ymax = None):
        x = np.linspace(self.min_age,self.max_age,self.n_time)

        demog2 = self.empstats.get_demog()

        scaled1 = virta1*demog2/self.episodestats.alive #/self.episodestats.alive
        scaled2 = virta2*demog2/self.episodestats.alive #/self.episodestats.alive

        fig,ax = plt.subplots()
        plt.xlim(min_time,max_time)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(virta_label+'virta')
        ax.plot(x,scaled1,label = label1)
        ax.plot(x,scaled2,label = label2)
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin,ymax)

        plt.show()

    def plot_family(self,printtaa = True):
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        ax.plot(x,100*(self.episodestats.empstate[:,5]+self.episodestats.empstate[:,6]+self.episodestats.empstate[:,7])/self.episodestats.alive[:,0],label = 'vanhempainvapailla')
        #emp_statsratio = 100*self.empstats.outsider_stats()
        #ax.plot(x,emp_statsratio,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,5,3:6]+self.episodestats.gempstate[:,6,3:6]+self.episodestats.gempstate[:,7,3:6],1,
            keepdims = True)/np.sum(self.episodestats.galive[:,3:6],1,keepdims = True),label = 'vanhempainvapailla, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,5,0:3]+self.episodestats.gempstate[:,6,0:3]+self.episodestats.gempstate[:,7,0:3],1,
            keepdims = True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims = True),label = 'vanhempainvapailla, miehet')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,7,3:6],1,
            keepdims = True)/np.sum(self.episodestats.galive[:,3:6],1,keepdims = True),label = 'kht, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,7,0:3],1,
            keepdims = True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims = True),label = 'kht, miehet')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,5,3:6]+self.episodestats.gempstate[:,6,3:6],1,
            keepdims = True)/np.sum(self.episodestats.galive[:,3:6],1,keepdims = True),label = 'äitiysvapaa, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,5,0:3]+self.episodestats.gempstate[:,6,0:3],1,
            keepdims = True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims = True),label = 'isyysvapaa, miehet')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()                  

        ratio_label = self.labels['ratio']
        empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive
        self.plot_states(empstate_ratio,ylabel = ratio_label,parent = True,stack = False)

    def plot_outsider(self,printtaa = True):
        '''
        plottaa työvoiman ulkopuolella olevat
        mukana ei isyysvapaat, opiskelijat, armeijassa olevat eikä alle 3 kk kestäneet äitiysvpaat
        '''
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        outsiders = self.episodestats.empstate[:,11]+self.episodestats.empstate[:,7]
        ax.plot(x,100*outsiders/self.episodestats.alive[:,0],label = 'työvoiman ulkopuolella, ei opiskelija, sis. kht')
        emp_statsratio = 100*self.empstats.outsider_stats()
        ax.plot(x,emp_statsratio,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(x,100*(self.episodestats.empstate[:,11])/self.episodestats.alive[:,0],
            label = 'työvoiman ulkopuolella, ei opiskelija, ei vanh.vapaat')
        emp_statsratio = 100*self.empstats.outsider_stats()
        ax.plot(x,emp_statsratio,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(x,100*(self.episodestats.empstate[:,14])/self.episodestats.alive[:,0],
            label = 'sv-päivärahalla')
        #emp_statsratio = 100*self.empstats.outsider_stats()
        #ax.plot(x,emp_statsratio,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        
        fig,ax = plt.subplots()
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,14,3:6],1,keepdims = True)/np.sum(self.episodestats.galive[:,3:6],1,keepdims = True),label = 'sv-päivärahalla, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,14,0:3],1,keepdims = True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims = True),label = 'sv-päivärahalla, miehet')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        
        fig,ax = plt.subplots()
        naisia = np.sum(self.episodestats.galive[:,3:6],1,keepdims = True)
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,11,3:6]+self.episodestats.gempstate[:,7,3:6],1,keepdims = True)/naisia,
            label = 'työvoiman ulkopuolella, naiset')
        miehia = np.sum(self.episodestats.galive[:,0:3],1,keepdims = True)
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,11,0:3]+self.episodestats.gempstate[:,7,0:3],1,keepdims = True)/miehia,
            label = 'työvoiman ulkopuolella, miehet')
        emp_statsratio = 100*self.empstats.outsider_stats(g = 1)
        ax.plot(x,emp_statsratio,label = self.labels['havainto, naiset'])
        emp_statsratio = 100*self.empstats.outsider_stats(g = 2)
        ax.plot(x,emp_statsratio,label = self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
            
    def plot_tulot(self):
        '''
        plot net income per person
        '''
        x = np.linspace(self.min_age+self.timestep,self.max_age-1,self.n_time-2)
        fig,ax = plt.subplots()
        tulot = self.episodestats.infostats_tulot_netto[1:-1,0]/self.timestep/self.episodestats.alive[1:-1,0]
        ax.plot(x,tulot,label = 'tulot netto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Tulot netto [e/v]')
        ax.legend()
        plt.show()

        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack = False)
        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack = False,unemp = True)
        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack = False,emp = True)
        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack = False,all_emp = True)

    def plot_pinkslip(self):
        pink = 100*self.episodestats.infostats_pinkslip/np.maximum(1,self.episodestats.empstate)
        self.plot_states(pink,'Karenssittomien osuus tilassa [%]',stack = False,unemp = True,add_student = False)

        print([np.max((self.episodestats.infostats_pinkslip/np.maximum(1,self.episodestats.empstate))[:,k]) for k in range(self.n_groups)])

    def plot_student(self):
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        ax.plot(x+self.timestep,100*(self.episodestats.empstate[:,12]+self.episodestats.empstate[:,16])/self.episodestats.alive[:,0],label = 'opiskelija tai armeijassa')
        emp_statsratio = 100*self.empstats.student_stats()
        ax.plot(x,emp_statsratio,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        ax.plot(x+self.timestep,100*self.episodestats.empstate[:,16]/self.episodestats.alive[:,0],label = 'opiskelija tai armeijassa + työ')
        emp_statsratio = 100*self.empstats.student_stats()
        ax.plot(x,emp_statsratio,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax = plt.subplots()
        emp_statsratio = 100*self.empstats.student_stats()
        diff = 100.0*(self.episodestats.empstate[:,12]+self.episodestats.empstate[:,16])/self.episodestats.alive[:,0]-emp_statsratio
        ax.plot(x+self.timestep,diff,label = 'opiskelija tai armeijassa, virhe')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

    def plot_kassanjasen(self,fig=None,ax=None):
        x2,vrt = self.empstats.get_kassanjasenyys_rate()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        nofig = False
        if fig is None:
            nofig = True
            fig,ax = plt.subplots()
        jasenia = 100.0*self.episodestats.infostats_kassanjasen/self.episodestats.alive
        ax.plot(x+self.timestep,jasenia,label = 'työttömyyskassan jäsenien osuus kaikista')
        ax.plot(x2,100.0*vrt,label = 'havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        if nofig:
            plt.show()
        mini = np.nanmin(jasenia)
        maxi = np.nanmax(jasenia)
        print('Kassanjäseniä min {:1f} % max {:1f} %'.format(mini,maxi))

    def plot_group_student(self):
        fig,ax = plt.subplots()
        for gender in range(2):
            if gender== 0:
                leg = 'Opiskelijat+Armeija Miehet'
                opiskelijat = np.sum(self.episodestats.gempstate[:,12,0:3],axis = 1)+np.sum(self.episodestats.gempstate[:,16,0:3],axis = 1)
                alive = np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0] = np.sum(self.episodestats.galive[:,0:3],1)
            else:
                leg = 'Opiskelijat+Armeija Naiset'
                opiskelijat = np.sum(self.episodestats.gempstate[:,12,3:6],axis = 1)+np.sum(self.episodestats.gempstate[:,16,3:6],axis = 1)
                alive = np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0] = np.sum(self.episodestats.galive[:,3:6],1)

            opiskelijat = np.reshape(opiskelijat,(self.episodestats.galive.shape[0],1))
            osuus = 100*opiskelijat/alive
            x = np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label = leg)

        emp_statsratio = 100*self.empstats.student_stats(g = 1)
        ax.plot(x,emp_statsratio,label = self.labels['havainto, naiset'])
        emp_statsratio = 100*self.empstats.student_stats(g = 2)
        ax.plot(x,emp_statsratio,label = self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.show()

    def plot_group_disab(self,xstart = None,xend = None):
        fig,ax = plt.subplots()
        for gender in range(2):
            if gender== 0:
                leg = 'TK Miehet'
                opiskelijat = np.sum(self.episodestats.gempstate[:,3,0:3],axis = 1)
                alive = np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0] = np.sum(self.episodestats.galive[:,0:3],1)
            else:
                leg = 'TK Naiset'
                opiskelijat = np.sum(self.episodestats.gempstate[:,3,3:6],axis = 1)
                alive = np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0] = np.sum(self.episodestats.galive[:,3:6],1)

            opiskelijat = np.reshape(opiskelijat,(self.episodestats.galive.shape[0],1))
            osuus = 100*opiskelijat/alive
            x = np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label = leg)

        ax.plot(x,100*self.empstats.disab_stat(g = 1),label = self.labels['havainto, naiset'])
        ax.plot(x,100*self.empstats.disab_stat(g = 2),label = self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        if xstart is not None:
            ax.set_xlim([xstart,xend])
        plt.show()

    def plot_group_mort(self,xstart = None,xend = None):
        fig,ax = plt.subplots()
        for gender in range(2):
            alive = np.zeros((self.episodestats.galive.shape[0],1))
            if gender== 0:
                leg = 'mort men'
                alive[:,0] = np.sum(self.episodestats.galive[:,0:3],1)
            else:
                leg = 'mort women'
                alive[:,0] = np.sum(self.episodestats.galive[:,3:6],1)

            osuus = 100*(1.0-alive/self.episodestats.n_pop*2.0)
            x = np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,osuus,label = leg)

        ax.plot(x,100*self.empstats.get_mortdata(g = 1),label = self.labels['havainto, naiset'])
        ax.plot(x,100*self.empstats.get_mortdata(g = 2),label = self.labels['havainto, miehet'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        if xstart is not None:
            xstart = self.min_age+self.timestep
        if xend is not None:
            xend = self.max_age+self.timestep
        ax.set_xlim([xstart,xend])
        plt.show()        

    def plot_taxes(self,figname = None):
        valtionvero_ratio = 100*self.episodestats.infostats_valtionvero_distrib/np.reshape(np.sum(self.episodestats.infostats_valtionvero_distrib,1),(-1,1))
        kunnallisvero_ratio = 100*self.episodestats.infostats_kunnallisvero_distrib/np.reshape(np.sum(self.episodestats.infostats_kunnallisvero_distrib,1),(-1,1))
        vero_ratio = 100*(self.episodestats.infostats_kunnallisvero_distrib+self.episodestats.infostats_valtionvero_distrib)/(np.reshape(np.sum(self.episodestats.infostats_valtionvero_distrib,1),(-1,1))+np.reshape(np.sum(self.episodestats.infostats_kunnallisvero_distrib,1),(-1,1)))

        if figname is not None:
            self.plot_states(vero_ratio,ylabel = 'Valtioneronmaksajien osuus tilassa [%]',stack = True,figname = figname+'_stack')
        else:
            self.plot_states(vero_ratio,ylabel = 'Valtioneronmaksajien osuus tilassa [%]',stack = True)

        if figname is not None:
            self.plot_states(valtionvero_ratio,ylabel = 'Veronmaksajien osuus tilassa [%]',stack = True,figname = figname+'_stack')
        else:
            self.plot_states(valtionvero_ratio,ylabel = 'Veronmaksajien osuus tilassa [%]',stack = True)

        if figname is not None:
            self.plot_states(kunnallisvero_ratio,ylabel = 'Kunnallisveron maksajien osuus tilassa [%]',stack = True,figname = figname+'_stack')
        else:
            self.plot_states(kunnallisvero_ratio,ylabel = 'Kunnallisveron maksajien osuus tilassa [%]',stack = True)

        valtionvero_osuus,kunnallisvero_osuus,vero_osuus = self.episodestats.comp_taxratios()

        print('Valtionveron maksajien osuus\n{}'.format(self.v2_groupstates(valtionvero_osuus)))
        print('Kunnallisveron maksajien osuus\n{}'.format(self.v2_groupstates(kunnallisvero_osuus)))
        print('Veronmaksajien osuus\n{}'.format(self.v2_groupstates(vero_osuus)))

    def group_taxes(self,ratios):
        if len(ratios.shape)>1:
            vv_osuus = np.zeros((ratios.shape[0],5))
            vv_osuus[:,0] = ratios[:,0]+ratios[:,4]+ratios[:,5]+ratios[:,6]+\
                          ratios[:,7]+ratios[:,8]+ratios[:,9]+ratios[:,11]+\
                          ratios[:,12]+ratios[:,13]+ratios[:,16]
            vv_osuus[:,1] = ratios[:,1]+ratios[:,10]
            vv_osuus[:,2] = ratios[:,2]+ratios[:,3]+ratios[:,8]+ratios[:,9]
            vv_osuus[:,3] = ratios[:,1]+ratios[:,10]+ratios[:,8]+ratios[:,9]
        else:
            vv_osuus = np.zeros((4))
            vv_osuus[0] = ratios[0]+ratios[4]+ratios[5]+ratios[6]+\
                          ratios[7]+ratios[8]+ratios[9]+ratios[11]+\
                          ratios[12]+ratios[13]+ratios[16]
            vv_osuus[1] = ratios[1]+ratios[10]
            vv_osuus[2] = ratios[2]+ratios[3]+ratios[8]+ratios[9]
            vv_osuus[3] = ratios[1]+ratios[10]+ratios[8]+ratios[9]
        return vv_osuus


    def v2_states(self,x):
        return 'Ansiosidonnaisella {:.2f}\nKokoaikatyössä {:.2f}\nVanhuuseläkeläiset {:.2f}\nTyökyvyttömyyseläkeläiset {:.2f}\n'.format(x[0],x[1],x[2],x[3])+\
          'Putkessa {:.2f}\nÄitiysvapaalla {:.2f}\nIsyysvapaalla {:.2f}\nKotihoidontuella {:.2f}\n'.format(x[4],x[5],x[6],x[7])+\
          'VE+OA {:.2f}\nVE+kokoaika {:.2f}\nOsa-aikatyö {:.2f}\nTyövoiman ulkopuolella {:.2f}\n'.format(x[8],x[9],x[10],x[11])+\
          'Opiskelija/Armeija {:.2f}\nTM-tuki {:.2f}\n'.format(x[12]+x[16],x[13])

    def v2_groupstates(self,xx):
        x = self.group_taxes(xx)
        return 'Etuudella olevat {:.2f}\nTyössä {:.2f}\nEläkkeellä {:.2f}\n'.format(x[0],x[1],x[2])

    def plot_children(self,figname = None):
        c3,c7,c18,c_vrt = self.episodestats.comp_children()

        print('Alle 3v lapsia {:.0f}, alle 7v lapsia {:.0f}, alle 18v lapsia {:.0f} vrt {:.0f}'.format(np.sum(c3),np.sum(c7),np.sum(c18),np.sum(c_vrt)))
        
        self.plot_y(c3,label = 'Alle 3v lapset',
                    y2 = c7,label2 = 'Alle 7v lapset',
                    y3 = c18,label3 = 'Alle 18v lapset',
                    ylabel = 'Lapsia (lkm)',
                    show_legend = True)

        x,c3,c7,c18,c_vrt = self.episodestats.comp_children_ages()
        vrt_c18 = self.empstats.children_ages()
        fig,ax = plt.subplots()
        plt.title('lasten lkm vrt havainto')
        ax.plot(x,c3,label = 'alle 3v lapset')
        ax.plot(x,c_vrt,label = 'vrt syntyneet')
        ax.plot(vrt_c18[:,0],vrt_c18[:,1],label = 'syntyneet, havainto')
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.show()

        if self.episodestats.save_pop:
            h, edges = self.episodestats.comp_children_hist()
            plt.title('Lapsia per kotitalous')
            plt.bar(edges[:-1], h)
            plt.show()

            cemp_n,cemp_htv = self.episodestats.comp_emp_by_children()

            df_cemp_n = pd.DataFrame(cemp_n,columns = ['naiset','miehet','yhteensä'])
            df_cemp_n.index = ['alle 3v','3-6v','7-18v','ei lapsia']
            df_cemp_n.plot.bar(ylim = (65,95))

            df_cemp_htv = pd.DataFrame(cemp_htv,columns = ['naiset','miehet','yhteensä'])
            df_cemp_htv.index = ['alle 3v','3-6v','7-18v','ei lapsia']
            df_cemp_htv.plot.bar(ylim = (65,95))

            fig,ax = plt.subplots()
            plt.title('työllisyysaste lasten lkm mukaan')
            #ax.plot(cemp_n[:,0],label = 'yhteensä')
            #ax.plot(cemp_n[:,1],label = 'naiset')
            #ax.plot(cemp_n[:,2],label = 'miehet')
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            plt.show()

            fig,ax = plt.subplots()
            plt.title('htv lasten lkm mukaan')
            ax.plot(cemp_htv[:,0],label = 'yhteensä')
            ax.plot(cemp_htv[:,1],label = 'naiset')
            ax.plot(cemp_htv[:,2],label = 'miehet')
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            plt.show()

    def plot_emp_vs_workforce(self,empstate=None,alive=None,figname = None,ax = None,legend_infig = False):
        '''
        Plot employment as a proportion of workforce
        '''
        if empstate is None:
            empstate = self.episodestats.empstate
        if alive is None:
            alive = self.episodestats.alive
        
        workforce = self.episodestats.comp_workforce(empstate,alive)
        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = \
            self.episodestats.comp_empratios(empstate,alive,unempratio = False)

        tyolliset_osuus,tyottomat_osuus,vella_osuus = self.episodestats.comp_decisionratios(empstate,alive)

        pred_empratio = 1.0-tyottomyysaste

        age_label = self.labels['age']
        ratio_label = self.labels['osuus']

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        if ax is None:
            fig,ax = plt.subplots()
            noshow = False
        else:
            noshow = True
        ax.stackplot(x,100*tyolliset_osuus,100*tyottomat_osuus,100*vella_osuus,
            labels = (self.labels['työssä'],self.labels['työttömänä'],self.labels['eläkkeellä'])) #, colors = pal) pal = sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        unemp_statsratio = self.empstats.unempratio_stats(g = 0)
        wf_stats = self.empstats.workforce_stats(g = 0)
        empl_stats = self.empstats.emp_stats_tyossakayntitutkimus(g = 0)
        unemps_stats = self.empstats.unemp_stats_tyossakayntitutkimus(g = 0)
        pen_stats = self.empstats.pensioner_stats(g = 0)
        wf_statsratio = (unemps_stats + empl_stats)/(unemps_stats + empl_stats + pen_stats)*100
        emp_statsratio = (empl_stats)/(unemps_stats + empl_stats + pen_stats)*100
        #emp_statsratio = 100*(1.0-unemp_statsratio*wf_stats-pen_stats)
        ax.plot(x,wf_statsratio,ls = '--',label = 'Work force obs')
        ax.plot(x,emp_statsratio,ls = '-.',color = 'w',label = 'Employment obs')
        ax.set_xlabel(age_label)
        ax.set_ylabel('Proportion [%]')
        if legend_infig:
            leg = ax.legend(bbox_to_anchor = (0.05, 0.4), loc = 2, borderaxespad = 0.)
            #leg.get_frame().set_edgecolor('b')
            #leg.get_frame().set_linewidth(0.0)
            #leg.get_frame().set_facecolor('none')
        else:
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        ax.set_ylim([0,100])
        ax.set_xlim([18,75])
        if not noshow:
            if figname is not None:
                plt.savefig(figname+'workforce_vs_emp.'+self.figformat, format = self.figformat)
            plt.show()                    

    def plot_workforce(self,empstate=None,alive=None,figname = None,ax = None):
        '''
        Plot workforce ratio
        '''
        workforce = 100*self.episodestats.comp_workforce(self.episodestats.empstate,self.episodestats.alive,ratio = True)

        age_label = self.labels['age']
        ratio_label = self.labels['osuus']

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        if ax is None:
            fig,ax = plt.subplots()
            noshow = True
        else:
            noshow = False
        ax.plot(x,workforce,label = self.labels['malli'])
        emp_statsratio = 100*self.empstats.workforce_stats()
        ax.plot(x,emp_statsratio,ls = '--',label = self.labels['havainto'])
        ax.set_xlabel(age_label)
        ax.set_ylabel(self.labels['työvoima %'])
        ax.legend()
        if not noshow:
            if figname is not None:
                plt.savefig(figname+'workforce.'+self.figformat, format = self.figformat)
            plt.show()

    def plot_emp(self,figname = None,tyovoimatutkimus = False,palette_EK = False):

        if palette_EK:
            csfont,pal = setup_EK_fonts()
        else:
            csfont = {}

        tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = \
            self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio = False)

        age_label = self.labels['age']
        ratio_label = self.labels['osuus']

        x = np.linspace(self.min_age,self.max_age,self.n_time)

        emp_statsratio_tkt = 100*self.empstats.emp_stats(tyossakayntitutkimus = True)

        if palette_EK and False:
            lineplot(x,tyollisyysaste,y2 = emp_statsratio_tkt,label1 = self.labels['malli'],xlabel = age_label,ylabel = self.labels['tyollisyysaste %'])
        else:
            fig,ax = plt.subplots()
            ax.plot(x,tyollisyysaste,label = self.labels['malli'])
            #ax.plot(x,tyottomyysaste,label = self.labels['tyottomien osuus'])
            if tyovoimatutkimus:
                emp_statsratio_tvt = 100*self.empstats.emp_stats(tyossakayntitutkimus = False)
                ax.plot(x,emp_statsratio_tvt,ls = '--',label = self.labels['havainto']+' työvoimatilasto')
            ax.plot(x,emp_statsratio_tkt,ls = '--',label = self.labels['havainto']+' työssäkäyntitilasto')
            ax.set_xlabel(age_label,**csfont)
            ax.set_ylabel(self.labels['tyollisyysaste %'],**csfont)
            ax.legend()
            if figname is not None:
                plt.savefig(figname+'tyollisyysaste.'+self.figformat, format = self.figformat)
            plt.show()

        #if self.version in set([1,2,3]):
        fig,ax = plt.subplots()
        ax.stackplot(x,osatyoaste,100-osatyoaste,
                    labels = ('osatyössä','kokoaikaisessa työssä')) #, colors = pal) pal = sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        ax.set_ylim([0,100])
        plt.show()

    def plot_savings(self):
        savings_0 = np.zeros((self.n_time,1))
        savings_1 = np.zeros((self.n_time,1))
        savings_2 = np.zeros((self.n_time,1))
        act_savings_0 = np.zeros((self.n_time,1))
        act_savings_1 = np.zeros((self.n_time,1))
        act_savings_2 = np.zeros((self.n_time,1))

        for t in range(self.n_time):
            state_0 = np.argwhere(self.episodestats.popempstate[t,:]== 0)
            savings_0[t] = np.mean(self.episodestats.infostats_savings[t,state_0[:]])
            act_savings_0[t] = np.mean(self.sav_actions[t,state_0[:]])
            state_1 = np.argwhere(self.episodestats.popempstate[t,:]== 1)
            savings_1[t] = np.mean(self.episodestats.infostats_savings[t,state_1[:]])
            act_savings_1[t] = np.mean(self.sav_actions[t,state_1[:]])
            state_2 = np.argwhere(self.episodestats.popempstate[t,:]== 2)
            savings_2[t] = np.mean(self.episodestats.infostats_savings[t,state_2[:]])
            act_savings_2[t] = np.mean(self.sav_actions[t,state_2[:]])

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        savings = np.mean(self.episodestats.infostats_savings,axis = 1)
        ax.plot(x,savings,label = 'savings all')
        ax.legend()
        plt.title('Savings all')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        savings = np.mean(self.episodestats.infostats_savings,axis = 1)
        ax.plot(x,savings_0,label = 'unemp')
        ax.plot(x,savings_1,label = 'emp')
        ax.plot(x,savings_2,label = 'retired')
        plt.title('Savings by emp state')
        ax.legend()
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        savings = np.mean(self.sav_actions,axis = 1)
        savings_plus = np.nanmean(np.where(self.sav_actions>0,self.sav_actions,np.nan),axis = 1)[:,None]
        savings_minus = np.nanmean(np.where(self.sav_actions<0,self.sav_actions,np.nan),axis = 1)[:,None]
        ax.plot(x[1:],savings[1:],label = 'savings action')
        ax.plot(x[1:],savings_plus[1:],label = '+savings')
        ax.plot(x[1:],savings_minus[1:],label = '-savings')
        ax.legend()
        plt.title('Saving action')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        pops = np.random.randint(self.episodestats.n_pop,size = 20)
        ax.plot(x,self.episodestats.infostats_savings[:,pops],label = 'savings all')
        #ax.legend()
        plt.title('Savings all')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        savings = self.sav_actions[:,pops]
        ax.plot(x[1:],savings[1:,:],label = 'savings action')
        #ax.legend()
        plt.title('Saving action')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        savings = np.mean(self.episodestats.infostats_savings,axis = 1)
        ax.plot(x[1:],act_savings_0[1:],label = 'unemp')
        ax.plot(x[1:],act_savings_1[1:],label = 'emp')
        ax.plot(x[1:],act_savings_2[1:],label = 'retired')
        plt.title('Saving action by emp state')
        ax.legend()
        plt.show()

    def plot_emp_by_gender(self,figname = None):
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        for gender in range(2):
            if gender<1:
                empstate_ratio = 100*np.sum(self.episodestats.gempstate[:,:,0:3],axis = 2)/(np.sum(self.episodestats.galive[:,0:3],axis = 1)[:,None])
                genderlabel = 'miehet'
            else:
                empstate_ratio = 100*np.sum(self.episodestats.gempstate[:,:,3:6],axis = 2)/(np.sum(self.episodestats.galive[:,3:6],axis = 1)[:,None])
                genderlabel = 'naiset'
            if figname is not None:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),stack = True,figname = figname+'_stack')
            else:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),stack = True)

            if self.version in self.complex_models:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),ylimit = 20,stack = False)
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),parent = True,stack = False)
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),unemp = True,stack = False)

            if figname is not None:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),start_from = 60,stack = True,figname = figname+'_stack60')
            else:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),start_from = 60,stack = True)

    def plot_emp_by_group(self,figname = None):
        '''
        Tarpeen?
        '''
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        for g in range(self.n_groups):
            empstate_ratio = 100*np.sum(self.episodestats.gempstate[:,:,g],axis = 2)/(np.sum(self.episodestats.galive[:,g],axis = 1)[:,None])
            genderlabel = str(g)

            if figname is not None:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),stack = True,figname = figname+'_stack')
            else:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),stack = True)

            if self.version in self.complex_models:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),ylimit = 20,stack = False)
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),parent = True,stack = False)
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),unemp = True,stack = False)

            if figname is not None:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),start_from = 60,stack = True,figname = figname+'_stack60')
            else:
                self.plot_states(empstate_ratio,ylabel = self.labels['osuus tilassa x'].format(genderlabel),start_from = 60,stack = True)                

    def plot_parents_in_work(self):
        empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive
        self.plot_y(empstate_ratio[:,5],label = 'mothers in workforce',show_legend = True,
            ylabel = self.labels['ratio'],y2 = empstate_ratio[:,6],label2 = 'isyysvapaa')

    def plot_spouse(self,figname = None,grayscale = False):
        demog2 = self.empstats.get_demog()
        scalex = demog2/self.episodestats.alive*self.timestep
        #min_cage = self.episodestats.map_age(start)
        #max_cage = self.episodestats.map_age(end)+1

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        puolisoita = np.sum(self.episodestats.infostats_puoliso,axis = 1,keepdims = True)
        women,men = self.episodestats.comp_genders()
        spouseratio = puolisoita/self.episodestats.alive
        n_spouses = np.sum(puolisoita*scalex)*0.5
        av_spouseratio = np.mean(spouseratio)

        print(f'Naisia {women:.0f} miehiä {men:.0f}')
        print(f'Aviopareja {n_spouses:.0f}, osuus kaikista {av_spouseratio:.2f}')
            
        if figname is not None:
            fname = figname+'spouses.'+self.figformat
        else:
            fname = None

        self.plot_y(spouseratio,ylabel = self.labels['spouses'],figname = fname)

        if self.episodestats.save_pop:
            print(self.episodestats.comp_family_matrix())

    def plot_unemp_all(self,unempratio = True,figname = None,grayscale = False,tyovoimatutkimus = False):
        '''
        Plottaa työttömyysaste (unempratio = True) tai työttömien osuus väestöstö (False)
        '''
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        if unempratio:
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio = True)
            unempratio_stat_tvt = 100*self.empstats.unempratio_stats(g = 0,tyossakayntitutkimus = False)
            unempratio_stat_tkt = 100*self.empstats.unempratio_stats(g = 0,tyossakayntitutkimus = True)
            if self.language== 'Finnish':
                labeli = 'keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)
                ylabeli = self.labels['tyottomyysaste']
                labeli2 = 'työttömyysaste'
            else:
                labeli = 'average unemployment rate '+str(ka_tyottomyysaste)
                ylabeli = self.labels['tyottomyysaste']
                labeli2 = 'Unemployment rate'
        else:
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio = False)
            unempratio_stat_tvt = 100*self.empstats.unemp_stats(g = 0,tyossakayntitutkimus = False)
            unempratio_stat_tkt = 100*self.empstats.unemp_stats(g = 0,tyossakayntitutkimus = True)
            if self.language== 'Finnish':
                labeli = 'keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
                ylabeli = 'Työttömien osuus väestöstö [%]'
                labeli2 = 'työttömien osuus väestöstö'
            else:
                labeli = 'proportion of unemployed'+str(ka_tyottomyysaste)
                ylabeli = 'Proportion of unemployed [%]'
                labeli2 = 'proportion of unemployed'

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])

        ax.set_ylabel(ylabeli)
        ax.plot(x,tyottomyysaste,label = self.labels['malli'])
        if tyovoimatutkimus:
            ax.plot(x,unempratio_stat_tvt,ls = '--',label = self.labels['havainto']+',työvoimatutkimus')
        ax.plot(x,unempratio_stat_tkt,ls = '--',label = self.labels['havainto']+',työssäkäyntitutkimus')
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste.'+self.figformat, format = self.figformat)
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if tyovoimatutkimus:
            ax.plot(x,unempratio_stat_tvt,label = self.labels['havainto']+',työvoimatutkimus')
        ax.plot(x,unempratio_stat_tkt,label = self.labels['havainto']+',työssäkäyntitutkimus')
        ax.legend()
        if grayscale:
            pal = sns.light_palette("black", 8, reverse = True)
        else:
            pal = sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        ax.stackplot(x,tyottomyysaste,colors = pal) #,label = self.labels['malli'])
        #ax.plot(x,tyottomyysaste)
        plt.show()

    def plot_unemp_gender(self,unempratio = True,figname = None,grayscale = False,tyovoimatutkimus = False):
        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        for gender in range(2):
            if gender== 0:
                leg = 'Miehet'
                color = 'darkgray'
                labeli2 = 'proportion of unemployed'
                gendertext = 'men'
            else:
                leg = 'Naiset'
                color = 'black'
                labeli2 = 'proportion of unemployed'
                gendertext = 'women'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios_gender(unempratio = unempratio,gender = gendertext)

            ax.plot(x,tyottomyysaste,color = color,label = '{} {}'.format(labeli2,leg))

        if grayscale:
            lstyle = '--'
        else:
            lstyle = '--'

        if self.version in self.complex_models:
            if unempratio:
                ax.plot(x,100*self.empstats.unempratio_stats(g = 1,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, naiset'])
                ax.plot(x,100*self.empstats.unempratio_stats(g = 2,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, miehet'])
                labeli = 'keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)
                ylabeli = self.labels['tyottomyysaste']
            else:
                ax.plot(x,100*self.empstats.unemp_stats(g = 1,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, naiset'])
                ax.plot(x,100*self.empstats.unemp_stats(g = 2,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, miehet'])
                labeli = 'keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
                ylabeli = self.labels['tyottomien osuus']

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste_spk.'+self.figformat, format = self.figformat)
        plt.show()

    def plot_unemp_group(self,unempratio = True,figname = None,grayscale = False,tyovoimatutkimus = False):
        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        for g in range(self.n_groups):
            #if g== 0:
            #    color = 'darkgray'
            #else:
            color = 'black'
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios_groups(unempratio = unempratio,g = g)

            #print(tyottomyysaste.shape)

            ax.plot(x,tyottomyysaste,label = 'ryhmä {}'.format(g)) # color = color,

        if grayscale:
            lstyle = '--'
        else:
            lstyle = '--'

        if self.version in self.complex_models:
            if unempratio:
                ax.plot(x,100*self.empstats.unempratio_stats(g = 1,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, naiset'])
                ax.plot(x,100*self.empstats.unempratio_stats(g = 2,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, miehet'])
                labeli = 'keskimääräinen työttömyysaste '+str(ka_tyottomyysaste)
                ylabeli = self.labels['tyottomyysaste']
            else:
                ax.plot(x,100*self.empstats.unemp_stats(g = 1,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, naiset'])
                ax.plot(x,100*self.empstats.unemp_stats(g = 2,tyossakayntitutkimus = True),ls = lstyle,label = self.labels['havainto, miehet'])
                labeli = 'keskimääräinen työttömien osuus väestöstö '+str(ka_tyottomyysaste)
                ylabeli = self.labels['tyottomien osuus']

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste_spk.'+self.figformat, format = self.figformat)
        plt.show()        

    def plot_unemp(self,unempratio = True,figname = None,grayscale = False,tyovoimatutkimus = False):
        self.plot_unemp_all(unempratio = unempratio,figname = figname,grayscale = grayscale,tyovoimatutkimus = tyovoimatutkimus)
        self.plot_unemp_gender(unempratio = unempratio,figname = figname,grayscale = grayscale,tyovoimatutkimus = tyovoimatutkimus)
        self.plot_unemp_group(unempratio = unempratio,figname = figname,grayscale = grayscale,tyovoimatutkimus = tyovoimatutkimus)

    def plot_parttime_ratio(self,grayscale = True,figname = None):
        '''
        Plottaa osatyötä tekevien osuus väestöstö
        '''
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        labeli2 = 'Osatyötä tekevien osuus'
        fig,ax = plt.subplots()
        for gender in range(2):
            if gender== 0:
                leg = 'Miehet'
                g = 'men'
                pstyle = '-'
            else:
                g = 'women'
                leg = 'Naiset'
                pstyle = ''

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios_gender(gender = g,unempratio = False)

            ax.plot(x,osatyoaste,'{}'.format(pstyle),label = '{} {}'.format(labeli2,leg))


        o_x,m_osatyo,f_osatyo = self.empstats.stats_parttimework()
        if grayscale:
            ax.plot(o_x,f_osatyo,ls = '--',label = self.labels['havainto, naiset'])
            ax.plot(o_x,m_osatyo,ls = '--',label = self.labels['havainto, miehet'])
        else:
            ax.plot(o_x,f_osatyo,label = self.labels['havainto, naiset'])
            ax.plot(o_x,m_osatyo,label = self.labels['havainto, miehet'])
        labeli = 'osatyöaste '#+str(ka_tyottomyysaste)
        ylabeli = 'Osatyön osuus työnteosta [%]'

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        if False:
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'osatyoaste_spk.'+self.figformat, format = self.figformat)
        plt.show()


    def plot_unemp_shares(self,ax=None,empstate=None):
        if empstate is None:
            empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive
        else:
            empstate_ratio = 100*empstate/self.episodestats.alive
        print(empstate_ratio)
        self.plot_states(empstate_ratio,ylabel = 'Osuus tilassa [%]',onlyunemp = True,stack = True,ax=ax)

    def plot_gender_emp(self,grayscale = False,figname = None,cc = None,diff = None,label1 = '',label2 = '',ax=None):
        if ax is None:
            fig,ax = plt.subplots()
            show = True
        else:
            show = False

        if grayscale:
            lstyle = '--'
        else:
            lstyle = '--'

        for gender in range(2):
            if gender== 0:
                leg = self.labels['Miehet']
                color = 'darkgray'
                gender = 'men'
            else:
                leg = self.labels['Naiset']
                color = 'black'
                gender = 'women'

            #tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios(gempstate,alive)
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios_gender(unempratio = True,gender = gender)

            if diff and cc is not None:
                x = np.linspace(self.min_age,self.max_age,self.n_time)
                tyollisyysaste2,osatyoaste,tyottomyysaste,ka_tyottomyysaste = cc.episodestats.comp_empratios_gender(unempratio = True,gender = gender)
                ax.plot(x,tyollisyysaste-tyollisyysaste2,color = color,label = '{} {} {}'.format(self.labels['tyollisyysaste %'],leg,label1+' - '+label2))
            else:
                if cc is not None:
                    x = np.linspace(self.min_age,self.max_age,self.n_time)
                    ax.plot(x,tyollisyysaste,color = color,label = '{} {} {}'.format(self.labels['tyollisyysaste %'],leg,label1))
                    tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = cc.episodestats.comp_empratios_gender(unempratio = True,gender = gender)
                    ax.plot(x,tyollisyysaste,color = color,ls = '-.',label = '{} {} {}'.format(self.labels['tyollisyysaste %'],leg,label2))
                else:
                    x = np.linspace(self.min_age,self.max_age,self.n_time)
                    ax.plot(x,tyollisyysaste,color = color,label = '{} {}'.format(self.labels['tyollisyysaste %'],leg))

        if not diff:
            emp_statsratio = 100*self.empstats.emp_stats(g = 2)
            ax.plot(x,emp_statsratio,ls = lstyle,color = 'darkgray',label = self.labels['havainto, miehet'])
            emp_statsratio = 100*self.empstats.emp_stats(g = 1)
            ax.plot(x,emp_statsratio,ls = lstyle,color = 'black',label = self.labels['havainto, naiset'])

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        if False:
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste_spk.'+self.figformat, format = self.figformat)

        if show:
            plt.show()
        
    def plot_group_emp(self,grayscale = False,figname = None):
        fig,ax = plt.subplots()
        if grayscale:
            lstyle = '--'
        else:
            lstyle = '--'

        for group in range(self.n_groups):
            if group<3:
                leg = self.labels['Miehet']+' group '+str(group)
                color = 'darkgray'
            else:
                leg = self.labels['Naiset']+' group '+str(group)
                color = 'black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios_groups(unempratio = True,g = group)
            #tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = self.episodestats.comp_empratios(gempstate,alive)

            x = np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,tyollisyysaste,label = '{} {}'.format(self.labels['tyollisyysaste %'],leg)) #color = color,

        emp_statsratio = 100*self.empstats.emp_stats(g = 2)
        ax.plot(x,emp_statsratio,ls = lstyle,color = 'darkgray',label = self.labels['havainto, miehet'])
        emp_statsratio = 100*self.empstats.emp_stats(g = 1)
        ax.plot(x,emp_statsratio,ls = lstyle,color = 'black',label = self.labels['havainto, naiset'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        if True:
            ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        else:
            ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyysaste_spk.'+self.figformat, format = self.figformat)

        plt.show()
        
    def plot_pensions(self):
        if self.version in self.complex_models:
            self.plot_states(self.episodestats.stat_pension,ylabel = 'Tuleva eläke [e/v]',stack = False)
            self.plot_states(self.episodestats.stat_paidpension,ylabel = 'Alkanut eläke [e/v]',stack = False)

    def plot_career(self):
        if self.version in self.complex_models:
            self.plot_states(self.episodestats.stat_tyoura,ylabel = 'Työuran pituus [v]',stack = False)

    def plot_ratiostates(self,statistic,ylabel = '',ylimit = None, show_legend = True, parent = False,work60 = False,\
                         unemp = False,stack = False,no_ve = False,figname = None,emp = False,oa_unemp = False,\
                         add_student = True,start_from = None,end_at = None):
        self.plot_states(statistic/self.episodestats.empstate[:statistic.shape[0],:statistic.shape[1]],ylabel = ylabel,ylimit = ylimit,no_ve = no_ve,\
                        show_legend = show_legend,parent = parent,unemp = unemp,add_student = add_student,\
                        stack = stack,figname = figname,emp = emp,oa_unemp = oa_unemp,work60 = work60,start_from = start_from,end_at = end_at)

    def count_putki(self,emps = None):
        if emps is None:
            piped = np.reshape(self.episodestats.empstate[:,4],(self.episodestats.empstate[:,4].shape[0],1))
            demog2 = self.empstats.get_demog()
            putkessa = self.timestep*np.nansum(piped[1:]/self.episodestats.alive[1:]*demog2[1:])
            return putkessa
        else:
            piped = np.reshape(emps[:,4],(emps[:,4].shape[0],1))
            demog2 = self.empstats.get_demog()
            alive = np.sum(emps,axis = 1,keepdims = True)
            putkessa = self.timestep*np.nansum(piped[1:]/alive[1:]*demog2[1:])
            return putkessa

    def plot_y(self,y1,y2 = None,y3 = None,y4 = None,label = '',ylabel = '',label2 = None,label3 = None,label4 = None,
            ylimit = None,show_legend = False,start_from = None,end_at = None,figname = None,
            yminlim = None,ymaxlim = None,grayscale = False,title = None,reverse = False):
        
        fig,ax = plt.subplots()
        if start_from is None:
            x = np.linspace(self.min_age,self.max_age,self.n_time)
        else:
            if end_at is None:
                end_at = self.max_age
            x_n = end_at-start_from+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))#+2
            x = np.linspace(start_from,self.max_age,x_t)
            y1 = y1[self.map_age(start_from):self.map_age(end_at)]
            if y2 is not None:
                y2 = y2[self.map_age(start_from):self.map_age(end_at)]
            if y3 is not None:
                y3 = y3[self.map_age(start_from):self.map_age(end_at)]
            if y4 is not None:
                y4 = y4[self.map_age(start_from):self.map_age(end_at)]

        if grayscale:
            pal = sns.light_palette("black", 8, reverse = True)
        else:
            pal = sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            
        if start_from is None:
            ax.set_xlim(self.min_age,self.max_age)
        else:
            ax.set_xlim(start_from,end_at)

        if title is None:
            plt.title(title)

        if ymaxlim is not None or yminlim is not None:
            ax.set_ylim(yminlim,ymaxlim)


        ax.plot(x,y1,label = label)
        if y2 is not None:
            ax.plot(x,y2,label = label2)
        if y3 is not None:
            ax.plot(x,y3,label = label3)
        if y4 is not None:
            ax.plot(x,y4,label = label4)

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabel)
        if show_legend:
            if not reverse:
                lgd = ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            else:
                handles, labels = ax.get_legend_handles_labels()
                lgd = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            
        #fig.tight_layout()
        if figname is not None:
            if show_legend:
                plt.savefig(figname,bbox_inches = 'tight',bbox_extra_artists = (lgd,), format = self.figformat)
            else:
                plt.savefig(figname,bbox_inches = 'tight', format = self.figformat)
        plt.show()

    def plot_groups(self,y1,y2 = None,y3 = None,y4 = None,label = '',ylabel = '',label2 = None,label3 = None,label4 = None,
            ylimit = None,show_legend = True,start_from = None,figname = None,
            yminlim = None,ymaxlim = None,grayscale = False,title = None,reverse = False):

        fig,ax = plt.subplots()
        if start_from is None:
            x = np.linspace(self.min_age,self.max_age,self.n_time)
        else:
            x_n = self.max_age-start_from+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))+1
            x = np.linspace(start_from,self.max_age,x_t)
            y1 = y1[self.map_age(start_from):,:]
            if y2 is not None:
                y2 = y2[self.map_age(start_from):,:]
            if y3 is not None:
                y3 = y3[self.map_age(start_from):,:]
            if y4 is not None:
                y4 = y4[self.map_age(start_from):,:]

        if grayscale:
            pal = sns.light_palette("black", 8, reverse = True)
        else:
            pal = sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
            
        if start_from is None:
            ax.set_xlim(self.min_age,self.max_age)
        else:
            ax.set_xlim(start_from,self.max_age)

        if title is None:
            plt.title(title)

        if ymaxlim is not None or yminlim is not None:
            ax.set_ylim(yminlim,ymaxlim)


        for g in range(6):
            ax.plot(x,y1[:,g],label = 'group '+str(g))
        if y2 is not None:
            ax.plot(x,y2,label = label2)
        if y3 is not None:
            ax.plot(x,y3,label = label3)
        if y4 is not None:
            ax.plot(x,y4,label = label4)

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabel)
        if show_legend:
            if not reverse:
                lgd = ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            else:
                handles, labels = ax.get_legend_handles_labels()
                lgd = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            
        #fig.tight_layout()
        if figname is not None:
            if show_legend:
                plt.savefig(figname,bbox_inches = 'tight',bbox_extra_artists = (lgd,), format = self.figformat)
            else:
                plt.savefig(figname,bbox_inches = 'tight', format = self.figformat)
        plt.show()

    def plot_states(self,statistic,ylabel = '',ylimit = None,show_legend = True,parent = False,unemp = False,no_ve = False,
                    start_from = None,end_at = None,stack = True,figname = None,yminlim = None,ymaxlim = None,work60 = False,
                    onlyunemp = False,reverse = False,grayscale = False,emp = False,oa_emp = False,oa_unemp = False,
                    all_emp = False,sv = False,normalize = False,add_student = True,ax = None,legend_infig = False):
        if start_from is None:
            x = np.linspace(self.min_age,self.max_age,self.n_time)
            x = x[:statistic.shape[0]]
        else:
            x_n = self.max_age-start_from+1
            x_t = int(np.round((x_n-1)*self.inv_timestep))+1
            x = np.linspace(start_from,self.max_age,x_t)
            statistic = statistic[self.map_age(start_from):]
            x = x[:statistic.shape[0]]

        ura_emp = statistic[:,1]
        ura_ret = statistic[:,2]
        ura_unemp = statistic[:,0]
        if self.version in self.complex_models:
            ura_disab = statistic[:,3]
            ura_pipe = statistic[:,4]
            ura_mother = statistic[:,5]
            ura_dad = statistic[:,6]
            ura_kht = statistic[:,7]
            ura_vetyo = statistic[:,9]
            ura_veosatyo = statistic[:,8]
            ura_osatyo = statistic[:,10]
            ura_outsider = statistic[:,11]
            if add_student:
                ura_student = statistic[:,12]+statistic[:,16]
            else:
                ura_student = statistic[:,12]
            ura_tyomarkkinatuki = statistic[:,13]
            ura_svpaiva = statistic[:,14]
        else:
            ura_osatyo = 0 #statistic[:,3]

        if normalize:
            ymaxlim = 1.00
            scale = np.sum(statistic,1)
            ura_emp /= scale
            ura_ret /= scale
            ura_unemp /= scale
            if self.version in self.complex_models:
                ura_disab /= scale
                ura_pipe /= scale
                ura_mother /= scale
                ura_dad /= scale
                ura_kht /= scale
                ura_vetyo /= scale
                ura_veosatyo /= scale
                ura_osatyo /= scale
                ura_outsider /= scale
                ura_student /= scale
                ura_tyomarkkinatuki /= scale
                ura_svpaiva /= scale
            else:
                ura_osatyo = 0 #statistic[:,3]

        if no_ve:
            ura_ret[-2:-1] = None

        if ax is None:
            fig,ax = plt.subplots()
            noshow = False
        else:
            noshow = True
            #fig,ax = plt.subplots()

        if stack:
            if grayscale:
                pal = sns.light_palette("black", 8, reverse = True)
            else:
                pal = sns.color_palette("colorblind", self.n_employment)  # hls, husl, cubehelix
            reverse = True

            if parent:
                if self.version in self.complex_models:
                    ax.stackplot(x,ura_mother,ura_dad,ura_kht,
                        labels = (self.labels['äitiysvapaa'],self.labels['isyysvapaa'],self.labels['khtuki']), colors = pal)
            elif unemp:
                if self.version in self.complex_models:
                    ax.stackplot(x,ura_unemp,ura_pipe,ura_student,ura_outsider,ura_tyomarkkinatuki,ura_svpaiva,
                        labels = (self.labels['tyött'],self.labels['putki'],self.labels['opiskelija'],self.labels['ulkona'],self.labels['tm-tuki'],self.labels['svpaivaraha']), colors = pal)
                else:
                    ax.stackplot(x,ura_unemp,labels = (self.labels['tyött']), colors = pal)
            elif onlyunemp:
                if self.version in self.complex_models:
                    #urasum = np.nansum(statistic[:,[0,4,11,13]],axis = 1)/100
                    urasum = np.nansum(statistic[:,[0,4,13]],axis = 1)/100
                    osuus = (1.0-np.array([0.84,0.68,0.62,0.58,0.57,0.55,0.53,0.50,0.29]))*100
                    xx = np.array([22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5])
                    ax.stackplot(x,ura_unemp/urasum,ura_pipe/urasum,ura_tyomarkkinatuki/urasum,
                        labels = (self.labels['ansiosidonnainen'],self.labels['lisäpäivät'],self.labels['tm-tuki']), colors = pal)
                    ax.plot(xx,osuus,color = 'k')
                else:
                    ax.stackplot(x,ura_unemp,labels = (self.labels['tyött']), colors = pal)
            else:
                if self.version in self.complex_models:
                    ax.stackplot(x,ura_emp,ura_osatyo,ura_vetyo,ura_veosatyo,ura_unemp,ura_tyomarkkinatuki,ura_pipe,ura_ret,ura_disab,ura_mother,ura_dad,ura_kht,ura_student,ura_outsider,ura_svpaiva,
                        labels = (self.labels['työssä'],self.labels['osatyö'],self.labels['ve+työ'],self.labels['ve+osatyö'],self.labels['työtön'],
                                self.labels['tm-tuki'],self.labels['työttömyysputki'],self.labels['vanhuuseläke'],self.labels['tk-eläke'],self.labels['äitiysvapaa'],
                                self.labels['isyysvapaa'],self.labels['khtuki'],self.labels['student'],self.labels['outsider'],self.labels['svpaivaraha']),
                        colors = pal)
                else:
                    ax.stackplot(x,ura_emp,ura_unemp,ura_ret,
                        labels = (self.labels['työssä'],self.labels['työtön'],self.labels['vanhuuseläke']), colors = pal)

            if ymaxlim is None:
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(yminlim,ymaxlim)
        else:
            if parent:
                if self.version in self.complex_models:
                    ax.plot(x,ura_mother,label = self.labels['äitiysvapaa'])
                    ax.plot(x,ura_dad,label = self.labels['isyysvapaa'])
                    ax.plot(x,ura_kht,label = self.labels['khtuki'])
            elif unemp:
                ax.plot(x,ura_unemp,label = self.labels['tyött'])
                if self.version in self.complex_models:
                    ax.plot(x,ura_tyomarkkinatuki,label = self.labels['tm-tuki'])
                    ax.plot(x,ura_student,label = self.labels['student'])
                    ax.plot(x,ura_outsider,label = self.labels['outsider'])
                    ax.plot(x,ura_pipe,label = self.labels['putki'])
            elif oa_unemp:
                ax.plot(x,ura_unemp,label = self.labels['tyött'])
                if self.version in self.complex_models:
                    ax.plot(x,ura_tyomarkkinatuki,label = self.labels['tm-tuki'])
                    ax.plot(x,ura_student,label = self.labels['student'])
                    ax.plot(x,ura_outsider,label = self.labels['outsider'])
                    ax.plot(x,ura_pipe,label = self.labels['putki'])
                    ax.plot(x,ura_osatyo,label = self.labels['osa-aika'])
            elif emp:
                ax.plot(x,ura_emp,label = self.labels['kokoaikatyö'])
                ax.plot(x,ura_osatyo,label = self.labels['osatyö'])
            elif oa_emp:
                ax.plot(x,ura_veosatyo,label = self.labels['ve+kokoaikatyö'])
                ax.plot(x,ura_vetyo,label = self.labels['ve+osatyö'])
            elif all_emp:
                ax.plot(x,ura_emp,label = self.labels['kokoaikatyö'])
                ax.plot(x,ura_osatyo,label = self.labels['osatyö'])
                ax.plot(x,ura_veosatyo,label = self.labels['ve+osatyö'])
                ax.plot(x,ura_vetyo,label = self.labels['ve+työ'])
            elif sv:
                ax.plot(x,ura_svpaiva,label = self.labels['sv-päiväraha'])
            elif work60:
                ax.plot(x,ura_ret,label = self.labels['eläke'])
                ax.plot(x,ura_emp,label = self.labels['kokoaikatyö'])
                if self.version in self.complex_models:
                    ax.plot(x,ura_osatyo,label = self.labels['osatyö'])
                    ax.plot(x,ura_vetyo,label = self.labels['ve+kokoaikatyö'])
                    ax.plot(x,ura_veosatyo,label = self.labels['ve+osatyö'])
            else:
                ax.plot(x,ura_unemp,label = self.labels['tyött'])
                ax.plot(x,ura_ret,label = self.labels['eläke'])
                ax.plot(x,ura_emp,label = self.labels['kokoaikatyö'])
                if self.version in self.complex_models:
                    ax.plot(x,ura_osatyo,label = self.labels['osatyö'])
                    ax.plot(x,ura_disab,label = self.labels['tk'])
                    ax.plot(x,ura_pipe,label = self.labels['putki'])
                    ax.plot(x,ura_tyomarkkinatuki,label = self.labels['tm-tuki'])
                    ax.plot(x,ura_mother,label = self.labels['äitiysvapaa'])
                    ax.plot(x,ura_dad,label = self.labels['isyysvapaa'])
                    ax.plot(x,ura_kht,label = self.labels['khtuki'])
                    ax.plot(x,ura_vetyo,label = self.labels['ve+kokoaikatyö'])
                    ax.plot(x,ura_veosatyo,label = self.labels['ve+osatyö'])
                    ax.plot(x,ura_student,label = self.labels['student'])
                    ax.plot(x,ura_outsider,label = self.labels['outsider'])
                    ax.plot(x,ura_svpaiva,label = self.labels['svpaivaraha'])
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabel)
        if show_legend:
            if legend_infig:
                lgd = ax.legend(bbox_to_anchor = (0.10, 0.5), loc = 2, borderaxespad = 0.)
            else:
                if not reverse:
                    lgd = ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
                else:
                    handles, labels = ax.get_legend_handles_labels()
                    lgd = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)

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
        if not noshow:
            if figname is not None:
                if show_legend:
                    plt.savefig(figname,bbox_inches = 'tight',bbox_extra_artists = (lgd,), format = self.figformat)
                else:
                    plt.savefig(figname,bbox_inches = 'tight', format = self.figformat)
            plt.show()

    def plot_toe(self):
        if self.version in self.complex_models:
            self.plot_ratiostates(self.episodestats.stat_toe,'työssäolo-ehdon pituus 28 kk aikana [v]',stack = False)

    def plot_sal(self):
        self.plot_states(self.episodestats.salaries_emp,'Keskipalkka [e/v]',stack = False)
        self.plot_states(self.episodestats.salaries_emp,'Keskipalkka [e/v]',stack = False,emp = True)
        self.plot_states(self.episodestats.salaries_emp,'Keskipalkka [e/v]',stack = False,all_emp = True)

    def plot_moved_to(self,state,name=None):
        if name is None:
            name = 'state '+str(state)
        siirtyneet_ratio = self.episodestats.siirtyneet_det[:,:,state]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel = f'Siirtyneet {name} tilasta',stack = True,
                        yminlim = 0,ymaxlim = min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        self.plot_states(siirtyneet_ratio,ylabel = f'Siirtyneet {name} tilasta',stack = True,
                        yminlim = 0,ymaxlim = 1,normalize = True)

    def plot_moved_from(self,state,name=None):
        if name is None:
            name = 'state '+str(state)
        siirtyneet_ratio = self.episodestats.siirtyneet_det[:,state,:]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel = f'Siirtyneet {name} tilasta',stack = True,
                        yminlim = 0,ymaxlim = min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        self.plot_states(siirtyneet_ratio,ylabel = f'Siirtyneet {name} tilasta',stack = True,
                        yminlim = 0,ymaxlim = 1,normalize = True)

    def plot_moved(self):
        siirtyneet_ratio = self.episodestats.siirtyneet/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel = 'Siirtyneet tilasta',stack = True,
                        yminlim = 0,ymaxlim = min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        pysyneet_ratio = self.episodestats.pysyneet/self.episodestats.alive*100
        self.plot_states(pysyneet_ratio,ylabel = 'Pysyneet tilassa',stack = True,
                        yminlim = 0,ymaxlim = min(100,1.1*np.nanmax(np.cumsum(pysyneet_ratio,1))))
        self.plot_moved_to(1,'työssä')
        self.plot_moved_to(0,'työtön (tila 0)')
        self.plot_moved_from(0,'työtön (tila 0)')
        self.plot_moved_from(13,'tm-tuki (tila 13)')
        self.plot_moved_to(13,'tm-tuki (tila 13)')
        self.plot_moved_to(10,'osa-aikatyö')
        self.plot_moved_to(11,'outsider')
        self.plot_moved_from(11,'outsider')
        self.plot_moved_from(12,'opiskelija')
        self.plot_moved_to(12,'opiskelija')
        self.plot_moved_to(14,'sv-päiväraha')
        self.plot_moved_from(14,'sv-päiväraha')
        self.plot_moved_to(2,'vanhuuseläke')
        self.plot_moved_from(7,'kotihoidontuki')
        self.plot_moved_from(8,'ve+oatyö')
        self.plot_moved_from(9,'ve+työ')
        self.plot_moved_from(4,'putki')
        self.plot_moved_to(4,'putki')

    def plot_ave_stay(self):
        self.plot_states(self.episodestats.time_in_state,ylabel = 'Ka kesto tilassa',stack = False)
        self.plot_y(self.episodestats.time_in_state[:,1],ylabel = 'Ka kesto työssä')
        self.plot_y(self.episodestats.time_in_state[:,0],ylabel = 'Ka kesto työttömänä')

    def plot_ove(self):
        self.plot_ratiostates(self.episodestats.infostats_ove,ylabel = 'Ove',stack = False,start_from = 60,add_student = False)
        #self.plot_ratiostates(np.sum(self.episodestats.infostats_ove,axis = 1),ylabel = 'Ove',stack = False)
        self.plot_y(np.sum(self.episodestats.infostats_ove,axis = 1)/self.episodestats.alive[:,0],ylabel = 'Oven ottaneet',start_from = 60,end_at = 70)
        self.plot_y((self.episodestats.infostats_ove[:,1]+self.episodestats.infostats_ove[:,10])/(self.episodestats.empstate[:,1]+self.episodestats.empstate[:,10]),label = 'Kaikista työllisistä',
            y2 = (self.episodestats.infostats_ove[:,1])/(self.episodestats.empstate[:,1]),label2 = 'Kokotyöllisistä',
            y3 = (self.episodestats.infostats_ove[:,10])/(self.episodestats.empstate[:,10]),label3 = 'Osatyöllisistä',
            ylabel = 'Oven ottaneet',show_legend = True,start_from = 60,end_at = 70)
        self.plot_y((self.episodestats.infostats_ove[:,0]+self.episodestats.infostats_ove[:,13]+self.episodestats.infostats_ove[:,4])/(self.episodestats.empstate[:,0]+self.episodestats.empstate[:,13]+self.episodestats.empstate[:,4]),label = 'Kaikista työttömistä',
            y2 = (self.episodestats.infostats_ove[:,0])/(self.episodestats.empstate[:,0]),label2 = 'Ansiosid.',
            y3 = (self.episodestats.infostats_ove[:,4])/(self.episodestats.empstate[:,4]),label3 = 'Putki',
            y4 = (self.episodestats.infostats_ove[:,13])/(self.episodestats.empstate[:,13]),label4 = 'TM-tuki',
            ylabel = 'Oven ottaneet',show_legend = True,start_from = 60,end_at = 70)
        
        ovex = np.sum(self.episodestats.infostats_ove_g*np.maximum(1,self.episodestats.gempstate),axis = 1)/np.maximum(1,np.sum(self.episodestats.gempstate,axis = 1))
        self.plot_groups(ovex,start_from = 60,ylabel = 'Osuus')

    def plot_reward(self):
        self.plot_ratiostates(self.episodestats.rewstate,ylabel = 'Keskireward tilassa',stack = False)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel = 'Keskireward tilassa',stack = False,no_ve = True)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel = 'Keskireward tilassa',stack = False,oa_unemp = True)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel = 'Keskireward tilassa',stack = False,oa_unemp = True,start_from = 60)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel = 'Keskireward tilassa',stack = False,start_from = 60)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel = 'Keskireward tilassa',stack = False,start_from = 60,work60 = True)
        #self.plot_y(np.sum(self.episodestats.rewstate,axis = 1),label = 'Koko reward tilassa')

    def vector_to_array(self,x):
        return x[:,None]

    def plot_wage_reduction_compare(self,cc2,label1 = 'HO',label2 = 'baseline'):
        emp = np.array([1,10])
        unemp = np.array([0,13,4])
        gen_red = np.sum(self.episodestats.stat_wage_reduction*self.episodestats.empstate,axis = 1)[:,None]/np.maximum(1,self.episodestats.alive)
        gen_red2 = np.sum(cc2.episodestats.stat_wage_reduction*cc2.episodestats.empstate,axis = 1)[:,None]/np.maximum(1,cc2.episodestats.alive)
        print('average wage reduction',np.mean(self.episodestats.stat_wage_reduction),np.mean(cc2.episodestats.stat_wage_reduction))
        print(label1,label2)
        self.plot_y(gen_red,y2 = gen_red2,ylabel = 'Average wage reduction',label = label1,label2 = label2,show_legend = True)
        for k in range(16):
            print(f'state {k}',np.mean(self.episodestats.stat_wage_reduction[:,k]),np.mean(cc2.episodestats.stat_wage_reduction[:,k]))
        print('Ages 30-60')
        for k in range(16):
            wr1 = np.mean(self.episodestats.stat_wage_reduction[self.map_age(30):self.map_age(60),k])
            wr2 = np.mean(cc2.episodestats.stat_wage_reduction[self.map_age(30):self.map_age(60),k])
            print(f'state {k}',wr1,wr2,wr1-wr2)

        for k in range(16):
            gen_red = self.episodestats.stat_wage_reduction[:,k]#/self.episodestats.empstate[:,k]
            gen_red2 = cc2.episodestats.stat_wage_reduction[:,k]#/cc2.episodestats.empstate[:,k]
            self.plot_y(gen_red,y2 = gen_red2,ylabel = f'Average wage reduction state {k}',label = label1,label2 = label2,show_legend = True)

    def plot_wage_reduction(self):
        emp = np.array([1,10])
        unemp = np.array([0,13,4])
        gen_red = np.sum(self.episodestats.stat_wage_reduction*self.episodestats.empstate,axis = 1)[:,None]/np.maximum(1,self.episodestats.alive)
        print(np.mean(self.episodestats.stat_wage_reduction),np.sum(self.episodestats.stat_wage_reduction*self.episodestats.empstate),np.sum(self.episodestats.alive))
        for k in range(16):
            print(k,np.mean(self.episodestats.stat_wage_reduction[:,k]))
        for k in range(16):
            print(k,np.mean(self.episodestats.stat_wage_reduction[self.map_age(30):self.map_age(60),k]))
        self.plot_y(gen_red,ylabel = 'Average wage reduction')
        gen_red_w = np.sum(np.sum(self.episodestats.stat_wage_reduction_g[:,:,0:3]*self.episodestats.gempstate[:,:,0:3],axis = 2),axis = 1)[:,None]/np.maximum(1,np.sum(self.episodestats.galive[:,0:3],axis = 1))[:,None]
        gen_red_m = np.sum(np.sum(self.episodestats.stat_wage_reduction_g[:,:,3:6]*self.episodestats.gempstate[:,:,3:6],axis = 2),axis = 1)[:,None]/np.maximum(1,np.sum(self.episodestats.galive[:,3:6],axis = 1))[:,None]
        self.plot_y(gen_red_w,ylabel = 'Average wage reduction by gender',label = 'women',y2 = gen_red_m,label2 = 'men',show_legend = True)
        #gen_red = np.sum(self.episodestats.stat_wage_reduction[:,emp]*self.episodestats.empstate[:,emp],axis = 1)[:,None]/np.maximum(1,np.sum(self.episodestats.empstate[:,emp],axis = 1))
        #gen_red = self.episodestats.stat_wage_reduction[:,emp]/np.maximum(1,self.episodestats.empstate[:,emp])
        #self.plot_y(gen_red,ylabel = 'Employed wage reduction',show_legend = True)
        #gen_red = np.sum(self.episodestats.stat_wage_reduction[:,unemp]*self.episodestats.empstate[:,unemp],axis = 1)[:,None]/np.maximum(1,np.sum(self.episodestats.empstate[:,unemp],axis = 1))
        #gen_red = self.episodestats.stat_wage_reduction[:,unemp]/np.maximum(1,self.episodestats.empstate[:,unemp])
        #self.plot_y(gen_red,ylabel = 'Unemployed wage reduction',show_legend = True)
        self.plot_states(self.episodestats.stat_wage_reduction,ylabel = 'wage-reduction tilassa',stack = False)
        self.plot_states(self.episodestats.stat_wage_reduction,ylabel = 'wage-reduction tilassa',stack = False,unemp = True)
        self.plot_states(self.episodestats.stat_wage_reduction,ylabel = 'wage-reduction tilassa',stack = False,emp = True)
        #self.plot_ratiostates(np.log(1.0+self.episodestats.stat_wage_reduction),ylabel = 'log 5wage-reduction tilassa',stack = False)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,0:3],axis = 2),ylabel = 'wage-reduction tilassa naiset',stack = False)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,3:6],axis = 2),ylabel = 'wage-reduction tilassa miehet',stack = False)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,0:3],axis = 2),ylabel = 'wage-reduction tilassa, naiset',stack = False,unemp = True)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,3:6],axis = 2),ylabel = 'wage-reduction tilassa, miehet',stack = False,unemp = True)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,0:3],axis = 2),ylabel = 'wage-reduction tilassa, naiset',stack = False,emp = True)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,3:6],axis = 2),ylabel = 'wage-reduction tilassa, miehet',stack = False,emp = True)

    def plot_distrib(self,label = '',plot_emp = False,plot_bu = False,ansiosid = False,tmtuki = False,putki = False,outsider = False,max_age = 500,laaja = False,max = 4,figname = None):
        unemp_distrib,emp_distrib,unemp_distrib_bu = self.episodestats.comp_empdistribs(ansiosid = ansiosid,tmtuki = tmtuki,putki = putki,outsider = outsider,max_age = max_age,laaja = laaja)
        tyoll_distrib,tyoll_distrib_bu = self.episodestats.comp_tyollistymisdistribs(ansiosid = ansiosid,tmtuki = tmtuki,putki = putki,outsider = outsider,max_age = max_age,laaja = laaja)

        print(label)
        if plot_emp:
            self.plot_empdistribs(emp_distrib)
        if plot_bu:
            self.plot_unempdistribs_bu(unemp_distrib_bu)
        else:
            self.plot_unempdistribs(unemp_distrib,figname = figname)

        #self.plot_tyolldistribs(unemp_distrib,tyoll_distrib,tyollistyneet = False)
        if plot_bu:
            self.plot_tyolldistribs_both_bu(unemp_distrib_bu,tyoll_distrib_bu,max = max)
        else:
            self.plot_tyolldistribs_both(unemp_distrib,tyoll_distrib,max = max,figname = figname)

    def plot_irr(self,figname = '',grayscale = False):
        self.plot_aggirr()
        self.plot_aggirr(gender = 1)
        self.plot_aggirr(gender = 2)
        if self.episodestats.save_pop:
            self.episodestats.comp_irr()
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_reduced,figname = figname+'_reduced',reduced = True,grayscale = grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_full,figname = figname+'_full',grayscale = grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_full,figname = figname+'_full_naiset',gender = 1,grayscale = grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_full,figname = figname+'_full_miehet',gender = 2,grayscale = grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_reduced,figname = figname+'_red_naiset',gender = 1,grayscale = grayscale,reduced = True)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_reduced,figname = figname+'_red_miehet',gender = 2,grayscale = grayscale,reduced = True)

    def plot_aggirr(self,gender = None):
        '''
        Laskee aggregoidun sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        
        if not self.episodestats.save_pop:
            print('test_salaries: not enough data (save_pop = False)')
            return

        if gender is None:
            gendername = 'Kaikki'
        else:
            gendername = self.get_gendername(gender)

        agg_irr_tyel_full,agg_irr_tyel_reduced,agg_premium,agg_pensions_reduced,agg_pensions_full,maxnpv = self.episodestats.comp_aggirr(gender = gender,full = True)

        print('{}: aggregate irr tyel reduced {:.4f} % reaalisesti'.format(gendername,agg_irr_tyel_reduced))
        print('{}: aggregate irr tyel full {:.4f} % reaalisesti'.format(gendername,agg_irr_tyel_full))
        
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('pension/premium')
        ax.plot(x[:-1],agg_pensions_full[:-1],label = 'työeläkemeno')
        ax.plot(x[:-1],agg_pensions_reduced[:-1],label = 'yhteensovitettu työeläkemeno')
        ax.plot(x[:-1],agg_premium[:-1],label = 'työeläkemaksu')
        ax.legend()
        plt.show()
        
    def compare_irr(self,cc2 = None,cc3 = None,cc4 = None,cc5 = None,cc6 = None,label1 = '',label2 = '',label3 = '',label4 = '',label5 = '',label6 = '',figname = '',grayscale = False):
        self.episodestats.comp_irr()
        full1 = self.episodestats.infostats_irr_tyel_full
        reduced1 = self.episodestats.infostats_irr_tyel_reduced
        full2 = None
        reduced2 = None
        full3 = None
        reduced3 = None
        full4 = None
        reduced4 = None
        full5 = None
        reduced5 = None
        full6 = None
        reduced6 = None
        if cc2 is not None:
            cc2.episodestats.comp_irr()
            full2 = cc2.episodestats.infostats_irr_tyel_full
            reduced2 = cc2.episodestats.infostats_irr_tyel_reduced
        if cc3 is not None:
            cc3.episodestats.comp_irr()
            full3 = cc3.episodestats.infostats_irr_tyel_full
            reduced3 = cc3.episodestats.infostats_irr_tyel_reduced
        if cc4 is not None:
            cc4.episodestats.comp_irr()
            full4 = cc4.episodestats.infostats_irr_tyel_full
            reduced4 = cc4.episodestats.infostats_irr_tyel_reduced
        if cc5 is not None:
            cc5.episodestats.comp_irr()
            full5 = cc5.episodestats.infostats_irr_tyel_full
            reduced5 = cc5.episodestats.infostats_irr_tyel_reduced
        if cc6 is not None:
            cc6.episodestats.comp_irr()
            full6 = cc6.episodestats.infostats_irr_tyel_full
            reduced6 = cc6.episodestats.infostats_irr_tyel_reduced
        
        self.compare_irrdistrib(cc2 = cc2,cc3 = cc3,cc4 = cc4,cc5 = cc5,cc6 = cc6,label1 = label1,label2 = label2,label3 = label3,label4 = label4,label5 = label5,label6 = label6,reduced = False,grayscale = grayscale)
        self.compare_irrdistrib(cc2 = cc2,cc3 = cc3,cc4 = cc4,cc5 = cc5,cc6 = cc6,figname = figname+'_reduced',reduced = True,label1 = label1,label2 = label2,label3 = label3,label4 = label4,label5 = label5,label6 = label6,grayscale = grayscale)
        self.compare_irrdistrib(cc2 = cc2,cc3 = cc3,cc4 = cc4,cc5 = cc5,cc6 = cc6,figname = figname+'_full_naiset',label1 = label1,label2 = label2,label3 = label3,label4 = label4,label5 = label5,label6 = label6,reduced = False,gender = 1,grayscale = grayscale)
        self.compare_irrdistrib(cc2 = cc2,cc3 = cc3,cc4 = cc4,cc5 = cc5,cc6 = cc6,figname = figname+'_full_miehet',label1 = label1,label2 = label2,label3 = label3,label4 = label4,label5 = label5,label6 = label6,reduced = False,gender = 2,grayscale = grayscale)

    def filter_irrdistrib(self,cc,gender = None,reduced = False):
        '''
        Suodata irrit sukupuolen mukaan
        '''
        mortstate = self.env.get_mortstate()
        
        if reduced:
            irr_distrib = cc.episodestats.infostats_irr_tyel_reduced
        else:
            irr_distrib = cc.episodestats.infostats_irr_tyel_full
        
        gendername = self.get_gendername(gender)
        gendermask = self.episodestats.get_gendermask(gender)
        #if gender is None:
        #    gendermask = np.ones_like(irr_distrib)
        #else:
        #    if gender== 1: # naiset
        #        gendermask = cc.episodestats.infostats_group>2
        #    else: # miehet
        #        gendermask = cc.episodestats.infostats_group<3
                
                
        nanmask = ma.mask_or(np.isnan(irr_distrib),gendermask.astype(bool))
        #nanmask = (np.isnan(irr_distrib).astype(int)+gendermask.astype(int)).astype(bool)
        v2_irrdata = ma.array(irr_distrib,mask = nanmask)
        v2data = v2_irrdata.compressed() # nans dropped
        
        return v2data

    def compare_takuu_kansanelake(self):
        suhde,n_minus,lkm_takuuelake,lkm_kansanelake = self.episodestats.comp_pop_takuuelake()
        print('Kansaneläkkeen saajia {:.2f} tukineljännesvuotta'.format(lkm_kansanelake))
        print('Takuueläkkeen saajia {:.2f} tukineljännesvuotta'.format(lkm_takuuelake))
        print('Takuueläke > 0, kansaneläke = 0, lkm {:.2f} tukineljännesvuotta'.format(n_minus))
        print('Takuueläke / Kansaneläke {:.2f} %'.format(100*suhde))
    
    def get_gendername(self,gender):
        if gender is None:
            gendername = ''
        else:
            if gender== 1: # naiset
                gendername = ' (Naiset)'
            else: # miehet
                gendername = ' (Miehet)'
                
        return gendername

    def plot_irrdistrib(self,irr_distrib,grayscale = True,figname = '',reduced = False,gender = None):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        mortstate = self.env.get_mortstate()
        
        gendername = self.get_gendername(gender)
        gendermask = self.episodestats.get_gendermask(gender)
        
        nanmask = ma.mask_or(np.isnan(irr_distrib),gendermask.astype(bool))
        #nanmask = (np.isnan(irr_distrib).astype(int)+gendermask.astype(int)).astype(bool)
        v2_irrdata = ma.array(irr_distrib,mask = nanmask)
        v2data = v2_irrdata.compressed() # nans dropped
        
        ika = 65
        alivemask = self.episodestats.popempstate[self.map_age(ika),:] != mortstate
        alivemask = np.reshape(alivemask,(-1,1))
        alivemask = ma.mask_or(alivemask,gendermask.astype(bool))
        #alivemask = (alivemask.astype(int)*gendermask.astype(int)).astype(bool)
        
        #nan_alivemask = alivemask*nanmask[:,0]
        #nanalive_irrdata = ma.array(irr_distrib,mask = nan_alivemask)
        #nanalive_data = nanalive_irrdata.compressed() # nans dropped

        if reduced:
            print('\nTyel-irr huomioiden kansan- ja takuueläkkeen yhteensovitus'+gendername)
        else:
            print('\nTyel-irr ILMAN kansan- ja takuueläkkeen yhteensovitusta'+gendername)

        #fig,ax = plt.subplots()
        #ax.set_xlabel('Sisäinen tuottoaste [%]')
        #lbl = ax.set_ylabel('Taajuus')
        #x = np.linspace(-7,7,100)
        #scaled,x2 = np.histogram(irr_distrib,x)
        #ax.bar(x2[1:-1],scaled[1:],align = 'center')
        #if figname is not None:
        #    plt.savefig(figname+'irrdistrib.'+self.figformat, bbox_inches = 'tight', format = self.figformat)
        #plt.show()

        self.plot_one_irrdistrib(v2data,label1 = 'Sisäinen tuotto [%]')

        # v2
        print('Kaikki havainnot ilman NaNeja:\n- Keskimääräinen irr {:.3f} % reaalisesti'.format(np.mean(v2data)))
        print('- Mediaani irr {:.3f} % reaalisesti'.format(np.median(v2data)))
        print('- Nans {} %'.format(100*np.sum(np.isnan(irr_distrib))/irr_distrib.shape[0]))

        # kaikki havainnot
        percent_m50 = 100*np.true_divide((irr_distrib  <= -50).sum(axis = 0),irr_distrib.shape[0])[0]
        percent_m10 = 100*np.true_divide((irr_distrib  <= -10).sum(axis = 0),irr_distrib.shape[0])[0]
        percent_0 =   100*np.true_divide((irr_distrib  <= 0).sum(axis = 0),irr_distrib.shape[0])[0]
        percent_p10 = 100*np.true_divide((irr_distrib  >= 10).sum(axis = 0),irr_distrib.shape[0])[0]
        print(f'Kaikki havainnot\nOsuudet\n- irr < -50% {percent_m50:.2f} %:lla\n- irr < -10% {percent_m10:.2f} %')
        print(f'- irr < 0% {percent_0:.2f} %:lla\n- irr > 10% {percent_p10:.2f} %\n')
        
        # v2
        percent_m50 = 100*((v2data  <= -50).sum(axis = 0)/v2data.shape[0])
        percent_m10 = 100*((v2data  <= -10).sum(axis = 0)/v2data.shape[0])
        percent_0 =   100*((v2data  <= 0).sum(axis = 0)/v2data.shape[0])
        percent_p10 = 100*((v2data  >= 10).sum(axis = 0)/v2data.shape[0])
        print(f'Ilman NaNeja\nOsuudet\n- irr < -50% {percent_m50:.2f} %:lla\n- irr < -10% {percent_m10:.2f} %')
        print(f'- irr < 0% {percent_0:.2f} %:lla\n- irr > 10% {percent_p10:.2f} %\n')

        count = (np.sum(self.episodestats.infostats_pop_paidpension,axis = 0)<0.1).sum(axis = 0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus mikään eläke ei lainkaan maksussa {:.2f} %'.format(100*percent))
        
        no_pension = (np.sum(self.episodestats.infostats_pop_tyoelake,axis = 0)<0.1)
        count = no_pension.sum(axis = 0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläkettä ei lainkaan maksussa {:.2f} %'.format(100*percent))
        
        nopremium = (np.sum(self.episodestats.infostats_tyelpremium,axis = 0)<0.1)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei lainkaan maksua maksettu {:.2f} %'.format(100*percent))
        
        count = (np.sum(self.episodestats.infostats_paid_tyel_pension,axis = 0)<0.1).sum(axis = 0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei vastinetta maksulle {:.2f} %'.format(100*percent))
        
        nopp = no_pension*nopremium
        count = nopp.sum(axis = 0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei lainkaan maksua eikä maksettua eläkettä {:.2f} %'.format(100*percent))

        print('\nOsuudet\n')
        arri = ma.sum(ma.array(self.episodestats.infostats_pop_paidpension[self.map_age(ika),:],mask = alivemask))<0.1
        percent = np.true_divide(ma.sum(arri),self.episodestats.alive[self.map_age(ika),0])
        print('{}v osuus eläke ei maksussa, ei kuollut {:.2f} %'.format(ika,100*percent))

        alivemask = self.episodestats.popempstate[self.map_age(ika),:] != mortstate
        count = (np.sum(self.episodestats.infostats_pop_tyoelake,axis = 0)<0.1).sum(axis = 0)
        percent = np.true_divide(count,self.episodestats.alive[self.map_age(ika),0])
        print('{}v osuus ei työeläkekarttumaa, ei kuollut {:.2f} %'.format(ika,100*percent))

        count = 1-self.episodestats.alive[self.map_age(ika),0]/self.episodestats.n_pop
        print('{}v osuus kuolleet {:.2f} % '.format(ika,100*count))

        count = 1-self.episodestats.alive[-1,0]/self.episodestats.n_pop
        print('Lopussa osuus kuolleet {:.2f} % '.format(100*count))
        
    def plot_one_irrdistrib(self,irr_distrib1,label1 = '1',
                                 irr2 = None,label2 = '2',
                                 irr3 = None,label3 = '2',
                                 irr4 = None,label4 = '2',
                                 irr5 = None,label5 = '2',
                                 irr6 = None,label6 = '2',
                                 grayscale = False,figname = ''):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
        
        df1 = pd.DataFrame(irr_distrib1,columns = ['Sisäinen tuottoaste [%]'])
        df1.loc[:,'label'] = label1
        if irr2 is not None:
            df2 = pd.DataFrame(irr2,columns = ['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label'] = label2
            df1 = pd.concat([df1,df2],ignore_index = True, sort = False)
        if irr3 is not None:
            df2 = pd.DataFrame(irr3,columns = ['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label'] = label3
            df1 = pd.concat([df1,df2],ignore_index = True, sort = False)
        if irr4 is not None:
            df2 = pd.DataFrame(irr4,columns = ['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label'] = label4
            df1 = pd.concat([df1,df2],ignore_index = True, sort = False)
        if irr5 is not None:
            df2 = pd.DataFrame(irr5,columns = ['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label'] = label5
            df1 = pd.concat([df1,df2],ignore_index = True, sort = False)
        if irr6 is not None:
            df2 = pd.DataFrame(irr6,columns = ['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label'] = label6
            df1 = pd.concat([df1,df2],ignore_index = True, sort = False)

        sns.displot(df1, x = "Sisäinen tuottoaste [%]", hue = "label", kind = "kde", fill = True,gridsize = 10000, bw_adjust = .05)
        plt.xlim(-10, 10)
        if figname is not None:
            plt.savefig(figname+'_kde.'+self.figformat, format = self.figformat)
        plt.show()
        
        sns.displot(df1, x = "Sisäinen tuottoaste [%]", hue = "label", stat = "density", fill = True,common_norm = False)
        plt.xlim(-10, 10)
        if figname is not None:
            plt.savefig(figname+'density.'+self.figformat, format = self.figformat)
        plt.show()

    def compare_gender_irr(self,figname = None,reduced = False,grayscale = False):
        self.episodestats.comp_irr()
        irr1 = self.filter_irrdistrib(self,reduced = reduced,gender = 1)
        irr2 = self.filter_irrdistrib(self,reduced = reduced,gender = 2)

        display(irr1)
        display(irr2)

        self.plot_one_irrdistrib(irr1,label1 = 'naiset',irr2 = irr2,label2 = 'miehet',
                                 grayscale = grayscale,figname = figname)

    def compare_reduced_irr(self,figname = None,gender = None,grayscale = False):
        self.episodestats.comp_irr()
        irr1 = self.filter_irrdistrib(self,reduced = True,gender = gender)
        irr2 = self.filter_irrdistrib(self,reduced = False,gender = gender)

        display(irr1)
        display(irr2)

        self.plot_one_irrdistrib(irr1,label1 = 'Yhteensovitettu',irr2 = irr2,label2 = 'Pelkkä työeläke',
                                 grayscale = grayscale,figname = figname)
        
    def compare_irrdistrib(self,cc2 = None,cc3 = None,cc4 = None,cc5 = None,cc6 = None,
            label1 = '',label2 = '',label3 = '',label4 = '',label5 = '',label6 = '',figname = None,reduced = False,gender = None,grayscale = False):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        gendername = self.get_gendername(gender)
        if reduced:
            print('\nTyel-irr huomioiden kansan- ja takuueläkkeen yhteensovitus'+gendername)
        else:
            print('\nTyel-irr ILMAN kansan- ja takuueläkkeen yhteensovitusta'+gendername)
            
        irr_distrib = self.filter_irrdistrib(self,reduced = reduced,gender = gender)
        aggirr,aggirr_reduced = self.episodestats.comp_aggirr()
        print(f'Aggregate irr for {label1}: {aggirr} (reduced {aggirr_reduced})')
        irr2 = None
        irr3 = None
        irr4 = None
        irr5 = None
        irr6 = None
        if cc2 is not None:
            irr2 = self.filter_irrdistrib(cc2,reduced = reduced,gender = gender)
            aggirr,aggirr_reduced = cc2.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label2}: {aggirr} (reduced aggirr_reduced)')
        if cc3 is not None:
            irr3 = self.filter_irrdistrib(cc3,reduced = reduced,gender = gender)
            aggirr,aggirr_reduced = cc3.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label3}: {aggirr} (reduced aggirr_reduced)')
        if cc4 is not None:
            irr4 = self.filter_irrdistrib(cc4,reduced = reduced,gender = gender)
            aggirr,aggirr_reduced = cc4.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label4}: {aggirr} (reduced aggirr_reduced)')
        if cc5 is not None:
            irr5 = self.filter_irrdistrib(cc5,reduced = reduced,gender = gender)
            aggirr,aggirr_reduced = cc5.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label5}: {aggirr} (reduced aggirr_reduced)')
        if cc6 is not None:
            irr6 = self.filter_irrdistrib(cc6,reduced = reduced,gender = gender)
            aggirr,aggirr_reduced = cc6.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label6}: {aggirr} (reduced aggirr_reduced)')


        self.plot_one_irrdistrib(irr_distrib,label1 = label1,
                                 irr2 = irr2,label2 = label2,
                                 irr3 = irr3,label3 = label3,
                                 irr4 = irr4,label4 = label4,
                                 irr5 = irr5,label5 = label5,
                                 irr6 = irr6,label6 = label6,
                                 grayscale = grayscale,figname = figname)
                                 
    def plot_scaled_irr(self,label = '',figname = None,reduced = False,gender = None,grayscale = False):
        '''
        Laskee annetulla maksulla irr:t eri maksuille skaalaamalla
        '''
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        irr_distrib = self.filter_irrdistrib(self,gender = gender)
        aggirr,aggirr_reduced,agg_premium,agg_pensions_reduced,agg_pensions_full,maxnpv = self.episodestats.comp_aggirr(full = True)
        print(f'Aggregate irr for {label}: {aggirr} (reduced {aggirr_reduced})')
        
        x = np.arange(0.01,0.35,0.01)
        agg_irr_tyel_full = np.zeros_like(x)
        agg_irr_tyel_reduced = np.zeros_like(x)
        for m,r in enumerate(x):
            agg_premium2 = agg_premium*r/0.244
            agg_irr_tyel_full[m] = self.reaalinen_palkkojenkasvu*100+self.episodestats.comp_annual_irr(maxnpv,agg_premium2,agg_pensions_full)
            agg_irr_tyel_reduced[m] = self.reaalinen_palkkojenkasvu*100+self.episodestats.comp_annual_irr(maxnpv,agg_premium2,agg_pensions_reduced)
            
        df1 = pd.DataFrame(agg_irr_tyel_full,columns = ['pelkkä työeläke'])
        df1.loc[:,'x'] = x
        df1.loc[:,'yhteensovitettu'] = agg_irr_tyel_reduced
            
        ps = np.zeros_like(agg_irr_tyel_full)+1.6
        
        if reduced:
            lineplot(x*100,agg_irr_tyel_full,y2 = agg_irr_tyel_reduced,y3 = ps,xlim = [10,30],ylim = [-1,6],
                ylabel = 'sisäinen tuotto [%]',xlabel = 'maksutaso [% palkoista]',selite = True,
                label = 'Pelkkä työeläke',label2 = 'Yhteensovitettu eläke',label3 = 'Palkkasumman kasvu',figname = figname)
        else:
            lineplot(x*100,agg_irr_tyel_full,y2 = ps,xlim = [10,30],ylim = [-1,6],
                ylabel = 'sisäinen tuotto [%]',xlabel = 'maksutaso [% palkoista]',selite = True,
                label = 'Pelkkä työeläke',label2 = 'Palkkasumman kasvu',figname = figname)

    def plot_img(self,img,xlabel = "eläke",ylabel = "Palkka",title = "Employed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        heatmap = plt.pcolor(img)
        plt.colorbar(heatmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()

    def scatter_density(self,x,y,label1 = '',label2 = ''):
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        #idx = z.argsort()
        #x, y, z = x[idx], y[idx], z[idx]
        fig,ax = plt.subplots()
        ax.scatter(x,y,c = z)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        #plt.legend()
#        plt.title('number of agents by pension level')
        plt.show()

    def plot_Vk(self,k,getV = None):
        obsV = np.zeros(((self.n_time,1)))

        #obsV[-1] = self.poprewstate[-1,k]
        for t in range(self.n_time-2,-1,-1):
            obsV[t] = self.poprewstate[t+1,k]+self.gamma*obsV[t+1]
            rewerr = self.poprewstate[t+1,k]-self.pop_predrew[t+1,k]
            delta = obsV[t]-self.aveV[t+1,k]
            wage = self.episodestats.infostats_pop_wage[t,k]
            old_wage = self.episodestats.infostats_pop_wage[max(0,t-1),k]
            age = self.map_t_to_age(t)
            old_age = int(self.map_t_to_age(max(0,t-1)))
            emp = self.episodestats.popempstate[t,k]
            predemp = self.episodestats.popempstate[max(0,t-1),k]
            pen = self.episodestats.infostats_pop_pension[t,k]
            predwage = self.env.wage_process_mean(old_wage,old_age,state = predemp)
            print(f'{age}: {obsV[t]:.4f} vs {self.aveV[t+1,k]:.4f} d {delta:.4f} re {rewerr:.6f} in state {emp} ({k},{wage:.2f},{pen:.2f}) ({predwage:.2f}) ({self.poprewstate[t+1,k]:.5f},{self.pop_predrew[t+1,k]:.5f})')


        err = obsV-self.aveV[t,k]
        obsV_error = np.abs(obsV-self.aveV[t,k])


    def plot_Vstats(self):
        obsV = np.zeros((self.n_time,self.episodestats.n_pop))
        obsVemp = np.zeros((self.n_time,3))
        obsVemp_error = np.zeros((self.n_time,3))
        obsVemp_abserror = np.zeros((self.n_time,3))

        obsV[self.n_time-1,:] = self.poprewstate[self.n_time-1,:]
        for t in range(self.n_time-2,0,-1):
            obsV[t,:] = self.poprewstate[t,:]+self.gamma*obsV[t+1,:]
            delta = obsV[t,:]-self.aveV[t,:]
            for k in range(self.episodestats.n_pop):
                if np.abs(delta[k])>0.2:
                    wage = self.episodestats.infostats_pop_wage[t,k]
                    pen = self.episodestats.infostats_pop_pension[t,k]

            for state in range(2):
                s = np.asarray(self.episodestats.popempstate[t,:]== state).nonzero()
                obsVemp[t,state] = np.mean(obsV[t,s])
                obsVemp_error[t,state] = np.mean(obsV[t,s]-self.aveV[t,s])
                obsVemp_abserror[t,state] = np.mean(np.abs(obsV[t,s]-self.aveV[t,s]))

        err = obsV[:self.n_time]-self.aveV
        obsV_error = np.abs(err)

        mean_obsV = np.mean(obsV,axis = 1)
        mean_predV = np.mean(self.aveV,axis = 1)
        mean_abs_errorV = np.mean(np.abs(err),axis = 1)
        mean_errorV = np.mean(err,axis = 1)
        fig,ax = plt.subplots()
        ax.plot(mean_abs_errorV[1:],label = 'abs. error')
        ax.plot(mean_errorV[1:],label = 'error')
        ax.plot(np.max(err,axis = 1),label = 'max')
        ax.plot(np.min(err,axis = 1),label = 'min')
        ax.set_xlabel('time')
        ax.set_ylabel('error (pred-obs)')
        plt.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(mean_obsV[1:],label = 'observed')
        ax.plot(mean_predV[1:],label = 'predicted')
        ax.set_xlabel('time')
        ax.set_ylabel('V')
        plt.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(obsVemp[1:,0],label = 'state 0')
        ax.plot(obsVemp[1:,1],label = 'state 1')
        ax.set_xlabel('time')
        ax.set_ylabel('V')
        plt.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(obsVemp_error[1:,0],label = 'state 0')
        ax.plot(obsVemp_error[1:,1],label = 'state 1')
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        plt.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(obsVemp_abserror[1:,0],label = 'state 0')
        ax.plot(obsVemp_abserror[1:,1],label = 'state 1')
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        plt.legend()
        plt.show()

    def render(self,load = None,figname = None,grayscale = False):
        if load is not None:
            self.load_sim(load)

        self.plot_results(figname = figname,grayscale = grayscale)

    def compare_with(self,cc2,label1 = 'perus',label2 = 'vaihtoehto',grayscale = True,figname = None,dash = False,palette_EK = True):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        if palette_EK:
            csfont,pal = setup_EK_fonts()
        else:
            csfont = {}

        diff_emp = self.episodestats.empstate/self.episodestats.alive-cc2.episodestats.empstate/cc2.episodestats.alive
        diff_emp = diff_emp*100
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        real1 = self.episodestats.comp_presentvalue()
        real2 = cc2.episodestats.comp_presentvalue()
        mean_real1 = np.mean(real1,axis = 1)
        mean_real2 = np.mean(real2,axis = 1)
        initial1 = np.mean(real1[1,:])
        initial2 = np.mean(real2[1,:])

        self.compare_disab(cc = cc2, xstart = None,xend = None, label1 = label1,label2 = label2,figname=figname)

        self.plot_wage_reduction_compare(cc2,label1=label1,label2=label2)

        if self.episodestats.save_pop:
            rew1 = self.episodestats.comp_total_reward(discounted = True)
            rew2 = cc2.episodestats.comp_total_reward(discounted = True)
            net1,eqnet1 = self.episodestats.comp_total_netincome()
            net2,eqnet2 = cc2.episodestats.comp_total_netincome()

            print(f'{label1} reward {rew1} netincome {net1:.2f} eq {eqnet1:.3f} initial {initial1}')
            print(f'{label2} reward {rew2} netincome {net2:.2f} eq {eqnet2:.3f} initial {initial2}')

        gini1 = self.episodestats.comp_gini()
        gini2 = cc2.episodestats.comp_gini()
        print(f'Gini coefficient is {label1} {gini1:.3f} vs {label2} {gini2:.3f} ')

        if self.minimal>0:
            s = 20
            e = 70
        else:
            s = 21
            e = 60 #63.5

        emp_htv1 = np.sum(cc2.episodestats.emp_htv,axis = 1)
        emp_htv2 = np.sum(self.episodestats.emp_htv,axis = 1)

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1 = \
            self.episodestats.comp_employed_ratio(self.episodestats.empstate,emp_htv = emp_htv1)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2 = \
            cc2.episodestats.comp_employed_ratio(cc2.episodestats.empstate,emp_htv = emp_htv2)
        htv1,tyolliset1,tyottomat1,osatyolliset1,kokotyolliset1,tyollaste1,osatyolaste1,kokotyolaste1,tyottomyys_aste1 = \
            self.episodestats.comp_tyollisyys_stats(self.episodestats.empstate,scale_time = True,start = s,end = e,emp_htv = emp_htv1,agegroups = False)
        htv2,tyolliset2,tyottomat2,osatyolliset2,kokotyolliset2,tyollaste2,osatyolaste2,kokotyolaste2,tyottomyys_aste2 = \
            cc2.episodestats.comp_tyollisyys_stats(cc2.episodestats.empstate,scale_time = True,start = s,end = e,emp_htv = emp_htv2,agegroups = False)

        tyollaste1 = tyollaste1*100
        tyollaste2 = tyollaste2*100
        tyotaste1 = self.episodestats.comp_unemp_stats_agg(per_pop = False)*100
        tyotaste2 = cc2.episodestats.comp_unemp_stats_agg(per_pop = False)*100
        tyovoimatutk_tyollaste = self.empstats.get_tyollisyysaste_tyovoimatutkimus(self.year)
        tyovoimatutk_tytaste = self.empstats.get_tyottomyysaste_tyovoimatutkimus(self.year)
        tyossakayntitutk_tyollaste = self.empstats.get_tyollisyysaste_tyossakayntitutkimus(self.year)
        tyossakayntitutk_tytaste = self.empstats.get_tyottomyysaste_tyossakayntitutkimus(self.year)
        print('\nSic! Työllisyysaste vastaa työvoimatilaston laskutapaa!')
        print(f'Työllisyysaste1 {s}-{e}: {tyollaste1:.2f}% (työvoimatutkimus {tyovoimatutk_tyollaste:.2f}%)')
        print(f'Työllisyysaste2 {s}-{e}: {tyollaste2:.2f}% (työvoimatutkimus {tyovoimatutk_tyollaste:.2f}%)')
        print(f'Työttömyysaste1 {s}-{e}: {tyotaste1:.2f}% (työvoimatutkimus {tyovoimatutk_tytaste:.2f}%)')
        print(f'Työttömyysaste2 {s}-{e}: {tyotaste2:.2f}% (työvoimatutkimus {tyovoimatutk_tytaste:.2f}%)')


        htv1_full,tyolliset1_full,tyottomat1_full,osata1_full,kokota1_full,tyollaste1_full,osatyo_osuus1_full,kokotyo_osuus1_full,tyot_osuus1_full,tyot_aste1 = \
            self.episodestats.comp_tyollisyys_stats(self.episodestats.empstate,scale_time = True,start = 18,end = 75,emp_htv = emp_htv1,agegroups = True)
        htv2_full,tyolliset2_full,tyottomat2_full,osata2_full,kokota2_full,tyollaste2_full,osatyo_osuus2_full,kokotyo_osuus2_full,tyot_osuus2_full,tyot_aste2 = \
            cc2.episodestats.comp_tyollisyys_stats(cc2.episodestats.empstate,scale_time = True,start = 18,end = 75,emp_htv = emp_htv2,agegroups = True)
        haj1 = self.episodestats.comp_uncertainty(self.episodestats.empstate,emp_htv = emp_htv1)
        haj2 = cc2.episodestats.comp_uncertainty(cc2.episodestats.empstate,emp_htv = emp_htv2)

        ansiosid_osuus1,tm_osuus1 = self.episodestats.comp_unemployed_detailed(self.episodestats.empstate)
        ansiosid_osuus2,tm_osuus2 = cc2.episodestats.comp_unemployed_detailed(cc2.episodestats.empstate)
        #khh_osuus1 = self.episodestats.comp_kht(self.episodestats.empstate)
        #khh_osuus2 = self.episodestats.comp_kht(cc2.empstate)

        #self.episodestats.comp_employment_stats()
        #cc2.episodestats.comp_employment_stats()

        self.compare_against(cc = cc2,cctext = label2)

#         q1 = self.episodestats.comp_budget(scale = True)
#         q2 = cc2.comp_budget(scale = True)
#         
#         df1 = pd.DataFrame.from_dict(q1,orient = 'index',columns = [label1])
#         df2 = pd.DataFrame.from_dict(q2,orient = 'index',columns = ['one'])
#         df = df1.copy()
#         df[label2] = df2['one']
#         df['diff'] = df1[label1]-df2['one']

        fig,ax = plt.subplots()
        ax.plot(x[1:self.n_time],mean_real1[1:self.n_time]-mean_real2[1:self.n_time],label = label1+'-'+label2)
        ax.legend()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('real diff')
        plt.show()

        fig,ax = plt.subplots()
        c1 = self.episodestats.comp_cumurewstate()
        c2 = cc2.episodestats.comp_cumurewstate()
        ax.plot(x,c1,label = label1)
        ax.plot(x,c2,label = label2)
        ax.legend()
        ax.set_xlabel('rev age')
        ax.set_ylabel('rew')
        plt.show()

        fig,ax = plt.subplots()
        ax.plot(x,c1-c2,label = label1+'-'+label2)
        ax.legend()
        ax.set_xlabel('rev age')
        ax.set_ylabel('rew diff')
        plt.show()

        if dash:
            ls = '--'
        else:
            ls = None

        lineplot(x,100*tyollaste1_full,y2 = 100*tyollaste2_full,xlim = [20,75],ylim = [0,100],label = label1,label2 = label2,xlabel = self.labels['age'],ylabel = self.labels['tyollisyysaste %'],selite = True,figname = figname+'emp.'+self.figformat)

        self.plot_gender_emp(cc = cc2,diff = False,label1 = label1,label2 = label2)
        self.plot_gender_emp(cc = cc2,diff = True,label1 = label1,label2 = label2)

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['diff osuus'])
        #print(tyot_osuus1.shape,tyot_osuus2.shape,kokotyo_osuus1.shape,kokotyo_osuus2.shape,osatyo_osuus1.shape,osatyo_osuus2.shape,tyolliset1.shape,tyolliset2.shape,htv_osuus1.shape,htv_osuus2.shape)
        ax.plot(x,100*(tyot_osuus1_full-tyot_osuus2_full),label = 'unemployment')
        ax.plot(x,100*(kokotyo_osuus1_full-kokotyo_osuus2_full),label = 'fulltime work')
        if self.version in self.complex_models:
            ax.plot(x,100*(osatyo_osuus1_full-osatyo_osuus2_full),label = 'osa-aikatyö')
            ax.plot(x,100*(tyollaste1_full-tyollaste2_full),label = 'työ yhteensä')
        ax.legend()
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['diff määrissä'])
        #print(tyot_osuus1.shape,tyot_osuus2.shape,kokotyo_osuus1.shape,kokotyo_osuus2.shape,osatyo_osuus1.shape,osatyo_osuus2.shape,tyolliset1.shape,tyolliset2.shape,htv_osuus1.shape,htv_osuus2.shape)
        ax.plot(x,100*(tyottomat1_full-tyottomat2_full),label = 'unemployment')
        ax.plot(x,100*(kokota1_full-kokota2_full),label = 'fulltime work')
        if self.version in self.complex_models:
            ax.plot(x,100*(osata1_full-osata2_full),label = 'osa-aikatyö')
            ax.plot(x,100*(tyolliset1_full-tyolliset2_full),label = 'työ yhteensä')
            ax.plot(x,100*(htv1_full-htv2_full),label = 'htv yhteensä')
        ax.legend()
        plt.show()        

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomyysaste'])
        ax.plot(x,100*tyot_osuus1_full,label = label1)
        ax.plot(x,100*tyot_osuus2_full,ls = ls,label = label2)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'unemp.'+self.figformat, format = self.figformat)
        plt.show()

        if self.minimal<0:
            fig,ax = plt.subplots()
            ax.set_xlabel(self.labels['age'])
            ax.set_ylabel('Kotihoidontuki [%]')
            #ax.plot(x,100*khh_osuus1,label = label1)
            #ax.plot(x,100*khh_osuus2,ls = ls,label = label2)
            ax.set_ylim([0,100])
            ax.legend()
            if figname is not None:
                plt.savefig(figname+'kht.'+self.figformat, format = self.figformat)
            plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Osatyö [%]')
        ax.plot(x,100*osatyo_osuus1_full,label = label1)
        ax.plot(x,100*osatyo_osuus2_full,ls = ls,label = label2)
        ax.set_ylim([0,100])
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'osatyo_osuus.'+self.figformat, format = self.figformat)
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['diff osuus'])
        ax.plot(x,100*ansiosid_osuus2,ls = ls,label = 'ansiosid. työttömyys, '+label2)
        ax.plot(x,100*ansiosid_osuus1,label = 'ansiosid. työttömyys, '+label1)
        ax.plot(x,100*tm_osuus2,ls = ls,label = 'tm-tuki, '+label2)
        ax.plot(x,100*tm_osuus1,label = 'tm-tuki, '+label1)
        ax.legend()
        plt.show()

        if self.language== 'English':
            print('Influence on employment of {:.0f}-{:.0f} years old approx. {:.0f} man-years and {:.0f} employed'.format(s,e,htv1-htv2,tyolliset1-tyolliset2))
            print('- full-time {:.0f}-{:.0f} y olds {:.0f} employed ({:.0f} vs {:.0f})'.format(s,e,(kokotyolliset1-kokotyolliset2),kokotyolliset1,kokotyolliset2))
            print('- part-time {:.0f}-{:.0f} y olds {:.0f} employed ({:.0f} vs {:.0f})'.format(s,e,(osatyolliset1-osatyolliset2),osatyolliset1,osatyolliset2))
            print('Employed {:.0f} vs {:.0f} man-years'.format(htv1,htv2))
            print('Influence on employment rate for {:.0f}-{:.0f} y olds {:.2f} % ({:.2f} vs {:.2f})'.format(s,e,(tyollaste1-tyollaste2)*100,tyollaste1*100,tyollaste2*100))
            print('- full-time {:.0f}-{:.0f} y olds {:.2f} % ({:.2f} vs {:.2f})'.format(s,e,(kokota1-kokota2)*100,kokota1*100,kokota2*100))
            print('- part-time {:.0f}-{:.0f} y olds {:.2f} % ({:.2f} vs {:.2f})'.format(s,e,(osata1-osata2)*100,osata1*100,osata2*100))
        else:
            print('Työllisyysvaikutus {:.0f}-{:.0f}-vuotiaisiin noin {:.0f} htv ja {:.0f} työllistä'.format(s,e,htv1-htv2,tyolliset1-tyolliset2))
            print('- kokoaikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(kokotyolliset1-kokotyolliset2),kokotyolliset1,kokotyolliset2))
            print('- osa-aikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.0f} työllistä ({:.0f} vs {:.0f})'.format(s,e,(osatyolliset1-osatyolliset2),osatyolliset1,osatyolliset2))
            print(f'Työllisiä {s:.0f}-{e:.0f} -vuotiaissa: {htv1:.0f} vs {htv2:.0f} htv')
            print('Työllisyysastevaikutus {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(tyollaste1-tyollaste2),tyollaste1,tyollaste2))
            print('- kokoaikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(kokotyolaste1-kokotyolaste2)*100,kokotyolaste2*100,kokotyolaste2*100))
            print('- osa-aikaisiin {:.0f}-{:.0f}-vuotiailla noin {:.2f} prosenttia ({:.2f} vs {:.2f})'.format(s,e,(osatyolaste1-osatyolaste2)*100,osatyolaste1*100,osatyolaste2*100))

        self.episodestats.comp_employment_stats()
        cc2.episodestats.comp_employment_stats()
        if self.minimal>0:
            unemp_htv1 = np.nansum(self.episodestats.demogstates[:,0])
            unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,0])
            e_unemp_htv1 = np.nansum(self.episodestats.demogstates[:,0])
            e_unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,0])
            tm_unemp_htv1 = np.nansum(self.episodestats.demogstates[:,0])*0
            tm_unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,0])*0
            f_unemp_htv1 = np.nansum(self.episodestats.demogstates[:,0])*0
            f_unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,0])*0
        else:
            unemp_htv1 = np.nansum(self.episodestats.demogstates[:,0]+self.episodestats.demogstates[:,4]+self.episodestats.demogstates[:,13])
            unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,0]+cc2.episodestats.demogstates[:,4]+cc2.episodestats.demogstates[:,13])
            e_unemp_htv1 = np.nansum(self.episodestats.demogstates[:,0])
            e_unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,0])
            tm_unemp_htv1 = np.nansum(self.episodestats.demogstates[:,13])
            tm_unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,13])
            f_unemp_htv1 = np.nansum(self.episodestats.demogstates[:,4])
            f_unemp_htv2 = np.nansum(cc2.episodestats.demogstates[:,4])

        # epävarmuus
        delta = 1.96*1.0/np.sqrt(self.episodestats.n_pop)

        if self.language== 'English':
            print('Työttömyysvaikutus noin {:.0f} htv'.format(unemp_htv1-unemp_htv2))
            print('- ansiosidonnaiseen noin {:.0f} htv ({:.0f} vs {:.0f})'.format((e_unemp_htv1-e_unemp_htv2),e_unemp_htv1,e_unemp_htv2))
            print('- tm-tukeen {:.0f} työllistä ({:.0f} vs {:.0f})'.format((tm_unemp_htv1-tm_unemp_htv2),tm_unemp_htv1,tm_unemp_htv2))
            print('- putkeen {:.0f} työllistä ({:.0f} vs {:.0f})'.format((f_unemp_htv1-f_unemp_htv2),f_unemp_htv1,f_unemp_htv2))
            print('Uncertainty in employment rates {:.4f}, std {:.4f}'.format(delta,haj1))
        else:
            print('Työttömyysvaikutus {:.0f} htv'.format(unemp_htv1-unemp_htv2))
            print('- ansiosidonnaiseen {:.0f} htv ({:.0f} vs {:.0f})'.format((e_unemp_htv1-e_unemp_htv2),e_unemp_htv1,e_unemp_htv2))
            print('- tm-tukeen {:.0f} työllistä ({:.0f} vs {:.0f})'.format((tm_unemp_htv1-tm_unemp_htv2),tm_unemp_htv1,tm_unemp_htv2))
            print('- putkeen {:.0f} työllistä ({:.0f} vs {:.0f})'.format((f_unemp_htv1-f_unemp_htv2),f_unemp_htv1,f_unemp_htv2))
            print('epävarmuus työllisyysasteissa {:.4f}%, hajonta {:.4f}%'.format(100*delta,haj1))

        if self.episodestats.save_pop:
            unemp_distrib,emp_distrib,unemp_distrib_bu = self.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            tyoll_distrib,tyoll_distrib_bu = self.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2 = cc2.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            tyoll_distrib2,tyoll_distrib_bu2 = cc2.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1 = label1,label2 = label2)
            if self.language== 'English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2,logy = False)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2,logy = False,diff = True)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = False,label1 = label1,label2 = label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = True,label1 = label1,label2 = label2)

            unemp_distrib,emp_distrib,unemp_distrib_bu = self.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)
            tyoll_distrib,tyoll_distrib_bu = self.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2 = cc2.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)
            tyoll_distrib2,tyoll_distrib_bu2 = cc2.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1 = label1,label2 = label2)
            if self.language== 'English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = False,label1 = label1,label2 = label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = True,label1 = label1,label2 = label2)

        if self.episodestats.save_pop:
            print(label2)
            keskikesto = self.episodestats.comp_unemp_durations(return_q = False)
            self.plot_unemp_durdistribs(keskikesto)

            print(label1)
            keskikesto = cc2.episodestats.comp_unemp_durations(return_q = False)
            self.plot_unemp_durdistribs(keskikesto)

            tyoll_virta,tyot_virta = self.episodestats.comp_virrat(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            tyoll_virta2,tyot_virta2 = cc2.episodestats.comp_virrat(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            self.plot_compare_virrat(tyoll_virta,tyoll_virta2,virta_label = 'Työllisyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 40,max_time = 64,virta_label = 'Työttömyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 55,max_time = 64,virta_label = 'Työttömyys',label1 = label1,label2 = label2)

            tyoll_virta,tyot_virta = self.episodestats.comp_virrat(ansiosid = True,tmtuki = False,putki = True,outsider = False)
            tyoll_virta2,tyot_virta2 = cc2.episodestats.comp_virrat(ansiosid = True,tmtuki = False,putki = True,outsider = False)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 40,max_time = 64,virta_label = 'ei-tm-Työttömyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 55,max_time = 64,virta_label = 'ei-tm-Työttömyys',label1 = label1,label2 = label2)

            tyoll_virta,tyot_virta = self.episodestats.comp_virrat(ansiosid = False,tmtuki = True,putki = True,outsider = False)
            tyoll_virta2,tyot_virta2 = cc2.episodestats.comp_virrat(ansiosid = False,tmtuki = True,putki = True,outsider = False)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 40,max_time = 64,virta_label = 'tm-Työttömyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 55,max_time = 64,virta_label = 'tm-Työttömyys',label1 = label1,label2 = label2)

    def compare_simfig_no8(self,cc2,label1 = 'perus',label2 = 'vaihtoehto',grayscale = True,figname = None,dash = False,palette_EK = True):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        if palette_EK:
            csfont,pal = setup_EK_fonts()
        else:
            csfont = {}

        diff_emp = self.episodestats.empstate/self.episodestats.alive-cc2.episodestats.empstate/cc2.episodestats.alive
        diff_emp = diff_emp*100
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        real1 = self.episodestats.comp_presentvalue()
        real2 = cc2.episodestats.comp_presentvalue()
        mean_real1 = np.mean(real1,axis = 1)
        mean_real2 = np.mean(real2,axis = 1)
        initial1 = np.mean(real1[1,:])
        initial2 = np.mean(real2[1,:])

        fig,ax = plt.subplots(2,2)

        self.compare_disab(cc = cc2, xstart = None,xend = None, label1 = label1,label2 = label2,ax=ax[1,0])

        emp_htv1 = np.sum(cc2.episodestats.emp_htv,axis = 1)
        emp_htv2 = np.sum(self.episodestats.emp_htv,axis = 1)

        if self.minimal>0:
            s = 20
            e = 70
        else:
            s = 21
            e = 60 #63.5

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1 = \
            self.episodestats.comp_employed_ratio(self.episodestats.empstate,emp_htv = emp_htv1)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2 = \
            cc2.episodestats.comp_employed_ratio(cc2.episodestats.empstate,emp_htv = emp_htv2)
        htv1,tyolliset1,tyottomat1,osatyolliset1,kokotyolliset1,tyollaste1,osatyolaste1,kokotyolaste1,tyottomyys_aste1 = \
            self.episodestats.comp_tyollisyys_stats(self.episodestats.empstate,scale_time = True,start = s,end = e,emp_htv = emp_htv1,agegroups = False)
        htv2,tyolliset2,tyottomat2,osatyolliset2,kokotyolliset2,tyollaste2,osatyolaste2,kokotyolaste2,tyottomyys_aste2 = \
            cc2.episodestats.comp_tyollisyys_stats(cc2.episodestats.empstate,scale_time = True,start = s,end = e,emp_htv = emp_htv2,agegroups = False)

        tyollaste1 = tyollaste1*100
        tyollaste2 = tyollaste2*100
        tyotaste1 = self.episodestats.comp_unemp_stats_agg(per_pop = False)*100
        tyotaste2 = cc2.episodestats.comp_unemp_stats_agg(per_pop = False)*100
        tyovoimatutk_tyollaste = self.empstats.get_tyollisyysaste_tyovoimatutkimus(self.year)
        tyovoimatutk_tytaste = self.empstats.get_tyottomyysaste_tyovoimatutkimus(self.year)
        tyossakayntitutk_tyollaste = self.empstats.get_tyollisyysaste_tyossakayntitutkimus(self.year)
        tyossakayntitutk_tytaste = self.empstats.get_tyottomyysaste_tyossakayntitutkimus(self.year)

        htv1_full,tyolliset1_full,tyottomat1_full,osata1_full,kokota1_full,tyollaste1_full,osatyo_osuus1_full,kokotyo_osuus1_full,tyot_osuus1_full,tyot_aste1 = \
            self.episodestats.comp_tyollisyys_stats(self.episodestats.empstate,scale_time = True,start = 18,end = 75,emp_htv = emp_htv1,agegroups = True)
        htv2_full,tyolliset2_full,tyottomat2_full,osata2_full,kokota2_full,tyollaste2_full,osatyo_osuus2_full,kokotyo_osuus2_full,tyot_osuus2_full,tyot_aste2 = \
            cc2.episodestats.comp_tyollisyys_stats(cc2.episodestats.empstate,scale_time = True,start = 18,end = 75,emp_htv = emp_htv2,agegroups = True)
        haj1 = self.episodestats.comp_uncertainty(self.episodestats.empstate,emp_htv = emp_htv1)
        haj2 = cc2.episodestats.comp_uncertainty(cc2.episodestats.empstate,emp_htv = emp_htv2)

        ansiosid_osuus1,tm_osuus1 = self.episodestats.comp_unemployed_detailed(self.episodestats.empstate)
        ansiosid_osuus2,tm_osuus2 = cc2.episodestats.comp_unemployed_detailed(cc2.episodestats.empstate)

        if dash:
            ls = '--'
        else:
            ls = None

        lineplot(x,100*tyollaste1_full,y2 = 100*tyollaste2_full,xlim = [20,75],ylim = [0,100],label = label1,label2 = label2,
                 xlabel = self.labels['age'],ylabel = self.labels['tyollisyysaste %'],selite = True,ax=ax[0,0],show=False)

        self.plot_gender_emp(cc = cc2,diff = False,label1 = label1,label2 = label2,ax=ax[1,1])
        #self.plot_gender_emp(cc = cc2,diff = True,label1 = label1,label2 = label2)

        ax[0,1].set_xlabel(self.labels['age'])
        ax[0,1].set_ylabel(self.labels['tyottomyysaste'])
        ax[0,1].plot(x,100*tyot_osuus1_full,label = label1)
        ax[0,1].plot(x,100*tyot_osuus2_full,ls = ls,label = label2)
        ax[0,1].legend()

        # fig,ax = plt.subplots()
        # ax.set_xlabel(self.labels['age'])
        # ax.set_ylabel('Osatyö [%]')
        # ax.plot(x,100*osatyo_osuus1_full,label = label1)
        # ax.plot(x,100*osatyo_osuus2_full,ls = ls,label = label2)
        # ax.set_ylim([0,100])
        # ax.legend()
        if figname is not None:
            plt.savefig(figname+'_no8.'+self.figformat, format = self.figformat)
        plt.show()

        if self.episodestats.save_pop:
            unemp_distrib,emp_distrib,unemp_distrib_bu = self.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            tyoll_distrib,tyoll_distrib_bu = self.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2 = cc2.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            tyoll_distrib2,tyoll_distrib_bu2 = cc2.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1 = label1,label2 = label2)
            if self.language== 'English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, no max age')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2,logy = False)
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2,logy = False,diff = True)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = False,label1 = label1,label2 = label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = True,label1 = label1,label2 = label2)

            unemp_distrib,emp_distrib,unemp_distrib_bu = self.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)
            tyoll_distrib,tyoll_distrib_bu = self.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2 = cc2.episodestats.comp_empdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)
            tyoll_distrib2,tyoll_distrib_bu2 = cc2.episodestats.comp_tyollistymisdistribs(ansiosid = True,tmtuki = True,putki = True,outsider = False,max_age = 54)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1 = label1,label2 = label2)
            if self.language== 'English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1 = label1,label2 = label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = False,label1 = label1,label2 = label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet = True,label1 = label1,label2 = label2)

        if self.episodestats.save_pop:
            print(label2)
            keskikesto = self.episodestats.comp_unemp_durations(return_q = False)
            self.plot_unemp_durdistribs(keskikesto)

            print(label1)
            keskikesto = cc2.episodestats.comp_unemp_durations(return_q = False)
            self.plot_unemp_durdistribs(keskikesto)

            tyoll_virta,tyot_virta = self.episodestats.comp_virrat(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            tyoll_virta2,tyot_virta2 = cc2.episodestats.comp_virrat(ansiosid = True,tmtuki = True,putki = True,outsider = False)
            self.plot_compare_virrat(tyoll_virta,tyoll_virta2,virta_label = 'Työllisyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 40,max_time = 64,virta_label = 'Työttömyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 55,max_time = 64,virta_label = 'Työttömyys',label1 = label1,label2 = label2)

            tyoll_virta,tyot_virta = self.episodestats.comp_virrat(ansiosid = True,tmtuki = False,putki = True,outsider = False)
            tyoll_virta2,tyot_virta2 = cc2.episodestats.comp_virrat(ansiosid = True,tmtuki = False,putki = True,outsider = False)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 40,max_time = 64,virta_label = 'ei-tm-Työttömyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 55,max_time = 64,virta_label = 'ei-tm-Työttömyys',label1 = label1,label2 = label2)

            tyoll_virta,tyot_virta = self.episodestats.comp_virrat(ansiosid = False,tmtuki = True,putki = True,outsider = False)
            tyoll_virta2,tyot_virta2 = cc2.episodestats.comp_virrat(ansiosid = False,tmtuki = True,putki = True,outsider = False)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 40,max_time = 64,virta_label = 'tm-Työttömyys',label1 = label1,label2 = label2)
            self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time = 55,max_time = 64,virta_label = 'tm-Työttömyys',label1 = label1,label2 = label2)


    def plot_density(self,emtr,figname = None,etla_x_emtr = None,etla_emtr = None,xlabel = 'EMTR',foretitle = 'in all states:',
                     text_in_title=True,plot_mean=True,bins=101,ax=None):
        axvcolor = 'gray'
        lstyle = '--'
        if ax is None:
            fig,ax = plt.subplots()
            noshow = False
        else:
            noshow = True
        nc = ma.masked_where(np.isnan(emtr), emtr).compressed()
        bins = np.arange(-0.5,100.5,2)
        bins2 = np.arange(-0.5,100.5,1)
        ax.hist(nc,density = True,bins = bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')

        if nc.shape[0]>0:
            counts, bins = np.histogram(nc,bins = bins,range = (-0.5,100.5))
            #plt.stairs(counts/nc.shape[0], bins)
            ka = np.nanmean(nc)
            med = ma.median(nc)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Density')
            if etla_emtr is not None:
                ax.plot(etla_x_emtr,etla_emtr,'r')
            print('Median:',med)
            if plot_mean:
                plt.axvline(x = ka,ls = lstyle,color = axvcolor)
            #plt.xlim(0,100)
            if text_in_title and not noshow:
                plt.title(f'{foretitle} mean {ka:.2f} median {med:.2f}')
        else:
            plt.title(f'{foretitle}')
        ax.set_xlim(0,100)
        prop = np.count_nonzero(nc>80)/max(1,nc.shape[0])*100
        print(f'{foretitle} loukussa (80 %) {prop}')        

        if not noshow:
            if figname is not None:
                plt.savefig(figname+'emtr.pdf', format = 'pdf')
            plt.show()

    def plot_emtr_fig6(self,figname = None,ax1=None,ax2=None):
        '''
        modified emtr for fig6
        '''
        axvcolor = 'gray'
        lstyle = '--'
        maxt = self.map_age(64)
        #emtr_tilat = set([0,1,4,7,8,9,10,13]) # include 14?
        emtr_tilat = set([0,1,4,5,6,7,8,9,10,11,12,13,14])
        
        etla_x_ptr,etla_ptr,etla_x_emtr,etla_emtr = self.empstats.get_emtr()

        #alivemask = (self.episodestats.popempstate== self.env.get_mortstate()) # pois kuolleet
        
        alivemask = np.zeros_like(self.episodestats.infostats_pop_emtr)
        for t in range(alivemask.shape[0]):
            for k in range(alivemask.shape[1]):
                if self.episodestats.popempstate[t,k] in emtr_tilat:
                    alivemask[t,k] = False
                else:
                    alivemask[t,k] = True

        # emtr
        emtr = ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = alivemask[1:maxt,:]))
        self.plot_density(emtr,etla_x_emtr = etla_x_emtr,etla_emtr = etla_emtr,text_in_title=False,plot_mean=False,bins=51,ax=ax1)

        # ptr
        tvax = ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask = alivemask[1:maxt,:]))
        self.plot_density(tvax,etla_x_emtr = etla_x_ptr,etla_emtr = etla_ptr,xlabel = 'PTR',text_in_title=False,plot_mean=False,bins=51,ax=ax2)

    def plot_emtr(self,figname = None):
        axvcolor = 'gray'
        lstyle = '--'
        maxt = self.map_age(64)
        #emtr_tilat = set([0,1,4,7,8,9,10,13]) # include 14?
        emtr_tilat = set([0,1,4,5,6,7,8,9,10,11,12,13,14])
        
        etla_x_ptr,etla_ptr,etla_x_emtr,etla_emtr = self.empstats.get_emtr()

        print_html('<h2>EMTR</h2>')

        #alivemask = (self.episodestats.popempstate== self.env.get_mortstate()) # pois kuolleet
        
        alivemask = np.zeros_like(self.episodestats.infostats_pop_emtr)
        for t in range(alivemask.shape[0]):
            for k in range(alivemask.shape[1]):
                if self.episodestats.popempstate[t,k] in emtr_tilat:
                    alivemask[t,k] = False
                else:
                    alivemask[t,k] = True

        #alivemask = (self.episodestats.popempstate not in emtr_tilat) # pois kuolleet
        emtr = ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = alivemask[1:maxt,:]))
        self.plot_density(emtr,figname = figname,etla_x_emtr = etla_x_emtr,etla_emtr = etla_emtr)

        #fig,ax = plt.subplots()
        #ax.set_xlabel('EMTR')
        #ax.set_ylabel('Density')
        #ax.hist(emtr,density = True,bins = 100)
        #ax.hist(nc,density = True,bins = bins2)
        ##ax.plot(etla_x_emtr,etla_emtr,'r')
        #plt.xlim(0,100)
        #plt.show()

        for k in emtr_tilat:
            mask = self.episodestats.get_empstatemask(k) #(self.episodestats.popempstate != k) 
            emtr = ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = mask[1:maxt,:]))
            self.plot_density(emtr,figname = figname,etla_x_emtr = etla_x_emtr,etla_emtr = etla_emtr,foretitle = f'state {k}')

        print_html('<h2>PTR</h2>')

        tvax = ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask = alivemask[1:maxt,:]))
        self.plot_density(tvax,figname = figname,etla_x_emtr = etla_x_ptr,etla_emtr = etla_ptr,foretitle = f'kaikki',xlabel = 'PTR')

        for k in emtr_tilat:
            #mask = (self.episodestats.popempstate != k) 
            mask = self.episodestats.get_empstatemask(k)
            tvax = ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask = mask[1:maxt,:]))
            self.plot_density(tvax,figname = figname,etla_x_emtr = etla_x_ptr,etla_emtr = etla_ptr,foretitle = f'state {k}',xlabel = 'PTR')

        mask = self.episodestats.get_empstatemask(10)*self.episodestats.get_empstatemask(1)
        #mask = (self.episodestats.popempstate != 10)*(self.episodestats.popempstate != 1)
        #mask1 = (self.episodestats.popempstate != 1)
        #mask = ma.mask_or(mask10,mask1) # miehet pois        
        tvax = ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask = mask[1:maxt,:]))
        self.plot_density(tvax,figname = figname,etla_x_emtr = etla_x_ptr,etla_emtr = etla_ptr,foretitle = f'työssä (1,10)',xlabel = 'PTR')
        nc = ma.masked_where(np.isnan(tvax), tvax).compressed()
        prop = np.count_nonzero(nc>80)/max(1,nc.shape[0])*100
        print(f'työssä: työttömyysloukussa (80 %) {prop}')

        for k in emtr_tilat:
            #mask = (self.episodestats.popempstate != k) 
            mask = self.episodestats.get_empstatemask(k)
            tvax = ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask = mask[1:maxt,:]))
            nc = ma.masked_where(np.isnan(tvax), tvax).compressed()
            prop = np.count_nonzero(nc>80)/max(1,nc.shape[0])*100
            print(f'{k}: työttömyysloukussa (80 %) {prop}')

#         for k in emtr_tilat:
#             mask = (self.episodestats.popempstate != k) 
#             tvax = ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = mask[1:maxt,:]))
#             nc = ma.masked_where(np.isnan(tvax), tvax).compressed()
#             mask2 = nc >= 1
#             prop = np.count_nonzero(nc<1)/max(1,nc.shape[0])*100
#             print(f'{k}: (1 %) {prop}')
# 
#         for k in emtr_tilat:
#             mask = ma.make_mask(self.episodestats.popempstate != k) 
#             tvax = ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = mask[1:maxt,:])
#             nc = ma.masked_where(np.isnan(tvax), tvax)
#             mask2 = ma.mask_or(mask[1:maxt,:],nc >= 10)
#             w = ma.array(self.episodestats.infostats_pop_wage[1:maxt,:],mask = mask2).compressed()
#             pw = ma.array(self.episodestats.infostats_pop_potential_wage[1:maxt,:],mask = mask2).compressed()
#             em = ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = mask2).compressed()
#             netto = ma.array(self.episodestats.infostats_poptulot_netto[1:maxt,:],mask = mask2).compressed()
#             for s,v in enumerate(em):
#                 print(f'{k}:',w[s],pw[s],v,netto[s])

                    
    ## FROM simstats.py

    def test_emtr(self):
        maxt = self.map_age(64)
        emtr_tilat = set([0,1,4,7,8,9,10,13,14])
    
        for k in emtr_tilat:
            #mask = (self.episodestats.popempstate != k) 
            mask = self.episodestats.get_empstatemask(k)
            tvax = ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask = mask[1:maxt,:])
            mask2 = tvax >= 1
            w = ma.array(self.episodestats.infostats_pop_wage[1:maxt,:],mask = tvax)
            print('w',w)

    def plot_aggkannusteet(self,ben,loadfile,baseloadfile = None,figname = None,label = None,baselabel = None):
        '''
        FIXME
        '''
        f = h5py.File(loadfile, 'r')
        netto = f['netto'][()]
        eff = f['eff'][()]
        tva = f['tva'][()]
        osa_tva = f['osa_tva'][()]
        min_salary = f['min_salary'][()]
        max_salary = f['max_salary'][()]
        step_salary = f['step_salary'][()]
        n = f['n'][()]
        f.close()
        
        basic_marg = fin_benefits.Marginals(ben,year = self.year)

        if baseloadfile is not None:
            f = h5py.File(baseloadfile, 'r')
            basenetto = f['netto'][()]
            baseeff = f['eff'][()]
            basetva = f['tva'][()]
            baseosatva = f['osa_tva'][()]
            f.close()        
            
            basic_marg.plot_insentives(netto,eff,tva,osa_tva,min_salary = min_salary,max_salary = max_salary+step_salary,figname = figname,
                step_salary = step_salary,basenetto = basenetto,baseeff = baseeff,basetva = basetva,baseosatva = baseosatva,
                otsikko = label,otsikkobase = baselabel)
        else:
            basic_marg.plot_insentives(netto,eff,tva,osa_tva,min_salary = min_salary,max_salary = max_salary+step_salary,figname = figname,
                step_salary = step_salary,otsikko = label,otsikkobase = baselabel)
                    
               
    # def compare_epistats(self,filename1,cc2,label1 = 'perus',label2 = 'vaihtoehto',figname = None,greyscale = True):
    #     m_best1,m_median1,s_emp1,median_htv1,u_tmtuki1,u_ansiosid1,h_median1,mn_median1 = self.get_simstats(filename1)
    #     _,m_mean1,s_emp1,mean_htv1,u_tmtuki1,u_ansiosid1,h_mean1,mn_mean1 = self.get_simstats(filename1,use_mean = True)

    #     tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2 = self.episodestats.comp_employed_ratio(cc2.empstate)
    #     htv2,tyoll2,haj2,tyollaste2,tyolliset2,osatyolliset2,kokotyolliset2,osata2,kokota2 = \
    #         self.episodestats.comp_tyollisyys_stats(cc2.empstate,scale_time = True,start = s,end = e) # /cc2.n_pop
    #     ansiosid_osuus2,tm_osuus2 = self.episodestats.comp_employed_detailed(cc2.empstate)
        
    #     m_best2 = tyoll_osuus2
    #     m_median2 = tyoll_osuus2
    #     s_emp2 = s_emp1*0
    #     median_htv2 = htv_osuus2
    #     #u_tmtuki2,
    #     #u_ansiosid2,
    #     #h_median2,
    #     mn_median2 = tyoll_osuus2
    #     m_mean2 = tyoll_osuus2
    #     s_emp2 = 0*s_emp1
    #     mean_htv2 = htv_osuus2
    #     #u_tmtuki2,
    #     #u_ansiosid2,
    #     #h_mean2,
    #     #mn_mean2

    #     if greyscale:
    #         plt.style.use('grayscale')
    #         plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
        
    #     print('Vaikutus mediaanityöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(median_htv2-median_htv1,median_htv2,median_htv1))
    #     print('Vaikutus keskiarvotyöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(mean_htv2-mean_htv1,mean_htv2,mean_htv1))

    #     fig,ax = plt.subplots()
    #     ax.set_xlabel(self.labels['age'])
    #     ax.set_ylabel('Employment rate [%]')
    #     x = np.linspace(self.min_age,self.max_age,self.n_time)
    #     ax.plot(x[1:],100*m_mean2[1:],label = label2)
    #     ax.plot(x[1:],100*m_mean1[1:],ls = '--',label = label1)
    #     ax.set_ylim([0,100])  
    #     ax.legend()
    #     if figname is not None:
    #         plt.savefig(figname+'tyollisyys.pdf')        
    #     plt.show()

    #     fig,ax = plt.subplots()
    #     ax.set_xlabel(self.labels['age'])
    #     ax.set_ylabel('Työllisyysero [hlö/htv]')
    #     x = np.linspace(self.min_age,self.max_age,self.n_time)
    #     ax.plot(x[1:],mn_median2[1:]-mn_median1[1:],label = label2+' miinus '+label1)
    #     ax.plot(x[1:],h_median2[1:]-h_median1[1:],label = label2+' miinus '+label1+' htv')
    #     ax.legend()
    #     plt.show()

    #     fig,ax = plt.subplots()
    #     ax.set_xlabel(self.labels['age'])
    #     ax.set_ylabel(self.labels['tyottomien osuus'])
    #     x = np.linspace(self.min_age,self.max_age,self.n_time)
    #     ax.plot(x[1:],100*u_tmtuki1[1:],ls = '--',label = 'tm-tuki, '+label1)
    #     ax.plot(x[1:],100*u_tmtuki2[1:],label = 'tm-tuki, '+label2)
    #     ax.plot(x[1:],100*u_ansiosid1[1:],ls = '--',label = 'ansiosidonnainen, '+label1)
    #     ax.plot(x[1:],100*u_ansiosid2[1:],label = 'ansiosidonnainen, '+label2)
    #     ax.legend()
    #     if figname is not None:
    #         plt.savefig(figname+'tyottomyydet.pdf')        
    #     plt.show()

    #     fig,ax = plt.subplots()
    #     ax.set_xlabel(self.labels['age'])
    #     ax.set_ylabel(self.labels['tyottomien osuus'])
    #     x = np.linspace(self.min_age,self.max_age,self.n_time)
    #     ax.plot(x[1:],100*(m_median2[1:]-m_median1[1:]),label = label1)
    #     plt.show()

    def plot_compare_csvirta(self,m1,m2,lbl):
        nc1 = np.reshape(np.cumsum(m1),m1.shape)
        nc2 = np.reshape(np.cumsum(m2),m1.shape)
        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        plt.plot(x,nc1)
        plt.plot(x,nc2)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(lbl)
        plt.show()
        fig,ax = plt.subplots()
        plt.plot(x,nc1-nc2)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('diff '+lbl)
        plt.show()

    def plot_compare_virtadistribs(self,tyoll_virta1,tyoll_virta2,tyot_virta1,tyot_virta2,tyot_virta_ansiosid1,tyot_virta_ansiosid2,tyot_virta_tm1,tyot_virta_tm2,label1 = '',label2 = ''):
        m1 = np.mean(tyoll_virta1,axis = 0,keepdims = True).transpose()
        m2 = np.mean(tyoll_virta2,axis = 0,keepdims = True).transpose()
        fig,ax = plt.subplots()
        plt.plot(m1,label = label1)
        plt.plot(m2,label = label2)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Keskimääräinen työllisyysvirta')
        plt.show()
        self.plot_compare_virrat(m1,m2,virta_label = 'työllisyys',label1 = label1,label2 = label2,ymin = 0,ymax = 5000)
        self.plot_compare_csvirta(m1,m2,'cumsum työllisyysvirta')

        m1 = np.mean(tyot_virta1,axis = 0,keepdims = True).transpose()
        m2 = np.mean(tyot_virta2,axis = 0,keepdims = True).transpose()
        self.plot_compare_virrat(m1,m2,virta_label = 'työttömyys',label1 = label1,label2 = label2)
        self.plot_compare_csvirta(m1,m2,'cumsum työttömyysvirta')

        m1 = np.mean(tyot_virta_ansiosid1,axis = 0,keepdims = True).transpose()
        m2 = np.mean(tyot_virta_ansiosid2,axis = 0,keepdims = True).transpose()
        self.plot_compare_virrat(m1,m2,virta_label = 'ei-tm-työttömyys',label1 = label1,label2 = label2)
        m1 = np.mean(tyot_virta_tm1,axis = 0,keepdims = True).transpose()
        m2 = np.mean(tyot_virta_tm2,axis = 0,keepdims = True).transpose()
        self.plot_compare_virrat(m1,m2,virta_label = 'tm-työttömyys',label1 = label1,label2 = label2)
        n1 = (np.mean(tyoll_virta1,axis = 0,keepdims = True)-np.mean(tyot_virta1,axis = 0,keepdims = True)).transpose()
        n2 = (np.mean(tyoll_virta2,axis = 0,keepdims = True)-np.mean(tyot_virta2,axis = 0,keepdims = True)).transpose()
        self.plot_compare_virrat(n1,n2,virta_label = 'netto',label1 = label1,label2 = label2,ymin = -1000,ymax = 1000)
        self.plot_compare_csvirta(n1,n2,'cumsum nettovirta')

    def plot_unemp_durdistribs(self,kestot,kestot2 = None):
        if len(kestot.shape)>2:
            m1 = self.episodestats.empdur_to_dict(np.mean(kestot,axis = 0))
        else:
            m1 = self.episodestats.empdur_to_dict(kestot)

        if len(kestot.shape)>2:
            m1 = self.episodestats.empdur_to_dict(np.mean(kestot,axis = 0))
        else:
            m1 = self.episodestats.empdur_to_dict(kestot)


        df = pd.DataFrame.from_dict(m1,orient = 'index',columns = ['0-6 m','6-12 m','12-18 m','18-24 m','yli 24 m'])
        print(tabulate(df, headers = 'keys', tablefmt = 'psql', floatfmt = ",.2f"))

    def plot_compare_unemp_durdistribs(self,kestot1,kestot2,viimekesto1,viimekesto2,label1 = '',label2 = ''):
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        print(label1)
        self.plot_unemp_durdistribs(kestot1)
        print(label2)
        self.plot_unemp_durdistribs(kestot2)

        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        print(label1)
        self.plot_unemp_durdistribs(viimekesto1)
        print(label2)
        self.plot_unemp_durdistribs(viimekesto2)
                            
    def fit_norm(self,diff):
        diff_stdval = np.std(diff)
        diff_meanval = np.mean(diff)
        diff_minval = np.min(diff)
        diff_maxval = np.max(diff)
        sz = (diff_maxval-diff_minval)/10
        x = np.linspace(diff_minval,diff_maxval,1000)
        y = norm.pdf(x,diff_meanval,diff_stdval)*diff.shape[0]*sz
    
        return x,y

    def plot_simstat_unemp_all(self,ratio,unempratio = False,figname = None,grayscale = False,tyovoimatutkimus = False,fig=None,ax=None):
        '''
        Plottaa työttömyysaste (unempratio = True) tai työttömien osuus väestöstö (False)
        '''
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        if unempratio:
            unempratio_stat_tvt = 100*self.empstats.unempratio_stats(g = 0,tyossakayntitutkimus = False)
            unempratio_stat_tkt = 100*self.empstats.unempratio_stats(g = 0,tyossakayntitutkimus = True)
        else:
            unempratio_stat_tvt = 100*self.empstats.unemp_stats(g = 0,tyossakayntitutkimus = False)
            unempratio_stat_tkt = 100*self.empstats.unemp_stats(g = 0,tyossakayntitutkimus = True)

        if self.language== 'Finnish':
            labeli = 'keskimääräinen työttömien osuus väestöstö '            
            labeli2 = 'työttömien osuus väestöstö'
        else:
            labeli = 'proportion of unemployed'            
            labeli2 = 'Unemployment rate'

        if unempratio:
            ylabeli = self.labels['Työttömyysaste [%]']
        else:
            ylabeli = self.labels['Työttömien osuus [%]']

        if fig is None:
            fig,ax = plt.subplots()

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(ylabeli)
        ax.plot(x,ratio,label = self.labels['malli'])
        if tyovoimatutkimus:
            ax.plot(x,unempratio_stat_tvt,ls = '--',label = self.labels['havainto']+',työvoimatutkimus')
            ax.plot(x,unempratio_stat_tkt,ls = '--',label = self.labels['havainto']+',työssäkäyntitutkimus')
        else:
            ax.plot(x,unempratio_stat_tkt,ls = '--',label = self.labels['havainto'])
        ax.legend(frameon=False)
        if figname is not None:
            plt.savefig(figname+'tyottomyysaste.'+self.figformat, format = self.figformat)
        if fig is None:
            plt.show()

        #fig,ax = plt.subplots()
        #ax.set_xlabel(self.labels['age'])
        #ax.set_ylabel(ylabeli)
        #if tyovoimatutkimus:
        #    ax.plot(x,unempratio_stat_tvt,label = self.labels['havainto']+',työvoimatutkimus')
        #    ax.plot(x,unempratio_stat_tkt,label = self.labels['havainto']+',työssäkäyntitutkimus')
        #else:
        #    ax.plot(x,unempratio_stat_tkt,label = self.labels['havainto'])
        #ax.legend()
        #if grayscale:
        #    pal = sns.light_palette("black", 8, reverse = True)
        #else:
        #    pal = sns.color_palette("hls", self.n_employment)  # hls, husl, cubehelix
        #ax.stackplot(x,ratio,colors = pal) #,label = self.labels['malli'])
        #ax.plot(x,tyottomyysaste)
        #plt.show()
    
    def sel_subset(self,df,subset):
        if subset==0:
            list_ben=['tyottomyyspvraha','ansiopvraha','peruspvraha','asumistuki',
                      'tyoelakemeno','kansanelakemeno','takuuelakemeno','elatustuki','lapsilisa','opintotuki',
                      'isyyspaivaraha','aitiyspaivaraha','kotihoidontuki','sairauspaivaraha','toimeentulotuki','etuusmeno']
        elif subset==1:
            list_ben=['tyotulosumma','etuusmeno','nettotulot','valtionvero','kunnallisvero','ylevero','alv',
                      'ptel','tyottomyysvakuutusmaksu','tyoelakemaksu','sairausvakuutusmaksu','pvhoitomaksu','ta_maksut',
                      'julkinen talous, netto']
        elif subset==2:
            list_ben=['tyotulosumma','nettotulot','tyotulosumma osa-aika','tyotulosumma kokoaika',
                      'valtionvero','kunnallisvero','alv','ylevero',
                      'ptel','tyottomyysvakuutusmaksu','tyoelakemaksu','sairausvakuutusmaksu','pvhoitomaksu','ta_maksut',
                      'verotettava etuusmeno','julkinen talous, netto']
        elif subset==3:
            list_ben=['työssä 63+','työssä ja eläkkeellä','työllisiä 18-62','työllisiä','osaaikatyössä','kokoaikatyössä'] # 'ovella'
        elif subset==4:
            list_ben=['yhteensä','työllisiä','osaaikatyössä','kokoaikatyössä','työssä 63+','työssä ja eläkkeellä','työikäisiä 18-62',
                      'aikuisia','lapsia','pareja','yksinhuoltajia','lapsiperheitä'] # 'ovella'
        elif subset==5:
            list_ben=['tyotulosumma','tyotulosumma osa-aika','tyotulosumma kokoaika','etuusmeno','verot+maksut+alv','nettotulot']
        list_ben_eng=[self.output_labels[x] for x in list_ben]
        df2 = df.loc[list_ben_eng]
        df2 = df2.style.format(decimal='.', thousands=',', precision=0)

        return df2

    def plot_simtables(self,filename,grayscale = False,figname = None,cc2 = None, title='baseline'):
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
            best_emp,emps,agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,\
            alives,agg_empstate,agg_alives,agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,\
            galives,agg_galives,gempstate,agg_gempstate\
                = self.episodestats.load_simstats(filename)

        mean_htv,mean_tyoll,h_mean,m_mean,diff_htv,mean_rew,um_mean,m_mean,std_htv,h_std,mean_unempratio \
            = self.episodestats.get_simstats(filename,use_mean = True)
        median_htv,median_tyoll,h_median,m_median,median_rew,um_median,std_tyoll,s_tyoll,median_unempratio \
            = self.episodestats.get_simstats(filename,use_mean = False)

        if self.version>0:
            print('lisäpäivillä on {:.0f} henkilöä'.format(self.episodestats.count_putki_dist(emps)))

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        diff_htv = agg_htv-mean_htv
        diff_tyoll = agg_tyoll-median_tyoll
        mean_discounted_rew = np.mean(agg_discounted_rew)
        mean_netincome = np.mean(agg_netincome)
        mean_equi_netincome = np.mean(agg_equivalent_netincome)

        print(f'Mean undiscounted reward {mean_rew}')
        print(f'Mean discounted reward {mean_discounted_rew}')
        print(f'Mean net income {mean_netincome} mean equivalent net income {mean_equi_netincome}')
        
        m_median = np.median(emp_tyolliset_osuus,axis = 0)
        s_emp = np.std(emp_tyolliset_osuus,axis = 0)
        m_best = emp_tyolliset_osuus[best_emp,:]

        if True:
            q_stat = self.empstats.stat_participants(lkm = False)
            q_days = self.empstats.stat_days()
            df3 = pd.DataFrame.from_dict(q_days,orient = 'index',columns = ['htv_tot'])
            df_htv = htv_budget.copy()
            df_htv[self.output_labels['toteuma (htv)']] = df3['htv_tot']
            df_htv[self.output_labels['diff (htv)']] = df_htv['htv']-df_htv[self.output_labels['toteuma (htv)']]

            df_lkm = participants.copy()
            df2 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = ['toteuma'])
            df_lkm[self.output_labels['toteuma (#)']] = df2['toteuma']
            df_lkm[self.output_labels['diff (#)']] = df_lkm['lkm']-df_lkm[self.output_labels['toteuma (#)']]

            q_stat = self.empstats.stat_budget()
            cctext = 'budget'
            df4 = budget.copy() # pd.DataFrame.from_dict(budget,orient = 'index',columns = ['e/y'])
            df5 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = [cctext])
            df_budget = self.compare_df(df4,df5,cctext1 = 'e/v',cctext2 = cctext,cctext2_new = cctext)
        
        print(self.sel_subset(df_budget,0).to_latex())
        print(self.sel_subset(df_budget,2).to_latex())
        print(self.sel_subset(df_budget,1).to_latex())
        print(self.sel_subset(df_budget,5).to_latex())
        print(self.sel_subset(df_lkm,4).to_latex())
        print(self.sel_subset(df_htv,3).to_latex())
        print(df_budget)

    def plot_simstats(self,filename,grayscale = False,figname = None,cc2 = None):
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
            best_emp,emps,agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,\
            alives,agg_empstate,agg_alives,agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,\
            galives,agg_galives,gempstate,agg_gempstate\
                = self.episodestats.load_simstats(filename)

        mean_htv,mean_tyoll,h_mean,m_mean,diff_htv,mean_rew,um_mean,m_mean,std_htv,h_std,mean_unempratio \
            = self.episodestats.get_simstats(filename,use_mean = True)
        median_htv,median_tyoll,h_median,m_median,median_rew,um_median,std_tyoll,s_tyoll,median_unempratio \
            = self.episodestats.get_simstats(filename,use_mean = False)
        if self.version>0:
            print('lisäpäivillä on {:.0f} henkilöä'.format(self.episodestats.count_putki_dist(emps)))

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        diff_htv = agg_htv-mean_htv
        diff_tyoll = agg_tyoll-median_tyoll
        mean_discounted_rew = np.mean(agg_discounted_rew)
        mean_netincome = np.mean(agg_netincome)
        mean_equi_netincome = np.mean(agg_equivalent_netincome)

        fig,ax = plt.subplots()
        x = np.linspace(8,48,6)
        plt.bar(x,100*pt_agg[0,:],width=7)
        ax.set_xticks([8,16,24,32,40,48])
        ax.set_ylabel(self.labels['osuus %'])
        ax.set_xlabel(self.labels['Työaika [h]'])
        plt.show()

        print(f'Mean undiscounted reward {mean_rew}')
        print(f'Mean discounted reward {mean_discounted_rew}')
        print(f'Mean net income {mean_netincome} mean equivalent net income {mean_equi_netincome}')
        fig,ax = plt.subplots()
        ax.set_xlabel('Discounted rewards')
        ax.set_ylabel(self.labels['Lukumäärä'])
        ax.hist(agg_discounted_rew,color = 'lightgray')
        plt.show()
        
        x,y = self.fit_norm(diff_htv)
        
        m_median = np.median(emp_tyolliset_osuus,axis = 0)
        s_emp = np.std(emp_tyolliset_osuus,axis = 0)
        m_best = emp_tyolliset_osuus[best_emp,:]

        q_stat = self.empstats.stat_participants(lkm = False)
        q_days = self.empstats.stat_days()
        df3 = pd.DataFrame.from_dict(q_days,orient = 'index',columns = ['htv_tot'])
        df_htv = htv_budget.copy()
        df_htv[self.output_labels['toteuma (htv)']] = df3['htv_tot']
        df_htv[self.output_labels['diff (htv)']] = df_htv['htv']-df_htv[self.output_labels['toteuma (htv)']]

        df_lkm = participants.copy()
        df2 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = ['toteuma'])
        df_lkm[self.output_labels['toteuma (#)']] = df2['toteuma']
        df_lkm[self.output_labels['diff (#)']] = df_lkm['lkm']-df_lkm[self.output_labels['toteuma (#)']]

        q_stat = self.empstats.stat_budget()
        cctext = 'budget'
        df4 = budget.copy() # pd.DataFrame.from_dict(budget,orient = 'index',columns = ['e/y'])
        df5 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = [cctext])
        df_budget = self.compare_df(df4,df5,cctext1 = 'e/v',cctext2 = cctext,cctext2_new = cctext)

        print(tabulate(df_budget, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
        print(tabulate(df_lkm, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
        print('Henkilövuosia tiloissa skaalattuna väestötasolle henkilövuosina')
        #print(tabulate(htv_budget, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
        print(tabulate(df_htv, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))

        if self.minimal:
            print('Työllisyyden keskiarvo {:.0f} htv mediaani {:.0f} htv std {:.0f} htv'.format(mean_htv,median_htv,std_htv))
        else:
            print('Työllisyyden keskiarvo keskiarvo {:.0f} htv, mediaani {:.0f} htv std {:.0f} htv\n'
                  'keskiarvo {:.0f} työllistä, mediaani {:.0f} työllistä, std {:.0f} työllistä'.format(
                    mean_htv,median_htv,std_htv,mean_tyoll,median_tyoll,std_tyoll))

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['Poikkeama työllisyydessä [htv]'])
        ax.set_ylabel(self.labels['Lukumäärä'])
        ax.hist(diff_htv,color = 'lightgray')
        ax.plot(x,y,color = 'black')
        if figname is not None:
            plt.savefig(figname+'poikkeama.pdf')
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['Osatyön osuus %'])
        ax.set_ylabel(self.labels['age'])
        ax.hist(diff_htv,color = 'lightgray')
        ax.plot(x,y,color = 'black')
        plt.legend(frameon=False)
        if figname is not None:
            plt.savefig(figname+'pt_age.pdf')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        labeli2 = 'proportion working parttime, '
        for gender in range(2):
            if gender== 0:
                leg = self.labels['Miehet']
                g = 'men'
                pstyle = '-'
            else:
                g = 'women'
                leg = self.labels['Naiset']
                pstyle = ''

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = \
                self.episodestats.comp_empratios_gender(gempstate=agg_gempstate,galive=agg_galives,gender = g,unempratio = False)

            ax.plot(x,osatyoaste,'{}'.format(pstyle),label = 'LCM {}'.format(leg))

        o_x,m_osatyo,f_osatyo = self.empstats.stats_parttimework()
        if grayscale:
            ax.plot(o_x,f_osatyo,ls = '--',label = self.labels['havainto, naiset'])
            ax.plot(o_x,m_osatyo,ls = '--',label = self.labels['havainto, miehet'])
        else:
            ax.plot(o_x,f_osatyo,label = self.labels['havainto, naiset'])
            ax.plot(o_x,m_osatyo,label = self.labels['havainto, miehet'])
        labeli = 'osatyöaste '#+str(ka_tyottomyysaste)
        ylabeli = 'Osatyön osuus työnteosta [%]'
        ax.legend(frameon=False)
        ax.set_ylabel(self.labels['Osatyön osuus %'])
        ax.set_xlabel(self.labels['age'])
        if figname is not None:
            plt.savefig(figname+'pt_estimate.pdf')
        plt.show()


        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['Työllisyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*np.transpose(emp_tyolliset_osuus),linewidth = 0.4)
        ax.plot(x,100*m_mean,label = 'keskiarvo')
        ax.legend(frameon=False)
        if figname is not None:
            plt.savefig(figname+'tyollisyyshajonta.pdf')
        plt.show()

        if self.version>0:
            x,y = self.fit_norm(diff_tyoll)
            fig,ax = plt.subplots()
            ax.set_xlabel(self.labels['Poikkeama työllisyydessä [henkilöä]'])
            ax.set_ylabel(self.labels['Lukumäärä'])
            ax.hist(diff_tyoll,color = 'lightgray')
            ax.plot(x,y,color = 'black')
            plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['Palkkio'])
        ax.set_ylabel(self.labels['Lukumäärä'])
        ax.hist(agg_rew)
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['Työllisyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_mean,label = 'keskiarvo')
        ax.plot(x,100*m_median,label = 'mediaani')
        ax.plot(x,100*m_best,label = 'paras')
        ax.plot(x,100*self.empstats.emp_stats(),label = 'havainto')
        ax.legend()
        plt.show()

        self.plot_various_groups(empstate=agg_empstate,alive=agg_alives,figname = 'agg_empstate')
        self.plot_emp_vs_workforce(empstate=agg_empstate,alive=agg_alives,figname = 'agg_workforce')
        self.plot_simstat_unemp_all(100*um_mean,unempratio = False)
        self.plot_simstat_unemp_all(100*mean_unempratio,unempratio = True)

        if False:
            fig,ax = plt.subplots()
            ax.set_xlabel(self.labels['age'])
            ax.set_ylabel(self.labels['Hajonta työllisyysasteessa [%]'])
            x = np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,100*std_tyoll)
            plt.show()

        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,\
            tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
            unemp_dur,unemp_lastdur,unemp_basis_distrib = self.episodestats.load_simdistribs(filename)
       
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        self.plot_unemp_durdistribs(unemp_dur)
        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        self.plot_unemp_durdistribs(unemp_lastdur)

        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label = 'vaihtoehto')
        self.plot_unempdistribs(unemp_distrib1,figname = figname,max = 2.5,miny = 1e-5,maxy = 1)
        #self.plot_tyolldistribs(unemp_distrib1,tyoll_distrib1,tyollistyneet = True,figname = figname)
        self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max = 2.5,figname = figname)


    def plot_simfig_no8(self,filename,grayscale = True,figname = None,cc2 = None):
        '''
        Create fig no 7 to the paper
        '''

        # create benefits
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        fig,ax = plt.subplots(2,1)

        self.plot_simfig_unempbasis(filename,fig=fig,ax=ax[0])

        # create emtr and ptr
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,emps,\
            agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,alives,agg_empstate,agg_alives,\
            agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,galives,agg_galives,gempstate,agg_gempstate = \
            self.episodestats.load_simstats(filename)

        self.plot_unemp_shares(ax=ax[1],empstate=agg_empstate)
        add_label(ax)

        if figname is not None:
            plt.savefig(figname+'fig8.pdf')

    def plot_simfig_no2(self,grayscale = False,figname = None):
        '''
        Create fig no 2 to the paper
        '''

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        fig,ax = plt.subplots(1,2)

        empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive
        ratio_label = self.labels['osuus']
        self.plot_states(empstate_ratio,ylabel = ratio_label,stack = True,ax=ax[0],legend_infig=False)

        self.plot_emp_vs_workforce(empstate=None,alive=None,figname = None,ax = ax[1],legend_infig=True)
        fig.subplots_adjust(wspace=0.9)
        add_label(ax)
        if figname is not None:
                plt.savefig(figname, format = 'pdf')
        plt.show()

    def plot_simfig_no6(self,grayscale = False,figname = None):
        '''
        Create fig no 2 to the paper
        '''

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        fig,ax = plt.subplots(1,2)

        empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive
        ratio_label = self.labels['osuus']
        self.plot_states(empstate_ratio,ylabel = ratio_label,stack = True,ax=ax[0],legend_infig=False)

        self.plot_emp_vs_workforce(empstate=None,alive=None,figname = None,ax = ax[1],legend_infig=True)
        fig.subplots_adjust(wspace=0.9)
        add_label(ax)
        if figname is not None:
                plt.savefig(figname, format = 'pdf')
        plt.show()        

    def plot_simfig_no5(self,filename,grayscale = False,figname = None,cc2 = None):
        '''
        Create fig no 5 to the paper
        '''
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
            best_emp,emps,agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,\
            alives,agg_empstate,agg_alives,agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,\
            galives,agg_galives,gempstate,agg_gempstate\
                = self.episodestats.load_simstats(filename)

        mean_htv,mean_tyoll,h_mean,m_mean,diff_htv,mean_rew,um_mean,m_mean,std_htv,h_std,mean_unempratio \
            = self.episodestats.get_simstats(filename,use_mean = True)
        median_htv,median_tyoll,h_median,m_median,median_rew,um_median,std_tyoll,s_tyoll,median_unempratio \
            = self.episodestats.get_simstats(filename,use_mean = False)
        if self.version>0:
            print('lisäpäivillä on {:.0f} henkilöä'.format(self.episodestats.count_putki_dist(emps)))

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        diff_htv = agg_htv-mean_htv
        diff_tyoll = agg_tyoll-mean_tyoll
        mean_discounted_rew = np.mean(agg_discounted_rew)
        mean_netincome = np.mean(agg_netincome)
        mean_equi_netincome = np.mean(agg_equivalent_netincome)

        harju_x = np.array([10,20,30,40,50])
        harju_y = np.array([0.011825,0.027524,0.072397,0.860807,0.027448])

        fig,ax = plt.subplots(2,2)
        #fig.tight_layout()
        x = np.linspace(8,48,6)
        ax[0,0].bar(x,100*pt_agg[0,:]/np.sum(pt_agg[0,:]),width=7)
        #ax[0,0].bar(harju_x,100*harju_y,width=7)
        ax[0,0].set_xticks([8,16,24,32,40,48])
        ax[0,0].set_ylabel(self.labels['osuus %'])
        ax[0,0].set_xlabel(self.labels['Työaika [h]'])
        
        m_median = np.median(emp_tyolliset_osuus,axis = 0)
        s_emp = np.std(emp_tyolliset_osuus,axis = 0)
        m_best = emp_tyolliset_osuus[best_emp,:]

        if True:
            q_stat = self.empstats.stat_participants(lkm = False)
            q_days = self.empstats.stat_days()
            df3 = pd.DataFrame.from_dict(q_days,orient = 'index',columns = ['htv_tot'])
            df_htv = htv_budget.copy()
            df_htv[self.output_labels['toteuma (htv)']] = df3['htv_tot']
            df_htv[self.output_labels['diff (htv)']] = df_htv['htv']-df_htv[self.output_labels['toteuma (htv)']]

            df_lkm = participants.copy()
            df2 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = ['toteuma'])
            df_lkm[self.output_labels['toteuma (#)']] = df2['toteuma']
            df_lkm[self.output_labels['diff (#)']] = df_lkm['lkm']-df_lkm[self.output_labels['toteuma (#)']]

            q_stat = self.empstats.stat_budget()
            cctext = 'budget'
            df4 = budget.copy() # pd.DataFrame.from_dict(budget,orient = 'index',columns = ['e/y'])
            df5 = pd.DataFrame.from_dict(q_stat,orient = 'index',columns = [cctext])
            df_budget = self.compare_df(df4,df5,cctext1 = 'e/v',cctext2 = cctext,cctext2_new = cctext)

        x = np.linspace(self.min_age,self.max_age,self.n_time)
        labeli2 = 'proportion working parttime, '
        for gender in range(2):
            if gender== 0:
                leg = self.labels['Miehet']
                g = 'men'
                pstyle = '-'
            else:
                g = 'women'
                leg = self.labels['Naiset']
                pstyle = ''

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste = \
                self.episodestats.comp_empratios_gender(gempstate=agg_gempstate,galive=agg_galives,gender = g,unempratio = False)

            ax[0,1].plot(x,osatyoaste,'{}'.format(pstyle),label = 'LCM {}'.format(leg))

        o_x,m_osatyo,f_osatyo = self.empstats.stats_parttimework()
        if grayscale:
            ax[0,1].plot(o_x,f_osatyo,ls = '--',label = self.labels['havainto, naiset'])
            ax[0,1].plot(o_x,m_osatyo,ls = '--',label = self.labels['havainto, miehet'])
        else:
            ax[0,1].plot(o_x,f_osatyo,label = self.labels['havainto, naiset'])
            ax[0,1].plot(o_x,m_osatyo,label = self.labels['havainto, miehet'])
        labeli = 'osatyöaste '#+str(ka_tyottomyysaste)
        ylabeli = 'Osatyön osuus työnteosta [%]'
        ax[0,1].legend(frameon=False)
        ax[0,1].set_ylabel(self.labels['Osatyön osuus %'])
        ax[0,1].set_xlabel(self.labels['age'])

        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,\
            tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
            unemp_dur,unemp_lastdur,unemp_basis_distrib = self.episodestats.load_simdistribs(filename)

        self.plot_simstat_unemp_all(100*mean_unempratio,unempratio = True,fig=fig,ax=ax[1,0])
        self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max = 2.5,figname = figname,fig=fig,ax=ax[1,1],kuva1=False,kuva2=False,kuva3=False,kuva4=True,kyyra=True,
                                     label1='Employed',label2='Moved away')
       
        add_label(ax)
        if figname is not None:
            plt.savefig(figname+'fig5.pdf')

        #print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        #self.plot_unemp_durdistribs(unemp_dur)
        #print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        #self.plot_unemp_durdistribs(unemp_lastdur)

        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label = 'vaihtoehto')
        #self.plot_unempdistribs(unemp_distrib1,figname = figname,max = 2.5,miny = 1e-5,maxy = 1)
        #self.plot_tyolldistribs(unemp_distrib1,tyoll_distrib1,tyollistyneet = True,figname = figname)
        #self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max = 2.5,figname = figname)


    def plot_simfig_no7(self,filename,filename2,label1='Reform',label2='Baseline',figname='images/comparison_',grayscale = False,cc2 = None):
        '''
        Create fig no 7 to the paper
        '''
        mean_htv1,mean_tyoll1,h_mean1,m_mean1,diff_htv1,mean_rew1,um_mean1,mean_tyolliset_osuus1,std_htv,h_std,mean_unempratio1 \
            = self.episodestats.get_simstats(filename,use_mean = True)
        median_htv1,median_tyoll1,h_median1,m_median1,median_rew1,median_tyott1,std_tyoll1,s_tyoll1,median_unempratio1 \
            = self.episodestats.get_simstats(filename,use_mean = False)
        
        mean_htv2,mean_tyoll2,h_mean2,m_mean2,diff_htv2,mean_rew2,um_mean2,mean_tyolliset_osuus1,std_htv2,s_htv1,mean_unempratio2 \
            = self.episodestats.get_simstats(filename2,use_mean = True)
        median_htv2,median_tyoll2,h_median2,m_median2,median_rew2,median_tyott2,std_tyoll2,s_tyoll2,median_unempratio2 \
            = self.episodestats.get_simstats(filename2,use_mean = False)

        agg_htv1,agg_tyoll1,agg_rew1,agg_discounted_rew1,emp_tyolliset1,emp_tyolliset_osuus1,\
            emp_tyottomat1,emp_tyottomat_osuus1,emp_htv1,emps1,best_rew1,\
            best_emp1,emps1,agg_netincome1,agg_equivalent_netincome1,budget1,participants1,htv_budget1,\
            alives1,agg_empstate1,agg_alives1,agg_tyottomyysaste1,emp_tyottomyysaste1,pt_agg1,pt_agegroup1,\
            galives1,agg_galives1,gempstate1,agg_gempstate1\
                = self.episodestats.load_simstats(filename)

        agg_htv2,agg_tyoll2,agg_rew2,agg_discounted_rew2,emp_tyolliset2,emp_tyolliset_osuus2,\
            emp_tyottomat2,emp_tyottomat_osuus2,emp_htv2,emps2,best_rew2,\
            best_emp2,emps2,agg_netincome2,agg_equivalent_netincome2,budget2,participants2,htv_budget2,\
            alives2,agg_empstate2,agg_alives2,agg_tyottomyysaste2,emp_tyottomyysaste2,pt_agg2,pt_agegroup2,\
            galives2,agg_galives2,gempstate2,agg_gempstate2\
                = self.episodestats.load_simstats(filename2)
                
        color_reform = 'black'
        color_baseline = 'darkgray'
        color_obs = 'lightgray'

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        fig,ax = plt.subplots(2,2)
        ax[0,0].set_xlabel(self.labels['age'])
        ax[0,0].set_ylabel(self.labels['Työllisyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax[0,0].plot(x,100*m_mean1,label = label1,color=color_reform)
        ax[0,0].plot(x,100*m_mean2,label = label2,color=color_baseline)
        ax[0,0].plot(x,100*self.empstats.emp_stats(),'--',label = 'observation',color=color_obs)
        ax[0,0].legend(frameon=False)

        ax[0,1].set_xlabel(self.labels['age'])
        ax[0,1].set_ylabel(self.labels['Työttömyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax[0,1].plot(x,100*mean_unempratio1,label = label1, color=color_reform)
        ax[0,1].plot(x,100*mean_unempratio2,label = label2, color=color_baseline)
        ax[0,1].plot(x,100*self.empstats.unempratio_stats(tyossakayntitutkimus=True),'--',label = 'Observation',color=color_obs)
        ax[0,1].legend(frameon=False)

        x = np.linspace(8,48,6)
        wd=3
        ax[1,0].bar(x-wd,pt_agg1[0,:]*mean_tyoll1,width=3,label = label1, color=color_reform)
        ax[1,0].bar(x,pt_agg2[0,:]*mean_tyoll2,width=3,label = label2, color=color_baseline)
        ax[1,0].set_xticks([8,16,24,32,40,48])
        ax[1,0].legend(frameon=False)
        ax[1,0].set_ylabel(self.labels['lkm'])
        ax[1,0].set_xlabel(self.labels['Työaika [h]'])
        
        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,\
            tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta1,tyot_virta1,tyot_virta_ansiosid1,tyot_virta_tm1,\
            unemp_dur1,unemp_lastdur1,unemp_basis_distrib1 = self.episodestats.load_simdistribs(filename)

        unemp_distrib2,emp_distrib2,unemp_distrib_bu2,\
            tyoll_distrib2,tyoll_distrib_bu2,\
            tyoll_virta2,tyot_virta2,tyot_virta_ansiosid2,tyot_virta_tm2,\
            unemp_dur2,unemp_lastdur2,unemp_basis_distrib2 = self.episodestats.load_simdistribs(filename2)

        #print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        #self.plot_compare_unemp_durdistribs(unemp_dur1,unemp_dur2,unemp_lastdur1,unemp_lastdur2,label1 = label1,label2 = label2)

        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label1 = label1,label2 = label2)
        #self.plot_unempdistribs(unemp_distrib1,figname = figname,max = 4,miny = 1e-3,maxy = 2,unemp_distrib2 = unemp_distrib2,label1 = label1,label2 = label2,fig=fig,ax=ax[1,1])
        self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max = 2.5,figname = figname,emp_distrib2 = unemp_distrib2,tyoll_distrib2 = tyoll_distrib2,
                                     label1 = label1,label2 = label2,fig=fig,ax=ax[1,1],kuva1=False,kuva2=False,kuva3=True,kuva4=False)
        add_label(ax)

        if figname is not None:
            plt.savefig(figname+'fig7.pdf')

    def plot_simfig_unempbasis(self,filename,grayscale = False,figname = None,ax = None,fig = None):
        '''
        Compare unemp basis against observation
        '''

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,\
            tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
            unemp_dur,unemp_lastdur,unemp_basis_distrib = self.episodestats.load_simdistribs(filename)

        nofig = False
        if fig is None:
            nofig = True
            fig,ax = plt.subplots(1,1)

        self.plot_unempbasis_distrib(unemp_basis_distrib,figname = None,ax=ax,fig=fig) # ax = ax[1],
        fig.subplots_adjust(wspace=0.9)
        #add_label(ax)

        if nofig:
            if figname is not None:
                    plt.savefig(figname, format = 'pdf')
                    
            plt.show()

    def plot_simstat_workforce_stack(self,empstate,figname = None):
        empstate_ratio = 100*self.episodestats.empstate/self.episodestats.alive
        ratio_label = self.labels['osuus']
        if figname is not None:
            self.plot_states(empstate_ratio,ylabel = ratio_label,stack = True,figname = figname+'_stack')
        else:
            self.plot_states(empstate_ratio,ylabel = ratio_label,stack = True)

    
    def compare_simtables(self,filename,filename2,figname=None,label1='rev',label2='baseline',grayscale=False):
        mean_htv1,mean_tyoll1,h_mean1,m_mean1,diff_htv1,mean_rew1,um_mean1,mean_tyolliset_osuus1,std_htv,h_std,mean_unempratio1 \
            = self.episodestats.get_simstats(filename,use_mean = True)
        median_htv1,median_tyoll1,h_median1,m_median1,median_rew1,median_tyott1,std_tyoll1,s_tyoll1,mean_unempratio1 \
            = self.episodestats.get_simstats(filename,use_mean = False)
        
        mean_htv2,mean_tyoll2,h_mean2,m_mean2,diff_htv2,mean_rew2,um_mean2,mean_tyolliset_osuus1,std_htv2,s_htv1,mean_unempratio2 \
            = self.episodestats.get_simstats(filename2,use_mean = True)
        median_htv2,median_tyoll2,h_median2,m_median2,median_rew2,median_tyott2,std_tyoll2,s_tyoll2,median_unempratio \
            = self.episodestats.get_simstats(filename2,use_mean = False)

        agg_htv1,agg_tyoll1,agg_rew1,agg_discounted_rew1,emp_tyolliset1,emp_tyolliset_osuus1,\
            emp_tyottomat1,emp_tyottomat_osuus1,emp_htv1,emps1,best_rew1,\
            best_emp1,emps1,agg_netincome1,agg_equivalent_netincome1,budget1,participants1,htv_budget1,\
            alives1,agg_empstate1,agg_alives1,agg_tyottomyysaste1,emp_tyottomyysaste1,pt_agg1,pt_age1,\
            galives1,agg_galives1,gempstate1,agg_gempstate1\
                = self.episodestats.load_simstats(filename)

        agg_htv2,agg_tyoll2,agg_rew2,agg_discounted_rew2,emp_tyolliset2,emp_tyolliset_osuus2,\
            emp_tyottomat2,emp_tyottomat_osuus2,emp_htv2,emps2,best_rew2,\
            best_emp2,emps2,agg_netincome2,agg_equivalent_netincome2,budget2,participants2,htv_budget2,\
            alives2,agg_empstate2,agg_alives2,agg_tyottomyysaste2,emp_tyottomyysaste2,pt_agg2,pt_age2,\
            galives2,agg_galives2,gempstate2,agg_gempstate2\
                = self.episodestats.load_simstats(filename2)
                
        if self.version>0:
            print('{}: lisäpäivillä on {:.0f} henkilöä'.format(label1,self.episodestats.count_putki_dist(emps1)))
            print('{}: lisäpäivillä on {:.0f} henkilöä'.format(label2,self.episodestats.count_putki_dist(emps2)))

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        mean_discounted_rew1 = np.mean(agg_discounted_rew1)
        mean_netincome1 = np.mean(agg_netincome1)
        mean_equi_netincome1 = np.mean(agg_equivalent_netincome1)

        mean_discounted_rew2 = np.mean(agg_discounted_rew2)
        mean_netincome2 = np.mean(agg_netincome2)
        mean_equi_netincome2 = np.mean(agg_equivalent_netincome2)        

        print('Vaikutus mediaanityöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(median_htv2-median_htv1,median_htv2,median_htv1))
        print('Vaikutus keskiarvotyöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(mean_htv2-mean_htv1,mean_htv2,mean_htv1))

        print(f'Mean undiscounted reward {mean_rew1} ({label1}) vs {mean_rew2} ({label2})')
        #print(f'Mean discounted reward {mean_discounted_rew1} v2 {mean_discounted_rew2}')
        print(f'Mean net income {mean_netincome1} mean equivalent net income {mean_equi_netincome2}')
        
        dif_budget = self.compare_df(budget1,budget2,cctext1 = 'e/v',cctext2 = 'e/v',cctext1_new = label1+' e/v',cctext2_new = label2+' e/v')
        dif_participants = self.compare_df(participants1,participants2,cctext1 = 'lkm',cctext2 = 'lkm',cctext1_new = label1+' (#)',cctext2_new = label2+' (#)')
        dif_htv_budget = self.compare_df(htv_budget1,htv_budget2,cctext1 = 'htv',cctext2 = 'htv',cctext2_new = label2+' (py)')

        print(self.sel_subset(dif_budget,0).to_latex())
        print(self.sel_subset(dif_budget,1).to_latex())
        print(self.sel_subset(dif_budget,2).to_latex())
        print(self.sel_subset(dif_budget,5).to_latex())
        print(self.sel_subset(dif_participants,4).to_latex())
        print(self.sel_subset(dif_htv_budget,3).to_latex())

        if self.minimal:
            print('Työllisyyden keskiarvo {:.0f} htv mediaani {:.0f} htv std {:.0f} htv'.format(mean_htv1,median_htv1,std_htv1))
        else:
            print(f'{label1:s} Työllisyyden keskiarvo keskiarvo {mean_htv1:.0f} htv, mediaani {median_htv1:.0f} htv, std {mean_tyoll1:.0f} htv\n'
                  f'keskiarvo {mean_tyoll1:.0f} työllistä, mediaani {median_tyoll1:.0f} työllistä, std {std_tyoll1:.0f} työllistä')
            print(f'{label2:s} Työllisyyden keskiarvo keskiarvo {mean_htv2:.0f} htv, mediaani {median_htv2:.0f} htv, std {mean_tyoll2:.0f} htv\n'
                  f'keskiarvo {mean_tyoll2:.0f} työllistä, mediaani {median_tyoll2:.0f} työllistä, std {std_tyoll2:.0f} työllistä')

    def compare_simstats(self,filename,filename2,figname=None,label1='rev',label2='baseline',grayscale=False):
        mean_htv1,mean_tyoll1,h_mean1,m_mean1,diff_htv1,mean_rew1,um_mean1,mean_tyolliset_osuus1,std_htv1,h_std,mean_unempratio1 \
            = self.episodestats.get_simstats(filename,use_mean = True)
        median_htv1,median_tyoll1,h_median1,m_median1,median_rew1,median_tyott1,std_tyoll1,s_tyoll1,median_unempratio1 \
            = self.episodestats.get_simstats(filename,use_mean = False)
        
        mean_htv2,mean_tyoll2,h_mean2,m_mean2,diff_htv2,mean_rew2,um_mean2,mean_tyolliset_osuus1,std_htv2,s_htv1,mean_unempratio2 \
            = self.episodestats.get_simstats(filename2,use_mean = True)
        median_htv2,median_tyoll2,h_median2,m_median2,median_rew2,median_tyott2,std_tyoll2,s_tyoll2,median_unempratio2 \
            = self.episodestats.get_simstats(filename2,use_mean = False)

        agg_htv1,agg_tyoll1,agg_rew1,agg_discounted_rew1,emp_tyolliset1,emp_tyolliset_osuus1,\
            emp_tyottomat1,emp_tyottomat_osuus1,emp_htv1,emps1,best_rew1,\
            best_emp1,emps1,agg_netincome1,agg_equivalent_netincome1,budget1,participants1,htv_budget1,\
            alives1,agg_empstate1,agg_alives1,agg_tyottomyysaste1,emp_tyottomyysaste1,pt_agg1,pt_agegroup1,\
            galives1,agg_galives1,gempstate1,agg_gempstate1\
                = self.episodestats.load_simstats(filename)

        agg_htv2,agg_tyoll2,agg_rew2,agg_discounted_rew2,emp_tyolliset2,emp_tyolliset_osuus2,\
            emp_tyottomat2,emp_tyottomat_osuus2,emp_htv2,emps2,best_rew2,\
            best_emp2,emps2,agg_netincome2,agg_equivalent_netincome2,budget2,participants2,htv_budget2,\
            alives2,agg_empstate2,agg_alives2,agg_tyottomyysaste2,emp_tyottomyysaste2,pt_agg2,pt_agegroup2,\
            galives2,agg_galives2,gempstate2,agg_gempstate2\
                = self.episodestats.load_simstats(filename2)
                
        color_reform = 'black'
        color_baseline = 'darkgray'
        color_obs = 'lightgray'

        if self.version>0:
            print('{}: lisäpäivillä on {:.0f} henkilöä'.format(label1,self.episodestats.count_putki_dist(emps1)))
            print('{}: lisäpäivillä on {:.0f} henkilöä'.format(label2,self.episodestats.count_putki_dist(emps2)))

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        mean_discounted_rew1 = np.mean(agg_discounted_rew1)
        mean_netincome1 = np.mean(agg_netincome1)
        mean_equi_netincome1 = np.mean(agg_equivalent_netincome1)

        mean_discounted_rew2 = np.mean(agg_discounted_rew2)
        mean_netincome2 = np.mean(agg_netincome2)
        mean_equi_netincome2 = np.mean(agg_equivalent_netincome2)        

        print('Vaikutus mediaanityöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(median_htv1-median_htv2,median_htv1,median_htv2))
        print('Vaikutus keskiarvotyöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(mean_htv1-mean_htv2,mean_htv1,mean_htv2))

        print(f'Mean undiscounted reward {mean_rew1} ({label1}) vs {mean_rew2} ({label2})')
        #print(f'Mean discounted reward {mean_discounted_rew1} v2 {mean_discounted_rew2}')
        print(f'Mean net income {mean_netincome1} mean equivalent net income {mean_equi_netincome2}')
        
        dif_budget = self.compare_df(budget1,budget2,cctext1 = 'e/v',cctext2 = 'e/v',cctext1_new = label1+' e/v',cctext2_new = label2+' e/v')
        dif_participants = self.compare_df(participants1,participants2,cctext1 = 'lkm',cctext2 = 'lkm',cctext1_new = label1+' (#)',cctext2_new = label2+' (#)')
        dif_htv_budget = self.compare_df(htv_budget1,htv_budget2,cctext1 = 'htv',cctext2 = 'htv',cctext2_new = label2+' (py)')

        print(tabulate(dif_budget, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
        print(tabulate(dif_participants, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))
        print('Henkilövuosia tiloissa skaalattuna väestötasolle henkilövuosina')
        print(tabulate(dif_htv_budget, headers = 'keys', tablefmt = 'psql', floatfmt = ",.0f"))

        if self.minimal:
            print('Työllisyyden keskiarvo {:.0f} htv mediaani {:.0f} htv std {:.0f} htv'.format(mean_htv1,median_htv1,std_htv1))
        else:
            print(f'{label1:s} Työllisyyden keskiarvo keskiarvo {mean_htv1:.0f} htv, med. {median_htv1:.0f} htv, lkm {mean_tyoll1:.0f}, std {std_htv1:.0f} htv \n'
                  f'keskiarvo {mean_tyoll1:.0f} työllistä, mediaani {median_tyoll1:.0f} työllistä, std {std_tyoll1:.0f} työllistä')
            print(f'{label2:s} Työllisyyden keskiarvo keskiarvo {mean_htv2:.0f} htv, med. {median_htv2:.0f} htv, lkm {mean_tyoll2:.0f}, std {std_htv2:.0f} htv\n'
                  f'keskiarvo {mean_tyoll2:.0f} työllistä, mediaani {median_tyoll2:.0f} työllistä, std {std_tyoll2:.0f} työllistä')
            N1 = agg_htv1.shape[0]
            N2 = agg_htv1.shape[0]
            diff0 = np.abs(mean_tyoll1-mean_tyoll2)
            std0 = np.sqrt(std_htv1**2/N1+std_htv2**2/N2)
            t95=norm.ppf(0.95)
            t99=norm.ppf(0.99)
            t999=norm.ppf(0.999)
            conf_interval95 = t95*std0            
            conf_interval99 = t99*std0            
            conf_interval999 = t999*std0            
            signif = norm.cdf(diff0/std0)
            print(f'conf_interval at 95 % (N1: {N1}, N2: {N2}): {conf_interval95:.0f}')
            print(f'conf_interval at 99 % (N1: {N1}, N2: {N2}): {conf_interval99:.0f}')
            print(f'conf_interval at 999 % (N1: {N1}, N2: {N2}): {conf_interval999:.0f}')
            print(f'significance at {signif:.6f}')

        fig,ax = plt.subplots()
        ax.set_xlabel('Poikkeama työllisyydessä [htv]')
        ax.set_ylabel(self.labels['Lukumäärä'])
        ax.hist(diff_htv1,color = color_reform)
        ax.hist(diff_htv2,color = color_baseline)
        x1,y1 = self.fit_norm(diff_htv1)
        x2,y2 = self.fit_norm(diff_htv2)
        ax.plot(x1,y1,color = 'black')
        ax.plot(x2,y2,color = 'gray')
        if figname is not None:
            plt.savefig(figname+'poikkeama.pdf')
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel('Työllisyys [htv]')
        ax.set_ylabel(self.labels['Lukumäärä'])
        ax.hist(h_mean1,color = color_reform)
        ax.hist(h_mean2,color = color_baseline)
        x1,y1 = self.fit_norm(h_mean1)
        x2,y2 = self.fit_norm(h_mean2)
        ax.plot(x1,y1,color = 'black')
        ax.plot(x2,y2,color = color_baseline)
        if figname is not None:
            plt.savefig(figname+'_htv_vrt.pdf')
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel('Palkkio')
        ax.set_ylabel(self.labels['Lukumäärä'])
        ax.hist(agg_rew1,color = color_reform)
        ax.hist(agg_rew2,color = color_baseline)
        plt.show()
        
        #fig,ax = plt.subplots()
        #ax.set_xlabel('Discounted rewards')
        #ax.set_ylabel('Lukumäärä')
        #ax.hist(agg_discounted_rew1,color = 'lightgray')
        #ax.hist(agg_discounted_rew2,color = 'darkgray')
        #plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['Työllisyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_mean1,label = label1,color=color_reform)
        #ax.plot(x,100*m_median1,label = 'mediaani '+label1)
        ax.plot(x,100*m_mean2,label = label2,color=color_baseline)
        #ax.plot(x,100*m_median2,label = 'mediaani '+label2)
        #ax.plot(x,100*(m_emp+s_emp),label = 'ka+std')
        #ax.plot(x,100*(m_emp-s_emp),label = 'ka-std')
        #ax.plot(x,100*m_best1,label = 'paras '+label1)
        #ax.plot(x,100*m_best2,label = 'paras '+label2)
        ax.plot(x,100*self.empstats.emp_stats(),'--',label = 'observation',color=color_obs)
        ax.legend(frameon=False)
        if figname is not None:
            plt.savefig(figname+'_emprate.pdf')
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['Työllisyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_mean1-100*m_mean2,color='black')
        if figname is not None:
            plt.savefig(figname+'_empdiff.pdf')

        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['Työttömyys [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*um_mean1,label = label1,color=color_reform)
        ax.plot(x,100*um_mean2,label = label2,color=color_baseline)
        ax.plot(x,100*self.empstats.unemp_stats(),'--',label = 'observation',color=color_obs)
        ax.legend(frameon=False)
        if figname is not None:
            plt.savefig(figname+'_unemp.pdf')

        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['Työttömyysaste [%]'])
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*mean_unempratio1,label = label1, color=color_reform)
        ax.plot(x,100*mean_unempratio2,label = label2, color=color_baseline)
        #ax.plot(x,100*self.empstats.unempratio_stats(),label = 'havainto')
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'_unemprate.pdf')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(8,48,6)
        wd=3
        plt.bar(x-wd,100*pt_agg1[0,:],width=3,label = label1, color=color_reform)
        plt.bar(x,100*pt_agg2[0,:],width=3,label = label2, color=color_baseline)
        ax.set_xticks([8,16,24,32,40,48])
        ax.legend()
        ax.set_ylabel(self.labels['osuus'])
        ax.set_xlabel(self.labels['Työaika [h]'])
        if figname is not None:
            plt.savefig(figname+'_worktime_prop.pdf')
        plt.show()

        fig,ax = plt.subplots()
        x = np.linspace(8,48,6)
        wd=3
        plt.bar(x-wd,pt_agg1[0,:]*mean_tyoll1,width=3,label = label1, color=color_reform)
        plt.bar(x,pt_agg2[0,:]*mean_tyoll2,width=3,label = label2, color=color_baseline)
        ax.set_xticks([8,16,24,32,40,48])
        ax.legend()
        ax.set_ylabel(self.labels['lkm'])
        ax.set_xlabel(self.labels['Työaika [h]'])
        if figname is not None:
            plt.savefig(figname+'_worktime_abs.pdf')
        plt.show()

        fig,ax = plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Hajonta työllisyysasteessa [%]')
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*s_tyoll1,label = label1,color=color_reform)
        ax.plot(x,100*s_tyoll2,label = label2,color=color_baseline)
        plt.show()

        if False:
            demog2 = self.empstats.get_demog()
            fig,ax = plt.subplots()
            ax.set_xlabel(self.labels['age'])
            ax.set_ylabel('cumsum työllisyys [lkm]')
            x = np.linspace(self.min_age,self.max_age,self.n_time)
            cs = np.cumsum(h_mean2[1:]-h_mean1[1:])
            c2 = np.cumsum(h_mean1[1:])
            c1 = np.cumsum(h_mean2[1:])
            ax.plot(x[1:],cs,label = label1)
            #emp_statsratio = 100*self.emp_stats()
            #ax.plot(x,emp_statsratio,label = 'havainto')
            #ax.legend()
            plt.show()

            for age in set([50,63,63.25,63.5]):
                mx = self.map_age(age)-1
                print('Kumulatiivinen työllisyysvaikutus {:.2f} vuotiaana {:.1f} htv ({:.0f} vs {:.0f})'.format(age,cs[mx],c1[mx],c2[mx]))
                    
        
        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,\
            tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta1,tyot_virta1,tyot_virta_ansiosid1,tyot_virta_tm1,\
            unemp_dur1,unemp_lastdur1,unemp_basis_distrib1 = self.episodestats.load_simdistribs(filename)

        unemp_distrib2,emp_distrib2,unemp_distrib_bu2,\
            tyoll_distrib2,tyoll_distrib_bu2,\
            tyoll_virta2,tyot_virta2,tyot_virta_ansiosid2,tyot_virta_tm2,\
            unemp_dur2,unemp_lastdur2,unemp_basis_distrib2 = self.episodestats.load_simdistribs(filename2)

        #print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        self.plot_compare_unemp_durdistribs(unemp_dur1,unemp_dur2,unemp_lastdur1,unemp_lastdur2,label1 = label1,label2 = label2)
        #print('Keskikestot viimeisimmän työttömyysjakson mukaan')

        self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label1 = label1,label2 = label2)
        self.plot_unempdistribs(unemp_distrib1,figname = figname,max = 4,miny = 1e-3,maxy = 2,unemp_distrib2 = unemp_distrib2,label1 = label1,label2 = label2)
        #self.plot_tyolldistribs(unemp_distrib1,tyoll_distrib1,tyollistyneet = True,figname = figname)
        self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max = 4,figname = figname,emp_distrib2 = unemp_distrib2,tyoll_distrib2 = tyoll_distrib2,label1 = label1,label2 = label2)
                                                        
    def plot_ps(self,cc2,label1 = 'oma',label2 = 'baseline'):
        scalex = self.empstats.get_demog()/self.episodestats.alive #n_pop
        scalex2 = cc2.episodestats.empstats.get_demog()/cc2.episodestats.alive #n_pop

        ps1 = self.episodestats.infostats_palkkatulo * scalex
        ps2 = cc2.episodestats.infostats_palkkatulo * scalex2

        print('summa1',np.sum(ps1),'summa2',np.sum(ps2))

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,ps1,label = label1)
        ax.plot(x,ps2,label = label2)
        ax.legend()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('palkkasummma')
        plt.show()

        cum_ps1_p2 = np.cumsum(ps1-ps2)
        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,cum_ps1_p2,label = label1)
        ax.legend()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('cumsum diff palkkasummma')
        plt.show()

        htv1 = np.sum(np.sum(self.episodestats.emp_htv,axis = 2),axis = 1) * scalex[:,0] * self.timestep
        htv2 = np.sum(np.sum(cc2.episodestats.emp_htv,axis = 2),axis = 1) * scalex2[:,0] * self.timestep

        print('htv1',np.sum(htv1),'htv2',np.sum(htv2))
        print('ps1/htv1',np.sum(ps1)/np.sum(htv1),'ps2/htv2',np.sum(ps2)/np.sum(htv2))
        print('dps/dhtv',(np.sum(ps1)-np.sum(ps2))/(np.sum(htv1)-np.sum(htv2)))
        print('dps1',np.sum(ps1)-np.sum(ps2),'hypot dps1',(np.sum(htv1)-np.sum(htv2))*np.sum(ps1)/np.sum(htv1))
        print('dps1 * veroaste',0.405*(np.sum(ps1)-np.sum(ps2)),'hypot dps1',0.405*(np.sum(htv1)-np.sum(htv2))*np.sum(ps1)/np.sum(htv1))
        print('dps/dhtv * veroaste',0.405*(np.sum(ps1)-np.sum(ps2))/(np.sum(htv1)-np.sum(htv2)),'hypot dps1',0.405*np.sum(ps1)/np.sum(htv1))

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,htv1,label = label1)
        ax.plot(x,htv2,label = label2)
        ax.legend()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('htv')
        plt.show()

        suhde1 = ps1[:,0]/htv1
        suhde2 = ps2[:,0]/htv2

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,suhde1,label = label1)
        ax.plot(x,suhde2,label = label2)
        ax.legend()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('ps/htv')
        plt.show()

        suhde = suhde1/suhde2

        fig,ax = plt.subplots()
        x = np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,suhde,label = 'suhteiden suhde')
        ax.legend()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('suhteiden suhde')
        plt.show()

        print('suhde 18-63',np.nanmean(suhde[18-18:63-18]))
        print('suhde',np.nanmean(suhde))
