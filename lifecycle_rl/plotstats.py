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
from fin_benefits import Labels
from scipy.stats import gaussian_kde
from .utils import empirical_cdf,print_html,modify_offsettext,add_source,setup_EK_fonts


#locale.setlocale(locale.LC_ALL, 'fi_FI')

class PlotStats():
    def __init__(self,stats,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year=2018,version=3,params=None,gamma=0.92,lang='English'):
        self.version=version
        self.gamma=gamma
        self.params=params
        self.params['n_time']=n_time
        self.params['n_emps']=n_emps
        self.episodestats=stats
        
        self.lab=Labels()
        self.reset(timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,params=params,lang=lang)

    def set_episodestats(self,stats):
        self.episodestat=stats

    def reset(self,timestep,n_time,n_emps,n_pop,env,minimal,min_age,max_age,min_retirementage,year,version=None,params=None,lang=None,dynprog=False):
        self.min_age=min_age
        self.max_age=max_age
        self.min_retirementage=min_retirementage
        self.minimal=minimal

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
        #self.episodestats.n_pop=n_pop
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
        
    def plot_various_groups(self,figname=None):
        empstate_ratio=100*self.episodestats.empstate/self.episodestats.alive
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

        if self.version in set([1,2,3,4,5,104]):
            self.plot_states(empstate_ratio,ylabel=ratio_label,ylimit=20,stack=False)
            self.plot_states(empstate_ratio,ylabel=ratio_label,unemp=True,stack=False)
            
    def compare_against(self,cc=None,cctext='toteuma'):
        if self.version in set([1,2,3,4,5,104]):
            q=self.episodestats.comp_budget(scale=True)
            if cc is None:
                q_stat=self.empstats.stat_budget()
            else:
                q_stat=cc.episodestats.comp_budget(scale=True)

            df1 = pd.DataFrame.from_dict(q,orient='index',columns=['e/v'])
            df2 = pd.DataFrame.from_dict(q_stat,orient='index',columns=[cctext])
            df=df1.copy()
            df[cctext]=df2[cctext]
            df['ero']=df1['e/v']-df2[cctext]

            print('Rahavirrat skaalattuna väestötasolle')
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

            q=self.episodestats.comp_participants(scale=True,lkm=False)
            q_lkm=self.episodestats.comp_participants(scale=True,lkm=True)
            if cc is None:
                q_stat=self.empstats.stat_participants()
                q_days=self.empstats.stat_days()
            else:
                q_stat=cc.episodestats.comp_participants(scale=True,lkm=True)
                q_days=cc.episodestats.comp_participants(scale=True,lkm=False)

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
            q=self.episodestats.comp_participants(scale=True)
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
        
        self.episodestats.comp_total_netincome()
        #self.plot_rewdist()

        if self.version in set([1,2,3,4,5,104]):
            self.compare_against()
        else:
            q=self.episodestats.comp_participants(scale=True)
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

        print_html('<h2>Simulation stats</h2>')
        print('Simulated individuals',self.episodestats.n_pop)

        print_html('<h2>Tilastot</h2>')

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.episodestats.comp_employed_ratio(self.episodestats.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1,osatyolliset1,kokotyolliset1,osata1,kokota1=self.episodestats.comp_tyollisyys_stats(self.episodestats.empstate/self.episodestats.n_pop,scale_time=True,start=18,end=65,full=True)

        tyollaste=tyollaste1*100
        print('\nSic! Työllisyysaste vastaa työvoimatilaston laskutapaa!')
        print(f'Työllisyysaste 18-64: {tyollaste:.2f}%')

        self.episodestats.comp_total_reward()
        self.episodestats.comp_total_reward(discounted=True)
        
        gini=self.empstats.get_gini(self.year)

        print('Gini coefficient is {:.3f} (havainto {:.3f})'.format(self.episodestats.comp_gini(),gini))

        print('\nSic! pienituloisuus lasketaan vain aikuisväestöstä!')
        abs_pienituloisuus=12000
        p50,p60,pt=self.episodestats.comp_pienituloisuus(level=abs_pienituloisuus)
        print('Pienituloisuus 50%: {:.2f}%; 60%: {:.2f}%; abs 1000 e/kk {:.2f}%'.format(100*p50,100*p60,100*pt))

        print_html('<h2>Sovite</h2>')

        #print('Real discounted reward {}'.format(self.episodestats.comp_realoptimrew()))
        #real=self.episodestats.comp_presentvalue()
        #print('Initial discounted reward {}'.format(np.mean(real[1,:])))

        print('Real discounted reward {}'.format(self.episodestats.get_average_discounted_reward()))
        print('Initial discounted reward {}'.format(self.episodestats.get_initial_reward()))

        print_html('<h2>Työssä</h2>')
        self.plot_emp(figname=figname)
        if self.version in set([1,2,3,4,5,104]):
            self.plot_gender_emp(figname=figname)
            self.plot_group_emp()
            self.plot_workforce()
            print_html('<h2>Tekemätön työ</h2>')
            self.plot_tekematon_tyo()

        print_html('<h2>Osa-aika</h2>')
        if self.version in set([5]):
            self.plot_pt_act()
        if self.version in set([1,2,3,4,5,104]):
            self.plot_parttime_ratio(figname=figname)
            
        if self.version in set([1,2,3,4,5,104]):
            print_html('<h2>Ryhmät</h2>')
            self.plot_outsider()        
            self.plot_various_groups(figname=figname)
            self.plot_group_student()

        if self.version in set([4,5,104]):
            print_html('<h2>Lapset ja puolisot</h2>')
            self.plot_spouse()
            self.plot_children()
            self.plot_family()
            self.plot_parents_in_work()
            
        if self.version in set([101,104]):
            print_html('<h2>Säästöt</h2>')
            self.plot_savings()

        print_html('<h2>Tulot</h2>')
        if self.version in set([3,4,5,104]):
            self.plot_tulot()
            
        self.plot_sal()

        print_html('<h2>Työttömyys</h2>')
        self.plot_toe()
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        keskikesto=self.episodestats.comp_unemp_durations()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        keskikesto=self.episodestats.comp_unemp_durations_v2()
        df = pd.DataFrame.from_dict(keskikesto,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

        self.plot_unemp_after_ra()

        if self.version in set([1,2,3,4,5,104]):
            print('Lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki()))

        self.plot_unemp(unempratio=True,figname=figname)
        self.plot_unemp(unempratio=False)

        if self.version in set([1,2,3,4,5,104]):
            self.plot_unemp_shares()
            self.plot_kassanjasen()
            self.plot_pinkslip()

        #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, no max age',ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää',ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50,figname=figname)

        if self.version in set([1,2,3,4,5,104]):
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=True,ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=50)
            self.plot_distrib(label='Jakauma ansiosidonnainen+putki, jakso päättynyt ennen 50v ikää, jäljellä oleva aika',plot_bu=False,ansiosid=True,tmtuki=False,putki=True,outsider=False,max_age=50)
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea',ansiosid=True,tmtuki=True,putki=False,outsider=False)
            #self.plot_distrib(label='Jakauma ansiosidonnainen+tmtuki ilman putkea, max Ikä 50v',ansiosid=True,tmtuki=True,putki=False,outsider=False,max_age=50)
            self.plot_distrib(label='Jakauma tmtuki',ansiosid=False,tmtuki=True,putki=False,outsider=False)
            #self.plot_distrib(label='Jakauma työvoiman ulkopuoliset',ansiosid=False,tmtuki=False,putki=False,outsider=True)
            #self.plot_distrib(label='Jakauma laaja (ansiosidonnainen+tmtuki+putki+ulkopuoliset)',laaja=True)
            
        if self.version in set([0,1,2,3,4,5,104]):
            print_html('<h2>Eläkkeet</h2>')
            self.plot_all_pensions()
            print_html('<h2>Työkyvyttömyyseläke</h2>')
            self.plot_disab()

        if self.version in set([1,2,3,4,5,104]):
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
            self.plot_emtr()
            
            
    def plot_pt_act(self):
        mask_osaaika=(self.episodestats.popempstate!=10) # osa-aika
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask_osaaika)).compressed()
        plt.hist(arr,density=True)
        plt.title('Osa-aika, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        arr=ma.array(self.episodestats.infostats_pop_pt_act,mask=mask_osaaika)
        plt.plot(x,ma.mean(arr,axis=1))
        plt.title('Osa-aika, pt-tila')
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        pt,ft,vept,veft=self.episodestats.comp_ptproportions()
        plt.stackplot(x,pt.T)
        plt.legend(labels=['25%','50%','75%'])
        plt.title('Osa-aika, pt-tila')
        plt.show()
        plt.stackplot(x,ft.T)
        plt.legend(labels=['100%','125%','150%'])
        plt.title('Kokoaika, pt-tila')
        plt.show()
        plt.stackplot(x,vept.T)
        plt.legend(labels=['25%','50%','75%'])
        plt.title('Ve+Osa-aika, pt-tila')
        plt.show()
        plt.stackplot(x,veft.T)
        plt.legend(labels=['100%','125%','150%'])
        plt.title('Ve+Kokoaika, pt-tila')
        plt.show()

        men_mask=(self.episodestats.infostats_group<4).T
        women_mask=~men_mask

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        mask=ma.mask_or(mask_osaaika,men_mask) # miehet pois
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask)).compressed()
        plt.hist(arr,density=True)
        plt.title('Osa-aika naiset, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        mask=ma.mask_or(mask_osaaika,women_mask) # naiset pois
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask)).compressed()
        plt.hist(arr,density=True)
        plt.title('Osa-aika miehet, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        mask_osaaika=(self.episodestats.popempstate!=8) # ve+osa-aika
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask_osaaika)).compressed()
        plt.hist(arr,density=True)
        plt.title('ve+osa-aika, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        mask_osaaika=(self.episodestats.popempstate!=9) # ve+koko-aika
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask_osaaika)).compressed()
        plt.hist(arr,density=True)
        plt.title('ve+koko-aika, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()
        
        mask_kokoaika=(self.episodestats.popempstate!=1) # kokoaika
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask_kokoaika)).compressed()
        plt.hist(arr,density=True)
        plt.title('Kokoaika, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        mask=ma.mask_or(mask_kokoaika,men_mask) # naiset pois
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask)).compressed()
        plt.hist(arr,density=True)
        plt.title('Kokoaika naiset, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        mask=ma.mask_or(mask_kokoaika,women_mask) # naiset pois
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask_kokoaika)).compressed()
        plt.hist(arr,density=True)
        plt.title('Kokoaika miehet, pt-tila: ave {}'.format(ma.mean(arr)))
        plt.show()

        mask=mask_kokoaika
        arr=ma.array(self.episodestats.infostats_pop_pt_act,mask=mask)
        plt.plot(x,ma.mean(arr,axis=1))
        plt.title('Kokoaika, pt-tila')
        plt.show()

        mask=mask_osaaika
        arr=ma.ravel(ma.array(self.episodestats.infostats_pop_pt_act,mask=mask)).compressed()
        plt.hist(arr,density=True)
        plt.title('Kokoaika, unemp')
        plt.show()


    def plot_unemp_after_ra(self):
        self.plot_states(self.episodestats.stat_unemp_after_ra,ylabel='Unemp after ret.age',stack=False,start_from=60,end_at=70)

    def plot_disab(self):
        w1,w2,n_tk=self.episodestats.comp_tkstats()
        print(f'Työkyvyttömyyseläkkeisiin menetetty palkkasumma {w1:,.2f} ja työpanoksen arvo {w2:,.2f}')
        self.plot_group_disab()
        self.plot_group_disab(xstart=60,xend=67)

    def plot_tekematon_tyo(self):
        w1,wplt=self.episodestats.comp_potential_palkkasumma(grouped=True,full=True)
        wplt2=wplt.copy()
        for k in range(15):
            w2=2.1*w1[k] # kerroin 2,1 muuttaa palkan työpanoksen arvoksi
            if k in [0,2,3,4,5,6,7,11,12,13,14]:
                print(f'Tilaan {k} menetetty palkkasumma {w1[k]:,.2f} ja työpanoksen arvo {w2:,.2f}')
            else:
                print(f'Tilan {k} palkkasumma {w1[k]:,.2f} ja työpanoksen arvo {w2:,.2f}')
                
        ps=np.sum(w1[[1,8,9,10]])
        tpa=2.1*ps
        nops=np.sum(w1[[0,2,3,4,5,6,7,11,12,13,14]])
        notpa=2.1*nops
        print(f'Palkkasumma {ps:,.2f} ja työpanoksen arvo {tpa:,.2f}')
        print(f'Menetetty palkkasumma {nops:,.2f} ja työpanoksen arvo {notpa:,.2f}')
        wplt[:,[1,8,9,10]]=0
        self.plot_states(wplt,ylabel='Menetetty palkkasumma',stack=True,ymaxlim=np.max(np.sum(wplt,axis=1)))
        wplt=wplt/np.sum(wplt,axis=1,keepdims=True)*100
        self.plot_states(wplt,ylabel='Menetetty palkkasumma [%]',stack=True)
        wplt[:,[0,2,4,5,6,7,11,12,13]]=0
        self.plot_states(wplt,ylabel='Menetetty palkkasumma',stack=True,ymaxlim=np.max(np.nansum(wplt,axis=1)))
        wplt=wplt/np.sum(wplt,axis=1,keepdims=True)*100
        self.plot_states(wplt,ylabel='Menetetty palkkasumma [%]',stack=True)

        wplt2[:,[1,2,3,5,6,7,8,9,10,11,12,14]]=0
        self.plot_states(wplt2,ylabel='Menetetty palkkasumma',stack=True,ymaxlim=np.max(np.nansum(wplt2,axis=1)))
        wplt2=wplt2/np.sum(wplt2,axis=1,keepdims=True)*100
        self.plot_states(wplt2,ylabel='Menetetty palkkasumma [%]',stack=True)

    def plot_all_pensions(self):
        alivemask=(self.episodestats.popempstate==self.env.get_mortstate()) # pois kuolleet
        kemask=(self.episodestats.infostats_pop_kansanelake<0.1)
        kemask=ma.mask_or(kemask,alivemask)
        temask=(self.episodestats.infostats_pop_kansanelake>0.1) # pois kansaneläkkeen saajat
        temask=ma.mask_or(temask,alivemask)
        notemask=(self.episodestats.infostats_pop_tyoelake>10.0) # pois kansaneläkkeen saajat
        notemask=ma.mask_or(notemask,alivemask)
        self.plot_pensions()
        self.plot_pension_stats(self.episodestats.stat_pop_paidpension/self.timestep,65,'kokoeläke ilman kuolleita',mask=alivemask)
        self.plot_pension_stats(self.episodestats.infostats_pop_tyoelake/self.timestep,65,'työeläke')
        self.plot_pension_stats(self.episodestats.infostats_paid_tyel_pension/self.timestep,65,'työeläkemaksun vastine')
        self.plot_pension_stats(self.episodestats.infostats_paid_tyel_pension/self.timestep,65,'työeläkemaksun vastine, vain työeläke',mask=temask)
        self.plot_pension_stats(self.episodestats.infostats_pop_tyoelake/self.timestep,65,'vain työeläke',mask=temask)
        self.plot_pension_stats(self.episodestats.infostats_pop_kansanelake/self.timestep,65,'kansanelake kaikki',max_pen=10_000,plot_ke=True)
        self.plot_pension_stats(self.episodestats.infostats_pop_kansanelake/self.timestep,65,'kansanelake>0',max_pen=10_000,mask=kemask,plot_ke=True)
        self.plot_pension_stats(self.episodestats.infostats_pop_kansanelake/self.timestep,65,'kansaneläke, ei työeläkettä',max_pen=10_000,mask=notemask,plot_ke=True)
        self.plot_pension_stats(self.episodestats.infostats_pop_tyoelake/self.timestep,65,'työeläke, jos kansanelake>0',max_pen=20_000,mask=kemask)
        self.plot_pension_stats(self.episodestats.infostats_pop_pension,60,'tulevat eläkkeet')
        self.plot_pension_stats(self.episodestats.infostats_pop_pension,60,'tulevat eläkkeet, vain elossa',mask=alivemask)
        self.plot_pension_time()


    def min_max(self):
        min_wage=np.min(self.episodestats.infostats_pop_wage)
        max_wage=np.max(self.episodestats.infostats_pop_wage)
        max_pension=np.max(self.episodestats.infostats_pop_pension)
        min_pension=np.min(self.episodestats.infostats_pop_pension)
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

#     def plot_ratiostats(self,t):
#         '''
#         Tee kuvia tuloksista
#         '''
#         x=np.linspace(self.min_age,self.max_age,self.n_time)
#         fig,ax=plt.subplots()
#         ax.set_xlabel('palkat')
#         ax.set_ylabel('freq')
#         ax.hist(self.episodestats.infostats_pop_wage[t,:])
#         plt.show()
#         fig,ax=plt.subplots()
#         ax.set_xlabel('aika')
#         ax.set_ylabel('palkat')
#         meansal=np.mean(self.episodestats.infostats_pop_wage,axis=1)
#         stdsal=np.std(self.episodestats.infostats_pop_wage,axis=1)
#         ax.plot(x,meansal)
#         ax.plot(x,meansal+stdsal)
#         ax.plot(x,meansal-stdsal)
#         plt.show()

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
        alive=self.episodestats.alive/self.episodestats.n_pop
        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Alive [%]')
        nn_time = int(np.ceil((self.max_age-self.min_age)*self.inv_timestep))+1
        x=np.linspace(self.min_age,self.max_age,nn_time)
        ax.plot(x[1:],alive[1:]*100)
        plt.show()
        
    def plot_pension_time(self):
        self.plot_y(self.episodestats.infostats_tyoelake,label='työeläke',
            y2=self.episodestats.infostats_kansanelake,label2='kansaneläke',
            ylabel='eläke [e/v]',
            start_from=60,end_at=70,show_legend=True)

        demog2=self.empstats.get_demog()
        scalex=demog2/self.episodestats.n_pop
        tyoelake_meno=self.episodestats.infostats_tyoelake*scalex
        kansanelake_meno=self.episodestats.infostats_kansanelake*scalex
        print('työeläkemeno alle 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(tyoelake_meno[:self.map_age(63)]),1_807.1))
        print('työeläkemeno yli 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(tyoelake_meno[self.map_age(63):]),24_227.2))
        print('kansaneläkemeno alle 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(kansanelake_meno[:self.map_age(63)]),679.6))
        print('kansaneläkemeno yli 63: {:.2f} vs tilasto {:.2f}'.format(np.sum(kansanelake_meno[self.map_age(63):]),1_419.9))
        
        self.plot_y(self.episodestats.infostats_kansanelake,label='kansaneläke',
            ylabel='kansaneläke [e/v]',
            start_from=60,end_at=70,show_legend=True)

        self.plot_y(self.episodestats.infostats_kansanelake/self.episodestats.infostats_tyoelake*100,label='suhde',
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
        ax.plot(x2[1:],scaled)
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

    def plot_unempdistribs(self,unemp_distrib,max=2.5,figname=None,miny=None,maxy=None):
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
                scaled,x=np.histogram(self.episodestats.infostats_pop_wage[t,:],bins=bins)
                x2=0.5*(x[1:]+x[0:-1])
                ax.plot(x2,scaled,label=t)
            plt.legend()
            plt.show()
        else:
            if sum:
                scaled,x=np.histogram(np.sum(self.episodestats.infostats_pop_wage,axis=0),bins=bins)
                x2=0.5*(x[1:]+x[0:-1])
                plt.plot(x2,scaled)
            else:
                fig,ax=plt.subplots()
                for t1 in range(t,t+n,1):
                    scaled,x=np.histogram(self.episodestats.infostats_pop_wage[t1,:],bins=bins)
                    x2=0.5*(x[1:]+x[0:-1])
                    ax.plot(x2,scaled,label=t1)
                plt.legend()
                plt.show()

    def test_salaries(self):

        def kuva(sal,ika,m,p,palkka):
            plt.hist(sal[:m],bins=50,density=True)
            ave=np.mean(sal[:m])/12
            palave=np.sum(palkka*p)/12/np.sum(palkka)
            plt.title('{}: ave {:,.2f} vs {:,.2f}'.format(ika,ave,palave))
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

    
        n=self.episodestats.n_pop
        
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
        for k in range(self.episodestats.n_pop):
            g=int(self.episodestats.infostats_group[k])
            for t in range(self.n_time-2):
                if self.episodestats.popempstate[t,k] in set([1]): # 9,8,10
                    wage=self.episodestats.infostats_pop_wage[t,k]
                    salx[t]=salx[t]+wage
                    saln[t]=saln[t]+1
                    salgx[t,g]=salgx[t,g]+wage
                    salgn[t,g]=salgn[t,g]+1
                    
            if self.episodestats.popempstate[self.map_age(20),k] in set([1]):
                sal20[m20]=self.episodestats.infostats_pop_wage[self.map_age(20),k]
                m20=m20+1
            if self.episodestats.popempstate[self.map_age(25),k] in set([1]):
                sal25[m25]=self.episodestats.infostats_pop_wage[self.map_age(25),k]
                m25=m25+1
            if self.episodestats.popempstate[self.map_age(30),k] in set([1]):
                sal30[m30]=self.episodestats.infostats_pop_wage[self.map_age(30),k]
                m30=m30+1
            if self.episodestats.popempstate[self.map_age(40),k] in set([1]):
                sal40[m40]=self.episodestats.infostats_pop_wage[self.map_age(40),k]
                m40=m40+1
            if self.episodestats.popempstate[self.map_age(50),k] in set([1]):
                sal50[m50]=self.episodestats.infostats_pop_wage[self.map_age(50),k]
                m50=m50+1
            if self.episodestats.popempstate[self.map_age(60),k] in set([1]):
                sal60[m60]=self.episodestats.infostats_pop_wage[self.map_age(60),k]
                m60=m60+1
            if self.episodestats.popempstate[self.map_age(65),k] in set([1,9]):
                sal65[m65]=self.episodestats.infostats_pop_wage[self.map_age(65),k]
                m65=m65+1

        salx_f=np.sum(salgx[:,3:6],axis=1)
        saln_f=np.sum(salgn[:,3:6],axis=1)
        salx_m=np.sum(salgx[:,0:3],axis=1)
        saln_m=np.sum(salgn[:,0:3],axis=1)

        salx=salx/np.maximum(1,saln)
        salgx=salgx/np.maximum(1,salgn)
        salx_f=salx_f/np.maximum(1,saln_f)
        salx_m=salx_m/np.maximum(1,saln_m)
        
        alivemask = (self.episodestats.popempstate==15).astype(bool)
        wdata=ma.array(self.episodestats.infostats_pop_wage,mask=alivemask)#.compressed()

        workmask = (self.episodestats.popempstate==2)
        workmask = ma.mask_or(workmask,self.episodestats.popempstate==15)
        workmask = ma.mask_or(workmask,self.episodestats.popempstate==3)
        wdata2=ma.array(self.episodestats.infostats_pop_wage,mask=workmask)#.compressed()
            
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
        plt.plot(data_range,ma.mean(wdata[::4,:],axis=1),label='malli alive')
        plt.plot(data_range,ma.mean(wdata2[::4,:],axis=1),label='malli not ret')
        plt.plot(data_range,salx[::4],label='malli töissä')
        data_range_b=np.arange(self.min_age,self.max_age+2)
        plt.plot(data_range_b,0.5*palkat_ika_miehet+0.5*palkat_ika_naiset,label='data')
        plt.legend()
        plt.show()

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
        n=self.episodestats.n_pop
        
        data_range=np.arange(self.min_age,self.max_age)
        minwage=78_400

        salx=np.zeros((self.episodestats.n_pop*self.n_time,1))
        saln=0
        salgx=np.zeros((self.n_groups,self.episodestats.n_pop*self.n_time))
        salgn=np.zeros(self.n_groups,dtype=int)
        for k in range(self.episodestats.n_pop):
            for t in range(self.n_time-2):
                wage=self.episodestats.infostats_pop_wage[t,k]
                if self.episodestats.popempstate[t,k] in set([1]) and wage>minwage: # 9,8,10
                    salx[saln]=wage
                    saln+=1
                    g=int(self.episodestats.infostats_group[k])
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

        scaled1=virta1*demog2/self.episodestats.n_pop #/self.episodestats.alive
        scaled2=virta2*demog2/self.episodestats.n_pop #/self.episodestats.alive

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
        ax.plot(x,100*(self.episodestats.empstate[:,5]+self.episodestats.empstate[:,6]+self.episodestats.empstate[:,7])/self.episodestats.alive[:,0],label='vanhempainvapailla')
        #emp_statsratio=100*self.empstats.outsider_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,5,3:6]+self.episodestats.gempstate[:,6,3:6]+self.episodestats.gempstate[:,7,3:6],1,
            keepdims=True)/np.sum(self.episodestats.galive[:,3:6],1,keepdims=True),label='vanhempainvapailla, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,5,0:3]+self.episodestats.gempstate[:,6,0:3]+self.episodestats.gempstate[:,7,0:3],1,
            keepdims=True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims=True),label='vanhempainvapailla, miehet')
#        emp_statsratio=100*self.empstats.outsider_stats(g=1)
#        ax.plot(x,emp_statsratio,label=self.labels['havainto, naiset'])
#        emp_statsratio=100*self.empstats.outsider_stats(g=2)
#        ax.plot(x,emp_statsratio,label=self.labels['havainto, miehet'])

        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        
        if printtaa:
            nn=np.sum(self.episodestats.galive[:,3:6],1,keepdims=True)
            n=np.sum(100*(self.episodestats.gempstate[:,5,3:6]+self.episodestats.gempstate[:,6,3:6]+self.episodestats.gempstate[:,7,3:6]),1,keepdims=True)/nn
            mn=np.sum(self.episodestats.galive[:,0:3],1,keepdims=True)
            m=np.sum(100*(self.episodestats.gempstate[:,5,0:3]+self.episodestats.gempstate[:,6,0:3]+self.episodestats.gempstate[:,7,0:3]),1,keepdims=True)/mn
            
        ratio_label=self.labels['ratio']
        empstate_ratio=100*self.episodestats.empstate/self.episodestats.alive
        self.plot_states(empstate_ratio,ylabel=ratio_label,parent=True,stack=False)

    def plot_outsider(self,printtaa=True):
        '''
        plottaa työvoiman ulkopuolella olevat
        mukana ei isyysvapaat, opiskelijat, armeijassa olevat eikä alle 3 kk kestäneet äitiysvpaat
        '''
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,100*(self.episodestats.empstate[:,11]+self.episodestats.empstate[:,5]+self.episodestats.empstate[:,7]+self.episodestats.empstate[:,14]
            -self.episodestats.infostats_mother_in_workforce[:,0])/self.episodestats.alive[:,0],
            label='työvoiman ulkopuolella, ei opiskelija, sis. vanh.vapaat')
        emp_statsratio=100*self.empstats.outsider_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(x,100*(self.episodestats.empstate[:,11])/self.episodestats.alive[:,0],
            label='työvoiman ulkopuolella, ei opiskelija, ei vanh.vapaat')
        emp_statsratio=100*self.empstats.outsider_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.plot(x,100*(self.episodestats.empstate[:,14])/self.episodestats.alive[:,0],
            label='sv-päivärahalla')
        #emp_statsratio=100*self.empstats.outsider_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        
        fig,ax=plt.subplots()
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,14,3:6],1,keepdims=True)/np.sum(self.episodestats.galive[:,3:6],1,keepdims=True),label='sv-päivärahalla, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,14,0:3],1,keepdims=True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims=True),label='sv-päivärahalla, miehet')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()
        
        fig,ax=plt.subplots()
        ax.plot(x,100*(np.sum(self.episodestats.gempstate[:,11,3:6]+self.episodestats.gempstate[:,5,3:6]+self.episodestats.gempstate[:,7,3:6]+self.episodestats.gempstate[:,14,3:6],1,keepdims=True)
            -self.episodestats.infostats_mother_in_workforce)/np.sum(self.episodestats.galive[:,3:6],1,keepdims=True),
            label='työvoiman ulkopuolella, naiset')
        ax.plot(x,100*np.sum(self.episodestats.gempstate[:,11,0:3]+self.episodestats.gempstate[:,7,0:3]+self.episodestats.gempstate[:,14,0:3],1,keepdims=True)/np.sum(self.episodestats.galive[:,0:3],1,keepdims=True),
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
            nn=np.sum(self.episodestats.galive[:,3:6],1,keepdims=True)
            n=np.sum(100*(self.episodestats.gempstate[:,5,3:6]+self.episodestats.gempstate[:,6,3:6]+self.episodestats.gempstate[:,7,3:6]),1,keepdims=True)/nn
            mn=np.sum(self.episodestats.galive[:,0:3],1,keepdims=True)
            m=np.sum(100*(self.episodestats.gempstate[:,5,0:3]+self.episodestats.gempstate[:,6,0:3]+self.episodestats.gempstate[:,7,0:3]),1,keepdims=True)/mn
            
    def plot_tulot(self):
        '''
        plot net income per person
        '''
        x=np.linspace(self.min_age+self.timestep,self.max_age-1,self.n_time-2)
        fig,ax=plt.subplots()
        tulot=self.episodestats.infostats_tulot_netto[1:-1,0]/self.timestep/self.episodestats.alive[1:-1,0]
        ax.plot(x,tulot,label='tulot netto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Tulot netto [e/v]')
        ax.legend()
        plt.show()

        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack=False)
        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack=False,unemp=True)
        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack=False,emp=True)
        self.plot_states(self.episodestats.infostats_tulot_netto_emp[1:-1,:]/self.timestep/self.episodestats.empstate[1:-1,:],'Tulot netto tiloittain [e/v]',stack=False,all_emp=True)

    def plot_pinkslip(self):
        pink=100*self.episodestats.infostats_pinkslip/np.maximum(1,self.episodestats.empstate)
        self.plot_states(pink,'Karenssittomien osuus tilassa [%]',stack=False,unemp=True)

    def plot_student(self):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x+self.timestep,100*self.episodestats.empstate[:,12]/self.episodestats.alive[:,0],label='opiskelija tai armeijassa')
        emp_statsratio=100*self.empstats.student_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        emp_statsratio=100*self.empstats.student_stats()
        diff=100.0*self.episodestats.empstate[:,12]/self.episodestats.alive[:,0]-emp_statsratio
        ax.plot(x+self.timestep,diff,label='opiskelija tai armeijassa, virhe')
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ratio'])
        ax.legend()
        plt.show()

    def plot_kassanjasen(self):
    
        x2,vrt=self.empstats.get_kassanjasenyys_rate()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        jasenia=100.0*self.episodestats.infostats_kassanjasen/self.episodestats.alive
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
                opiskelijat=np.sum(self.episodestats.gempstate[:,12,0:3],axis=1)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,0:3],1)
            else:
                leg='Opiskelijat+Armeija Naiset'
                opiskelijat=np.sum(self.episodestats.gempstate[:,12,3:6],axis=1)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,3:6],1)

            opiskelijat=np.reshape(opiskelijat,(self.episodestats.galive.shape[0],1))
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
                opiskelijat=np.sum(self.episodestats.gempstate[:,3,0:3],axis=1)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,0:3],1)
            else:
                leg='TK Naiset'
                opiskelijat=np.sum(self.episodestats.gempstate[:,3,3:6],axis=1)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,3:6],1)

            opiskelijat=np.reshape(opiskelijat,(self.episodestats.galive.shape[0],1))
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
        valtionvero_ratio=100*self.episodestats.infostats_valtionvero_distrib/np.reshape(np.sum(self.episodestats.infostats_valtionvero_distrib,1),(-1,1))
        kunnallisvero_ratio=100*self.episodestats.infostats_kunnallisvero_distrib/np.reshape(np.sum(self.episodestats.infostats_kunnallisvero_distrib,1),(-1,1))
        vero_ratio=100*(self.episodestats.infostats_kunnallisvero_distrib+self.episodestats.infostats_valtionvero_distrib)/(np.reshape(np.sum(self.episodestats.infostats_valtionvero_distrib,1),(-1,1))+np.reshape(np.sum(self.episodestats.infostats_kunnallisvero_distrib,1),(-1,1)))

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

        valtionvero_osuus,kunnallisvero_osuus,vero_osuus=self.episodestats.comp_taxratios()

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


    def v2_states(self,x):
        return 'Ansiosidonnaisella {:.2f}\nKokoaikatyössä {:.2f}\nVanhuuseläkeläiset {:.2f}\nTyökyvyttömyyseläkeläiset {:.2f}\n'.format(x[0],x[1],x[2],x[3])+\
          'Putkessa {:.2f}\nÄitiysvapaalla {:.2f}\nIsyysvapaalla {:.2f}\nKotihoidontuella {:.2f}\n'.format(x[4],x[5],x[6],x[7])+\
          'VE+OA {:.2f}\nVE+kokoaika {:.2f}\nOsa-aikatyö {:.2f}\nTyövoiman ulkopuolella {:.2f}\n'.format(x[8],x[9],x[10],x[11])+\
          'Opiskelija/Armeija {:.2f}\nTM-tuki {:.2f}\n'.format(x[12],x[13])

    def v2_groupstates(self,xx):
        x=self.group_taxes(xx)
        return 'Etuudella olevat {:.2f}\nTyössä {:.2f}\nEläkkeellä {:.2f}\n'.format(x[0],x[1],x[2])

    def plot_children(self,figname=None):
        c3,c7,c18=self.episodestats.comp_children()
        
        self.plot_y(c3,label='Alle 3v lapset',
                    y2=c7,label2='Alle 7v lapset',
                    y3=c18,label3='Alle 18v lapset',
                    ylabel='Lapsia (lkm)',
                    show_legend=True)

    def plot_workforce(self,figname=None):

        workforce=self.episodestats.comp_workforce(self.episodestats.empstate,self.episodestats.alive)

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
            self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio=False) #,mother_in_workforce=self.episodestats.infostats_mother_in_workforce)
        #tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio=False,mother_in_workforce=self.episodestats.infostats_mother_in_workforce)

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
            state_0=np.argwhere(self.episodestats.popempstate[t,:]==0)
            savings_0[t]=np.mean(self.episodestats.infostats_savings[t,state_0[:]])
            act_savings_0[t]=np.mean(self.sav_actions[t,state_0[:]])
            state_1=np.argwhere(self.episodestats.popempstate[t,:]==1)
            savings_1[t]=np.mean(self.episodestats.infostats_savings[t,state_1[:]])
            act_savings_1[t]=np.mean(self.sav_actions[t,state_1[:]])
            state_2=np.argwhere(self.episodestats.popempstate[t,:]==2)
            savings_2[t]=np.mean(self.episodestats.infostats_savings[t,state_2[:]])
            act_savings_2[t]=np.mean(self.sav_actions[t,state_2[:]])

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=np.mean(self.episodestats.infostats_savings,axis=1)
        ax.plot(x,savings,label='savings all')
        ax.legend()
        plt.title('Savings all')
        plt.show()

        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        savings=np.mean(self.episodestats.infostats_savings,axis=1)
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
        pops=np.random.randint(self.episodestats.n_pop,size=20)
        ax.plot(x,self.episodestats.infostats_savings[:,pops],label='savings all')
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
        savings=np.mean(self.episodestats.infostats_savings,axis=1)
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
                empstate_ratio=100*np.sum(self.episodestats.gempstate[:,:,0:3],axis=2)/(np.sum(self.episodestats.galive[:,0:3],axis=1)[:,None])
                genderlabel='miehet'
            else:
                empstate_ratio=100*np.sum(self.episodestats.gempstate[:,:,3:6],axis=2)/(np.sum(self.episodestats.galive[:,3:6],axis=1)[:,None])
                genderlabel='naiset'
            if figname is not None:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),stack=True,figname=figname+'_stack')
            else:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),stack=True)

            if self.version in set([1,2,3,4,5,104]):
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),ylimit=20,stack=False)
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),parent=True,stack=False)
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),unemp=True,stack=False)

            if figname is not None:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),start_from=60,stack=True,figname=figname+'_stack60')
            else:
                self.plot_states(empstate_ratio,ylabel=self.labels['osuus tilassa x'].format(genderlabel),start_from=60,stack=True)

    def plot_parents_in_work(self):
        empstate_ratio=100*self.episodestats.empstate/self.episodestats.alive
        ml=100*self.episodestats.infostats_mother_in_workforce/self.episodestats.alive
        self.plot_y(ml,label='mothers in workforce',show_legend=True,
            y2=empstate_ratio[:,6],ylabel=self.labels['ratio'],label2='isyysvapaa')

    def plot_spouse(self,figname=None,grayscale=False):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        puolisoita=np.sum(self.episodestats.infostats_puoliso,axis=1)
            
        spouseratio=puolisoita/self.episodestats.alive[:,0]
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
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio=True)
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
            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(self.episodestats.empstate,self.episodestats.alive,unempratio=False)
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
                gempstate=np.sum(self.episodestats.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.episodestats.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,3:6],1)
                leg='Naiset'
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(gempstate,alive,unempratio=unempratio)

            ax.plot(x,tyottomyysaste,color=color,label='{} {}'.format(labeli2,leg))

        if grayscale:
            lstyle='--'
        else:
            lstyle='--'

        if self.version in set([1,2,3,4,5,104]):
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

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_gempratios(gender=g,unempratio=False)

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
        empstate_ratio=100*self.episodestats.empstate/self.episodestats.alive
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
                gempstate=np.sum(self.episodestats.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.episodestats.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,3:6],1)
                leg=self.labels['Naiset']
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(gempstate,alive)

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
                gempstate=np.sum(self.episodestats.gempstate[:,:,0:3],axis=2)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,0:3],1)
                color='darkgray'
            else:
                gempstate=np.sum(self.episodestats.gempstate[:,:,3:6],axis=2)
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=np.sum(self.episodestats.galive[:,3:6],1)
                leg=self.labels['Naiset']
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(gempstate,alive)

            x=np.linspace(self.min_age,self.max_age,self.n_time)
            #ax.plot(x,tyollisyysaste,color=color,label='{} {}'.format(self.labels['tyollisyysaste %'],leg))

        for group in range(self.n_groups):
            if group<3:
                leg=self.labels['Miehet']+'group '+str(group)
                gempstate=self.episodestats.gempstate[:,:,group]
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=self.episodestats.galive[:,group]
                color='darkgray'
            else:
                gempstate=self.episodestats.gempstate[:,:,group]
                alive=np.zeros((self.episodestats.galive.shape[0],1))
                alive[:,0]=self.episodestats.galive[:,group]
                leg=self.labels['Naiset']+'group '+str(group)
                color='black'

            tyollisyysaste,osatyoaste,tyottomyysaste,ka_tyottomyysaste=self.episodestats.comp_empratios(gempstate,alive)

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
        if self.version in set([1,2,3,4,5,104]):
            self.plot_states(self.episodestats.stat_pension,ylabel='Tuleva eläke [e/v]',stack=False)
            self.plot_states(self.episodestats.stat_paidpension,ylabel='Alkanut eläke [e/v]',stack=False)

    def plot_career(self):
        if self.version in set([1,2,3,4,5,104]):
            self.plot_states(self.episodestats.stat_tyoura,ylabel='Työuran pituus [v]',stack=False)

    def plot_ratiostates(self,statistic,ylabel='',ylimit=None, show_legend=True, parent=False,work60=False,\
                         unemp=False,stack=False,no_ve=False,figname=None,emp=False,oa_unemp=False,start_from=None,end_at=None):
        self.plot_states(statistic/self.episodestats.empstate[:statistic.shape[0],:statistic.shape[1]],ylabel=ylabel,ylimit=ylimit,no_ve=no_ve,\
                        show_legend=show_legend,parent=parent,unemp=unemp,\
                        stack=stack,figname=figname,emp=emp,oa_unemp=oa_unemp,work60=work60,start_from=start_from,end_at=end_at)

    def count_putki(self,emps=None):
        if emps is None:
            piped=np.reshape(self.episodestats.empstate[:,4],(self.episodestats.empstate[:,4].shape[0],1))
            demog2=self.empstats.get_demog()
            putkessa=self.timestep*np.nansum(piped[1:]/self.episodestats.alive[1:]*demog2[1:])
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
                    onlyunemp=False,reverse=False,grayscale=False,emp=False,oa_emp=False,oa_unemp=False,all_emp=False,sv=False):
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
        if self.version in set([1,2,3,4,5,104]):
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
                pal=sns.color_palette("colorblind", self.n_employment)  # hls, husl, cubehelix
            reverse=True

            if parent:
                if self.version in set([1,2,3,4,5,104]):
                    ax.stackplot(x,ura_mother,ura_dad,ura_kht,
                        labels=('äitiysvapaa','isyysvapaa','khtuki'), colors=pal)
            elif unemp:
                if self.version in set([1,2,3,4,5,104]):
                    ax.stackplot(x,ura_unemp,ura_pipe,ura_student,ura_outsider,ura_tyomarkkinatuki,ura_svpaiva,
                        labels=('tyött','putki','opiskelija','ulkona','tm-tuki','svpaivaraha'), colors=pal)
                else:
                    ax.stackplot(x,ura_unemp,labels=('tyött'), colors=pal)
            elif onlyunemp:
                if self.version in set([1,2,3,4,5,104]):
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
                if self.version in set([1,2,3,4,5,104]):
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
                if self.version in set([1,2,3,4,5,104]):
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
            elif unemp:
                ax.plot(x,ura_unemp,label='tyött')
                if self.version in set([1,2,3,4,5,104]):
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_pipe,label='putki')
            elif oa_unemp:
                ax.plot(x,ura_unemp,label='tyött')
                if self.version in set([1,2,3,4,5,104]):
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_student,label='student')
                    ax.plot(x,ura_outsider,label='outsider')
                    ax.plot(x,ura_pipe,label='putki')
                    ax.plot(x,ura_osatyo,label='osa-aika')
            elif emp:
                ax.plot(x,ura_emp,label='kokoaikatyö')
                ax.plot(x,ura_osatyo,label='osatyö')
            elif oa_emp:
                ax.plot(x,ura_veosatyo,label='ve+kokoaikatyö')
                ax.plot(x,ura_vetyo,label='ve+osatyö')
            elif all_emp:
                ax.plot(x,ura_emp,label='kokoaikatyö')
                ax.plot(x,ura_osatyo,label='osatyö')
                ax.plot(x,ura_veosatyo,label='ve+osatyö')
                ax.plot(x,ura_vetyo,label='ve+työ')
            elif sv:
                ax.plot(x,ura_svpaiva,label='sv-päiväraha')
            elif work60:
                ax.plot(x,ura_ret,label='eläke')
                ax.plot(x,ura_emp,label='kokoaikatyö')
                if self.version in set([1,2,3,4,5,104]):
                    ax.plot(x,ura_osatyo,label='osatyö')
                    ax.plot(x,ura_vetyo,label='ve+kokoaikatyö')
                    ax.plot(x,ura_veosatyo,label='ve+osatyö')
            else:
                ax.plot(x,ura_unemp,label='tyött')
                ax.plot(x,ura_ret,label='eläke')
                ax.plot(x,ura_emp,label='kokoaikatyö')
                if self.version in set([1,2,3,4,5,104]):
                    ax.plot(x,ura_osatyo,label='osatyö')
                    ax.plot(x,ura_disab,label='tk')
                    ax.plot(x,ura_pipe,label='putki')
                    ax.plot(x,ura_tyomarkkinatuki,label='tm-tuki')
                    ax.plot(x,ura_mother,label='äitiysvapaa')
                    ax.plot(x,ura_dad,label='isyysvapaa')
                    ax.plot(x,ura_kht,label='khtuki')
                    ax.plot(x,ura_vetyo,label='ve+kokoaikatyö')
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
        if self.version in set([1,2,3,4,5,104]):
            self.plot_ratiostates(self.episodestats.stat_toe,'työssäolo-ehdon pituus 28 kk aikana [v]',stack=False)

    def plot_sal(self):
        self.plot_states(self.episodestats.salaries_emp,'Keskipalkka [e/v]',stack=False)
        self.plot_states(self.episodestats.salaries_emp,'Keskipalkka [e/v]',stack=False,emp=True)
        self.plot_states(self.episodestats.salaries_emp,'Keskipalkka [e/v]',stack=False,all_emp=True)

    def plot_moved(self):
        siirtyneet_ratio=self.episodestats.siirtyneet/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        pysyneet_ratio=self.episodestats.pysyneet/self.episodestats.alive*100
        self.plot_states(pysyneet_ratio,ylabel='Pysyneet tilassa',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(pysyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,:,1]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet työhön tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,7,:]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet kotihoidontuelle',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,8,:]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet ve+oatyö tilaan',stack=True,start_from=60,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,9,:]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet ve+työ tilaan',stack=True,start_from=60,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,:,4]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,start_from=58,ylabel='Siirtyneet putkeen tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,4,:]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,start_from=58,ylabel='Siirtyneet putkesta tilaan',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,:,0]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet työttömäksi tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,:,13]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet tm-tuelle tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,:,10]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet osa-aikatyöhön tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,:,14]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet svpäivärahalle tilasta',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))
        siirtyneet_ratio=self.episodestats.siirtyneet_det[:,14,:]/self.episodestats.alive*100
        self.plot_states(siirtyneet_ratio,ylabel='Siirtyneet svpäivärahalta tilaan',stack=True,
                        yminlim=0,ymaxlim=min(100,1.1*np.nanmax(np.cumsum(siirtyneet_ratio,1))))

    def plot_ave_stay(self):
        self.plot_states(self.episodestats.time_in_state,ylabel='Ka kesto tilassa',stack=False)
        self.plot_y(self.episodestats.time_in_state[:,1],ylabel='Ka kesto työssä')
        self.plot_y(self.episodestats.time_in_state[:,0],ylabel='Ka kesto työttömänä')

    def plot_ove(self):
        self.plot_ratiostates(self.episodestats.infostats_ove,ylabel='Ove',stack=False,start_from=60)
        #self.plot_ratiostates(np.sum(self.episodestats.infostats_ove,axis=1),ylabel='Ove',stack=False)
        self.plot_y(np.sum(self.episodestats.infostats_ove,axis=1)/self.episodestats.alive[:,0],ylabel='Oven ottaneet',start_from=60,end_at=70)
        self.plot_y((self.episodestats.infostats_ove[:,1]+self.episodestats.infostats_ove[:,10])/(self.episodestats.empstate[:,1]+self.episodestats.empstate[:,10]),label='Kaikista työllisistä',
            y2=(self.episodestats.infostats_ove[:,1])/(self.episodestats.empstate[:,1]),label2='Kokotyöllisistä',
            y3=(self.episodestats.infostats_ove[:,10])/(self.episodestats.empstate[:,10]),label3='Osatyöllisistä',
            ylabel='Oven ottaneet',show_legend=True,start_from=60,end_at=70)
        self.plot_y((self.episodestats.infostats_ove[:,0]+self.episodestats.infostats_ove[:,13]+self.episodestats.infostats_ove[:,4])/(self.episodestats.empstate[:,0]+self.episodestats.empstate[:,13]+self.episodestats.empstate[:,4]),label='Kaikista työttömistä',
            y2=(self.episodestats.infostats_ove[:,0])/(self.episodestats.empstate[:,0]),label2='Ansiosid.',
            y3=(self.episodestats.infostats_ove[:,4])/(self.episodestats.empstate[:,4]),label3='Putki',
            y4=(self.episodestats.infostats_ove[:,13])/(self.episodestats.empstate[:,13]),label4='TM-tuki',
            ylabel='Oven ottaneet',show_legend=True,start_from=60,end_at=70)
        
        ovex=np.sum(self.episodestats.infostats_ove_g*np.maximum(1,self.episodestats.gempstate),axis=1)/np.maximum(1,np.sum(self.episodestats.gempstate,axis=1))
        self.plot_groups(ovex,start_from=60,ylabel='Osuus')

    def plot_reward(self):
        self.plot_ratiostates(self.episodestats.rewstate,ylabel='Keskireward tilassa',stack=False)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,no_ve=True)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,oa_unemp=True)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,oa_unemp=True,start_from=60)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,start_from=60)
        self.plot_ratiostates(self.episodestats.rewstate[:-1],ylabel='Keskireward tilassa',stack=False,start_from=60,work60=True)
        #self.plot_y(np.sum(self.episodestats.rewstate,axis=1),label='Koko reward tilassa')

    def vector_to_array(self,x):
        return x[:,None]


    def plot_wage_reduction(self):
        emp=np.array([1,10])
        unemp=np.array([0,13,4])
        gen_red=np.sum(self.episodestats.stat_wage_reduction*self.episodestats.empstate,axis=1)[:,None]/np.maximum(1,self.episodestats.alive)
        self.plot_y(gen_red,ylabel='Average wage reduction')
        gen_red_w=np.sum(np.sum(self.episodestats.stat_wage_reduction_g[:,:,0:3]*self.episodestats.gempstate[:,:,0:3],axis=2),axis=1)[:,None]/np.maximum(1,np.sum(self.episodestats.galive[:,0:3],axis=1))[:,None]
        gen_red_m=np.sum(np.sum(self.episodestats.stat_wage_reduction_g[:,:,3:6]*self.episodestats.gempstate[:,:,3:6],axis=2),axis=1)[:,None]/np.maximum(1,np.sum(self.episodestats.galive[:,3:6],axis=1))[:,None]
        self.plot_y(gen_red_w,ylabel='Average wage reduction by gender',label='women',y2=gen_red_m,label2='men',show_legend=True)
        #gen_red=np.sum(self.episodestats.stat_wage_reduction[:,emp]*self.episodestats.empstate[:,emp],axis=1)[:,None]/np.maximum(1,np.sum(self.episodestats.empstate[:,emp],axis=1))
        #gen_red=self.episodestats.stat_wage_reduction[:,emp]/np.maximum(1,self.episodestats.empstate[:,emp])
        #self.plot_y(gen_red,ylabel='Employed wage reduction',show_legend=True)
        #gen_red=np.sum(self.episodestats.stat_wage_reduction[:,unemp]*self.episodestats.empstate[:,unemp],axis=1)[:,None]/np.maximum(1,np.sum(self.episodestats.empstate[:,unemp],axis=1))
        #gen_red=self.episodestats.stat_wage_reduction[:,unemp]/np.maximum(1,self.episodestats.empstate[:,unemp])
        #self.plot_y(gen_red,ylabel='Unemployed wage reduction',show_legend=True)
        self.plot_states(self.episodestats.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False)
        self.plot_states(self.episodestats.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False,unemp=True)
        self.plot_states(self.episodestats.stat_wage_reduction,ylabel='wage-reduction tilassa',stack=False,emp=True)
        #self.plot_ratiostates(np.log(1.0+self.episodestats.stat_wage_reduction),ylabel='log 5wage-reduction tilassa',stack=False)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,0:3],axis=2),ylabel='wage-reduction tilassa naiset',stack=False)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,3:6],axis=2),ylabel='wage-reduction tilassa miehet',stack=False)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,0:3],axis=2),ylabel='wage-reduction tilassa, naiset',stack=False,unemp=True)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,3:6],axis=2),ylabel='wage-reduction tilassa, miehet',stack=False,unemp=True)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,0:3],axis=2),ylabel='wage-reduction tilassa, naiset',stack=False,emp=True)
        self.plot_states(np.mean(self.episodestats.stat_wage_reduction_g[:,:,3:6],axis=2),ylabel='wage-reduction tilassa, miehet',stack=False,emp=True)

    def plot_distrib(self,label='',plot_emp=False,plot_bu=False,ansiosid=False,tmtuki=False,putki=False,outsider=False,max_age=500,laaja=False,max=4,figname=None):
        unemp_distrib,emp_distrib,unemp_distrib_bu=self.episodestats.comp_empdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
        tyoll_distrib,tyoll_distrib_bu=self.episodestats.comp_tyollistymisdistribs(ansiosid=ansiosid,tmtuki=tmtuki,putki=putki,outsider=outsider,max_age=max_age,laaja=laaja)
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

    def plot_irr(self,figname='',grayscale=False):
        self.plot_aggirr()
        self.plot_aggirr(gender=1)
        self.plot_aggirr(gender=2)
        self.episodestats.comp_irr()
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_reduced,figname=figname+'_reduced',reduced=True,grayscale=grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_full,figname=figname+'_full',grayscale=grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_full,figname=figname+'_full_naiset',gender=1,grayscale=grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_full,figname=figname+'_full_miehet',gender=2,grayscale=grayscale)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_reduced,figname=figname+'_red_naiset',gender=1,grayscale=grayscale,reduced=True)
        self.plot_irrdistrib(self.episodestats.infostats_irr_tyel_reduced,figname=figname+'_red_miehet',gender=2,grayscale=grayscale,reduced=True)

    def plot_aggirr(self,gender=None):
        '''
        Laskee aggregoidun sisäisen tuottoasteen (IRR)
        Indeksointi puuttuu npv:n osalta
        Tuloksiin lisättävä inflaatio+palkkojen reaalikasvu = palkkojen nimellinen kasvu
        '''
        
        if gender is None:
            gendername='Kaikki'
        else:
            gendername=self.get_gendername(gender)

        agg_irr_tyel_full,agg_irr_tyel_reduced,agg_premium,agg_pensions_reduced,agg_pensions_full,maxnpv=self.episodestats.comp_aggirr(gender=gender,full=True)

        print('{}: aggregate irr tyel reduced {:.4f} % reaalisesti'.format(gendername,agg_irr_tyel_reduced))
        print('{}: aggregate irr tyel full {:.4f} % reaalisesti'.format(gendername,agg_irr_tyel_full))
        
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('age')
        ax.set_ylabel('pension/premium')
        ax.plot(x[:-1],agg_pensions_full[:-1],label='työeläkemeno')
        ax.plot(x[:-1],agg_pensions_reduced[:-1],label='yhteensovitettu työeläkemeno')
        ax.plot(x[:-1],agg_premium[:-1],label='työeläkemaksu')
        ax.legend()
        plt.show()
        
    def compare_irr(self,cc2=None,cc3=None,cc4=None,cc5=None,cc6=None,label1='',label2='',label3='',label4='',label5='',label6='',figname='',grayscale=False):
        self.episodestats.comp_irr()
        full1=self.episodestats.infostats_irr_tyel_full
        reduced1=self.episodestats.infostats_irr_tyel_reduced
        full2=None
        reduced2=None
        full3=None
        reduced3=None
        full4=None
        reduced4=None
        full5=None
        reduced5=None
        full6=None
        reduced6=None
        if cc2 is not None:
            cc2.episodestats.comp_irr()
            full2=cc2.episodestats.infostats_irr_tyel_full
            reduced2=cc2.episodestats.infostats_irr_tyel_reduced
        if cc3 is not None:
            cc3.episodestats.comp_irr()
            full3=cc3.episodestats.infostats_irr_tyel_full
            reduced3=cc3.episodestats.infostats_irr_tyel_reduced
        if cc4 is not None:
            cc4.episodestats.comp_irr()
            full4=cc4.episodestats.infostats_irr_tyel_full
            reduced4=cc4.episodestats.infostats_irr_tyel_reduced
        if cc5 is not None:
            cc5.episodestats.comp_irr()
            full5=cc5.episodestats.infostats_irr_tyel_full
            reduced5=cc5.episodestats.infostats_irr_tyel_reduced
        if cc6 is not None:
            cc6.episodestats.comp_irr()
            full6=cc6.episodestats.infostats_irr_tyel_full
            reduced6=cc6.episodestats.infostats_irr_tyel_reduced
        
        self.compare_irrdistrib(cc2=cc2,cc3=cc3,cc4=cc4,cc5=cc5,cc6=cc6,label1=label1,label2=label2,label3=label3,label4=label4,label5=label5,label6=label6,reduced=False,grayscale=grayscale)
        self.compare_irrdistrib(cc2=cc2,cc3=cc3,cc4=cc4,cc5=cc5,cc6=cc6,figname=figname+'_reduced',reduced=True,label1=label1,label2=label2,label3=label3,label4=label4,label5=label5,label6=label6,grayscale=grayscale)
        self.compare_irrdistrib(cc2=cc2,cc3=cc3,cc4=cc4,cc5=cc5,cc6=cc6,figname=figname+'_full_naiset',label1=label1,label2=label2,label3=label3,label4=label4,label5=label5,label6=label6,reduced=False,gender=1,grayscale=grayscale)
        self.compare_irrdistrib(cc2=cc2,cc3=cc3,cc4=cc4,cc5=cc5,cc6=cc6,figname=figname+'_full_miehet',label1=label1,label2=label2,label3=label3,label4=label4,label5=label5,label6=label6,reduced=False,gender=2,grayscale=grayscale)

    def filter_irrdistrib(self,cc,gender=None,reduced=False):
        '''
        Suodata irrit sukupuolen mukaan
        '''
        mortstate=self.env.get_mortstate()
        
        if reduced:
            irr_distrib=cc.episodestats.infostats_irr_tyel_reduced
        else:
            irr_distrib=cc.episodestats.infostats_irr_tyel_full
        
        gendername=self.get_gendername(gender)
        gendermask=self.episodestats.get_gendermask(gender)
        #if gender is None:
        #    gendermask=np.ones_like(irr_distrib)
        #else:
        #    if gender==1: # naiset
        #        gendermask = cc.episodestats.infostats_group>2
        #    else: # miehet
        #        gendermask = cc.episodestats.infostats_group<3
                
                
        nanmask = ma.mask_or(np.isnan(irr_distrib),gendermask.astype(bool))
        #nanmask = (np.isnan(irr_distrib).astype(int)+gendermask.astype(int)).astype(bool)
        v2_irrdata=ma.array(irr_distrib,mask=nanmask)
        v2data=v2_irrdata.compressed() # nans dropped
        
        return v2data
    
    def get_gendername(self,gender):
        if gender is None:
            gendername=''
        else:
            if gender==1: # naiset
                gendername=' (Naiset)'
            else: # miehet
                gendername=' (Miehet)'
                
        return gendername

    def plot_irrdistrib(self,irr_distrib,grayscale=True,figname='',reduced=False,gender=None):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        mortstate=self.env.get_mortstate()
        
        gendername=self.get_gendername(gender)
        gendermask=self.episodestats.get_gendermask(gender)
        
        nanmask = ma.mask_or(np.isnan(irr_distrib),gendermask.astype(bool))
        #nanmask = (np.isnan(irr_distrib).astype(int)+gendermask.astype(int)).astype(bool)
        v2_irrdata=ma.array(irr_distrib,mask=nanmask)
        v2data=v2_irrdata.compressed() # nans dropped
        
        ika=65
        alivemask = self.episodestats.popempstate[self.map_age(ika),:]!=mortstate
        alivemask = np.reshape(alivemask,(-1,1))
        alivemask = ma.mask_or(alivemask,gendermask.astype(bool))
        #alivemask = (alivemask.astype(int)*gendermask.astype(int)).astype(bool)
        
        #nan_alivemask=alivemask*nanmask[:,0]
        #nanalive_irrdata=ma.array(irr_distrib,mask=nan_alivemask)
        #nanalive_data=nanalive_irrdata.compressed() # nans dropped

        if reduced:
            print('\nTyel-irr huomioiden kansan- ja takuueläkkeen yhteensovitus'+gendername)
        else:
            print('\nTyel-irr ILMAN kansan- ja takuueläkkeen yhteensovitusta'+gendername)

        #fig,ax=plt.subplots()
        #ax.set_xlabel('Sisäinen tuottoaste [%]')
        #lbl=ax.set_ylabel('Taajuus')
        #x=np.linspace(-7,7,100)
        #scaled,x2=np.histogram(irr_distrib,x)
        #ax.bar(x2[1:-1],scaled[1:],align='center')
        #if figname is not None:
        #    plt.savefig(figname+'irrdistrib.'+self.figformat, bbox_inches='tight', format=self.figformat)
        #plt.show()

        self.plot_one_irrdistrib(v2data,label1='Sisäinen tuotto [%]')

        # v2
        print('Kaikki havainnot ilman NaNeja:\n- Keskimääräinen irr {:.3f} % reaalisesti'.format(np.mean(v2data)))
        print('- Mediaani irr {:.3f} % reaalisesti'.format(np.median(v2data)))
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

        count = (np.sum(self.episodestats.stat_pop_paidpension,axis=0)<0.1).sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus mikään eläke ei lainkaan maksussa {:.2f} %'.format(100*percent))
        
        no_pension = (np.sum(self.episodestats.infostats_pop_tyoelake,axis=0)<0.1)
        count = no_pension.sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläkettä ei lainkaan maksussa {:.2f} %'.format(100*percent))
        
        nopremium = (np.sum(self.episodestats.infostats_tyelpremium,axis=0)<0.1)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei lainkaan maksua maksettu {:.2f} %'.format(100*percent))
        
        count = (np.sum(self.episodestats.infostats_paid_tyel_pension,axis=0)<0.1).sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei vastinetta maksulle {:.2f} %'.format(100*percent))
        
        nopp = no_pension*nopremium
        count=nopp.sum(axis=0)
        percent = np.true_divide(count,irr_distrib.shape[0])
        print('Osuus työeläke ei lainkaan maksua eikä maksettua eläkettä {:.2f} %'.format(100*percent))

        print('\nOsuudet\n')
        arri=ma.sum(ma.array(self.episodestats.stat_pop_paidpension[self.map_age(ika),:],mask=alivemask))<0.1
        percent = np.true_divide(ma.sum(arri),self.episodestats.alive[self.map_age(ika),0])
        print('{}v osuus eläke ei maksussa, ei kuollut {:.2f} %'.format(ika,100*percent))

        alivemask = self.episodestats.popempstate[self.map_age(ika),:]!=mortstate
        count = (np.sum(self.episodestats.infostats_pop_tyoelake,axis=0)<0.1).sum(axis=0)
        percent = np.true_divide(count,self.episodestats.alive[self.map_age(ika),0])
        print('{}v osuus ei työeläkekarttumaa, ei kuollut {:.2f} %'.format(ika,100*percent))

        count = 1-self.episodestats.alive[self.map_age(ika),0]/self.episodestats.n_pop
        print('{}v osuus kuolleet {:.2f} % '.format(ika,100*count))

        count = 1-self.episodestats.alive[-1,0]/self.episodestats.n_pop
        print('Lopussa osuus kuolleet {:.2f} % '.format(100*count))
        
    def plot_one_irrdistrib(self,irr_distrib1,label1='1',
                                 irr2=None,label2='2',
                                 irr3=None,label3='2',
                                 irr4=None,label4='2',
                                 irr5=None,label5='2',
                                 irr6=None,label6='2',
                                 grayscale=False,figname=''):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
        
        df1=pd.DataFrame(irr_distrib1,columns=['Sisäinen tuottoaste [%]'])
        df1.loc[:,'label']=label1
        if irr2 is not None:
            df2=pd.DataFrame(irr2,columns=['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label']=label2
            df1=pd.concat([df1,df2],ignore_index=True, sort=False)
        if irr3 is not None:
            df2=pd.DataFrame(irr3,columns=['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label']=label3
            df1=pd.concat([df1,df2],ignore_index=True, sort=False)
        if irr4 is not None:
            df2=pd.DataFrame(irr4,columns=['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label']=label4
            df1=pd.concat([df1,df2],ignore_index=True, sort=False)
        if irr5 is not None:
            df2=pd.DataFrame(irr5,columns=['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label']=label5
            df1=pd.concat([df1,df2],ignore_index=True, sort=False)
        if irr6 is not None:
            df2=pd.DataFrame(irr6,columns=['Sisäinen tuottoaste [%]'])
            df2.loc[:,'label']=label6
            df1=pd.concat([df1,df2],ignore_index=True, sort=False)

        sns.displot(df1, x="Sisäinen tuottoaste [%]", hue="label", kind="kde", fill=True,gridsize=10000, bw_adjust=.05)
        plt.xlim(-10, 10)
        if figname is not None:
            plt.savefig(figname+'_kde.'+self.figformat, format=self.figformat)
        plt.show()
        
        sns.displot(df1, x="Sisäinen tuottoaste [%]", hue="label", stat="density", fill=True,common_norm=False)
        plt.xlim(-10, 10)
        if figname is not None:
            plt.savefig(figname+'density.'+self.figformat, format=self.figformat)
        plt.show()

    def compare_gender_irr(self,figname=None,reduced=False,grayscale=False):
        self.episodestats.comp_irr()
        irr1=self.filter_irrdistrib(self,reduced=reduced,gender=1)
        irr2=self.filter_irrdistrib(self,reduced=reduced,gender=2)

        display(irr1)
        display(irr2)

        self.plot_one_irrdistrib(irr1,label1='naiset',irr2=irr2,label2='miehet',
                                 grayscale=grayscale,figname=figname)

    def compare_reduced_irr(self,figname=None,gender=None,grayscale=False):
        self.episodestats.comp_irr()
        irr1=self.filter_irrdistrib(self,reduced=True,gender=gender)
        irr2=self.filter_irrdistrib(self,reduced=False,gender=gender)

        display(irr1)
        display(irr2)

        self.plot_one_irrdistrib(irr1,label1='Yhteensovitettu',irr2=irr2,label2='Pelkkä työeläke',
                                 grayscale=grayscale,figname=figname)
        
    def compare_irrdistrib(self,cc2=None,cc3=None,cc4=None,cc5=None,cc6=None,
            label1='',label2='',label3='',label4='',label5='',label6='',figname=None,reduced=False,gender=None,grayscale=False):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        gendername=self.get_gendername(gender)
        if reduced:
            print('\nTyel-irr huomioiden kansan- ja takuueläkkeen yhteensovitus'+gendername)
        else:
            print('\nTyel-irr ILMAN kansan- ja takuueläkkeen yhteensovitusta'+gendername)
            
        irr_distrib=self.filter_irrdistrib(self,reduced=reduced,gender=gender)
        aggirr,aggirr_reduced=self.episodestats.comp_aggirr()
        print(f'Aggregate irr for {label1}: {aggirr} (reduced {aggirr_reduced})')
        irr2=None
        irr3=None
        irr4=None
        irr5=None
        irr6=None
        if cc2 is not None:
            irr2=self.filter_irrdistrib(cc2,reduced=reduced,gender=gender)
            aggirr,aggirr_reduced=cc2.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label2}: {aggirr} (reduced aggirr_reduced)')
        if cc3 is not None:
            irr3=self.filter_irrdistrib(cc3,reduced=reduced,gender=gender)
            aggirr,aggirr_reduced=cc3.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label3}: {aggirr} (reduced aggirr_reduced)')
        if cc4 is not None:
            irr4=self.filter_irrdistrib(cc4,reduced=reduced,gender=gender)
            aggirr,aggirr_reduced=cc4.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label4}: {aggirr} (reduced aggirr_reduced)')
        if cc5 is not None:
            irr5=self.filter_irrdistrib(cc5,reduced=reduced,gender=gender)
            aggirr,aggirr_reduced=cc5.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label5}: {aggirr} (reduced aggirr_reduced)')
        if cc6 is not None:
            irr6=self.filter_irrdistrib(cc6,reduced=reduced,gender=gender)
            aggirr,aggirr_reduced=cc6.episodestats.comp_aggirr()
            print(f'Aggregate irr for {label6}: {aggirr} (reduced aggirr_reduced)')


        self.plot_one_irrdistrib(irr_distrib,label1=label1,
                                 irr2=irr2,label2=label2,
                                 irr3=irr3,label3=label3,
                                 irr4=irr4,label4=label4,
                                 irr5=irr5,label5=label5,
                                 irr6=irr6,label6=label6,
                                 grayscale=grayscale,figname=figname)
                                 
    def plot_scaled_irr(self,label='',figname=None,reduced=False,gender=None,grayscale=False):
        '''
        Laskee annetulla maksulla irr:t eri maksuille skaalaamalla
        '''
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        irr_distrib=self.filter_irrdistrib(self,gender=gender)
        aggirr,aggirr_reduced,agg_premium,agg_pensions_reduced,agg_pensions_full,maxnpv=self.episodestats.comp_aggirr(full=True)
        print(f'Aggregate irr for {label}: {aggirr} (reduced {aggirr_reduced})')
        
        x=np.arange(0.01,0.35,0.01)
        agg_irr_tyel_full=np.zeros_like(x)
        agg_irr_tyel_reduced=np.zeros_like(x)
        for m,r in enumerate(x):
            agg_premium2=agg_premium*r/0.244
            agg_irr_tyel_full[m]=self.reaalinen_palkkojenkasvu*100+self.episodestats.comp_annual_irr(maxnpv,agg_premium2,agg_pensions_full)
            agg_irr_tyel_reduced[m]=self.reaalinen_palkkojenkasvu*100+self.episodestats.comp_annual_irr(maxnpv,agg_premium2,agg_pensions_reduced)
            
        df1 = pd.DataFrame(agg_irr_tyel_full,columns=['pelkkä työeläke'])
        df1.loc[:,'x']=x
        df1.loc[:,'yhteensovitettu']=agg_irr_tyel_reduced
            
        ps=np.zeros_like(agg_irr_tyel_full)+1.6
        
        if reduced:
            self.lineplot(x*100,agg_irr_tyel_full,y2=agg_irr_tyel_reduced,y3=ps,xlim=[10,30],ylim=[-1,6],
                ylabel='sisäinen tuotto [%]',xlabel='maksutaso [% palkoista]',selite=True,
                label='Pelkkä työeläke',label2='Yhteensovitettu eläke',label3='Palkkasumman kasvu',figname=figname)
        else:
            self.lineplot(x*100,agg_irr_tyel_full,y2=ps,xlim=[10,30],ylim=[-1,6],
                ylabel='sisäinen tuotto [%]',xlabel='maksutaso [% palkoista]',selite=True,
                label='Pelkkä työeläke',label2='Palkkasumman kasvu',figname=figname)
        
        
    def lineplot(x,y,y2=None,y3=None,y4=None,y5=None,y6=None,
                 label=None,label2=None,label3=None,label4=None,label5=None,label6=None,
                 xlabel='',ylabel='',selite=False,source=None,xlim=None,ylim=None,figname=None):
        csfont,pal=setup_EK_fonts()

        linestyle={'linewidth': 3}
        legendstyle={'frameon': False}

        fig, axs = plt.subplots()
        axs.plot(x,y,label=label,**linestyle)
        if y2 is not None:
            axs.plot(x,y2,label=label2,**linestyle)
        if y3 is not None:
            axs.plot(x,y3,label=label3,**linestyle)
        if y4 is not None:
            axs.plot(x,y4,label=label4,**linestyle)
        if y5 is not None:
            axs.plot(x,y5,label=label5,**linestyle)
        if y6 is not None:
            axs.plot(x,y6,label=label6,**linestyle)

        if selite:
            axs.legend(loc='upper right',**legendstyle)

        axs.set_xlabel(xlabel,**csfont)
        axs.set_ylabel(ylabel,**csfont)
        axs.grid(True,color='black',fillstyle='top',lw=0.5,axis='y',alpha=1.0)
        if xlim is not None:
            axs.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            axs.set_ylim(ylim[0], ylim[1])
        if source is not None:
            add_source(source,**csfont)
        if figname is not None:
            plt.savefig(figname+'.png', format='png')

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
            wage=self.episodestats.infostats_pop_wage[t,k]
            old_wage=self.episodestats.infostats_pop_wage[max(0,t-1),k]
            age=self.map_t_to_age(t)
            old_age=int(self.map_t_to_age(max(0,t-1)))
            emp=self.episodestats.popempstate[t,k]
            predemp=self.episodestats.popempstate[max(0,t-1),k]
            pen=self.episodestats.infostats_pop_pension[t,k]
            predwage=self.env.wage_process_mean(old_wage,old_age,state=predemp)
            print(f'{age}: {obsV[t]:.4f} vs {self.aveV[t+1,k]:.4f} d {delta:.4f} re {rewerr:.6f} in state {emp} ({k},{wage:.2f},{pen:.2f}) ({predwage:.2f}) ({self.poprewstate[t+1,k]:.5f},{self.pop_predrew[t+1,k]:.5f})')


        err=obsV-self.aveV[t,k]
        obsV_error=np.abs(obsV-self.aveV[t,k])


    def plot_Vstats(self):
        obsV=np.zeros((self.n_time,self.episodestats.n_pop))
        obsVemp=np.zeros((self.n_time,3))
        obsVemp_error=np.zeros((self.n_time,3))
        obsVemp_abserror=np.zeros((self.n_time,3))

        obsV[self.n_time-1,:]=self.poprewstate[self.n_time-1,:]
        for t in range(self.n_time-2,0,-1):
            obsV[t,:]=self.poprewstate[t,:]+self.gamma*obsV[t+1,:]
            delta=obsV[t,:]-self.aveV[t,:]
            for k in range(self.episodestats.n_pop):
                if np.abs(delta[k])>0.2:
                    wage=self.episodestats.infostats_pop_wage[t,k]
                    pen=self.episodestats.infostats_pop_pension[t,k]

            for state in range(2):
                s=np.asarray(self.episodestats.popempstate[t,:]==state).nonzero()
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

    def compare_with(self,cc2,label1='perus',label2='vaihtoehto',grayscale=True,figname=None,dash=False):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        diff_emp=self.episodestats.empstate/self.episodestats.n_pop-cc2.episodestats.empstate/cc2.episodestats.n_pop
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        real1=self.episodestats.comp_presentvalue()
        real2=cc2.episodestats.comp_presentvalue()
        mean_real1=np.mean(real1,axis=1)
        mean_real2=np.mean(real2,axis=1)
        initial1=np.mean(real1[1,:])
        initial2=np.mean(real2[1,:])

        rew1=self.episodestats.comp_total_reward(output=False,discounted=True)
        rew2=cc2.episodestats.comp_total_reward(output=False,discounted=True)
        net1,eqnet1=self.episodestats.comp_total_netincome(output=False)
        net2,eqnet2=cc2.episodestats.comp_total_netincome(output=False)

        print(f'{label1} reward {rew1} netincome {net1:.2f} eq {eqnet1:.3f} initial {initial1}')
        print(f'{label2} reward {rew2} netincome {net2:.2f} eq {eqnet2:.3f} initial {initial2}')

        if self.minimal>0:
            s=20
            e=70
        else:
            s=21
            e=60 #63.5

        tyoll_osuus1,htv_osuus1,tyot_osuus1,kokotyo_osuus1,osatyo_osuus1=self.episodestats.comp_employed_ratio(self.episodestats.empstate)
        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.episodestats.comp_employed_ratio(cc2.episodestats.empstate)
        htv1,tyoll1,haj1,tyollaste1,tyolliset1,osatyolliset1,kokotyolliset1,osata1,kokota1=self.episodestats.comp_tyollisyys_stats(self.episodestats.empstate/self.episodestats.n_pop,scale_time=True,start=s,end=e,full=True)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2,osatyolliset2,kokotyolliset2,osata2,kokota2=self.episodestats.comp_tyollisyys_stats(cc2.episodestats.empstate/cc2.episodestats.n_pop,scale_time=True,start=s,end=e,full=True)
        ansiosid_osuus1,tm_osuus1=self.episodestats.comp_unemployed_detailed(self.episodestats.empstate)
        ansiosid_osuus2,tm_osuus2=self.episodestats.comp_unemployed_detailed(cc2.episodestats.empstate)
        #khh_osuus1=self.episodestats.comp_kht(self.episodestats.empstate)
        #khh_osuus2=self.episodestats.comp_kht(cc2.empstate)

        self.episodestats.comp_employment_stats()
        cc2.episodestats.comp_employment_stats()

        self.compare_against(cc=cc2,cctext=label2)

#         q1=self.episodestats.comp_budget(scale=True)
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
        c1=self.episodestats.comp_cumurewstate()
        c2=cc2.episodestats.comp_cumurewstate()
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
        if self.version in set([1,2,3,4,5,104]):
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
            unemp_htv1=np.nansum(self.episodestats.demogstates[:,0])
            unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,0])
            e_unemp_htv1=np.nansum(self.episodestats.demogstates[:,0])
            e_unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,0])
            tm_unemp_htv1=np.nansum(self.episodestats.demogstates[:,0])*0
            tm_unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,0])*0
            f_unemp_htv1=np.nansum(self.episodestats.demogstates[:,0])*0
            f_unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,0])*0
        else:
            unemp_htv1=np.nansum(self.episodestats.demogstates[:,0]+self.episodestats.demogstates[:,4]+self.episodestats.demogstates[:,13])
            unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,0]+cc2.episodestats.demogstates[:,4]+cc2.episodestats.demogstates[:,13])
            e_unemp_htv1=np.nansum(self.episodestats.demogstates[:,0])
            e_unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,0])
            tm_unemp_htv1=np.nansum(self.episodestats.demogstates[:,13])
            tm_unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,13])
            f_unemp_htv1=np.nansum(self.episodestats.demogstates[:,4])
            f_unemp_htv2=np.nansum(cc2.episodestats.demogstates[:,4])

        # epävarmuus
        delta=1.96*1.0/np.sqrt(self.episodestats.n_pop)

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
            unemp_distrib,emp_distrib,unemp_distrib_bu=self.episodestats.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            tyoll_distrib,tyoll_distrib_bu=self.episodestats.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.episodestats.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)
            tyoll_distrib2,tyoll_distrib_bu2=cc2.episodestats.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False)

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

            unemp_distrib,emp_distrib,unemp_distrib_bu=self.episodestats.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            tyoll_distrib,tyoll_distrib_bu=self.episodestats.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            unemp_distrib2,emp_distrib2,unemp_distrib_bu2=cc2.episodestats.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)
            tyoll_distrib2,tyoll_distrib_bu2=cc2.episodestats.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=54)

            self.plot_compare_empdistribs(emp_distrib,emp_distrib2,label1=label1,label2=label2)
            if self.language=='English':
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            else:
                print('Jakauma ansiosidonnainen+tmtuki+putki, max age 54')
            self.plot_compare_unempdistribs(unemp_distrib,unemp_distrib2,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=False,label1=label1,label2=label2)
            self.plot_compare_tyolldistribs(unemp_distrib,tyoll_distrib,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2)

        print(label2)
        keskikesto=self.episodestats.comp_unemp_durations(return_q=False)
        self.plot_unemp_durdistribs(keskikesto)

        print(label1)
        keskikesto=cc2.episodestats.comp_unemp_durations(return_q=False)
        self.plot_unemp_durdistribs(keskikesto)

        tyoll_virta,tyot_virta=self.episodestats.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.episodestats.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        self.plot_compare_virrat(tyoll_virta,tyoll_virta2,virta_label='Työllisyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='Työttömyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='Työttömyys',label1=label1,label2=label2)

        tyoll_virta,tyot_virta=self.episodestats.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.episodestats.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='ei-tm-Työttömyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='ei-tm-Työttömyys',label1=label1,label2=label2)

        tyoll_virta,tyot_virta=self.episodestats.comp_virrat(ansiosid=False,tmtuki=True,putki=True,outsider=False)
        tyoll_virta2,tyot_virta2=cc2.episodestats.comp_virrat(ansiosid=False,tmtuki=True,putki=True,outsider=False)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=40,max_time=64,virta_label='tm-Työttömyys',label1=label1,label2=label2)
        self.plot_compare_virrat(tyot_virta,tyot_virta2,min_time=55,max_time=64,virta_label='tm-Työttömyys',label1=label1,label2=label2)

    def plot_emtr(self,figname=None):
        axvcolor='gray'
        lstyle='--'
        maxt=self.map_age(64)
        emtr_tilat=set([0,1,4,7,8,9,10,13]) # include 14?
        
        etla_x_ptr,etla_ptr,etla_x_emtr,etla_emtr=self.empstats.get_emtr()

        print_html('<h2>EMTR</h2>')

        fig,ax=plt.subplots()
        ax.set_xlabel('EMTR')
        ax.set_ylabel('Density')
        #alivemask=(self.episodestats.popempstate==self.env.get_mortstate()) # pois kuolleet
        
        alivemask=np.zeros_like(self.episodestats.infostats_pop_emtr)
        for t in range(alivemask.shape[0]):
            for k in range(alivemask.shape[1]):
                if self.episodestats.popempstate[t,k] in emtr_tilat:
                    alivemask[t,k]=False
                else:
                    alivemask[t,k]=True

        #alivemask=(self.episodestats.popempstate not in emtr_tilat) # pois kuolleet
        emtr=ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask=alivemask[1:maxt,:]))
        nc=ma.masked_where(np.isnan(emtr), emtr).compressed()
        bins=np.arange(-0.5,100.5,2)
        bins2=np.arange(-0.5,100.5,1)
        ax.hist(nc,density=True,bins=bins)
        ka=np.nanmean(nc)
        med=ma.median(nc)
        ax.plot(etla_x_emtr,etla_emtr,'r')
        plt.axvline(x=ka,ls=lstyle,color=axvcolor)
        plt.title(f'mean {ka:.2f} median {med:.2f}')
        plt.xlim(0,100)
        if figname is not None:
            plt.savefig(figname+'emtr.pdf', format='pdf')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('EMTR')
        ax.set_ylabel('Density')
        #ax.hist(emtr,density=True,bins=100)
        ax.hist(nc,density=True,bins=bins2)
        ax.plot(etla_x_emtr,etla_emtr,'r')
        plt.xlim(0,100)
        plt.show()


        for k in emtr_tilat:
            fig,ax=plt.subplots()
            ax.set_xlabel('EMTR')
            ax.set_ylabel('Density')
            mask=(self.episodestats.popempstate!=k) 
            emtr=ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask=mask[1:maxt,:]))
            nc=ma.masked_where(np.isnan(emtr), emtr).compressed()
            if nc.shape[0]>0:
                ax.hist(nc,density=True,bins=bins2)
                ka=ma.mean(nc)
                med=ma.median(nc)
                plt.title(f'state {k}: mean {ka:.2f} median {med:.2f}')
            else:
                plt.title(f'state {k}')
            plt.xlim(0,100)
            plt.show()

        print_html('<h2>PTR</h2>')

        fig,ax=plt.subplots()
        ax.set_xlabel('TVA')
        ax.set_ylabel('Density')
        tva=ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask=alivemask[1:maxt,:]))
        nc=ma.masked_where(np.isnan(tva), tva).compressed()
        ax.hist(nc,density=True,bins=200)
        ax.plot(etla_x_ptr,etla_ptr,'r')
        plt.xlim(0,100)
        ka=np.mean(nc)
        med=np.median(nc)
        plt.title(f'mean {ka:.2f} median {med:.2f}')
        plt.axvline(x=ka,ls=lstyle,color=axvcolor)
        if figname is not None:
            plt.savefig(figname+'ptr.pdf', format='pdf')
        plt.show()
        prop=np.count_nonzero(nc>80)/max(1,nc.shape[0])*100
        print(f'kaikki: työttömyysloukussa (80 %) {prop}')        

        for k in emtr_tilat:
            fig,ax=plt.subplots()
            ax.set_xlabel('TVA')
            ax.set_ylabel('Density')
            mask=(self.episodestats.popempstate!=k) 
            tvax=ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask=mask[1:maxt,:]))
            nc=ma.masked_where(np.isnan(tvax), tvax).compressed()
            #if k==0:
            #    print(nc)
            if nc.shape[0]>0:
                ax.hist(nc,density=True,bins=200)
                ka=ma.mean(nc)
                med=ma.median(nc)
                plt.title(f'state {k}: mean {ka:.2f} median {med:.2f}')
            else:
                plt.title(f'state {k}')
            plt.xlim(10,100)
            plt.show()

        for k in emtr_tilat:
            mask=(self.episodestats.popempstate!=k) 
            tvax=ma.ravel(ma.array(self.episodestats.infostats_pop_tva[1:maxt,:],mask=mask[1:maxt,:]))
            nc=ma.masked_where(np.isnan(tvax), tvax).compressed()
            prop=np.count_nonzero(nc>80)/max(1,nc.shape[0])*100
            print(f'{k}: työttömyysloukussa (80 %) {prop}')

#         for k in emtr_tilat:
#             mask=(self.episodestats.popempstate!=k) 
#             tvax=ma.ravel(ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask=mask[1:maxt,:]))
#             nc=ma.masked_where(np.isnan(tvax), tvax).compressed()
#             mask2=nc>=1
#             prop=np.count_nonzero(nc<1)/max(1,nc.shape[0])*100
#             print(f'{k}: (1 %) {prop}')
# 
#         for k in emtr_tilat:
#             mask=ma.make_mask(self.episodestats.popempstate!=k) 
#             tvax=ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask=mask[1:maxt,:])
#             nc=ma.masked_where(np.isnan(tvax), tvax)
#             mask2=ma.mask_or(mask[1:maxt,:],nc>=10)
#             w=ma.array(self.episodestats.infostats_pop_wage[1:maxt,:],mask=mask2).compressed()
#             pw=ma.array(self.episodestats.infostats_pop_potential_wage[1:maxt,:],mask=mask2).compressed()
#             em=ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask=mask2).compressed()
#             netto=ma.array(self.episodestats.infostats_poptulot_netto[1:maxt,:],mask=mask2).compressed()
#             for s,v in enumerate(em):
#                 print(f'{k}:',w[s],pw[s],v,netto[s])

                    
    ## FROM simstats.py

    def test_emtr(self):
        maxt=self.map_age(64)
        emtr_tilat=set([0,1,4,7,8,9,10,13,14])
    
        for k in emtr_tilat:
            mask=(self.episodestats.popempstate!=k) 
            tvax=ma.array(self.episodestats.infostats_pop_emtr[1:maxt,:],mask=mask[1:maxt,:])
            mask2=tvax>=1
            w=ma.array(self.episodestats.infostats_pop_wage[1:maxt,:],mask=tvax)
            print('w',w)

    def plot_aggkannusteet(self,ben,loadfile,baseloadfile=None,figname=None,label=None,baselabel=None):
        '''
        FIXME
        '''
        f = h5py.File(loadfile, 'r')
        netto=f['netto'][()]
        eff=f['eff'][()]
        tva=f['tva'][()]
        osa_tva=f['osa_tva'][()]
        min_salary=f['min_salary'][()]
        max_salary=f['max_salary'][()]
        step_salary=f['step_salary'][()]
        n=f['n'][()]
        f.close()
        
        basic_marg=fin_benefits.Marginals(ben,year=self.year)

        if baseloadfile is not None:
            f = h5py.File(baseloadfile, 'r')
            basenetto=f['netto'][()]
            baseeff=f['eff'][()]
            basetva=f['tva'][()]
            baseosatva=f['osa_tva'][()]
            f.close()        
            
            basic_marg.plot_insentives(netto,eff,tva,osa_tva,min_salary=min_salary,max_salary=max_salary+step_salary,figname=figname,
                step_salary=step_salary,basenetto=basenetto,baseeff=baseeff,basetva=basetva,baseosatva=baseosatva,
                otsikko=label,otsikkobase=baselabel)
        else:
            basic_marg.plot_insentives(netto,eff,tva,osa_tva,min_salary=min_salary,max_salary=max_salary+step_salary,figname=figname,
                step_salary=step_salary,otsikko=label,otsikkobase=baselabel)
                    
               
    def compare_simstats(self,filename1,filename2,label1='perus',label2='vaihtoehto',figname=None,greyscale=True):
        m_best1,m_median1,s_emp1,median_htv1,u_tmtuki1,u_ansiosid1,h_median1,mn_median1=self.get_simstats(filename1)
        _,m_mean1,s_emp1,mean_htv1,u_tmtuki1,u_ansiosid1,h_mean1,mn_mean1=self.get_simstats(filename1,use_mean=True)
        
        m_best2,m_median2,s_emp2,median_htv2,u_tmtuki2,u_ansiosid2,h_median2,mn_median2=self.get_simstats(filename2)
        _,m_mean2,s_emp2,mean_htv2,u_tmtuki2,u_ansiosid2,h_mean2,mn_mean2=self.get_simstats(filename2,use_mean=True)

        if greyscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
        
        print('Vaikutus mediaanityöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(median_htv2-median_htv1,median_htv2,median_htv1))
        print('Vaikutus keskiarvotyöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(mean_htv2-mean_htv1,mean_htv2,mean_htv1))

        if False: # mediaani
            fig,ax=plt.subplots()
            ax.set_xlabel(self.labels['age'])
            ax.set_ylabel( self.labels['tyollisyysaste %'])
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x[1:],100*m_median1[1:],label=label1)
            ax.plot(x[1:],100*m_median2[1:],label=label2)
            #emp_statsratio=100*self.emp_stats()
            #ax.plot(x,emp_statsratio,label='havainto')
            ax.legend()
            plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel( self.labels['tyollisyysaste %'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*m_mean2[1:],label=label2)
        ax.plot(x[1:],100*m_mean1[1:],ls='--',label=label1)
        ax.set_ylim([0,100])  
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'emp.pdf', format='pdf')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ero osuuksissa'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],mn_median2[1:]-mn_median1[1:],label=label2+' miinus '+label1)
        ax.plot(x[1:],h_median2[1:]-h_median1[1:],label=label2+' miinus '+label1+' htv')
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Earning-related Unemployment rate [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #ax.plot(x[1:],100*u_tmtuki1[1:],ls='--',label='tm-tuki, '+label1)
        #ax.plot(x[1:],100*u_tmtuki2[1:],label='tm-tuki, '+label2)
        #ax.plot(x[1:],100*u_ansiosid1[1:],ls='--',label='ansiosidonnainen, '+label1)
        #ax.plot(x[1:],100*u_ansiosid2[1:],label='ansiosidonnainen, '+label2)
        ax.plot(x[1:],100*u_ansiosid2[1:],label=label2)
        ax.plot(x[1:],100*u_ansiosid1[1:],ls='--',label=label1)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'unemp.pdf', format='pdf')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomien osuus'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*u_tmtuki1[1:],ls='--',label='tm-tuki, '+label1)
        ax.plot(x[1:],100*u_tmtuki2[1:],label='tm-tuki, '+label2)
        ax.plot(x[1:],100*u_ansiosid1[1:],ls='--',label='ansiosidonnainen, '+label1)
        ax.plot(x[1:],100*u_ansiosid2[1:],label='ansiosidonnainen, '+label2)
        ax.plot(x[1:],100*(u_tmtuki1[1:]+u_ansiosid1[1:]),ls='--',label='yhteensä, '+label1)
        ax.plot(x[1:],100*(u_tmtuki2[1:]+u_ansiosid2[1:]),label='yhteensä, '+label2)
        #ax.plot(x[1:],100*u_ansiosid2[1:],label=label2)
        #ax.plot(x[1:],100*u_ansiosid1[1:],ls='--',label=label1)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyydet.eps', format='eps')
            plt.savefig(figname+'tyottomyydet.png', format='png',dpi=300)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomien osuus'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*(u_tmtuki1[1:]+u_ansiosid1[1:]),ls='--',label=label1)
        ax.plot(x[1:],100*(u_tmtuki2[1:]+u_ansiosid2[1:]),label=label2)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyydet2.eps', format='eps')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomien osuus'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*(u_tmtuki1[1:]+u_ansiosid1[1:]),ls='--',label=label1)
        ax.plot(x[1:],100*(u_tmtuki2[1:]+u_ansiosid2[1:]),label=label2)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyydet3.eps', format='eps')
            plt.savefig(figname+'tyottomyydet3.png', format='png',dpi=300)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['ero osuuksissa'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*(m_median2[1:]-m_median1[1:]),label=label1)
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        #ax.legend()
        plt.show()

        demog2=self.empstats.get_demog()
        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('cumsum työllisyys [lkm]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        cs=np.cumsum(h_mean2[1:]-h_mean1[1:])
        c2=np.cumsum(h_mean1[1:])
        c1=np.cumsum(h_mean2[1:])
        ax.plot(x[1:],cs,label=label1)
        #emp_statsratio=100*self.emp_stats()
        #ax.plot(x,emp_statsratio,label='havainto')
        #ax.legend()
        plt.show()

        for age in set([50,63,63.25,63.5]):
            mx=self.map_age(age)-1
            print('Kumulatiivinen työllisyysvaikutus {:.2f} vuotiaana {:.1f} htv ({:.0f} vs {:.0f})'.format(age,cs[mx],c1[mx],c2[mx]))
            
        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta1,tyot_virta1,tyot_virta_ansiosid1,tyot_virta_tm1,kestot1,viimkesto1=self.episodestats.load_simdistribs(filename1)
        unemp_distrib2,emp_distrib2,unemp_distrib_bu2,tyoll_distrib2,tyoll_distrib_bu2,\
            tyoll_virta2,tyot_virta2,tyot_virta_ansiosid2,tyot_virta_tm2,kestot2,viimkesto2=self.episodestats.load_simdistribs(filename2)
        
        self.plot_compare_unemp_durdistribs(kestot1,kestot2,viimkesto1,viimkesto2,label1='',label2='')
        
        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label='vaihtoehto')
        self.plot_compare_unempdistribs(unemp_distrib1,unemp_distrib2,label1=label1,label2=label2,figname=figname)
        self.plot_compare_tyolldistribs(unemp_distrib1,tyoll_distrib1,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2,figname=figname)
        self.plot_compare_virtadistribs(tyoll_virta1,tyoll_virta2,tyot_virta1,tyot_virta2,tyot_virta_ansiosid1,tyot_virta_ansiosid2,tyot_virta_tm1,tyot_virta_tm2,label1=label1,label2=label2)

    def compare_epistats(self,filename1,cc2,label1='perus',label2='vaihtoehto',figname=None,greyscale=True):
        m_best1,m_median1,s_emp1,median_htv1,u_tmtuki1,u_ansiosid1,h_median1,mn_median1=self.get_simstats(filename1)
        _,m_mean1,s_emp1,mean_htv1,u_tmtuki1,u_ansiosid1,h_mean1,mn_mean1=self.get_simstats(filename1,use_mean=True)

        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.episodestats.comp_employed_ratio(cc2.empstate)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2,osatyolliset2,kokotyolliset2,osata2,kokota2=self.episodestats.comp_tyollisyys_stats(cc2.empstate/cc2.n_pop,scale_time=True,start=s,end=e,full=True)
        ansiosid_osuus2,tm_osuus2=self.episodestats.comp_employed_detailed(cc2.empstate)
        
        m_best2=tyoll_osuus2
        m_median2=tyoll_osuus2
        s_emp2=s_emp1*0
        median_htv2=htv_osuus2
        #u_tmtuki2,
        #u_ansiosid2,
        #h_median2,
        mn_median2=tyoll_osuus2
        m_mean2=tyoll_osuus2
        s_emp2=0*s_emp1
        mean_htv2=htv_osuus2
        #u_tmtuki2,
        #u_ansiosid2,
        #h_mean2,
        #mn_mean2

        if greyscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
        
        print('Vaikutus mediaanityöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(median_htv2-median_htv1,median_htv2,median_htv1))
        print('Vaikutus keskiarvotyöllisyyteen {:.0f} htv ({:.0f} vs {:.0f})'.format(mean_htv2-mean_htv1,mean_htv2,mean_htv1))

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Employment rate [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*m_mean2[1:],label=label2)
        ax.plot(x[1:],100*m_mean1[1:],ls='--',label=label1)
        ax.set_ylim([0,100])  
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyys.eps')        
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Työllisyysero [hlö/htv]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],mn_median2[1:]-mn_median1[1:],label=label2+' miinus '+label1)
        ax.plot(x[1:],h_median2[1:]-h_median1[1:],label=label2+' miinus '+label1+' htv')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomien osuus'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*u_tmtuki1[1:],ls='--',label='tm-tuki, '+label1)
        ax.plot(x[1:],100*u_tmtuki2[1:],label='tm-tuki, '+label2)
        ax.plot(x[1:],100*u_ansiosid1[1:],ls='--',label='ansiosidonnainen, '+label1)
        ax.plot(x[1:],100*u_ansiosid2[1:],label='ansiosidonnainen, '+label2)
        ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyottomyydet.eps')        
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(self.labels['tyottomien osuus'])
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x[1:],100*(m_median2[1:]-m_median1[1:]),label=label1)
        plt.show()

    def plot_compare_csvirta(self,m1,m2,lbl):
        nc1=np.reshape(np.cumsum(m1),m1.shape)
        nc2=np.reshape(np.cumsum(m2),m1.shape)
        fig,ax=plt.subplots()
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        plt.plot(x,nc1)
        plt.plot(x,nc2)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel(lbl)
        plt.show()
        fig,ax=plt.subplots()
        plt.plot(x,nc1-nc2)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('diff '+lbl)
        plt.show()

    def plot_compare_virtadistribs(self,tyoll_virta1,tyoll_virta2,tyot_virta1,tyot_virta2,tyot_virta_ansiosid1,tyot_virta_ansiosid2,tyot_virta_tm1,tyot_virta_tm2,label1='',label2=''):
        m1=np.mean(tyoll_virta1,axis=0,keepdims=True).transpose()
        m2=np.mean(tyoll_virta2,axis=0,keepdims=True).transpose()
        fig,ax=plt.subplots()
        plt.plot(m1,label=label1)
        plt.plot(m2,label=label2)
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Keskimääräinen työllisyysvirta')
        plt.show()
        self.plot_compare_virrat(m1,m2,virta_label='työllisyys',label1=label1,label2=label2,ymin=0,ymax=5000)
        self.plot_compare_csvirta(m1,m2,'cumsum työllisyysvirta')

        m1=np.mean(tyot_virta1,axis=0,keepdims=True).transpose()
        m2=np.mean(tyot_virta2,axis=0,keepdims=True).transpose()
        self.plot_compare_virrat(m1,m2,virta_label='työttömyys',label1=label1,label2=label2)
        self.plot_compare_csvirta(m1,m2,'cumsum työttömyysvirta')

        m1=np.mean(tyot_virta_ansiosid1,axis=0,keepdims=True).transpose()
        m2=np.mean(tyot_virta_ansiosid2,axis=0,keepdims=True).transpose()
        self.plot_compare_virrat(m1,m2,virta_label='ei-tm-työttömyys',label1=label1,label2=label2)
        m1=np.mean(tyot_virta_tm1,axis=0,keepdims=True).transpose()
        m2=np.mean(tyot_virta_tm2,axis=0,keepdims=True).transpose()
        self.plot_compare_virrat(m1,m2,virta_label='tm-työttömyys',label1=label1,label2=label2)
        n1=(np.mean(tyoll_virta1,axis=0,keepdims=True)-np.mean(tyot_virta1,axis=0,keepdims=True)).transpose()
        n2=(np.mean(tyoll_virta2,axis=0,keepdims=True)-np.mean(tyot_virta2,axis=0,keepdims=True)).transpose()
        self.plot_compare_virrat(n1,n2,virta_label='netto',label1=label1,label2=label2,ymin=-1000,ymax=1000)
        self.plot_compare_csvirta(n1,n2,'cumsum nettovirta')

    def plot_unemp_durdistribs(self,kestot):
        if len(kestot.shape)>2:
            m1=self.episodestats.empdur_to_dict(np.mean(kestot,axis=0))
        else:
            m1=self.episodestats.empdur_to_dict(kestot)

        df = pd.DataFrame.from_dict(m1,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

    def plot_compare_unemp_durdistribs(self,kestot1,kestot2,viimekesto1,viimekesto2,label1='',label2=''):
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        self.plot_unemp_durdistribs(kestot1)
        self.plot_unemp_durdistribs(kestot2)

        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        self.plot_unemp_durdistribs(viimekesto1)
        self.plot_unemp_durdistribs(viimekesto2)
                            
    def fit_norm(self,diff):
        diff_stdval=np.std(diff)
        diff_meanval=np.mean(diff)
        diff_minval=np.min(diff)
        diff_maxval=np.max(diff)
        sz=(diff_maxval-diff_minval)/10
        x=np.linspace(diff_minval,diff_maxval,1000)
        y=norm.pdf(x,diff_meanval,diff_stdval)*diff.shape[0]*sz
    
        return x,y
               
    def plot_simstats(self,filename,grayscale=False,figname=None):
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
            best_emp,emps,agg_netincome,agg_equivalent_netincome=self.episodestats.load_simstats(filename)

        if self.version>0:
            print('lisäpäivillä on {:.0f} henkilöä'.format(self.episodestats.count_putki_dist(emps)))

        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        std_tyoll=np.std(agg_tyoll)
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-median_tyoll
        mean_rew=np.mean(agg_rew)
        mean_discounted_rew=np.mean(agg_discounted_rew)
        mean_netincome=np.mean(agg_netincome)
        mean_equi_netincome=np.mean(agg_equivalent_netincome)

        print(f'Mean undiscounted reward {mean_rew}')
        print(f'Mean discounted reward {mean_discounted_rew}')
        print(f'Mean net income {mean_netincome} mean equivalent net income {mean_equi_netincome}')
        fig,ax=plt.subplots()
        ax.set_xlabel('Discounted rewards')
        ax.set_ylabel('Lukumäärä')
        ax.hist(agg_discounted_rew,color='lightgray')
        plt.show()
        
        x,y=self.fit_norm(diff_htv)
        
        m_mean=np.mean(emp_tyolliset_osuus,axis=0)
        m_median=np.median(emp_tyolliset_osuus,axis=0)
        s_emp=np.std(emp_tyolliset_osuus,axis=0)
        m_best=emp_tyolliset_osuus[best_emp,:]
        um_mean=np.mean(emp_tyottomat_osuus,axis=0)
        um_median=np.median(emp_tyottomat_osuus,axis=0)

        if self.minimal:
            print('Työllisyyden keskiarvo {:.0f} htv mediaani {:.0f} htv std {:.0f} htv'.format(mean_htv,median_htv,std_htv))
        else:
            print('Työllisyyden keskiarvo keskiarvo {:.0f} htv, mediaani {:.0f} htv std {:.0f} htv\n'
                  'keskiarvo {:.0f} työllistä, mediaani {:.0f} työllistä, std {:.0f} työllistä'.format(
                    mean_htv,median_htv,std_htv,mean_tyoll,median_tyoll,std_tyoll))

        fig,ax=plt.subplots()
        ax.set_xlabel('Poikkeama työllisyydessä [htv]')
        ax.set_ylabel('Lukumäärä')
        ax.hist(diff_htv,color='lightgray')
        ax.plot(x,y,color='black')
        if figname is not None:
            plt.savefig(figname+'poikkeama.eps')
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Työllisyysaste [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*np.transpose(emp_tyolliset_osuus),linewidth=0.4)
        emp_statsratio=100*m_mean #100*self.emp_stats()
        ax.plot(x,emp_statsratio,label='keskiarvo')
        #ax.legend()
        if figname is not None:
            plt.savefig(figname+'tyollisyyshajonta.eps')
        plt.show()

        if self.version>0:
            x,y=self.fit_norm(diff_tyoll)
            fig,ax=plt.subplots()
            ax.set_xlabel('Poikkeama työllisyydessä [henkilöä]')
            ax.set_ylabel('Lukumäärä')
            ax.hist(diff_tyoll,color='lightgray')
            ax.plot(x,y,color='black')
            plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel('Palkkio')
        ax.set_ylabel('Lukumäärä')
        ax.hist(agg_rew)
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Työllisyysaste [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*m_mean,label='keskiarvo')
        ax.plot(x,100*m_median,label='mediaani')
        #ax.plot(x,100*(m_emp+s_emp),label='ka+std')
        #ax.plot(x,100*(m_emp-s_emp),label='ka-std')
        ax.plot(x,100*m_best,label='paras')
        emp_statsratio=100*self.empstats.emp_stats()
        ax.plot(x,emp_statsratio,label='havainto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Työttömyysaste [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*um_mean,label='keskiarvo')
        ax.plot(x,100*um_median,label='mediaani')
        #unemp_statsratio=100*self.unemp_stats()
        #ax.plot(x,unemp_statsratio,label='havainto')
        ax.legend()
        plt.show()

        fig,ax=plt.subplots()
        ax.set_xlabel(self.labels['age'])
        ax.set_ylabel('Hajonta työllisyysasteessa [%]')
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        ax.plot(x,100*s_emp)
        plt.show()
        
        unemp_distrib1,emp_distrib1,unemp_distrib_bu1,\
            tyoll_distrib1,tyoll_distrib_bu1,\
            tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
            unemp_dur,unemp_lastdur=self.episodestats.load_simdistribs(filename)
       
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        self.plot_unemp_durdistribs(unemp_dur)
        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        self.plot_unemp_durdistribs(unemp_lastdur)

        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label='vaihtoehto')
        self.plot_unempdistribs(unemp_distrib1,figname=figname,max=10,miny=1e-5,maxy=2)
        #self.plot_tyolldistribs(unemp_distrib1,tyoll_distrib1,tyollistyneet=True,figname=figname)
        self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max=4,figname=figname)
                            