'''

    simhelper.py

    implements methods that help in interpreting multiple simulation results with a variety of parameters

'''

import h5py
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm
import locale
from tabulate import tabulate
import pandas as pd
#import smoothfit
from tqdm import tqdm_notebook as tqdm
from . lifecycle_v2 import Lifecycle
from . episodestats import Labels,modify_offsettext

class SimHelper():
    def __init__(self):
        epi=Labels()
        self.labels=epi.get_labels()

    def plot_stats(self,datafile,baseline=None,ref=None,xlabel=None,dire=None,
                    plot_kiila=True,percent_scale_x=False,grayscale=False,label1='ref',
                    label2='baseline',grouped=False):
                    
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
            pal=sns.dark_palette("darkgray", 6, reverse=True)
            reverse=True
        else:
            pal=sns.color_palette()            
            
        
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,kiila,muut,tyollaste,tyotaste,tyossa,\
            osatyoratio,kiila_kaikki,total_ps_norw,total_tyossa_norw,total_htv_norw,\
            osuus_vero,osuus_kunnallisvero,osuus_valtionvero,total_etuusmeno,\
            total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno\
            =self.load_data(datafile)
        rew,verot,ps,htv,muut,kiila,tyossa,osatyoratio,tyotaste,\
            ps_norw,htv_norw,tyossa_norw,etuusmeno,\
            tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno=self.comp_means_norw(datafile)

        self.plot_tyossa(additional_income_tax,tyossa,htv,xlabel=xlabel,percent_scale_x=True,dire=dire)
        self.plot_tyossa_vertaa(additional_income_tax,tyossa,htv,tyossa_norw,htv_norw,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire)
        self.plot_tyoton(additional_income_tax,tyotaste,xlabel=xlabel,percent_scale_x=True,percent_scale_y=True,dire=dire)
        self.plot_osatyossa(additional_income_tax,osatyoratio,xlabel=xlabel,percent_scale_x=True,dire=dire)

        if baseline is not None:
            baseline_tax,baseline_verot,baseline_rew,baseline_ps,baseline_htv,baseline_kiila,baseline_muut,_,_,_,\
                baseline_osatyoratio,baseline_kiila_kaikki,baseline_total_ps_norw,baseline_total_tyossa_norw,baseline_total_htv_norw,\
                baseline_osuus_vero,baseline_osuus_kunnallisvero,baseline_osuus_valtionvero,baseline_total_etuusmeno,\
                baseline_total_tyottomyysmeno,baseline_total_asumistukimeno,baseline_total_elakemeno,baseline_total_toimeentulotukimeno,baseline_total_muutmeno\
                =self.load_data(baseline)
            brew,bverot,bps,bhtv,bmuut,bkiila,btyossa,bosatyoratio,btyotaste,\
                bps_norw,bhtv_norw,btyossa_norw,betuus,\
                btyottomyysmeno,basumistukimeno,belakemeno,btoimeentulotukimeno,bmuutmeno=self.comp_means_norw(baseline)
                
            self.plot_etuusmeno(additional_income_tax,etuusmeno,ref_additional_tax=baseline_tax,ref_etuusmeno=betuus,xlabel=xlabel,label1=label1,label2=label2,dire=dire,percent_scale_x=percent_scale_x)
            self.plot_etuusmeno(additional_income_tax,total_etuusmeno,ref_additional_tax=baseline_tax,ref_etuusmeno=betuus,xlabel=xlabel,label1=label1,label2=label2,dire=None,percent_scale_x=percent_scale_x)
            self.plot_detetuusmeno(additional_income_tax,tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno,xlabel=xlabel,label1=label1,label2=label2,dire=None,percent_scale_x=percent_scale_x)
            self.plot_tyossa(additional_income_tax,tyossa,htv,ref_additional_tax=baseline_tax,ref_htv=bhtv,label1=label1,label2=label2,
                    ref_tyossa=btyossa,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire)
            if plot_kiila:
                self.plot_verokiila(additional_income_tax,kiila,ref_additional_tax=baseline_tax,ref_kiila=baseline_kiila)

            self.plot_taxes(additional_income_tax,verot,muut,muut,
                       ref_additional_tax=baseline_tax,ref_verot=bverot,ref_muut=bmuut,ref_total_muut=baseline_muut,
                       xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x,label1=label1,label2=label2)
            self.plot_reward(additional_income_tax,rew,ref_additional_tax=baseline_tax,ref_rew=brew,
                       xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x,label1=label1,label2=label2)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,ref_tax=baseline_tax,ref_ps=bps,
                        percent_scale_x=percent_scale_x,dire=dire,label1=label1,label2=label2,)
            self.plot_veroosuudet(additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero,label1=label1,label2=label2)
            self.plot_veroosuudet_abs(additional_income_tax,osuus_vero,verot,
                ref_tax=baseline_tax,bosuus_vero=baseline_osuus_vero,btotal_vero=baseline_verot,label1=label1,label2=label2)
        elif ref is not None:
            ref_rew,ref_verot,ref_ps,ref_htv,ref_muut,ref_kiila=self.get_refstats(ref,scale=additional_income_tax)
            self.plot_etuusmeno(additional_income_tax,etuusmeno,xlabel=xlabel,dire=dire)
            self.plot_etuusmeno(additional_income_tax,total_etuusmeno,xlabel=xlabel,dire=dire)
            if plot_kiila:
                self.plot_verokiila(additional_income_tax,kiila,ref_additional_tax=additional_income_tax,ref_kiila=ref_kiila)
            self.plot_taxes(additional_income_tax,verot,muut,muut,
                       ref_additional_tax=additional_income_tax,ref_verot=ref_verot,ref_muut=ref_muut,ref_rew=ref_rew,xlabel=xlabel,dire=dire)
            self.plot_reward(additional_income_tax,rew,ref_additional_tax=additional_income_tax,ref_rew=ref_rew,
                       xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x,label1=label1,label2=label2)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,percent_scale_x=percent_scale_x,dire=dire)
            self.plot_veroosuudet(additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
            self.plot_veroosuudet_abs(additional_income_tax,osuus_vero,verot)
        else:
            self.plot_etuusmeno(additional_income_tax,etuusmeno,xlabel=xlabel,dire=dire)
            self.plot_etuusmeno(additional_income_tax,total_etuusmeno,xlabel=xlabel,dire=dire)
            self.plot_detetuusmeno(additional_income_tax,tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno,xlabel=xlabel,label1=label1,label2=label2,dire=None,percent_scale_x=percent_scale_x)
            self.plot_verokiila(additional_income_tax,kiila)
            self.plot_taxes(additional_income_tax,verot,muut,muut,xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x)
            self.plot_reward(additional_income_tax,rew,xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,percent_scale_x=percent_scale_x,dire=dire)
            #self.plot_taxes_prop(additional_income_tax,TELosuus_vero,xlabel='Verokiila')
            self.plot_veroosuudet(additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
            self.plot_veroosuudet_abs(additional_income_tax,osuus_vero,verot)
            
        #self.plot_elasticity(kiila,ps,xlabel='Verokiila',ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_elasticity(verot,ps,xlabel='verot',ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_elasticity(verot,tyossa,xlabel='verot',ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x)
        
        #self.plot_elasticity(kiila,htv,xlabel='Verokiila',ylabel='Työnmäärän jousto',dire=dire,percent_scale_x=percent_scale_x)
        #self.plot_elasticity(kiila,tyossa,xlabel='Verokiila',ylabel='Osallistumisjousto',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_elasticity(additional_income_tax,ps,xlabel=xlabel,ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        self.plot_elasticity(additional_income_tax,htv,xlabel=xlabel,ylabel='Työnmäärän jousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        self.plot_elasticity(additional_income_tax,tyossa,xlabel=xlabel,ylabel='Osallistumisjousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        self.plot_osatyo(additional_income_tax,osatyoratio,xlabel=xlabel,ylabel='Osuus [%-yks]',dire=dire,percent_scale_x=percent_scale_x)

        self.plot_elasticity2d(additional_income_tax,htv,tyossa,xlabel=xlabel,ylabel='Jousto',label1='Työnmäärän jousto',label2='Osallistumisjousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        
        self.plot_total(additional_income_tax,total_rew,total_verot,total_htv) #,percent_scale_x=percent_scale_x)
        
    def add_gender(self,x,mean=True):
        y=np.zeros((x.shape[0],x.shape[1],2))
        y[:,:,0]=np.sum(x[:,:,0:2],axis=2)
        y[:,:,1]=np.sum(x[:,:,3:6],axis=2)
            
        if mean:
            y=np.mean(y,axis=1)

        return y
        
    def plot_stats_groups(self,datafile,baseline=None,ref=None,xlabel=None,dire=None,gender=False,
                    plot_kiila=True,percent_scale_x=False,grayscale=False,label1='ref',label2='baseline'):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
            pal=sns.dark_palette("darkgray", 6, reverse=True)
            reverse=True
        else:
            pal=sns.color_palette()            
            
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,kiila,muut,tyollaste,tyotaste,tyossa,\
            osatyoratio,kiila_kaikki,total_ps_norw,total_tyossa_norw,total_htv_norw,\
            osuus_vero,osuus_kunnallisvero,osuus_valtionvero,total_etuusmeno,\
            total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno\
            =self.load_data(datafile)
        
        if gender:
            rew,verot,ps,htv,muut,kiila,tyossa,osatyoratio,tyotaste,\
                ps_norw,htv_norw,tyossa_norw,etuusmeno,tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno\
                =self.comp_means_norw(datafile,gender=True)
        else:
            rew,verot,ps,htv,muut,kiila,tyossa,osatyoratio,tyotaste,\
                ps_norw,htv_norw,tyossa_norw,etuusmeno,tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno\
                =self.comp_means_norw(datafile,grouped=True)
            total_etuusmeno=np.mean(total_etuusmeno,axis=2)
        #kiila=np.mean(kiila,axis=2)

        #self.plot_tyossa(additional_income_tax,np.sum(tyossa,axis=1),np.sum(htv,axis=1),xlabel=xlabel,percent_scale_x=percent_scale_x,dire=None)
        self.plot_tyossa(additional_income_tax,tyossa,htv,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire,grouped=True,gender=gender)
        self.plot_tyossa_vertaa(additional_income_tax,tyossa,htv,tyossa_norw,htv_norw,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire,grouped=True,gender=gender)
        self.plot_tyoton(additional_income_tax,tyotaste,xlabel=xlabel,percent_scale_x=percent_scale_x,percent_scale_y=True,dire=dire,gender=gender,grouped=True)
        self.plot_osatyossa(additional_income_tax,osatyoratio,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire,gender=gender,grouped=True)

        if baseline is not None:
            baseline_tax,baseline_verot,baseline_rew,baseline_ps,baseline_htv,baseline_kiila,baseline_muut,_,_,_,\
                baseline_osatyoratio,baseline_kiila_kaikki,baseline_total_ps_norw,baseline_total_tyossa_norw,baseline_total_htv_norw,\
                baseline_osuus_vero,baseline_osuus_kunnallisvero,baseline_osuus_valtionvero,baseline_total_etuusmeno,\
                baseline_total_tyottomyysmeno,baseline_total_asumistukimeno,baseline_total_elakemeno,baseline_total_toimeentulotukimeno,baseline_total_muutmeno\
                =self.load_data(baseline)
            brew,bverot,bps,bhtv,bmuut,bkiila,btyossa,bosatyoratio,btyotaste,\
                bps_norw,bhtv_norw,btyossa_norw,betuus,btyottomyysmeno,basumistukimeno,belakemeno,btoimeentulotukimeno,bmuutmeno\
                =self.comp_means_norw(baseline,grouped=True)
                
            print(bmuut)
            
            # ei ryhmittäin    
            self.plot_etuusmeno(additional_income_tax,etuusmeno,ref_additional_tax=baseline_tax,ref_etuusmeno=betuus,xlabel=xlabel,label1=label1,label2=label2,dire=dire,percent_scale_x=percent_scale_x)
            # ei ryhmittäin    
            self.plot_etuusmeno(additional_income_tax,total_etuusmeno,ref_additional_tax=baseline_tax,ref_etuusmeno=betuus,xlabel=xlabel,label1=label1,label2=label2,dire=None,percent_scale_x=percent_scale_x)
            self.plot_tyossa(additional_income_tax,tyossa,htv,ref_additional_tax=baseline_tax,ref_htv=bhtv,label1=label1,label2=label2,
                    ref_tyossa=btyossa,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire)
            if plot_kiila:
                self.plot_verokiila(additional_income_tax,kiila,ref_additional_tax=baseline_tax,ref_kiila=baseline_kiila)

            # ei ryhmittäin    
            self.plot_taxes(additional_income_tax,verot,muut,muut,
                       ref_additional_tax=baseline_tax,ref_verot=bverot,ref_ps=bps,
                       ref_muut=bmuut,ref_rew=brew,ref_total_muut=baseline_muut,
                       xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x,label1=label1,label2=label2)
            # ei ryhmittäin    
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,ref_tax=baseline_tax,ref_ps=bps,
                        percent_scale_x=percent_scale_x,dire=dire,label1=label1,label2=label2,)
        elif ref is not None:
            ref_rew,ref_verot,ref_ps,ref_htv,ref_muut,ref_kiila=self.get_refstats(ref,scale=additional_income_tax)
            self.plot_etuusmeno(additional_income_tax,etuusmeno,xlabel=xlabel,dire=dire)
            self.plot_etuusmeno(additional_income_tax,total_etuusmeno,xlabel=xlabel,dire=dire)
            if plot_kiila:
                self.plot_verokiila(additional_income_tax,kiila,ref_additional_tax=additional_income_tax,ref_kiila=ref_kiila)
            self.plot_taxes(additional_income_tax,verot,muut,muut,
                       ref_additional_tax=additional_income_tax,ref_verot=ref_verot,
                       ref_muut=ref_muut,xlabel=xlabel,dire=dire)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,percent_scale_x=percent_scale_x,dire=dire)
            #self.plot_veroosuudet(additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
            #self.plot_veroosuudet_abs(additional_income_tax,osuus_vero,verot)
        else:
            self.plot_etuusmeno(additional_income_tax,etuusmeno,xlabel=xlabel,dire=dire)
            #self.plot_etuusmeno(additional_income_tax,total_etuusmeno,xlabel=xlabel,dire=dire)
            #self.plot_verokiila(additional_income_tax,kiila)
            self.plot_taxes(additional_income_tax,verot,muut,muut,xlabel=xlabel,dire=dire,percent_scale_x=percent_scale_x)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,percent_scale_x=percent_scale_x,dire=dire)
            #self.plot_taxes_prop(additional_income_tax,TELosuus_vero,xlabel='Verokiila')
            #self.plot_veroosuudet(additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
            #self.plot_veroosuudet_abs(additional_income_tax,osuus_vero,verot)
            
        #self.plot_elasticity(verot,ps,xlabel='verot',ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x)
        #self.plot_elasticity(verot,tyossa,xlabel='verot',ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x)
        #self.plot_elasticity(additional_income_tax,ps,xlabel=xlabel,ylabel='Palkkasumman jousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        #self.plot_elasticity(additional_income_tax,htv,xlabel=xlabel,ylabel='Työnmäärän jousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        #self.plot_elasticity(additional_income_tax,tyossa,xlabel=xlabel,ylabel='Osallistumisjousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        self.plot_osatyo(additional_income_tax,osatyoratio,xlabel=xlabel,ylabel='Osuus [%-yks]',dire=dire,percent_scale_x=percent_scale_x)

        #self.plot_elasticity2d(additional_income_tax,htv,tyossa,xlabel=xlabel,ylabel='Jousto',label1='Työnmäärän jousto',label2='Osallistumisjousto',dire=dire,percent_scale_x=percent_scale_x,diff=False)
        
        #self.plot_total(additional_income_tax,total_rew,total_verot,total_htv) #,percent_scale_x=percent_scale_x)
        
    def plot_stats_agegroups(self,datafile,baseline=None,ref=None,xlabel=None,dire=None,
                    plot_kiila=True,percent_scale_x=False,grayscale=False,label1='ref',label2='baseline'):
        if grayscale:
            plt.style.use('grayscale')
            plt.rcParams['figure.facecolor'] = 'white' # Or any suitable colour...
            pal=sns.dark_palette("darkgray", 6, reverse=True)
            reverse=True
        else:
            pal=sns.color_palette()            
            
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,kiila,muut,tyollaste,tyotaste,tyossa,\
            osatyoratio,kiila_kaikki,total_ps_norw,total_tyossa_norw,total_htv_norw,\
            osuus_vero,osuus_kunnallisvero,osuus_valtionvero,total_etuusmeno,\
            total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno\
            =self.load_data(datafile)
            
        rew,verot,ps,htv,muut,kiila,tyossa,osatyoratio,tyotaste,\
            ps_norw,htv_norw,tyossa_norw,etuusmeno,tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno\
            =self.comp_means_norw(datafile,grouped=True)

        self.plot_tyossa(additional_income_tax,tyossa,htv,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire,grouped=True,agegroups=True)
        self.plot_tyossa_vertaa(additional_income_tax,tyossa,htv,tyossa_norw,htv_norw,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=None,grouped=True,agegroups=True)
        self.plot_tyoton(additional_income_tax,tyotaste,xlabel=xlabel,percent_scale_x=True,percent_scale_y=True,dire=dire,grouped=True,agegroups=True)
        #self.plot_osatyossa(additional_income_tax,osatyoratio,xlabel=xlabel,percent_scale_x=True,dire=dire,grouped=True,agegroups=True)

        if baseline is not None:
            baseline_tax,baseline_verot,baseline_rew,baseline_ps,baseline_htv,baseline_kiila,baseline_muut,_,_,_,\
                baseline_osatyoratio,baseline_kiila_kaikki,baseline_total_ps_norw,baseline_total_tyossa_norw,baseline_total_htv_norw,\
                baseline_osuus_vero,baseline_osuus_kunnallisvero,baseline_osuus_valtionvero,baseline_total_etuusmeno,\
                baseline_total_tyottomyysmeno,baseline_total_asumistukimeno,baseline_total_elakemeno,baseline_total_toimeentulotukimeno,baseline_total_muutmeno\
                =self.load_data(baseline)
            brew,bverot,bps,bhtv,bmuut,bkiila,btyossa,bosatyoratio,btyotaste,\
                bps_norw,bhtv_norw,btyossa_norw,betuus,btyottomyysmeno,basumistukimeno,belakemeno,btoimeentulotukimeno,bmuutmeno\
                =self.comp_means_norw(baseline,grouped=True)
                
            print(bmuut)
            
            # ei ryhmittäin    
            self.plot_tyossa(additional_income_tax,tyossa,htv,ref_additional_tax=baseline_tax,ref_htv=bhtv,label1=label1,label2=label2,
                    ref_tyossa=btyossa,xlabel=xlabel,percent_scale_x=percent_scale_x,dire=dire)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,ref_tax=baseline_tax,ref_ps=bps,
                        percent_scale_x=percent_scale_x,dire=dire,label1=label1,label2=label2,)
        elif ref is not None:
            ref_rew,ref_verot,ref_ps,ref_htv,ref_muut,ref_kiila=self.get_refstats(ref,scale=additional_income_tax)
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,percent_scale_x=percent_scale_x,dire=dire,grouped=True,agegroups=True)
        else:
            self.plot_ps(additional_income_tax,total_ps,total_ps_norw,ps,ps_norw,percent_scale_x=percent_scale_x,dire=dire,grouped=True,agegroups=True)
            
        self.plot_osatyo(additional_income_tax,osatyoratio,xlabel=xlabel,ylabel='Osa-aikatyössä [%-yks]',dire=dire,
            percent_scale_x=percent_scale_x,percent_scale_y=True,grouped=True,agegroups=True)
        
    def plot_veroosuudet(self,additional_income_tax,osuus_vero,osuus_kunnallisvero,osuus_valtionvero,dire=None,label1=None,label2=None,
                        xlabel='Muutos [%-yks]',percent_scale_x=False,ylabel='Eläkeläisten osuus veroista',fname='elas'):
        vero=np.squeeze(osuus_vero[:,:,2])/np.squeeze(np.sum(osuus_vero,axis=2))
        kunnallis=np.squeeze(osuus_kunnallisvero[:,:,2])/np.squeeze(np.sum(osuus_kunnallisvero,axis=2))
        valtio=np.squeeze(osuus_valtionvero[:,:,2])/np.squeeze(np.sum(osuus_valtionvero,axis=2))
        self.plot_osuus(additional_income_tax,vero,label1=label1,xlabel=xlabel,ylabel='Eläkeläisten osuus veroista',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_osuus(additional_income_tax,kunnallis,label1=label1,xlabel=xlabel,ylabel='Eläkeläisten osuus kunnallisverosta',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_osuus(additional_income_tax,valtio,label1=label1,xlabel=xlabel,ylabel='Eläkeläisten osuus ansiotuloverosta',dire=dire,percent_scale_x=percent_scale_x)
        vero=np.squeeze(osuus_vero[:,:,1])/np.squeeze(np.sum(osuus_vero,axis=2))
        kunnallis=np.squeeze(osuus_kunnallisvero[:,:,1])/np.squeeze(np.sum(osuus_kunnallisvero,axis=2))
        valtio=np.squeeze(osuus_valtionvero[:,:,1])/np.squeeze(np.sum(osuus_valtionvero,axis=2))
        self.plot_osuus(additional_income_tax,vero,label1=label1,xlabel=xlabel,ylabel='Työn osuus veroista',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_osuus(additional_income_tax,kunnallis,label1=label1,xlabel=xlabel,ylabel='Työn osuus kunnallisverosta',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_osuus(additional_income_tax,valtio,label1=label1,xlabel=xlabel,ylabel='Työn osuus ansiotuloverosta',dire=dire,percent_scale_x=percent_scale_x)
        vero=np.squeeze(osuus_vero[:,:,0])/np.squeeze(np.sum(osuus_vero,axis=2))
        kunnallis=np.squeeze(osuus_kunnallisvero[:,:,0])/np.squeeze(np.sum(osuus_kunnallisvero,axis=2))
        valtio=np.squeeze(osuus_valtionvero[:,:,0])/np.squeeze(np.sum(osuus_valtionvero,axis=2))
        self.plot_osuus(additional_income_tax,vero,label1=label1,xlabel=xlabel,ylabel='Etuudensaajien osuus veroista',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_osuus(additional_income_tax,kunnallis,label1=label1,xlabel=xlabel,ylabel='Etuudensaajien osuus kunnallisverosta',dire=dire,percent_scale_x=percent_scale_x)
        self.plot_osuus(additional_income_tax,valtio,label1=label1,xlabel=xlabel,ylabel='Etuudensaajien osuus ansiotuloverosta',dire=dire,percent_scale_x=percent_scale_x)

    def plot_veroosuudet_abs(self,additional_income_tax,osuus_vero,total_vero,ref_tax=None,bosuus_vero=None,btotal_vero=None,
                        dire=None,xlabel='Muutos [%-yks]',percent_scale_x=False,ylabel='Eläkeläisten osuus veroista',fname='elas',
                        label1=None,label2=None):
        el_vero=osuus_vero[:,:,2]/np.sum(osuus_vero,axis=2)*total_vero
        tyo_vero=osuus_vero[:,:,1]/np.sum(osuus_vero,axis=2)*total_vero
        etuus_vero=osuus_vero[:,:,0]/np.sum(osuus_vero,axis=2)*total_vero
        self.plot_osuus(additional_income_tax,el_vero,y2=tyo_vero,y3=etuus_vero,label1='Eläkeläiset',label2='Työssä',label3='Etuudensaajat',
            xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
            
        if btotal_vero is not None:
            bel_vero=bosuus_vero[:,:,2]/np.sum(bosuus_vero,axis=2)*btotal_vero
            btyo_vero=bosuus_vero[:,:,1]/np.sum(bosuus_vero,axis=2)*btotal_vero
            betuus_vero=bosuus_vero[:,:,0]/np.sum(bosuus_vero,axis=2)*btotal_vero
            self.plot_osuus(additional_income_tax,el_vero,x2=ref_tax,y2=bel_vero,label1='Eläkeläiset',label2='baseline',
                xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
            self.plot_osuus(additional_income_tax,tyo_vero,x2=ref_tax,y2=btyo_vero,label1='Työssä',label2='baseline',
                xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
            self.plot_osuus(additional_income_tax,etuus_vero,x2=ref_tax,y2=betuus_vero,label1='Etuudensaajat',label2='baseline',
                xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
        else:
            self.plot_osuus(additional_income_tax,el_vero,label1='Eläkeläiset',
                xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
            self.plot_osuus(additional_income_tax,tyo_vero,label1='Työssä',
                xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
            self.plot_osuus(additional_income_tax,etuus_vero,label1='Etuudensaajat',
                xlabel=xlabel,ylabel='Verot (euroa)',dire=dire,percent_scale_x=percent_scale_x,legend=True)
    
    def plot_jees(self,ax,x,y,label,gender,grouped=False,agegroups=False,linestyle='solid',color=None):
        if label is None:
            label=''
        if grouped:
            if gender:
                ax.plot(x,y[:,0],label=label+'naiset')
                ax.plot(x,y[:,1],label=label+'miehet')
            elif agegroups:
                ax.plot(x,y[:,0],label=label+'19-35')
                ax.plot(x,y[:,1],label=label+'35-49')
                ax.plot(x,y[:,2],label=label+'50-65')
            else:
                for g in range(6):
                    ax.plot(x,y[:,g],label=label+str(g),linestyle=linestyle,color=color)
        else:
            ax.plot(x,y,label=label,linestyle=linestyle,color=color)
    
    def plot_osuus(self,x,y,x2=None,y2=None,x3=None,y3=None,x4=None,y4=None,x5=None,y5=None,
                   dire=None,label1=None,label2=None,label3=None,label4=None,label5=None,
                   ls1=None,ls2=None,ls3=None,ls4=None,
                   lc1=None,lc2=None,lc3=None,lc4=None,
                   xlabel='Muutos [%-yks]',modify_offset=False,modify_offset_text='',show_offset=True,
                   percent_scale_x=False,ylabel='',fname='elas',source=None,header=None,
                   percent_scale_y=False,legend=False,grouped=False,gender=False,agegroups=False):
        if percent_scale_x:
            scale=100
        else:
            scale=1
        if percent_scale_y:
            scaley=100
        else:
            scaley=1
            
        if header is not None:
            axs.title.set_text(header)
    
        fig,ax=plt.subplots()
        self.plot_jees(ax,scale*x,scaley*y,label=label1,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls1,color=lc1)
            
        if y2 is not None:
            if x2 is not None:
                self.plot_jees(ax,scale*x2,scaley*y2,label=label2,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls2,color=lc2)
            else:
                self.plot_jees(ax,scale*x,scaley*y2,label=label2,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls2,color=lc2)
        if y3 is not None:
            if x3 is not None:
                self.plot_jees(ax,scale*x3,scaley*y3,label=label3,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls3,color=lc3)
            else:
                self.plot_jees(ax,scale*x,scaley*y3,label=label3,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls3,color=lc3)
        if y4 is not None:
            if x4 is not None:
                self.plot_jees(ax,scale*x4,scaley*y4,label=label4,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls4,color=lc4)
            else:
                self.plot_jees(ax,scale*x,scaley*y4,label=label4,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls4,color=lc4)
        if y5 is not None:
            if x5 is not None:
                self.plot_jees(ax,scale*x3,scaley*y3,label=label5,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls5,color=lc5)
            else:
                self.plot_jees(ax,scale*x,scaley*y3,label=label5,gender=gender,grouped=grouped,agegroups=agegroups,linestyle=ls5,color=lc5)
        #plt.title(fname)
        #if ref_additional_tax is not None:
        #    ax.plot(scale*ref_additional_tax,ref_rew,label=label2)
        if legend:
            ax.legend()
            
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        #if modify_offset and show_offset:
        #    modify_offsettext(ax,modify_offset_text)
            
        #if not show_offset:
        #    offset = ax.yaxis.get_offset_text()
        #    offset.set_visible(False)            
            
        if dire is not None:
            plt.savefig(dire+fname+'.eps', format='eps')
            #plt.savefig(dire+fname+'_'+ylabel+'.png', format='png')
            #print('dire',dire)
            
        if source is not None:
            plt.annotate(source, (0,0), (80,-20), fontsize=8, 
                xycoords='axes fraction', textcoords='offset points', va='top')
        
        plt.show()

    def plot_elasticity(self,additional_income_tax,htv,dire=None,label1=None,label2=None,
                        xlabel='Muutos [%-yks]',percent_scale_x=False,ylabel='Elasticity',diff=True):
        el,elx=self.comp_elasticity(additional_income_tax,htv,diff=diff)
        #print(additional_income_tax,htv)
        #print(elx,el)
        self.plot_osuus(elx,el,label1=label1,xlabel=xlabel,ylabel=ylabel,dire=dire,percent_scale_x=percent_scale_x)

    def plot_tyoton(self,additional_income_tax,osuus,dire=None,label1=None,label2=None,xlabel='Tulovero [%-yks]',
                    percent_scale_x=True,percent_scale_y=True,ylabel='Työttömien osuus väestöstä [%-yks]',grouped=False,gender=False,agegroups=False):
        extraname=self.get_extraname(gender=gender,grouped=grouped,agegroups=agegroups)
        self.plot_osuus(additional_income_tax,osuus,label1=label1,xlabel=xlabel,ylabel=ylabel,dire=dire,percent_scale_x=percent_scale_x,
                        percent_scale_y=percent_scale_y,gender=gender,grouped=grouped,agegroups=agegroups,legend=False,fname=extraname+'tyoton')

    def plot_elasticity2d(self,additional_income_tax,htv,tyossa,dire=None,label1=None,label2=None,
                        xlabel='Muutos [%-yks]',percent_scale_x=False,ylabel='Elasticity',diff=True):
        el,elx=self.comp_elasticity(additional_income_tax,htv,diff=diff)
        print(el,el)
        el2,el2x=self.comp_elasticity(additional_income_tax,tyossa,diff=diff)
        for ind,e in enumerate(el):
            print(f'{elx[ind]}: {el[ind]} {el2[ind]}')
        self.plot_osuus(elx,el,y2=el2,label1=label1,label2=label2,xlabel=xlabel,ylabel=ylabel,dire=dire,percent_scale_x=percent_scale_x,legend=True)

    def plot_detetuusmeno(self,additional_tax,tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno,
        ref_additional_tax=None,ref_etuusmeno=None,dire=None,label1='',label2='',xlabel='Muutos [%-yks]',percent_scale_x=True):
        
        self.plot_etuusmeno(additional_tax,tyottomyysmeno,dire=dire,filename='tyottomyysmeno',ylabel='Työttömyysmeno',xlabel=xlabel,percent_scale_x=percent_scale_x)
        self.plot_etuusmeno(additional_tax,asumistukimeno,dire=dire,filename='asumistukimeno',ylabel='asumistukimeno',xlabel=xlabel,percent_scale_x=percent_scale_x)
        self.plot_etuusmeno(additional_tax,elakemeno,dire=dire,filename='elakemeno',ylabel='elakemeno',xlabel=xlabel,percent_scale_x=percent_scale_x)
        self.plot_etuusmeno(additional_tax,toimeentulotukimeno,dire=dire,filename='toimeentulotukimeno',ylabel='toimeentulotukimeno',xlabel=xlabel,percent_scale_x=percent_scale_x)
        self.plot_etuusmeno(additional_tax,muutmeno,dire=dire,filename='muutmeno',ylabel='muutmeno',xlabel=xlabel,percent_scale_x=percent_scale_x)

    def plot_etuusmeno(self,additional_tax,etuusmeno,ref_additional_tax=None,ref_etuusmeno=None,dire=None,label1='',label2='',
        xlabel='Muutos [%-yks]',ylabel='Etuusmeno',percent_scale_x=True,filename='etuusmeno.eps'):    

        fig,ax=plt.subplots()
        if percent_scale_x:
            scale=100
        else:
            scale=1
        ax.plot(scale*additional_tax,etuusmeno,label=label1)
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if ref_etuusmeno is not None:
            ax.plot(scale*ref_additional_tax,ref_etuusmeno,label=label1)
        if dire is not None:
            plt.savefig(dire+filename, format='eps')
        plt.show()
        

    def plot_reward(self,additional_tax,rew,
                   ref_additional_tax=None,ref_rew=None,percent_scale_x=False,legend=True,
                   dire=None,label1='',label2='baseline',xlabel='Muutos [%-yks]',nulline=None):    
        if percent_scale_x:
            scale=100
        else:
            scale=1
        fig,ax=plt.subplots()
        plt.title('Reward')
        if ref_additional_tax is not None:
            ax.plot(scale*additional_tax,rew,label=label1)
            ax.plot(scale*ref_additional_tax,ref_rew,'--',label=label2)
        else:
            ax.plot(scale*additional_tax,rew,label=label1)
        if legend:
            ax.legend()
        ax.set_ylabel('Hyöty')
        ax.set_xlabel(xlabel)
        if dire is not None:
            plt.savefig(dire+'reward.eps', format='eps')
        plt.show()        

    def plot_taxes(self,additional_tax,verot,muut,total_muut,
                   ref_additional_tax=None,ref_verot=None,ref_ps=None,ref_muut=None,
                   ref_total_muut=None,percent_scale_x=False,legend=False,
                   dire=None,label1='',label2='baseline',xlabel='Muutos [%-yks]',nulline=None):    
        if percent_scale_x:
            scale=100
        else:
            scale=1

        yscale=1e9

        fig,ax=plt.subplots()
        if ref_verot is not None:
            ax.plot(scale*additional_tax,verot/yscale,label=label1)
            ax.plot(scale*ref_additional_tax,ref_verot/yscale,'--',label=label2)
            legend=True
        else:
            ax.plot(scale*additional_tax,verot/yscale,label=label1)
            legend=False
        #plt.title('Verot ja veronluonteiset maksut')
        if legend:
            ax.legend()
            
        ax.set_ylabel(self.labels['Verokertymä [miljardia euroa]'])
        ax.set_xlabel(xlabel)
        
        if dire is not None:
            plt.savefig(dire+'verot.eps', format='eps')
            plt.savefig(dire+'verot.png', format='png',dpi=300)
            
        plt.show()
# 
#         fig,ax=plt.subplots()
#         if ref_additional_tax is not None:
#             ax.plot(scale*additional_tax,htv,label=label1)
#             ax.plot(scale*ref_additional_tax,ref_htv,'--',label=label2)
#         else:
#             ax.plot(scale*additional_tax,htv,label=label1)
#         #plt.title('Työnteko')
#         if legend:
#             ax.legend()
#         ax.set_ylabel(self.labels['Henkilöitä'])
#         ax.set_xlabel(xlabel)
#         if dire is not None:
#             plt.savefig(dire+'htv.eps', format='eps')
#         plt.show()

        fig,ax=plt.subplots()
        if ref_muut is not None:
            ax.plot(scale*additional_tax,muut/yscale,label=label1)
            ax.plot(scale*ref_additional_tax,ref_muut/yscale,'--',label=label2)
            legend=True
        else:
            ax.plot(scale*additional_tax,muut/yscale,label=label1)
            legend=False
        if legend:
            ax.legend()
        ax.set_ylabel(self.labels['Muut tarvittavat tulot [miljardia euroa]'])
        ax.set_xlabel(xlabel)
        #print(muut)
        #nulline=9.49009466e+09
        #if nulline is not None:
        #    ref_nulline=ref_muut*0+nulline
        #    ax.plot(scale*ref_additional_tax,ref_nulline,label='budjettitasapaino')
        if dire is not None:
            plt.savefig(dire+'muut.eps', format='eps')
            plt.savefig(dire+'muut.png', format='png',dpi=300)
        print(muut)
        plt.show()

        fig,ax=plt.subplots()
        if ref_muut is not None:
            ax.plot(scale*additional_tax,total_muut/yscale,label=label1)
            ax.plot(scale*ref_additional_tax,ref_muut/yscale,'--',label=label2)
            legend=True
        else:
            ax.plot(scale*additional_tax,total_muut/yscale,label=label1)
            legend=False
        if legend:
            ax.legend()
        ax.set_ylabel(self.labels['Muut tarvittavat tulot [miljardia euroa]'])
        ax.set_xlabel(xlabel)
        #if dire is not None:
        #    plt.savefig(dire+'muut.eps', format='eps')
        plt.show()

    def plot_osatyo(self,additional_income_tax,osuus,dire=None,label1=None,label2=None,xlabel='Tulovero [%-yks]',
            percent_scale_x=True,ylabel='',gender=False,grouped=False,agegroups=False,percent_scale_y=True):
        extraname=self.get_extraname(gender=gender,grouped=grouped,agegroups=agegroups)
            
        self.plot_osuus(additional_income_tax,osuus,label1=label1,xlabel=xlabel,ylabel=self.labels['Osatyönteko [%-yks]'],dire=dire,
            percent_scale_x=percent_scale_x,fname=extraname+'osatyo',gender=gender,grouped=grouped,agegroups=agegroups,legend=False,
            percent_scale_y=percent_scale_y)

    def get_extraname(self,gender=False,agegroups=False,grouped=False):
        if gender:
            return 'gender_'
        elif agegroups:
            return 'agegroups_'
        elif grouped:
            return 'grouped_'
        else:
            return ''

    def plot_osatyossa(self,additional_tax,tyossa,ref_htv=None,ref_additional_tax=None,
                    label1='',xlabel='',dire=None,percent_scale_x=True,gender=False,grouped=False,agegroups=False):
        
        extraname=self.get_extraname(gender=gender,grouped=grouped,agegroups=agegroups)
        if ref_htv is not None:
            self.plot_osuus(additional_tax,tyossa,
                            x2=ref_additional_tax,y2=ref_htv,
                            label1=label1,xlabel=xlabel,ylabel=self.labels['Osatyönteko [%-yks]'],dire=dire,percent_scale_x=percent_scale_x,
                            percent_scale_y=True,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'osatyossa')
        else:
            self.plot_osuus(additional_tax,tyossa,label1=label1,xlabel=xlabel,ylabel=self.labels['Osatyönteko [%-yks]'],dire=dire,percent_scale_x=percent_scale_x,
                            percent_scale_y=True,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'osatyossa')

    def plot_tyossa(self,additional_tax,tyossa,htv,ref_htv=None,ref_additional_tax=None,ref_tyossa=None,
                    label1='',label2='htv',xlabel='',dire=None,percent_scale_x=True,grouped=False,gender=False,agegroups=False,numerot=False):
        extraname=self.get_extraname(gender=gender,grouped=grouped,agegroups=agegroups)
        if ref_htv is not None:
            self.plot_osuus(additional_tax,tyossa/1e6,
                            x2=ref_additional_tax,y2=ref_tyossa/1e6,modify_offset=True,modify_offset_text='miljoonaa',show_offset=False,
                            label1=label1,xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa hlö]'],dire=dire,percent_scale_x=True,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'tyo')
            self.plot_osuus(additional_tax,htv/1e6,
                            x2=ref_additional_tax,y2=ref_htv/1e6,modify_offset=True,modify_offset_text='miljoonaa',show_offset=False,
                            label1=label1,xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa htv]'],dire=dire,percent_scale_x=True,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'htv')
        else:
            self.plot_osuus(additional_tax,tyossa/1e6,label1=label1,xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa hlö]'],dire=dire,percent_scale_x=True,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=False,fname=extraname+'tyo',
                            modify_offset=True,modify_offset_text='miljoonaa',show_offset=False)
            self.plot_osuus(additional_tax,htv/1e6,label1=label1,xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa htv]'],dire=dire,percent_scale_x=True,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'htv',
                            modify_offset=True,modify_offset_text='miljoonaa',show_offset=False)
            self.plot_osuus(additional_tax,tyossa/1e6,y2=htv/1e6,label1=label1+' hlö',label2=label1+' htv',xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa hlö/htv]'],dire=dire,percent_scale_x=percent_scale_x,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'tyohtv',
                            modify_offset=True,modify_offset_text='miljoonaa',show_offset=False)
                    

    def plot_tyossa_vertaa(self,additional_tax,tyossa,htv,tyossa_norw,htv_norw,
                    ref_htv=None,ref_additional_tax=None,
                    label1='Henkilöitä',label2='Htv',xlabel='',dire=None,percent_scale_x=True,
                    grouped=False,gender=False,agegroups=False):
        extraname=self.get_extraname(gender=gender,grouped=grouped,agegroups=agegroups)
        if ref_htv is not None:
            self.plot_osuus(additional_tax,tyossa/1e6,label1='henkilöitä',lc1='black',ls2='dashed',
                            x2=additional_tax,y2=tyossa_norw/1e6,label2='henkilöitä, ei ve',lc2='black',
                            x3=additional_tax,y3=htv/1e6,label3='htv',lc3='gray',
                            x4=additional_tax,y4=htv_norw/1e6,label4='htv, ei ve',lc4='gray',ls4='dashed',
                            xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa hlö]'],dire=dire,percent_scale_x=percent_scale_x,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'tyo_vertaa')
        else:
            self.plot_osuus(additional_tax,tyossa/1e6,label1='henkilöitä',lc1='black',
                            x2=additional_tax,y2=tyossa_norw/1e6,label2='henkilöitä, ei ve',lc2='black',ls2='dashed',
                            x3=additional_tax,y3=htv/1e6,label3='htv',lc3='gray',
                            x4=additional_tax,y4=htv_norw/1e6,label4='htv, ei ve',lc4='gray',ls4='dashed',
                            xlabel=xlabel,ylabel=self.labels['Työnteko [miljoonaa hlö]'],dire=dire,percent_scale_x=percent_scale_x,
                            percent_scale_y=False,gender=gender,grouped=grouped,agegroups=agegroups,legend=True,fname=extraname+'tyo_vertaa')
                    

    def plot_ps(self,additional_tax,total_ps,total_ps_norw,ps,ps_norw,ref_tax=None,
                ref_ps=None,dire=None,xlabel=None,percent_scale_x=False,label1='',label2='',
                grouped=False,gender=False,agegroups=False):
        if percent_scale_x:
            scale=100
        else:
            scale=1
            
        if gender:
            n_groups=2
        else:
            n_groups=6

        #plt.plot(scale*additional_tax,total_ps,'--',label='palkkasumma')
        #plt.plot(scale*additional_tax,total_ps_norw,label='palkkasumma, ei ve')
        #plt.title('palkkasumma')
        #plt.show()
        
        fig,ax=plt.subplots()
        ax.plot(scale*additional_tax,ps,label=label1+' kaikki')
        ax.plot(scale*additional_tax,ps_norw,'--',label=label1+' ilman ve+työ')
        if ref_ps is not None:
            ax.plot(scale*ref_tax,ref_ps,label=label2)
        plt.title('Palkkasumma')
        ax.set_ylabel(self.labels['palkkasumma'])
        ax.set_xlabel(xlabel)
        ax.legend()
        if dire is not None:
            plt.savefig(dire+'palkkasumma.eps', format='eps')
            #plt.savefig(dire+'palkkasumma.png', format='png')
        plt.show()

    def plot_total(self,additional_tax,total_rew,total_verot,total_htv,dire=None,xlabel=None,percent_scale_x=False):
        plt.plot(100*additional_tax,total_rew)
        plt.title('reward')
        plt.legend()
        plt.show()

        plt.plot(100*additional_tax,total_verot)
        plt.title('verot')
        plt.legend()
        if dire is not None:
            plt.savefig(dire+'verot_kaikki.png', format='png')
        plt.show()

        plt.plot(100*additional_tax,total_htv)
        plt.legend()
        plt.title('htv')
        plt.show()

    def plot_taxes_prop(self,additional_tax,osuus_vero,dire=None,xlabel=None):
        plt.plot(100*additional_tax,osuus_vero[:,:,0])
        plt.title('proportion of taxes paid by the not employed')
        plt.show()

        plt.plot(100*additional_tax,osuus_vero[:,:,1])
        plt.title('proportion of taxes paid by the employed')
        plt.show()

        plt.plot(100*additional_tax,osuus_vero[:,:,2])
        plt.title('proportion of taxes paid by the retired')
        plt.show()

    def plot_verokiila(self,additional_tax,kiila,xlabel=None,dire=None,ref_additional_tax=None,ref_kiila=None,label1='',label2='baseline'):
        fig,ax=plt.subplots()
        ax.plot(100*additional_tax,100*kiila)
        if ref_additional_tax is not None:
            ax.plot(100*ref_additional_tax,ref_kiila,label=label2)
        ax.set_ylabel(self.labels['Verokiila %'])
        ax.set_xlabel('Lisävero [%-yks]')
        if dire is not None:
            plt.savefig(dire+'verot_kaikki.png', format='png')
        plt.show()


    def save_data(self,filename,additional_income_tax,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut,total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,
                  total_kiila_kaikki_ansiot,total_ps_norw,total_tyossa_norw,total_htv_norw,
                  total_etuusmeno,total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno,
                  osuus_vero=None,osuus_kunnallisvero=None,osuus_valtionvero=None):
        f = h5py.File(filename, 'w')
        ftype='float64'

        _ = f.create_dataset('additional_income_tax', data=additional_income_tax, dtype=ftype)

        _ = f.create_dataset('total_verot', data=total_verot, dtype=ftype)
        _ = f.create_dataset('total_rew', data=total_rew, dtype=ftype)
        _ = f.create_dataset('total_ps', data=total_ps, dtype=ftype)
        _ = f.create_dataset('total_htv', data=total_htv, dtype=ftype)
        _ = f.create_dataset('total_kiila', data=total_kiila, dtype=ftype)
        _ = f.create_dataset('total_muut', data=total_muut, dtype=ftype)
        _ = f.create_dataset('osuus_vero', data=osuus_vero, dtype=ftype)
        _ = f.create_dataset('osuus_kunnallisvero', data=osuus_kunnallisvero, dtype=ftype)
        _ = f.create_dataset('osuus_valtionvero', data=osuus_valtionvero, dtype=ftype)
        _ = f.create_dataset('total_tyollaste', data=total_tyollaste, dtype=ftype)
        _ = f.create_dataset('total_tyotaste', data=total_tyotaste, dtype=ftype)
        _ = f.create_dataset('total_tyossa', data=total_tyossa, dtype=ftype)
        _ = f.create_dataset('ratio_osatyo', data=ratio_osatyo, dtype=ftype)
        _ = f.create_dataset('total_kiila_kaikki_ansiot', data=total_kiila_kaikki_ansiot, dtype=ftype)
        _ = f.create_dataset('total_ps_norw', data=total_ps_norw, dtype=ftype)
        _ = f.create_dataset('total_htv_norw', data=total_htv_norw, dtype=ftype)
        _ = f.create_dataset('total_tyossa_norw', data=total_tyossa_norw, dtype=ftype)
        _ = f.create_dataset('total_etuusmeno', data=total_etuusmeno, dtype=ftype)
        _ = f.create_dataset('total_elakemeno', data=total_elakemeno, dtype=ftype)
        _ = f.create_dataset('total_tyottomyysmeno', data=total_tyottomyysmeno, dtype=ftype)
        _ = f.create_dataset('total_asumistukimeno', data=total_asumistukimeno, dtype=ftype)
        _ = f.create_dataset('total_toimeentulotukimeno', data=total_toimeentulotukimeno, dtype=ftype)
        _ = f.create_dataset('total_muutmeno', data=total_muutmeno, dtype=ftype)
    

        f.close()

    def load_data(self,filename):
        f = h5py.File(filename, 'r')
        additional_income_tax=f.get('additional_income_tax')[:]
        total_verot=f.get('total_verot')[:]
        total_rew=f.get('total_rew')[:]
        total_ps=f.get('total_ps')[:]
        total_htv=f.get('total_htv')[:]
        total_kiila=f.get('total_kiila')[:]
        total_muut=f.get('total_muut')[:]
        if 'total_tyollaste' in f:
            total_tyollaste=f.get('total_tyollaste')[:]
            total_tyotaste=f.get('total_tyotaste')[:]
            total_tyossa=f.get('total_tyossa')[:]
        else:
            total_tyollaste=additional_income_tax*0
            total_tyotaste=additional_income_tax*0
            total_tyossa=additional_income_tax*0
        if 'ratio_osatyo' in f:
            ratio_osatyo=f.get('ratio_osatyo')[:]
        else:
            ratio_osatyo=additional_income_tax*0
        if 'total_ps_norw' in f:
            total_ps_norw=f.get('total_ps_norw')[:]
            total_tyossa_norw=f.get('total_tyossa_norw')[:]
            total_htv_norw=f.get('total_htv_norw')[:]
        else:
            total_ps_norw=total_ps*0
            total_tyossa_norw=total_tyossa*0
            total_htv_norw=total_htv*0
    
        if 'total_etuusmeno' in f:
            total_etuusmeno=f.get('total_etuusmeno')[:]
        else:
            total_etuusmeno=total_ps*0

        if 'total_tyottomyysmeno' in f:
            total_tyottomyysmeno=f.get('total_tyottomyysmeno')[:]
            total_asumistukimeno=f.get('total_asumistukimeno')[:]
            total_elakemeno=f.get('total_elakemeno')[:]
            total_toimeentulotukimeno=f.get('total_toimeentulotukimeno')[:]
            total_muutmeno=f.get('total_muutmeno')[:]
        else:
            total_tyottomyysmeno=total_ps*0
            total_asumistukimeno=total_ps*0
            total_elakemeno=total_ps*0
            total_toimeentulotukimeno=total_ps*0
            total_muutmeno=total_ps*0
    
        osuus_vero=f.get('osuus_vero')[:]
        osuus_kunnallisvero=f.get('osuus_kunnallisvero')[:]
        osuus_valtionvero=f.get('osuus_valtionvero')[:]
        total_kiila_kaikki_ansiot=f.get('total_kiila_kaikki_ansiot')[:]

        f.close()

        return additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,\
               total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,total_kiila_kaikki_ansiot,\
               total_ps_norw,total_tyossa_norw,total_htv_norw,\
               osuus_vero,osuus_kunnallisvero,osuus_valtionvero,total_etuusmeno,\
               total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno

    def comp_means(self,filename):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,\
            total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,total_kiila_kaikki_ansiot,\
            _,_,_,_,_,_,total_etuusmeno,total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno\
            =self.load_data(filename)
        rew=np.mean(total_rew,axis=1).reshape(-1, 1)
        verot=np.mean(total_verot,axis=1).reshape(-1, 1)
        ps=np.mean(total_ps,axis=1).reshape(-1, 1)
        htv=np.mean(total_htv,axis=1).reshape(-1, 1)
        muut=np.mean(total_muut,axis=1).reshape(-1, 1)
        kiila=np.mean(total_kiila,axis=1).reshape(-1, 1)
        tyossa=np.mean(total_tyossa,axis=1).reshape(-1, 1)
        ratio_osatyo=np.mean(ratio_osatyo,axis=1).reshape(-1, 1)
        tyotaste=np.mean(total_tyotaste,axis=1).reshape(-1, 1)
        etuusmeno=np.mean(total_etuusmeno,axis=1).reshape(-1, 1)
        tyottomyysmeno=np.mean(total_tyottomyysmeno,axis=1).reshape(-1, 1)
        asumistukimeno=np.mean(total_asumistukimeno,axis=1).reshape(-1, 1)
        elakemeno=np.mean(total_elakemeno,axis=1).reshape(-1, 1)
        toimeentulotukimeno=np.mean(total_toimeentulotukimeno,axis=1).reshape(-1, 1)
        muutmeno=np.mean(total_muutmeno,axis=1).reshape(-1, 1)

        return rew,verot,ps,htv,muut,kiila,tyossa,ratio_osatyo,tyotaste,etuusmeno,\
            tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno

    def comp_means_norw(self,filename,grouped=False,gender=False):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,\
            total_tyollaste,total_tyotaste,total_tyossa,ratio_osatyo,total_kiila_kaikki_ansiot,\
            total_ps_norw,total_tyossa_norw,total_htv_norw,_,_,_,total_etuusmeno,\
            total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno\
            =self.load_data(filename)
            
        if grouped:
            rew=np.mean(total_rew,axis=1)
            verot=np.mean(total_verot,axis=1)
            ps=np.mean(total_ps,axis=1)
            htv=np.mean(total_htv,axis=1)
            muut=np.mean(total_muut,axis=1)
            kiila=np.mean(total_kiila,axis=1)
            tyossa=np.mean(total_tyossa,axis=1)
            ratio_osatyo=np.mean(ratio_osatyo,axis=1)
            tyotaste=np.mean(total_tyotaste,axis=1)
            ps_norw=np.mean(total_ps_norw,axis=1)
            htv_norw=np.mean(total_htv_norw,axis=1)
            tyossa_norw=np.mean(total_tyossa_norw,axis=1)
            etuusmeno=np.mean(total_etuusmeno,axis=1)
            tyottomyysmeno=np.mean(total_tyottomyysmeno,axis=1)
            asumistukimeno=np.mean(total_asumistukimeno,axis=1)
            elakemeno=np.mean(total_elakemeno,axis=1)
            toimeentulotukimeno=np.mean(total_toimeentulotukimeno,axis=1)
            muutmeno=np.mean(total_muutmeno,axis=1)
        elif gender:
            verot=self.add_gender(total_verot)
            rew=self.add_gender(total_rew)
            ps=self.add_gender(total_ps)
            htv=self.add_gender(total_htv)
            kiila=self.add_gender(total_kiila)
            tyossa=self.add_gender(total_tyossa)
            muut=self.add_gender(total_muut)
            ratio_osatyo=np.mean(ratio_osatyo,axis=1)
            tyollaste=self.add_gender(total_tyollaste)
            tyotaste=self.add_gender(total_tyotaste)
            #osatyoratio=self.add_gender(osatyoratio)
            #kiila_kaikki=self.add_gender(kiila_kaikki)
            ps_norw=self.add_gender(total_ps_norw)
            tyossa_norw=self.add_gender(total_tyossa_norw)
            htv_norw=self.add_gender(total_htv_norw)
            etuusmeno=self.add_gender(total_etuusmeno)
            tyottomyysmeno=self.add_gender(total_tyottomyysmeno)
            asumistukimeno=self.add_gender(total_asumistukimeno)
            elakemeno=self.add_gender(total_elakemeno)
            toimeentulotukimeno=self.add_gender(total_toimeentulotukimeno)
            muutmeno=self.add_gender(total_muutmeno)
        else:
            rew=np.mean(total_rew,axis=1).reshape(-1, 1)
            verot=np.mean(total_verot,axis=1).reshape(-1, 1)
            ps=np.mean(total_ps,axis=1).reshape(-1, 1)
            htv=np.mean(total_htv,axis=1).reshape(-1, 1)
            muut=np.mean(total_muut,axis=1).reshape(-1, 1)
            kiila=np.mean(total_kiila,axis=1).reshape(-1, 1)
            tyossa=np.mean(total_tyossa,axis=1).reshape(-1, 1)
            ratio_osatyo=np.mean(ratio_osatyo,axis=1).reshape(-1, 1)
            tyotaste=np.mean(total_tyotaste,axis=1).reshape(-1, 1)
            ps_norw=np.mean(total_ps_norw,axis=1).reshape(-1, 1)
            htv_norw=np.mean(total_htv_norw,axis=1).reshape(-1, 1)
            tyossa_norw=np.mean(total_tyossa_norw,axis=1).reshape(-1, 1)
            etuusmeno=np.mean(total_etuusmeno,axis=1).reshape(-1, 1)
            tyottomyysmeno=np.mean(total_tyottomyysmeno,axis=1).reshape(-1, 1)
            asumistukimeno=np.mean(total_asumistukimeno,axis=1).reshape(-1, 1)
            elakemeno=np.mean(total_elakemeno,axis=1).reshape(-1, 1)
            toimeentulotukimeno=np.mean(total_toimeentulotukimeno,axis=1).reshape(-1, 1)
            muutmeno=np.mean(total_muutmeno,axis=1).reshape(-1, 1)

        return rew,verot,ps,htv,muut,kiila,tyossa,ratio_osatyo,tyotaste,\
                ps_norw,htv_norw,tyossa_norw,etuusmeno,\
                tyottomyysmeno,asumistukimeno,elakemeno,toimeentulotukimeno,muutmeno

    def get_refstats(self,filename,scale=None):
        additional_income_tax,total_verot,total_rew,total_ps,total_htv,total_kiila,total_muut,_,_,_,_=\
            self.load_data(filename)
        z=np.where(np.abs(additional_income_tax)<0.001)

        if scale is not None:
            scaling=np.squeeze(np.ones(scale.shape))
        else:
            scaling=1

        # baseline muut
        baseline_muut=total_muut[z]*scaling
        baseline_verot=total_verot[z]*scaling.T
        baseline_rew=total_rew[z]*scaling.T
        baseline_ps=total_ps[z]*scaling.T
        baseline_htv=total_htv[z]*scaling.T
        baseline_kiila=total_kiila[z]*scaling.T

        #print('baseline',baseline_htv.shape)

        return baseline_rew,baseline_verot,baseline_ps,baseline_htv,baseline_muut,baseline_kiila


    def comp_stats(self,taxresults,additional,datafile,repeats,minimal=False,mortality=False,perustulo=False,
                    randomness=True,pinkslip=True,plotdebug=False,start_repeat=0,grouped=False,g=0,env='unemployment-v3'):
        total_verot=np.zeros((additional.shape[0],repeats-start_repeat))
        total_rew=np.zeros((additional.shape[0],repeats-start_repeat))
        total_ps=np.zeros((additional.shape[0],repeats-start_repeat))
        total_htv=np.zeros((additional.shape[0],repeats-start_repeat))
        total_ps_norw=np.zeros((additional.shape[0],repeats-start_repeat))
        total_htv_norw=np.zeros((additional.shape[0],repeats-start_repeat))
        total_tyollaste=np.zeros((additional.shape[0],repeats-start_repeat))
        total_tyotaste=np.zeros((additional.shape[0],repeats-start_repeat))
        total_muut_tulot=np.zeros((additional.shape[0],repeats-start_repeat))
        total_kiila=np.zeros((additional.shape[0],repeats-start_repeat))
        total_kiila_kaikki_ansiot=np.zeros((additional.shape[0],repeats-start_repeat))
        total_tyossa=np.zeros((additional.shape[0],repeats-start_repeat))
        total_tyossa_norw=np.zeros((additional.shape[0],repeats-start_repeat))
        total_etuusmeno=np.zeros((additional.shape[0],repeats-start_repeat))
        ratio_osatyo=np.zeros((additional.shape[0],repeats-start_repeat))
        total_tyottomyysmeno=np.zeros((additional.shape[0],repeats-start_repeat))
        total_asumistukimeno=np.zeros((additional.shape[0],repeats-start_repeat))
        total_elakemeno=np.zeros((additional.shape[0],repeats-start_repeat))
        total_toimeentulotukimeno=np.zeros((additional.shape[0],repeats-start_repeat))
        total_muutmeno=np.zeros((additional.shape[0],repeats-start_repeat))

        osuus_valtionvero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        osuus_kunnallisvero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        osuus_vero=np.zeros(( additional.shape[0],repeats-start_repeat,4))

        for k,tax in enumerate(additional):
            cc=Lifecycle(env=env,minimal=False,mortality=mortality,perustulo=False,
                         randomness=randomness,pinkslip=pinkslip,plotdebug=plotdebug,
                         additional_income_tax=tax)
            for repeat in range(start_repeat,repeats):
                num=int(np.round(100*tax))
                print('computing extra tax {} repeat {}'.format(tax,repeat))
                results=taxresults+'_{}_{}'.format(num,repeat)
                print(results)
                cc.load_sim(results)
                r,qps,qv,h,muut,kiila,tyoll,tyot,lkm,osatyo_osuus,kiila_kaikki_ansiot,menot=cc.render_laffer(include_retwork=True,grouped=grouped,g=g)
                _,qps_norw,_,h_norw,_,kiila_norw,tyoll_norw,_,lkm_norw,_,kiila_kaikki_ansiot_norw,menot_norw=cc.render_laffer(include_retwork=False,grouped=grouped,g=g)
                vvosuus,kvosuus,vosuus=cc.comp_taxratios(grouped=True)
                total_verot[k,repeat-start_repeat]=qv
                total_rew[k,repeat-start_repeat]=r
                total_ps[k,repeat-start_repeat]=qps
                total_ps_norw[k,repeat-start_repeat]=qps_norw
                total_htv[k,repeat-start_repeat]=h
                total_htv_norw[k,repeat-start_repeat]=h_norw
                total_muut_tulot[k,repeat-start_repeat]=muut
                total_kiila[k,repeat-start_repeat]=kiila
                total_kiila_kaikki_ansiot[k,repeat-start_repeat]=kiila_kaikki_ansiot
                total_tyotaste[k,repeat-start_repeat]=tyot
                total_tyollaste[k,repeat-start_repeat]=tyoll
                total_tyossa[k,repeat-start_repeat]=lkm
                total_tyossa_norw[k,repeat-start_repeat]=lkm_norw
                total_etuusmeno[k,repeat-start_repeat]=menot['etuusmeno']
                total_tyottomyysmeno[k,repeat-start_repeat]=menot['tyottomyysmenot']
                total_asumistukimeno[k,repeat-start_repeat]=menot['asumistuki']
                total_elakemeno[k,repeat-start_repeat]=menot['kokoelake']
                total_toimeentulotukimeno[k,repeat-start_repeat]=menot['toimeentulotuki']
                total_muutmeno[k,repeat-start_repeat]=menot['muutmenot']
                ratio_osatyo[k,repeat-start_repeat]=osatyo_osuus
                osuus_vero[k,repeat-start_repeat,:]=vosuus
                osuus_kunnallisvero[k,repeat-start_repeat,:]=kvosuus
                osuus_valtionvero[k,repeat-start_repeat,:]=vvosuus
            
        self.save_data(datafile,additional,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut_tulot,total_tyollaste,total_tyotaste,total_tyossa,
                  ratio_osatyo,total_kiila_kaikki_ansiot,total_ps_norw,total_tyossa_norw,total_htv_norw,
                  total_etuusmeno,total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno,
                  osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
                  
    def test_stats_groups(self,taxresults,additional,datafile,repeats,minimal=False,mortality=False,perustulo=False,
                    randomness=True,pinkslip=True,plotdebug=False,start_repeat=0,n_groups=6,env='unemployment-v3'):

        for k,tax in enumerate(additional):
            cc=Lifecycle(env=env,minimal=False,mortality=mortality,perustulo=False,
                         randomness=randomness,pinkslip=pinkslip,plotdebug=plotdebug,
                         additional_income_tax=tax)
            for repeat in range(start_repeat,repeats):
                num=int(np.round(100*tax))
                print('testing extra tax {} repeat {}'.format(tax,repeat))
                results=taxresults+'_{}_{}'.format(num,repeat)
                cc.load_sim(results)
                cc.episodestats.test_emp()

    def comp_stats_groups(self,taxresults,additional,datafile,repeats,minimal=False,mortality=False,perustulo=False,
                    randomness=True,pinkslip=True,plotdebug=False,start_repeat=0,n_groups=6,env='unemployment-v3'):
        total_verot=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_rew=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_ps=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_htv=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_ps_norw=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_htv_norw=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyollaste=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyotaste=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_muut_tulot=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_kiila=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_kiila_kaikki_ansiot=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyossa=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyossa_norw=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_etuusmeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        ratio_osatyo=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        osuus_valtionvero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        osuus_kunnallisvero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        osuus_vero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        ratio_osatyo=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyottomyysmeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_asumistukimeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_elakemeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_toimeentulotukimeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_muutmeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))

        for k,tax in enumerate(additional):
            cc=Lifecycle(env=env,minimal=False,mortality=mortality,perustulo=False,
                         randomness=randomness,pinkslip=pinkslip,plotdebug=plotdebug,
                         additional_income_tax=tax)
            for repeat in range(start_repeat,repeats):
                num=int(np.round(100*tax))
                print('computing extra tax {} repeat {}'.format(tax,repeat))
                results=taxresults+'_{}_{}'.format(num,repeat)
                cc.load_sim(results)
                #cc.episodestats.test_emp()
                print(results)
                aps,aps_norw=cc.episodestats.comp_group_ps()
                for g in range(n_groups):
                    r,qps,qv,h,muut,kiila,tyoll,tyot,lkm,osatyo_osuus,kiila_kaikki_ansiot,menot=cc.render_laffer(include_retwork=True,grouped=True,g=g)
                    _,qps_norw,_,h_norw,_,kiila_norw,tyoll_norw,_,lkm_norw,_,kiila_kaikki_ansiot_norw,menot_norw=cc.render_laffer(include_retwork=False,grouped=True,g=g)
                    #vvosuus,kvosuus,vosuus=cc.comp_taxratios(grouped=True)
                    #print('lkm',lkm,lkm_norw)
                    total_verot[k,repeat-start_repeat,g]=qv
                    total_rew[k,repeat-start_repeat,g]=r
                    total_ps[k,repeat-start_repeat,g]=aps[g]
                    total_ps_norw[k,repeat-start_repeat,g]=aps_norw[g]
                    total_htv[k,repeat-start_repeat,g]=h
                    total_htv_norw[k,repeat-start_repeat,g]=h_norw
                    total_muut_tulot[k,repeat-start_repeat,g]=muut
                    total_kiila[k,repeat-start_repeat,g]=kiila
                    total_kiila_kaikki_ansiot[k,repeat-start_repeat,g]=kiila_kaikki_ansiot
                    total_tyotaste[k,repeat-start_repeat,g]=tyot
                    total_tyollaste[k,repeat-start_repeat,g]=tyoll
                    total_tyossa[k,repeat-start_repeat,g]=lkm
                    total_tyossa_norw[k,repeat-start_repeat,g]=lkm_norw
                    total_etuusmeno[k,repeat-start_repeat,g]=menot['etuusmeno']
                    total_tyottomyysmeno[k,repeat-start_repeat,g]=menot['tyottomyysmenot']
                    total_asumistukimeno[k,repeat-start_repeat,g]=menot['asumistuki']
                    total_elakemeno[k,repeat-start_repeat,g]=menot['kokoelake']
                    total_toimeentulotukimeno[k,repeat-start_repeat,g]=menot['toimeentulotuki']
                    total_muutmeno[k,repeat-start_repeat,g]=menot['muutmenot']
                    ratio_osatyo[k,repeat-start_repeat,g]=osatyo_osuus
                    #osuus_vero[k,repeat-start_repeat,:]=vosuus
                    #osuus_kunnallisvero[k,repeat-start_repeat,:]=kvosuus
                    #osuus_valtionvero[k,repeat-start_repeat,:]=vvosuus
                    
                #r,qps,qv,h,muut,kiila,tyoll,tyot,lkm,osatyo_osuus,kiila_kaikki_ansiot,menot=cc.render_laffer(include_retwork=True,grouped=False)
                #r,qps,qv,h,muut,kiila,tyoll,tyot,lkm_norw,osatyo_osuus,kiila_kaikki_ansiot,menot=cc.render_laffer(include_retwork=False,grouped=False)
                #print('actual',lkm,'found',np.sum(total_tyossa[k,repeat-start_repeat,:]),'delta',lkm-np.sum(total_tyossa[k,repeat-start_repeat,:]))
                #print('actual',lkm_norw,'found',np.sum(total_tyossa_norw[k,repeat-start_repeat,:]),'delta',lkm_norw-np.sum(total_tyossa_norw[k,repeat-start_repeat,:]))
            
        self.save_data(datafile,additional,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut_tulot,total_tyollaste,total_tyotaste,total_tyossa,
                  ratio_osatyo,total_kiila_kaikki_ansiot,total_ps_norw,total_tyossa_norw,total_htv_norw,
                  total_etuusmeno,total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno,
                  osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
                  
    def comp_stats_agegroups(self,taxresults,additional,datafile,repeats,minimal=False,mortality=False,perustulo=False,
                    randomness=True,pinkslip=True,plotdebug=False,start_repeat=0,n_groups=3,env='unemployment-v3'):
        total_verot=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_rew=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_ps=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_htv=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_ps_norw=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_htv_norw=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyollaste=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyotaste=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_muut_tulot=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_kiila=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_kiila_kaikki_ansiot=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyossa=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyossa_norw=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_etuusmeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        ratio_osatyo=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        osuus_valtionvero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        osuus_kunnallisvero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        osuus_vero=np.zeros(( additional.shape[0],repeats-start_repeat,4))
        ratio_osatyo=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_tyottomyysmeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_asumistukimeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_elakemeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_toimeentulotukimeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))
        total_muutmeno=np.zeros((additional.shape[0],repeats-start_repeat,n_groups))

        for k,tax in enumerate(additional):
            cc=Lifecycle(env=env,minimal=False,mortality=mortality,perustulo=False,
                         randomness=randomness,pinkslip=pinkslip,plotdebug=plotdebug,
                         additional_income_tax=tax)
            for repeat in range(start_repeat,repeats):
                num=int(np.round(100*tax))
                print('computing extra tax {} repeat {}'.format(tax,repeat))
                results=taxresults+'_{}_{}'.format(num,repeat)
                cc.load_sim(results)
                print(results)
                employed,htv,unemployed,parttimeratio,ps,ps_norw,unempratio,empratio=cc.episodestats.comp_stats_agegroup()
                ratio_osatyo[k,repeat-start_repeat,0:n_groups]=parttimeratio[0:n_groups]
                total_htv[k,repeat-start_repeat,0:n_groups]=htv[0:n_groups]
                total_tyotaste[k,repeat-start_repeat,0:n_groups]=unempratio[0:n_groups]
                total_tyollaste[k,repeat-start_repeat,0:n_groups]=employed[0:n_groups]
                total_tyossa[k,repeat-start_repeat,0:n_groups]=employed[0:n_groups]
                total_ps[k,repeat-start_repeat,0:n_groups]=ps[0:n_groups]
                total_ps_norw[k,repeat-start_repeat,0:n_groups]=ps_norw[0:n_groups]
            
        self.save_data(datafile,additional,total_verot,total_rew,total_ps,total_htv,
                  total_kiila,total_muut_tulot,total_tyollaste,total_tyotaste,total_tyossa,
                  ratio_osatyo,total_kiila_kaikki_ansiot,total_ps_norw,total_tyossa_norw,total_htv_norw,
                  total_etuusmeno,total_tyottomyysmeno,total_asumistukimeno,total_elakemeno,total_toimeentulotukimeno,total_muutmeno,
                  osuus_vero,osuus_kunnallisvero,osuus_valtionvero)
                      
    
    def comp_elasticity(self,x,y,diff=False):
        xl=x.shape[0]
        yl=x.shape[0]    
        el=np.zeros((xl-2,1))
        elx=np.zeros((xl-2,1))
    
        for k in range(1,xl-1):
            if diff:
                dx=(-x[k+1]+x[k-1])/(2*x[k])
            else:
                dx=-x[k+1]+x[k-1]
            dy=(y[k+1]-y[k-1])/(2*y[k])
            el[k-1]=dy/dx
            elx[k-1]=x[k]
            
            #print('{}: {} vs {}'.format(k,(x[k]-x[k-1]),dy))
        
        return el,elx
    