'''

    simstats.py

    implements statistic for multiple runs of a single model

'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm
import locale
from tabulate import tabulate
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from .episodestats import EpisodeStats
    
class SimStats(EpisodeStats):
    def run_simstats(self,results,save,n,plot=True,startn=0,max_age=54,singlefile=False,grouped=False,group=0):
        '''
        Laskee statistiikat ajoista
        '''
        
        print('computing simulation statistics...')
        #n=self.load_hdf(results+'_simut','n')
        e_rate=np.zeros((n,self.n_time))
        agg_htv=np.zeros(n)
        agg_tyoll=np.zeros(n)
        agg_rew=np.zeros(n)
        agg_discounted_rew=np.zeros(n)
        t_aste=np.zeros(self.n_time)
        emps=np.zeros((n,self.n_time,self.n_employment))
        emp_tyolliset=np.zeros((n,self.n_time))
        emp_tyottomat=np.zeros((n,self.n_time))
        emp_tyolliset_osuus=np.zeros((n,self.n_time))
        emp_tyottomat_osuus=np.zeros((n,self.n_time))
        emp_htv=np.zeros((n,self.n_time))
        tyoll_virta=np.zeros((n,self.n_time))
        tyot_virta=np.zeros((n,self.n_time))
        tyot_virta_ansiosid=np.zeros((n,self.n_time))
        tyot_virta_tm=np.zeros((n,self.n_time))
        unemp_dur=np.zeros((n,5,5))
        unemp_lastdur=np.zeros((n,5,5))
        agg_netincome=np.zeros(n)
        agg_equivalent_netincome=np.zeros(n)

        if singlefile:
            self.load_sim(results,print_pop=False)
        else:
            self.load_sim(results+'_'+str(100+startn),print_pop=False)

        if grouped:
            base_empstate=self.gempstate[:,:,group]/self.n_pop
        else:
            base_empstate=self.empstate/self.n_pop
        
        emps[0,:,:]=base_empstate
        htv_base,tyoll_base,haj_base,tyollaste_base,tyolliset_base=self.comp_tyollisyys_stats(base_empstate,scale_time=True)
        reward=self.get_reward()
        discounted_reward=self.get_reward(discounted=True)
        net,equiv=self.comp_total_netincome(output=False)
        agg_htv[0]=htv_base
        agg_tyoll[0]=tyoll_base
        agg_rew[0]=reward
        agg_discounted_rew[0]=discounted_reward
        agg_netincome[0]=net
        agg_equivalent_netincome[0]=equiv
        
        best_rew=reward
        best_emp=0
        t_aste[0]=tyollaste_base
        
        tyolliset_ika,tyottomat,htv_ika,tyolliset_osuus,tyottomat_osuus=self.comp_tyollisyys_stats(base_empstate,tyot_stats=True,shapes=True)

        emp_tyolliset[0,:]=tyolliset_ika[:]
        emp_tyottomat[0,:]=tyottomat[:]
        emp_tyolliset_osuus[0,:]=tyolliset_osuus[:]
        emp_tyottomat_osuus[0,:]=tyottomat_osuus[:]
        emp_htv[0,:]=htv_ika[:]
        
        unemp_distrib,emp_distrib,unemp_distrib_bu=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=max_age)
        tyoll_distrib,tyoll_distrib_bu=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=max_age)

        # virrat työllisyyteen ja työttömyyteen
        tyoll_virta0,tyot_virta0=self.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
        tyoll_virta_ansiosid0,tyot_virta_ansiosid0=self.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
        tyoll_virta_tm0,tyot_virta_tm0=self.comp_virrat(ansiosid=False,tmtuki=True,putki=False,outsider=False)

        tyoll_virta[0,:]=tyoll_virta0[:,0]
        tyot_virta[0,:]=tyot_virta0[:,0]
        tyot_virta_ansiosid[0,:]=tyot_virta_ansiosid0[:,0]
        tyot_virta_tm[0,:]=tyot_virta_tm0[:,0]
        
        unemp_dur0=self.comp_unemp_durations(return_q=False)
        unemp_lastdur0=self.comp_unemp_durations_v2(return_q=False)
        unemp_dur[0,:,:]=unemp_dur0[:,:]
        unemp_lastdur[0,:,:]=unemp_lastdur0[:,:]

        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel(self.labels['age'])
            ax.set_ylabel('Työllisyysaste [%]')
            x=np.linspace(self.min_age,self.max_age,self.n_time)
            ax.plot(x,100*tyolliset_base,alpha=0.9,lw=2.0)

        if not singlefile:
            tqdm_e = tqdm(range(int(n)), desc='Sim', leave=True, unit=" ")

            for i in range(startn+1,n): 
                self.load_sim(results+'_'+str(100+i),print_pop=False)
                if grouped:
                    empstate=self.gempstate[:,:,group]/self.n_pop
                else:
                    empstate=self.empstate/self.n_pop
                
                emps[i,:,:]=empstate
                reward=self.get_reward()
                discounted_reward=self.get_reward(discounted=True)
                
                net,equiv=self.comp_total_netincome(output=False)
                if reward>best_rew:
                    best_rew=reward
                    best_emp=i

                htv,tyollvaikutus,haj,tyollisyysaste,tyolliset=self.comp_tyollisyys_stats(empstate,scale_time=True)
                
                if plot:
                    ax.plot(x,100*tyolliset,alpha=0.5,lw=0.5)
    
                agg_htv[i]=htv
                agg_tyoll[i]=tyollvaikutus
                agg_rew[i]=reward
                agg_discounted_rew[i]=discounted_reward

                agg_netincome[i]=net
                agg_equivalent_netincome[i]=equiv
                t_aste[i]=tyollisyysaste

                #tyolliset_ika,tyottomat,htv_ika,tyolliset_osuus,tyottomat_osuus=self.comp_employed_number(empstate)
                tyolliset_ika,tyottomat,htv_ika,tyolliset_osuus,tyottomat_osuus=self.comp_tyollisyys_stats(empstate,tyot_stats=True)
            
                emp_tyolliset[i,:]=tyolliset_ika[:]
                emp_tyottomat[i,:]=tyottomat[:]
                emp_tyolliset_osuus[i,:]=tyolliset_osuus[:]
                emp_tyottomat_osuus[i,:]=tyottomat_osuus[:]
                emp_htv[i,:]=htv_ika[:]

                unemp_distrib2,emp_distrib2,unemp_distrib_bu2=self.comp_empdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=max_age)
                tyoll_distrib2,tyoll_distrib_bu2=self.comp_tyollistymisdistribs(ansiosid=True,tmtuki=True,putki=True,outsider=False,max_age=max_age)
            
                unemp_distrib.extend(unemp_distrib2)
                emp_distrib.extend(emp_distrib2)
                unemp_distrib_bu.extend(unemp_distrib_bu2)
                tyoll_distrib.extend(tyoll_distrib2)
                tyoll_distrib_bu.extend(tyoll_distrib_bu2)
            
                # virrat työllisyyteen ja työttömyyteen
                tyoll_virta0,tyot_virta0=self.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
                tyoll_virta_ansiosid0,tyot_virta_ansiosid0=self.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
                tyoll_virta_tm0,tyot_virta_tm0=self.comp_virrat(ansiosid=False,tmtuki=True,putki=False,outsider=False)

                tyoll_virta[i,:]=tyoll_virta0[:,0]
                tyot_virta[i,:]=tyot_virta0[:,0]
                tyot_virta_ansiosid[i,:]=tyot_virta_ansiosid0[:,0]
                tyot_virta_tm[i,:]=tyot_virta_tm0[:,0]

                unemp_dur0=self.comp_unemp_durations(return_q=False)
                unemp_lastdur0=self.comp_unemp_durations_v2(return_q=False)
                unemp_dur[i,:,:]=unemp_dur0[:,:]
                unemp_lastdur[i,:,:]=unemp_lastdur0[:,:]
                tqdm_e.update(1)
                tqdm_e.set_description("Pop " + str(n))

        self.save_simstats(save,agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,\
                            emp_tyolliset,emp_tyolliset_osuus,\
                            emp_tyottomat,emp_tyottomat_osuus,\
                            emp_htv,emps,\
                            best_rew,best_emp,\
                            unemp_distrib,emp_distrib,unemp_distrib_bu,\
                            tyoll_distrib,tyoll_distrib_bu,\
                            tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
                            unemp_dur,unemp_lastdur,agg_netincome,agg_equivalent_netincome)
                    
        #if not singlefile:
        #    # save the best
        #    self.load_sim(results+'_'+str(100+best_emp))
        #    self.save_sim(results+'_best')
                    
        print('done')
        print('best_emp',best_emp)
        
    def run_optimize_x(self,target,results,n,startn=0,averaged=False):
        '''
        Laskee statistiikat ajoista
        '''
        
        print('computing simulation statistics...')
        #n=self.load_hdf(results+'_simut','n')
        x_rate=np.zeros(n-startn)

        tqdm_e = tqdm(range(int(n-startn)), desc='Sim', leave=True, unit=" ")
        for i in range(startn,n): 
            self.load_sim(results+'_'+str(100+i),print_pop=False)
            x_rate[i-startn]=self.optimize_scale(target,averaged=averaged)
            tqdm_e.update(1)
            tqdm_e.set_description("Pop " + str(i-startn))
                    
        print(x_rate)
        print('Average x {}'.format(np.mean(x_rate)))
                    
        print('done')
        
    def fit_norm(self,diff):
        diff_stdval=np.std(diff)
        diff_meanval=np.mean(diff)
        diff_minval=np.min(diff)
        diff_maxval=np.max(diff)
        sz=(diff_maxval-diff_minval)/10
        x=np.linspace(diff_minval,diff_maxval,1000)
        y=norm.pdf(x,diff_meanval,diff_stdval)*diff.shape[0]*sz
    
        return x,y
        
    def count_putki_dist(self,emps):
        putki=[]
    
        for k in range(emps.shape[0]):
            putki.append(self.count_putki(emps[k,:,:]))
            
        putkessa=np.median(np.asarray(putki))
        return putkessa        
        
    def plot_simstats(self,filename,grayscale=False,figname=None):
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
            best_emp,emps,agg_netincome,agg_equivalent_netincome=self.load_simstats(filename)

        if self.version>0:
            print('lisäpäivillä on {:.0f} henkilöä'.format(self.count_putki_dist(emps)))

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
            unemp_dur,unemp_lastdur=self.load_simdistribs(filename)
       
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        self.plot_unemp_durdistribs(unemp_dur)
        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        self.plot_unemp_durdistribs(unemp_lastdur)

        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label='vaihtoehto')
        self.plot_unempdistribs(unemp_distrib1,figname=figname,max=10,miny=1e-5,maxy=2)
        #self.plot_tyolldistribs(unemp_distrib1,tyoll_distrib1,tyollistyneet=True,figname=figname)
        self.plot_tyolldistribs_both(unemp_distrib1,tyoll_distrib1,max=4,figname=figname)

    def get_simstats(self,filename1,plot=False,use_mean=False):
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
            emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
            best_emp,emps,agg_netincome,agg_equivalent_netincome=self.load_simstats(filename1)

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-mean_tyoll
        
        #print(filename1,emp_tyolliset_osuus)
        
        m_mean=np.mean(emp_tyolliset_osuus,axis=0)
        m_median=np.median(emp_tyolliset_osuus,axis=0)
        mn_median=np.median(emp_tyolliset,axis=0)
        mn_mean=np.median(emp_tyolliset,axis=0)
        s_emp=np.std(emp_tyolliset_osuus,axis=0)
        m_best=emp_tyolliset_osuus[best_emp,:]

        h_mean=np.mean(emp_htv,axis=0)
        h_median=np.median(emp_htv,axis=0)
        hs_emp=np.std(emp_htv,axis=0)
        h_best=emp_htv[best_emp,:]


        if self.minimal:
            u_tmtuki=0*np.median(emps[:,:,0],axis=0)
            u_ansiosid=np.median(emps[:,:,0],axis=0)
        else:
            u_tmtuki=np.median(emps[:,:,13],axis=0)
            u_ansiosid=np.median(emps[:,:,0]+emps[:,:,4],axis=0)
    
        if plot:
            fig,ax=plt.subplots()
            ax.set_xlabel('Poikkeama työllisyydessä [htv]')
            ax.set_ylabel('Lukumäärä')
            ax.hist(diff_htv)
            plt.show()

            if self.version>0:
                fig,ax=plt.subplots()
                ax.set_xlabel('Poikkeama työllisyydessä [henkilöä]')
                ax.set_ylabel('Lukumäärä')
                ax.hist(diff_tyoll)
                plt.show()

            fig,ax=plt.subplots()
            ax.set_xlabel('Palkkio')
            ax.set_ylabel('Lukumäärä')
            ax.hist(agg_rew)
            plt.show()    
    
        if use_mean:
            return m_best,m_mean,s_emp,mean_htv,u_tmtuki,u_ansiosid,h_mean,mn_mean
        else:
            return m_best,m_median,s_emp,median_htv,u_tmtuki,u_ansiosid,h_median,mn_median

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
            tyoll_virta1,tyot_virta1,tyot_virta_ansiosid1,tyot_virta_tm1,kestot1,viimkesto1=self.load_simdistribs(filename1)
        unemp_distrib2,emp_distrib2,unemp_distrib_bu2,tyoll_distrib2,tyoll_distrib_bu2,\
            tyoll_virta2,tyot_virta2,tyot_virta_ansiosid2,tyot_virta_tm2,kestot2,viimkesto2=self.load_simdistribs(filename2)
        
        self.plot_compare_unemp_durdistribs(kestot1,kestot2,viimkesto1,viimkesto2,label1='',label2='')
        
        #self.plot_compare_empdistribs(emp_distrib1,emp_distrib2,label='vaihtoehto')
        self.plot_compare_unempdistribs(unemp_distrib1,unemp_distrib2,label1=label1,label2=label2,figname=figname)
        self.plot_compare_tyolldistribs(unemp_distrib1,tyoll_distrib1,unemp_distrib2,tyoll_distrib2,tyollistyneet=True,label1=label1,label2=label2,figname=figname)
        self.plot_compare_virtadistribs(tyoll_virta1,tyoll_virta2,tyot_virta1,tyot_virta2,tyot_virta_ansiosid1,tyot_virta_ansiosid2,tyot_virta_tm1,tyot_virta_tm2,label1=label1,label2=label2)

    def compare_epistats(self,filename1,cc2,label1='perus',label2='vaihtoehto',figname=None,greyscale=True):
        m_best1,m_median1,s_emp1,median_htv1,u_tmtuki1,u_ansiosid1,h_median1,mn_median1=self.get_simstats(filename1)
        _,m_mean1,s_emp1,mean_htv1,u_tmtuki1,u_ansiosid1,h_mean1,mn_mean1=self.get_simstats(filename1,use_mean=True)

        tyoll_osuus2,htv_osuus2,tyot_osuus2,kokotyo_osuus2,osatyo_osuus2=self.comp_employed_ratio(cc2.empstate)
        htv2,tyoll2,haj2,tyollaste2,tyolliset2,osatyolliset2,kokotyolliset2,osata2,kokota2=self.comp_tyollisyys_stats(cc2.empstate/cc2.n_pop,scale_time=True,start=s,end=e,full=True)
        ansiosid_osuus2,tm_osuus2=self.comp_employed_detailed(cc2.empstate)
        
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

    def save_simstats(self,filename,agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
                        emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,\
                        unemp_distrib,emp_distrib,unemp_distrib_bu,\
                        tyoll_distrib,tyoll_distrib_bu,\
                        tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
                        unemp_dur,unemp_lastdur,agg_netincome,agg_equivalent_netincome):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset('agg_discounted_rew', data=agg_discounted_rew, dtype='float64')
        dset = f.create_dataset('agg_htv', data=agg_htv, dtype='float64')
        dset = f.create_dataset('agg_tyoll', data=agg_tyoll, dtype='float64')
        dset = f.create_dataset('agg_rew', data=agg_rew, dtype='float64')
        dset = f.create_dataset('emp_tyolliset', data=emp_tyolliset, dtype='float64')
        dset = f.create_dataset('emp_tyolliset_osuus', data=emp_tyolliset_osuus, dtype='float64')
        dset = f.create_dataset('emp_tyottomat', data=emp_tyottomat, dtype='float64')
        dset = f.create_dataset('emp_tyottomat_osuus', data=emp_tyottomat_osuus, dtype='float64')
        dset = f.create_dataset('emp_htv', data=emp_htv, dtype='float64')
        dset = f.create_dataset('emps', data=emps, dtype='float64')
        dset = f.create_dataset('best_rew', data=best_rew, dtype='float64')
        dset = f.create_dataset('best_emp', data=best_emp, dtype='float64')
        dset = f.create_dataset('unemp_distrib', data=unemp_distrib, dtype='float64')
        dset = f.create_dataset('emp_distrib', data=emp_distrib, dtype='float64')
        dset = f.create_dataset('unemp_distrib_bu', data=unemp_distrib_bu, dtype='float64')
        dset = f.create_dataset('tyoll_distrib', data=tyoll_distrib, dtype='float64')
        dset = f.create_dataset('tyoll_distrib_bu', data=tyoll_distrib_bu, dtype='float64')
        dset = f.create_dataset('tyoll_virta', data=tyoll_virta, dtype='float64')
        dset = f.create_dataset('tyot_virta', data=tyot_virta, dtype='float64')
        dset = f.create_dataset('tyot_virta_ansiosid', data=tyot_virta_ansiosid, dtype='float64')
        dset = f.create_dataset('tyot_virta_tm', data=tyot_virta_tm, dtype='float64')
        dset = f.create_dataset('unemp_dur', data=unemp_dur, dtype='float64')
        dset = f.create_dataset('unemp_lastdur', data=unemp_lastdur, dtype='float64')
        dset = f.create_dataset('agg_netincome', data=agg_netincome, dtype='float64')
        dset = f.create_dataset('agg_equivalent_netincome', data=agg_equivalent_netincome, dtype='float64')

    def load_simstats(self,filename):
        f = h5py.File(filename, 'r')
        agg_htv = f['agg_htv'][()]
        agg_tyoll = f['agg_tyoll'][()]
        agg_rew = f['agg_rew'][()]
        if 'agg_discounted_rew' in f.keys(): # older runs do not have these
            agg_discounted_rew = f['agg_discounted_rew'][()]
        else:
            agg_discounted_rew=0*agg_rew
        emps = f['emps'][()]
        best_rew = f['best_rew'][()]
        best_emp = int(f['best_emp'][()])
        emp_tyolliset = f['emp_tyolliset'][()]
        emp_tyolliset_osuus = f['emp_tyolliset_osuus'][()]
        emp_tyottomat = f['emp_tyottomat'][()]
        emp_tyottomat_osuus = f['emp_tyottomat_osuus'][()]
        emp_htv = f['emp_htv'][()]
        agg_netincome = f['agg_netincome'][()]
        agg_equivalent_netincome = f['agg_equivalent_netincome'][()]
        
        f.close()

        return agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
               emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,emps,\
               agg_netincome,agg_equivalent_netincome

    def load_simdistribs(self,filename):
        f = h5py.File(filename, 'r')
        if 'tyoll_virta' in f:
            unemp_distrib = f['unemp_distrib'][()] #f.get('unemp_distrib').value
        else:
            unemp_distrib=np.zeros((self.n_time,self.n_pop))
        
        if 'tyoll_virta' in f:
            emp_distrib = f['emp_distrib'][()] #f.get('emp_distrib').value
        else:
            emp_distrib=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            unemp_distrib_bu = f['unemp_distrib_bu'][()] #f.get('unemp_distrib_bu').value
        else:
            unemp_distrib_bu=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            tyoll_distrib =f['tyoll_distrib'][()] # f.get('tyoll_distrib').value
        else:
            tyoll_distrib=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            tyoll_distrib_bu = f['tyoll_distrib_bu'][()] #f.get('tyoll_distrib_bu').value
        else:
            tyoll_distrib_bu=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            tyoll_virta = f['tyoll_virta'][()] #f.get('tyoll_virta').value
        else:
            tyoll_virta=np.zeros((self.n_time,self.n_pop))
        if 'tyot_virta' in f:
            tyot_virta = f['tyot_virta'][()] #f.get('tyot_virta').value
        else:
            tyot_virta=np.zeros((self.n_time,self.n_pop))
        if 'tyot_virta_ansiosid' in f:
            tyot_virta_ansiosid = f['tyot_virta_ansiosid'][()] #f.get('tyot_virta_ansiosid').value
        else:
            tyot_virta_ansiosid=np.zeros((self.n_time,self.n_pop))
        if 'tyot_virta_tm' in f:
            tyot_virta_tm = f['tyot_virta_tm'][()] #f.get('tyot_virta_tm').value
        else:
            tyot_virta_tm=np.zeros((self.n_time,self.n_pop))
        if 'unemp_dur' in f:
            unemp_dur = f['unemp_dur'][()] #f.get('unemp_dur').value
        else:
            unemp_dur=np.zeros((1,5,5))
        if 'unemp_lastdur' in f:
            unemp_lastdur = f['unemp_lastdur'][()] #f.get('unemp_lastdur').value
        else:
            unemp_lastdur=np.zeros((1,5,5))
        
        f.close()

        return unemp_distrib,emp_distrib,unemp_distrib_bu,\
               tyoll_distrib,tyoll_distrib_bu,\
               tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
               unemp_dur,unemp_lastdur

    def plot_compare_csvirta(self,m1,m2,lbl):
        nc1=np.reshape(np.cumsum(m1),m1.shape)
        #print(m1.shape,np.cumsum(m1).shape)
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
        #print(tyoll_virta1.shape)
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
        #print(n1.shape,tyoll_virta1.shape)
        self.plot_compare_virrat(n1,n2,virta_label='netto',label1=label1,label2=label2,ymin=-1000,ymax=1000)
        self.plot_compare_csvirta(n1,n2,'cumsum nettovirta')

    def plot_unemp_durdistribs(self,kestot):
        if len(kestot.shape)>2:
            m1=self.empdur_to_dict(np.mean(kestot,axis=0))
        else:
            m1=self.empdur_to_dict(kestot)

        df = pd.DataFrame.from_dict(m1,orient='index',columns=['0-6 kk','6-12 kk','12-18 kk','18-24kk','yli 24 kk'])
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=",.2f"))

    def plot_compare_unemp_durdistribs(self,kestot1,kestot2,viimekesto1,viimekesto2,label1='',label2=''):
        print('Keskikestot käytettyjen ansiosidonnaisten päivärahojen mukaan')
        self.plot_unemp_durdistribs(kestot1)
        self.plot_unemp_durdistribs(kestot2)

        print('Keskikestot viimeisimmän työttömyysjakson mukaan')
        self.plot_unemp_durdistribs(viimekesto1)
        self.plot_unemp_durdistribs(viimekesto2)
        
    def comp_aggkannusteet(self,ben,min_salary=0,max_salary=6000,step_salary=50,n=None,savefile=None):
        n_salary=int((max_salary-min_salary)/step_salary)
        netto=np.zeros(n_salary)
        palkka=np.zeros(n_salary)
        tva=np.zeros(n_salary)
        osa_tva=np.zeros(n_salary)
        eff=np.zeros(n_salary)
        
        if n is None:
            n=self.n_pop
        
        tqdm_e = tqdm(range(int(n)), desc='Population', leave=True, unit=" p")
        
        num=0    
        for popp in range(n):
            for t in np.arange(0,self.n_time,10):
                employment_state=int(self.popempstate[t,popp])
            
                if employment_state in set([0,1,4,7,10,13]):
                    if employment_state in set([0,4]):
                        old_wage=self.infostats_unempwagebasis_acc[t,popp]
                    elif employment_state in set([13]):
                        old_wage=0
                    else:
                        old_wage=self.infostats_unempwagebasis[t,popp]
                        
                    if self.infostats_toe[t,popp]>0.5:
                        toe=1
                    else:
                        toe=0
                    wage=self.salaries[t,popp]
                    children_under3=int(self.infostats_children_under3[t,popp])
                    children_under7=int(self.infostats_children_under7[t,popp])
                    children_under18=children_under7
                    ika=self.map_t_to_age(t)
                    num=num+1
                    p=self.setup_p(wage,old_wage,0,employment_state,0,
                        children_under3,children_under7,children_under18,ika)
                        #irtisanottu=0,tyohistoria=0,karenssia_jaljella=0)
                    p2=self.setup_p_for_unemp(p,old_wage,toe,employment_state)
                    nettox,effx,tvax,osa_tvax=ben.comp_insentives(p=p,p2=p2,min_salary=min_salary,max_salary=max_salary,
                        step_salary=step_salary,dt=100)
                    netto+=nettox
                    eff+=effx
                    tva+=tvax
                    osa_tva+=osa_tvax
                    
                    #print('p={}\np2={}\nold_wage={}\ne={}\ntoe={}\n'.format(p,p2,old_wage,employment_state,toe))
                    #print('ika={} popp={} old_wage={} e={} toe={}\n'.format(ika,popp,old_wage,employment_state,toe))
                    
            tqdm_e.update(1)
            tqdm_e.set_description("Pop " + str(popp))

        netto=netto/num
        eff=eff/num
        tva=tva/num
        osa_tva=osa_tva/num
        
        if savefile is not None:
            f = h5py.File(savefile, 'w')
            ftype='float64'
            _ = f.create_dataset('netto', data=netto, dtype=ftype)
            _ = f.create_dataset('eff', data=eff, dtype=ftype)
            _ = f.create_dataset('tva', data=tva, dtype=ftype)
            _ = f.create_dataset('osa_tva', data=osa_tva, dtype=ftype)
            _ = f.create_dataset('min_salary', data=min_salary, dtype=ftype)
            _ = f.create_dataset('max_salary', data=max_salary, dtype=ftype)
            _ = f.create_dataset('step_salary', data=step_salary, dtype=ftype)
            _ = f.create_dataset('n', data=n, dtype=ftype)
            f.close()        


    def plot_aggkannusteet(self,ben,loadfile,baseloadfile=None,figname=None,label=None,baselabel=None):
        f = h5py.File(loadfile, 'r')
        netto=f.get('netto').value
        eff=f.get('eff').value
        tva=f.get('tva').value
        osa_tva=f.get('osa_tva').value
        min_salary=f.get('min_salary').value
        max_salary=f.get('max_salary').value
        step_salary=f.get('step_salary').value
        n=f.get('n').value
        f.close()        

        if baseloadfile is not None:
            f = h5py.File(baseloadfile, 'r')
            basenetto=f.get('netto').value
            baseeff=f.get('eff').value
            basetva=f.get('tva').value
            baseosatva=f.get('osa_tva').value
            f.close()        
            
            ben.plot_insentives(netto,eff,tva,osa_tva,min_salary=min_salary,max_salary=max_salary,figname=figname,
                step_salary=step_salary,basenetto=basenetto,baseeff=baseeff,basetva=basetva,baseosatva=baseosatva,
                otsikko=label,otsikkobase=baselabel)
        else:
            ben.plot_insentives(netto,eff,tva,osa_tva,min_salary=min_salary,max_salary=max_salary,figname=figname,
                step_salary=step_salary,otsikko=label,otsikkobase=baselabel)
            
    def setup_p(self,wage,old_wage,pension,employment_state,time_in_state,
                children_under3,children_under7,children_under18,ika,
                irtisanottu=0,tyohistoria=0,karenssia_jaljella=0,include_children=0):
        '''
        asettaa p:n parametrien mukaiseksi
        '''
        p={}

        p['opiskelija']=0
        p['toimeentulotuki_vahennys']=0
        p['ika']=ika
        p['tyoton']=0
        p['saa_ansiopaivarahaa']=0
        p['vakiintunutpalkka']=0
        
        if include_children:
            p['lapsia']=children_under18
            p['lapsia_paivahoidossa']=children_under7
            p['lapsia_alle_kouluikaisia']=children_under7
            p['lapsia_alle_3v']=children_under3
        else:
            p['lapsia']=children_under3
            p['lapsia_paivahoidossa']=children_under3
            p['lapsia_alle_kouluikaisia']=children_under3
            p['lapsia_alle_3v']=children_under3
        p['aikuisia']=1
        p['veromalli']=0
        p['kuntaryhma']=3
        p['lapsia_kotihoidontuella']=0
        p['tyottomyyden_kesto']=0
        p['puoliso_tyottomyyden_kesto']=10
        p['isyysvapaalla']=0
        p['aitiysvapaalla']=0
        p['kotihoidontuella']=0
        p['tyoelake']=0
        p['elakkeella']=0
        p['sairauspaivarahalla']=0
        p['disabled']=0
        if employment_state==1:
            p['tyoton']=0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p['t']=wage/12
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=0
        elif employment_state==0: # työtön, ansiopäivärahalla
            if ika<65:
                #self.render()
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['saa_ansiopaivarahaa']=1
                p['tyottomyyden_kesto']=time_in_state
                    
                if irtisanottu<1 and karenssia_jaljella>0:
                    p['saa_ansiopaivarahaa']=0
                    p['tyoton']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_state==13: # työmarkkinatuki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=0
                p['tyottomyyden_kesto']=12*21.5*time_in_state
                p['saa_ansiopaivarahaa']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_state==3: # tk
            p['t']=0
            p['elakkeella']=1 
            #p['elake']=pension
            p['tyoelake']=pension/12
            p['disabled']=1
        elif employment_state==4: # työttömyysputki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['saa_ansiopaivarahaa']=1
                p['tyottomyyden_kesto']=12*21.5*time_in_state
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_state==5: # ansiosidonnainen vanhempainvapaa, äidit
            p['aitiysvapaalla']=1
            p['aitiysvapaa_kesto']=0
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_state==6: # ansiosidonnainen vanhempainvapaa, isät
            p['isyysvapaalla']=1
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_state==7: # hoitovapaa
            p['kotihoidontuella']=1
            if include_children:
                p['lapsia_paivahoidossa']=0
                p['lapsia_kotihoidontuella']=children_under7
            else:
                p['lapsia_paivahoidossa']=0
                p['lapsia_kotihoidontuella']=children_under3
            p['kotihoidontuki_kesto']=time_in_state
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
        elif employment_state==2: # vanhuuseläke
            if ika>=self.min_retirementage:
                p['t']=0
                p['elakkeella']=1  
                p['tyoelake']=pension/12
            else:
                p['t']=0
                p['elakkeella']=0
                p['tyoelake']=0
        elif employment_state==8: # ve+osatyö
            p['t']=wage/12
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_state==9: # ve+työ
            p['t']=wage/12
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_state==10: # osa-aikatyö
            p['t']=wage/12
        elif employment_state==11: # työelämän ulkopuolella
            p['toimeentulotuki_vahennys']=0 # oletetaan että ei kieltäytynyt työstä
            p['t']=0
        elif employment_state==12: # opiskelija
            p['opiskelija']=1
            p['t']=0
        #elif employment_state==14: # armeijassa, korjaa! ei tosin vaikuta tuloksiin.
        #    p['opiskelija']=1
        #    p['t']=0
        elif employment_state==14: # kuollut
            p['t']=0
        else:
            print('Unknown employment_state ',employment_state)

        # tarkastellaan yksinasuvia henkilöitä
        if employment_state==12: # opiskelija
            p['asumismenot_toimeentulo']=250
            p['asumismenot_asumistuki']=250
        elif employment_state in set([2,8,9]): # eläkeläinen
            p['asumismenot_toimeentulo']=200
            p['asumismenot_asumistuki']=200
        else: # muu
            p['asumismenot_toimeentulo']=320 # per hlö, ehkä 500 e olisi realistisempi, mutta se tuottaa suuren asumistukimenon
            p['asumismenot_asumistuki']=320

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1
        p['puoliso_tulot']=0
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0
        
        return p
        
    def setup_p_for_unemp(self,p,old_wage,toe,employment_state):
        '''
        asettaa p:n parametrien mukaiseksi
        '''
        if employment_state in set([1,10]):
            p['tyoton']=1 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            if toe>0:
                p['saa_ansiopaivarahaa']=1
            else:
                p['saa_ansiopaivarahaa']=0
        else:
            pass
        
        return p