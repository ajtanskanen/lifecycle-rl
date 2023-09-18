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
import fin_benefits
    
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
            base_empstate=self.gempstate[:,:,group]/self.alive #/self.n_pop
        else:
            base_empstate=self.empstate/self.alive #self.n_pop
        
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
                    empstate=self.gempstate[:,:,group]/self.alive #self.n_pop
                else:
                    empstate=self.empstate/self.alive #self.n_pop
                
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
        
    def count_putki_dist(self,emps):
        putki=[]
    
        for k in range(emps.shape[0]):
            putki.append(self.count_putki(emps[k,:,:]))
            
        putkessa=np.median(np.asarray(putki))
        return putkessa        
 
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

    def comp_aggkannusteet(self,ben,min_salary=0,max_salary=6000,step_salary=50,n=None,savefile=None):
        n_salary=int((max_salary-min_salary)/step_salary)+1
        netto=np.zeros(n_salary)
        palkka=np.zeros(n_salary)
        tva=np.zeros(n_salary)
        osa_tva=np.zeros(n_salary)
        eff=np.zeros(n_salary)
        
        marg=fin_benefits.Marginals(ben,year=self.year)
        
        if n is None:
            n=self.n_pop
        
        tqdm_e = tqdm(range(int(n)), desc='Population', leave=True, unit=" p")
        
        num=0    
        for popp in range(n):
            for t in np.arange(1,self.n_time,10):
                employment_state=int(self.popempstate[t,popp])
            
                if employment_state in set([0,1,4,7,10,13,14]):
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
                        
                    wage=self.infostats_pop_wage[t,popp]
                    children_under3=int(self.infostats_children_under3[t,popp])
                    children_under7=int(self.infostats_children_under7[t,popp])
                    children_under18=children_under7
                    
                    print(f'{t}: e {employment_state} w {wage} ow {old_wage} c3 {children_under3} c7 {children_under7} c18 {children_under18}')
                    
                    ika=self.map_t_to_age(t)
                    num=num+1
                    
                    # ei huomioi parisuhteita! FIXME
                    if employment_state in set([1,10]): # töissä
                        p=self.setup_p(wage,old_wage,0,employment_state,0,
                            children_under3,children_under7,children_under18,ika)
                            #irtisanottu=0,tyohistoria=0,karenssia_jaljella=0)
                        p2=self.setup_p_for_unemp(p,old_wage,toe,employment_state)
                    else: # ei töissä
                        p=self.setup_p(wage,old_wage,0,employment_state,0,
                            children_under3,children_under7,children_under18,ika)
                            #irtisanottu=0,tyohistoria=0,karenssia_jaljella=0)
                        p2=p.copy()
                        
                    nettox,effx,tvax,osa_tvax,_=marg.comp_insentives(p0=p2,p=p,min_salary=min_salary,max_salary=max_salary,
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

    def plot_agg_emtr(self,ben,loadfile,baseloadfile=None,figname=None,label=None,baselabel=None):
        f = h5py.File(loadfile, 'r')
        netto=f['netto'][()]
        eff=f['eff'][()]
        tva=f['tva'][()]
        osa_tva=f['osa_tva'][()]
        min_salary=f['min_salary'][()]
        max_salary=f['max_salary'][()]
        salary=f['salary'][()]
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

        plt.hist(eff)