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
from tqdm.auto import tqdm # notebookissa widget, skriptissä tekstipalkki
from .episodestats import EpisodeStats
import fin_benefits
    
class SimStats(EpisodeStats):
    def run_simstats(self,results,save,n,startn=0,max_age=65,singlefile=False,grouped=False,group=0,include_distrib=True):
        '''
        Laskee statistiikat ajoista

        singlefile=True: yksi tiedosto results
        muuten ajot results+'_repeat_'+str(i)+'_combined', i = startn,...,n-1
        '''
        
        print('computing simulation statistics...')
        if singlefile:
            run_files = [results]
        else:
            run_files = [results+'_repeat_'+str(i)+'_combined' for i in range(startn,n)]
        n_runs = len(run_files)

        e_rate=np.zeros((n_runs,self.n_time))
        agg_htv=np.zeros(n_runs)
        agg_tyoll=np.zeros(n_runs)
        agg_tyottomyysaste=np.zeros(n_runs)
        agg_rew=np.zeros(n_runs)
        agg_discounted_rew=np.zeros(n_runs)
        t_aste=np.zeros(n_runs)
        emps=np.zeros((n_runs,self.n_time,self.n_states))
        gemps=np.zeros((n_runs,self.n_time,self.n_states,self.n_groups))
        emp_tyolliset=np.zeros((n_runs,self.n_time))
        emp_tyottomat=np.zeros((n_runs,self.n_time))
        emp_tyottomyysaste=np.zeros((n_runs,self.n_time))
        emp_tyolliset_osuus=np.zeros((n_runs,self.n_time))
        emp_tyottomat_osuus=np.zeros((n_runs,self.n_time))
        emp_htv=np.zeros((n_runs,self.n_time))
        tyoll_virta=np.zeros((n_runs,self.n_time))
        tyot_virta=np.zeros((n_runs,self.n_time))
        tyot_virta_ansiosid=np.zeros((n_runs,self.n_time))
        tyot_virta_tm=np.zeros((n_runs,self.n_time))
        unemp_dur=np.zeros((n_runs,5,5))
        unemp_lastdur=np.zeros((n_runs,5,5))
        agg_netincome=np.zeros(n_runs)
        agg_equivalent_netincome=np.zeros(n_runs)
        alives=np.zeros((n_runs,self.n_time),dtype = np.int64)
        galives=np.zeros((n_runs,self.n_time,self.n_groups),dtype = np.int64)
        agg_alives = np.zeros((self.n_time,1),dtype = np.float64)
        agg_empstate = np.zeros((self.n_time,self.n_states),dtype = np.float64)
        agg_galives = np.zeros((self.n_time,self.n_groups),dtype = np.float64)
        agg_gempstate = np.zeros((self.n_time,self.n_states,self.n_groups),dtype = np.float64)
        pt_agg = np.zeros((n_runs,6),dtype = np.float64)
        pt_agegroup = np.zeros((n_runs,self.n_time,6),dtype = np.float64)

        unemp_distrib,emp_distrib,unemp_distrib_bu = [],[],[]
        unemp_basis_distrib = []
        tyoll_distrib,tyoll_distrib_bu = [],[]
        budget,htv_budget,participants = None,None,None

        include_ansiosid = True
        include_tmtuki = False
        include_putki = True

        best_rew = -np.inf
        best_emp = 0

        tqdm_e = tqdm(range(n_runs), desc='Sim', leave=True, unit=" ")
        for j,filename in enumerate(run_files):
            self.load_sim(filename,print_pop=False)

            q=self.comp_budget(scale=True)
            budget2 = pd.DataFrame.from_dict(q,orient='index',columns=['e/v'])
            q=self.comp_participants(scale=True,lkm=False)
            htv2 = pd.DataFrame.from_dict(q,orient='index',columns=['htv'])
            q_lkm=self.comp_participants(scale=True,lkm=True)
            participants2 = pd.DataFrame.from_dict(q_lkm,orient='index',columns=['lkm'])
            if j == 0:
                budget,htv_budget,participants = budget2,htv2,participants2
            else:
                budget += budget2
                htv_budget += htv2
                participants += participants2

            if grouped:
                empstate = self.gempstate[:,:,group]
            else:
                empstate = self.empstate

            emps[j,:,:] = self.empstate
            gemps[j,:,:,:] = self.gempstate
            agg_empstate += self.empstate
            agg_gempstate += self.gempstate
            alives[j,:] = self.alive[:,0]
            galives[j,:,:] = self.galive[:,:]
            agg_discounted_rew[j],agg_rew[j] = self.get_reward()

            if include_distrib:
                agg_netincome[j],agg_equivalent_netincome[j] = self.comp_total_netincome()

            if agg_rew[j]>best_rew:
                best_rew = agg_rew[j]
                best_emp = j

            emp_htv2 = np.sum(self.emp_htv,axis=1)

            htv_ika,tyolliset_ika,tyottomat_ika,osatyolliset_ika,kokotyollvaikutus_ika,tyoll_osuus,osatyoll_osuus,\
                kokotyoll_osuus,tyottomien_osuus,tyottomyys_aste = self.comp_unemp_simstats_agegroups(empstate,start=self.min_age,end=self.max_age,scale_time=True,emp_htv=emp_htv2)
            htv,tyolliset,tyottomat,osatyolliset,kokotyollvaikutus,tyollaste,osatyollaste,\
                kokotyollaste,kokotyottomyysaste = self.comp_unemp_simstats_aggregate(empstate,scale_time=True,start=self.min_age,end=self.max_age,emp_htv=emp_htv2)

            agg_htv[j]=htv
            agg_tyoll[j]=tyolliset
            agg_tyottomyysaste[j]=kokotyottomyysaste
            t_aste[j]=tyollaste

            emp_tyolliset[j,:] = tyolliset_ika[:,0]
            emp_tyottomat[j,:] = tyottomat_ika[:,0]
            emp_tyottomyysaste[j,:] = tyottomyys_aste[:,0]
            emp_tyolliset_osuus[j,:] = tyoll_osuus[:,0]
            emp_tyottomat_osuus[j,:] = tyottomien_osuus[:,0]
            emp_htv[j,:] = htv_ika[:,0]

            if include_distrib:
                # samat jakaumafunktiot kaikille ajoille
                unemp_distrib2,emp_distrib2,unemp_distrib_bu2 = self.comp_empdistribs_v2(ansiosid=include_ansiosid,tmtuki=include_tmtuki,putki=include_putki,outsider=False,max_age=max_age)
                tyoll_distrib2,tyoll_distrib_bu2 = self.comp_tyollistymisdistribs(ansiosid=include_ansiosid,tmtuki=include_tmtuki,putki=include_putki,outsider=False,max_age=max_age)
                unemp_basis_distrib2 = self.comp_unempbasis_distribs(ansiosid=include_ansiosid,putki=True,max_age=max_age)

                unemp_distrib.extend(unemp_distrib2)
                unemp_basis_distrib.extend(unemp_basis_distrib2)
                emp_distrib.extend(emp_distrib2)
                unemp_distrib_bu.extend(unemp_distrib_bu2)
                tyoll_distrib.extend(tyoll_distrib2)
                tyoll_distrib_bu.extend(tyoll_distrib_bu2)
        
                # virrat työllisyyteen ja työttömyyteen
                tyoll_virta0,tyot_virta0 = self.comp_virrat(ansiosid=True,tmtuki=True,putki=True,outsider=False)
                tyoll_virta_ansiosid0,tyot_virta_ansiosid0 = self.comp_virrat(ansiosid=True,tmtuki=False,putki=True,outsider=False)
                tyoll_virta_tm0,tyot_virta_tm0 = self.comp_virrat(ansiosid=False,tmtuki=True,putki=False,outsider=False)

                tyoll_virta[j,:] = tyoll_virta0[:,0]
                tyot_virta[j,:] = tyot_virta0[:,0]
                tyot_virta_ansiosid[j,:] = tyot_virta_ansiosid0[:,0]
                tyot_virta_tm[j,:] = tyot_virta_tm0[:,0]

                unemp_dur[j,:,:] = self.comp_unemp_durations(return_q=False)
                unemp_lastdur[j,:,:] = self.comp_unemp_durations_v2(return_q=False)

            _,_,_,_,_,_,combisuhde,agg_combisuhde = self.comp_ptproportions()
            pt_agg[j,:] = agg_combisuhde[0,:]
            pt_agegroup[j,:,:] = combisuhde[0,:,:]

            tqdm_e.update(1)
            tqdm_e.set_description("Sim " + str(j+1))

        budget /= n_runs
        participants /= n_runs
        htv_budget /= n_runs
        agg_empstate = agg_empstate / n_runs
        agg_gempstate = agg_gempstate / n_runs
        agg_alives[:,0] = np.sum(alives,axis=0)/n_runs
        agg_galives[:,:] = np.sum(galives,axis=0)/n_runs

        self.save_simstats(save,agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,\
                            emp_tyolliset,emp_tyolliset_osuus,\
                            emp_tyottomat,emp_tyottomat_osuus,\
                            emp_htv,emps,\
                            best_rew,best_emp,\
                            unemp_distrib,emp_distrib,unemp_distrib_bu,\
                            tyoll_distrib,tyoll_distrib_bu,\
                            tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
                            unemp_dur,unemp_lastdur,agg_netincome,agg_equivalent_netincome,\
                            budget,participants,htv_budget,\
                            alives,agg_empstate,agg_alives,agg_tyottomyysaste,emp_tyottomyysaste,\
                            pt_agg,pt_agegroup,galives,agg_galives,gemps,agg_gempstate,\
                            unemp_basis_distrib)
                    
        print('done')
        
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
 
    def get_simstats(self,filename1,use_mean=True):
        agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
        emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,\
        best_emp,emps,agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,alives,agg_empstate,agg_alives,\
        agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,galives,agg_galives,gempstate,agg_gempstate\
            =self.load_simstats(filename1)

        mean_htv=np.mean(agg_htv)
        median_htv=np.median(agg_htv)
        mean_tyoll=np.mean(agg_tyoll)
        median_tyoll=np.median(agg_tyoll)
        std_htv=np.std(agg_htv)
        std_tyoll=np.std(agg_tyoll) # ajojen välinen hajonta
        diff_htv=agg_htv-mean_htv
        diff_tyoll=agg_tyoll-mean_tyoll
        mean_rew = np.mean(agg_rew)
        median_rew = np.median(agg_rew)
        
        m_mean=np.mean(emp_tyolliset_osuus,axis=0)
        m_median=np.median(emp_tyolliset_osuus,axis=0)
        mn_median=np.median(emp_tyolliset,axis=0)
        mn_mean=np.mean(emp_tyolliset,axis=0)
        s_tyoll=np.std(emp_tyolliset_osuus,axis=0)
        m_best=emp_tyolliset_osuus[best_emp,:]

        h_mean=np.mean(emp_htv,axis=0)
        h_median=np.median(emp_htv,axis=0)
        h_std=np.std(emp_htv,axis=0)
        h_best=emp_htv[best_emp,:]

        median_tyott = np.median(emp_tyottomat_osuus,axis=0)
        mean_tyott_osuus = np.mean(emp_tyottomat_osuus,axis=0)
        mean_tyolliset_osuus = np.mean(emp_tyolliset_osuus,axis=0)
        mean_unempratio = np.mean(emp_tyottomyysaste,axis=0)
        median_unempratio = np.median(emp_tyottomyysaste,axis=0)
    
        if use_mean:
            return mean_htv,mean_tyoll,h_mean,m_mean,diff_htv,mean_rew,mean_tyott_osuus,mean_tyolliset_osuus,std_htv,h_std,mean_unempratio
        else:
            return median_htv,median_tyoll,h_median,m_median,median_rew,median_tyott,std_tyoll,s_tyoll,median_unempratio
            
    def put_df5(self,f,htv,nimi):
        dt = h5py.special_dtype(vlen=str)
        _ = f.create_dataset(f'{nimi}_values', data=htv.values)
        _ = f.create_dataset(f'{nimi}_columns', data=htv.columns.values, dtype=dt)
        _ = f.create_dataset(f'{nimi}_index', data=htv.index.values, dtype=dt)

    def save_simstats(self,filename,agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
                        emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,\
                        unemp_distrib,emp_distrib,unemp_distrib_bu,\
                        tyoll_distrib,tyoll_distrib_bu,\
                        tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
                        unemp_dur,unemp_lastdur,agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,\
                        alives,agg_empstate,agg_alives,agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,\
                        galives,agg_galives,gempstate,agg_gempstate,\
                        unemp_basis_distrib):
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
        dset = f.create_dataset('emps', data=emps, dtype='int64')
        dset = f.create_dataset('gemps', data=gempstate, dtype='int64')
        dset = f.create_dataset('best_rew', data=best_rew, dtype='float64')
        dset = f.create_dataset('best_emp', data=best_emp, dtype='float64')
        dset = f.create_dataset('unemp_distrib', data=unemp_distrib, dtype='float64')
        dset = f.create_dataset('emp_distrib', data=emp_distrib, dtype='float64')
        dset = f.create_dataset('unemp_distrib_bu', data=unemp_distrib_bu, dtype='float64')
        dset = f.create_dataset('unemp_basis_distrib', data=unemp_basis_distrib, dtype='float64')
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
        dset = f.create_dataset('alives', data=alives, dtype='int64')
        dset = f.create_dataset('galives', data=galives, dtype='int64')
        dset = f.create_dataset('agg_empstate', data=agg_empstate, dtype='float64')
        dset = f.create_dataset('agg_gempstate', data=agg_gempstate, dtype='float64')
        dset = f.create_dataset('agg_alives', data=agg_alives, dtype='float64')
        dset = f.create_dataset('agg_galives', data=agg_galives, dtype='float64')        
        dset = f.create_dataset('agg_tyottomyysaste', data=agg_tyottomyysaste, dtype='float64')
        dset = f.create_dataset('emp_tyottomyysaste', data=emp_tyottomyysaste, dtype='float64')
        dset = f.create_dataset('pt_agg', data=pt_agg, dtype='float64')
        dset = f.create_dataset('pt_agegroup', data=pt_agegroup, dtype='float64')
        store = pd.HDFStore(filename+"_df.h5", 'w')  
        store.put('participants', participants, format='table')  
        store.put('budget', budget, format='table')  
        store.put('htv_budget', htv_budget, format='table')  
        store.close()
        #self.put_df5(f,participants,'participants')
        #self.put_df5(f,budget,'budget')
        #self.put_df5(f,htv_budget,'htv_budget')
        f.close()

    def get_df5(self,f,nimi):
        participants_values = f[f'{nimi}_values'][()]
        participants_index = f[f'{nimi}_index'][()].astype(str)
        participants_columns = f[f'{nimi}_columns'][()].astype(str)
        participants = pd.DataFrame(participants_values,index=participants_index,columns=participants_columns)
        return participants

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
        agg_empstate = f['agg_empstate'][()]
        alives = f['alives'][()]
        galives = f['galives'][()]
        agg_alives = f['agg_alives'][()]
        agg_galives = f['agg_galives'][()]
        agg_tyottomyysaste = f['agg_tyottomyysaste'][()]
        emp_tyottomyysaste = f['emp_tyottomyysaste'][()]
        pt_agg = f['pt_agg'][()]
        pt_agegroup = f['pt_agegroup'][()]
        gempstate = f['gemps'][()]
        agg_gempstate = f['agg_gempstate'][()]

        store = pd.HDFStore(filename+"_df.h5", 'r')  
        budget = store.get('budget')  
        participants = store.get('participants')  
        htv_budget = store.get('htv_budget')  
        store.close()

        #budget = self.get_df5(f,'budget')
        #participants = self.get_df5(f,'participants')
        #htv_budget = self.get_df5(f,'htv_budget')
        
        f.close()

        return agg_htv,agg_tyoll,agg_rew,agg_discounted_rew,emp_tyolliset,emp_tyolliset_osuus,\
               emp_tyottomat,emp_tyottomat_osuus,emp_htv,emps,best_rew,best_emp,emps,\
               agg_netincome,agg_equivalent_netincome,budget,participants,htv_budget,alives,agg_empstate,agg_alives,\
               agg_tyottomyysaste,emp_tyottomyysaste,pt_agg,pt_agegroup,galives,agg_galives,gempstate,agg_gempstate

    def load_simdistribs(self,filename):
        f = h5py.File(filename, 'r')
        if 'tyoll_virta' in f:
            unemp_distrib = f['unemp_distrib'][()] 
        else:
            unemp_distrib=np.zeros((self.n_time,self.n_pop))
        
        if 'tyoll_virta' in f:
            emp_distrib = f['emp_distrib'][()] 
        else:
            emp_distrib=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            unemp_distrib_bu = f['unemp_distrib_bu'][()] 
        else:
            unemp_distrib_bu=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            tyoll_distrib =f['tyoll_distrib'][()] 
        else:
            tyoll_distrib=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            tyoll_distrib_bu = f['tyoll_distrib_bu'][()] 
        else:
            tyoll_distrib_bu=np.zeros((self.n_time,self.n_pop))
        if 'tyoll_virta' in f:
            tyoll_virta = f['tyoll_virta'][()] 
        else:
            tyoll_virta=np.zeros((self.n_time,self.n_pop))
        if 'tyot_virta' in f:
            tyot_virta = f['tyot_virta'][()] 
        else:
            tyot_virta=np.zeros((self.n_time,self.n_pop))
        if 'tyot_virta_ansiosid' in f:
            tyot_virta_ansiosid = f['tyot_virta_ansiosid'][()] 
        else:
            tyot_virta_ansiosid=np.zeros((self.n_time,self.n_pop))
        if 'tyot_virta_tm' in f:
            tyot_virta_tm = f['tyot_virta_tm'][()] 
        else:
            tyot_virta_tm=np.zeros((self.n_time,self.n_pop))
        if 'unemp_dur' in f:
            unemp_dur = f['unemp_dur'][()]
        else:
            unemp_dur=np.zeros((1,5,5))
        if 'unemp_lastdur' in f:
            unemp_lastdur = f['unemp_lastdur'][()]
        else:
            unemp_lastdur=np.zeros((1,5,5))
        if 'unemp_basis_distrib' in f:
            unemp_basis_distrib = f['unemp_basis_distrib'][()]
        else:
            unemp_basis_distrib = np.zeros((self.n_time,self.n_pop))

        f.close()

        return unemp_distrib,emp_distrib,unemp_distrib_bu,\
               tyoll_distrib,tyoll_distrib_bu,\
               tyoll_virta,tyot_virta,tyot_virta_ansiosid,tyot_virta_tm,\
               unemp_dur,unemp_lastdur,unemp_basis_distrib

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

    def comp_elasticity(self,x,y,diff=False):
        xl=x.shape[0]
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
            
        return el,elx        