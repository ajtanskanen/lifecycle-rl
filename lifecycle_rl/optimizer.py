'''

Bayesian optimization for lifecycle models

The aim is to reproduce employment rate at each age

'''

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from pathlib import Path
from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .lifecycle_v2 import Lifecycle
        
class BalanceLifeCycle():
    def __init__(self,initargs=None,runargs=None,ref_muut=9.5e9,additional_income_tax=0,
            additional_kunnallisvero=0,additional_tyel_premium=0,additional_vat=0,
            logs=None,reset=False,debug=True):
        '''
        Alusta muuttujat
        '''
        self.runargs=runargs
        self.initargs=initargs
        self.ref_muut=ref_muut
        self.additional_income_tax=additional_income_tax
        self.additional_kunnallisvero=additional_kunnallisvero
        self.additional_tyel_premium=additional_tyel_premium
        self.additional_vat=additional_vat
        
        if logs is not None:
            self.logs=logs
        else:
            if debug:
                logs='test'
                #reset=True
            else:
                logs='log'

        LOG_DIR = Path().absolute() / 'bayes_opt_logs'
        #LOG_DIR.mkdir(exist_ok=True)
        
        num=int(self.additional_income_tax*100)
        self.logs=str(LOG_DIR / logs) + '_' + str(num) + '.json'
        
        pbounds={'x':(additional_income_tax-0.30,additional_income_tax+0.30)}

        if debug:
            f=self.test_black_box_function
        else:
            f=self.black_box_function

        self.optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1
        )
        from sklearn.gaussian_process.kernels import Matern,WhiteKernel
        
        kwargs={'alpha': 0.2, 'n_restarts_optimizer': 3,'kernel': Matern(nu=0.5)+WhiteKernel(noise_level=0.1)}
        self.optimizer.set_gp_params(**kwargs)
        
        # talletus
        logfile = self.logs
        logger = JSONLogger(path=logfile,reset=reset) #,reset=reset)

        if Path(logfile).exists():
            self.load_logs(self.optimizer)
        else:
            print(f'{logfile} not found')
            
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        
    def black_box_function(self,**x):
        """
        Function with unknown internals we wish to maximize.
        """
        print(x)
        initargs2=self.initargs
        initargs2['extra_ppr']=x['x']
        initargs2['additional_income_tax']=self.additional_income_tax
        initargs2['additional_kunnallisvero']=self.additional_kunnallisvero
        initargs2['additional_tyel_premium']=self.additional_tyel_premium
        initargs2['additional_vat']=self.additional_vat
        repeats=1
        err=np.empty(repeats)
        for r in range(repeats):
            cc=Lifecycle(**initargs2)
            cc.run_results(**self.runargs)
            err[r]=cc.L2BudgetError(self.ref_muut)
            
        ave=np.nanmean(err)
        print(f'ave {ave}')
            
        return ave

    def test_black_box_function(self,**x):
        """
        Function with unknown internals we wish to maximize.
        """
        initargs2=self.initargs
        initargs2['extra_ppr']=x['x']
        initargs2['additional_income_tax']=self.additional_income_tax
        e=np.random.normal(0,0.01)
        err=-(x['x']+e-0.1)**2
            
        print(f"{x['x']} ave {err}")
            
        return err
        
    def load_logs(self,optimizer):
        logfile=self.logs
        load_logs(self.optimizer, logs=[logfile]);
        
        print('loading',logfile)
        
        x_obs = np.array([[res["params"]["x"]] for res in self.optimizer.res])
        y_obs = np.array([res["target"] for res in self.optimizer.res])
        print(x_obs,y_obs)
        self.optimizer._gp.fit(x_obs, y_obs)
    
    def plot_confidence_intervals(self,min_ppr=-0.3,max_ppr=0.3):

        #logger = JSONLogger(path='foo')
        #self.load_logs(self.optimizer)
        
        print("Optimizer is now aware of {} points.".format(len(self.optimizer.space)))
    
        # Range of x to obtain the confidence intervals.
        x = np.linspace(min_ppr, max_ppr)
        
        # Obtain the corresponding mean and standard deviations.
        y_pred, y_std = self.optimizer._gp.predict(x.reshape(-1, 1), return_std=True)
    
        #y_pred = np.array(y_pred)
        #y_std = np.array(y_std)
        plt.figure(figsize = (10, 5))
        plt.plot(x, y_pred, "b-")
        y1=(y_pred-1.96*y_std).squeeze()
        y2=(y_pred+1.96*y_std).squeeze()
        print(y1.shape,y2.shape)
        plt.fill_between(x,y1,y2)
        plt.xlabel("$x$", fontsize = 14)
        plt.ylabel("$f(x)$", fontsize = 14)
        plt.legend(["$y = x^2$", "Observations", "Predictions", "95% Confidence Interval"], fontsize = 14)
        plt.grid(True)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()    
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        axis = plt.subplot(gs[0])
        acq = plt.subplot(gs[1])
    
        x_obs = np.array([[res["params"]["x"]] for res in self.optimizer.res])
        y_obs = np.array([res["target"] for res in self.optimizer.res])
    
        mu, sigma = y_pred, y_std
        #axis.plot(x, y, linewidth=3, label='Target')
        axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
        axis.plot(x, mu, '--', color='k', label='Prediction')

        axis.fill(np.concatenate([x, x[::-1]]), 
                  np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
        acq.set_xlim((-0.4, 0.4))
        axis.set_ylim((None, None))
        axis.set_ylabel('f(x)', fontdict={'size':20})
        axis.set_xlabel('x', fontdict={'size':20})
    
        utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
        utility = utility_function.utility(x.reshape(-1, 1), self.optimizer._gp, 0)
        acq.plot(x, utility, label='Utility Function', color='purple')
        acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
                 label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        acq.set_xlim((-0.4, 0.4))
        acq.set_ylim((0, np.max(utility) + 0.5))
        acq.set_ylabel('Utility', fontdict={'size':20})
        acq.set_xlabel('x', fontdict={'size':20})
    
        axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)        
        print(self.optimizer._gp.log_marginal_likelihood_value_)

    def optimize(self,reset=False,min_ppr=-0.3,max_ppr=0.3,debug=False,init_points=2,n_iter=5,
        fsnum=None,kappa=10,xi=1e-2):
        # Bounded region of parameter space
        pbounds = {'x': (min_ppr, max_ppr)} #, 'women_mu_scale': (0.01,0.3), 'women_mu_age': (57,62)}
        
        self.optimizer.maximize(init_points=5, n_iter=n_iter, acq='ucb', kappa=kappa, alpha=1.0)
        #self.optimizer.maximize(init_points=5, n_iter=n_iter, acq="ei", xi=1e-1)
        #self.optimizer.maximize(init_points=15, n_iter=n_iter, acq="poi", xi=1e-3, alpha=1)
        #self.optimizer.maximize(init_points=10, n_iter=n_iter, acq="ucb", kappa=10)
        

        print('The best parameters found {}'.format(self.optimizer.max))
        
        return self.optimizer.max

