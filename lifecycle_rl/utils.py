'''
Utils
'''

import gym
import numpy as np
from gym import spaces, logger, utils, error
from gym.utils import seeding
from IPython.core.display import display,HTML
#from stable_baselines.common import set_global_seeds

def make_env(env_id, rank, kwargs, seed=None, use_monitor=True):
    """
    Utility function for multiprocessed env.#

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id,kwargs=kwargs)
        if seed is not None:
            env.seed(seed + rank)
            env.env_seed(seed + rank + 100)

        # monitor enables various things, not used by default
        #print('monitor=',use_monitor)
        #if use_monitor:
        #    env = Monitor(env, self.log_dir, allow_early_resets=True)

        return env

#    if seed is not None:
#        set_global_seeds(seed)

    return _init()

def empirical_cdf(a):
    # a is the data array
    x = np.sort(a,axis=None)
    y = np.arange(len(x))/float(len(x))
    
    return x,y
    
def print_html(html):
    display(HTML(html))


def modify_offsettext(ax,text):
    '''
    For y axis
    '''
    x_pos = 0.0
    y_pos = 1.0
    horizontalalignment='left'
    verticalalignment='bottom'
    offset = ax.yaxis.get_offset_text()
    #value=offset.get_text()
#     value=float(value)
#     if value>=1e12:
#         text='biljoonaa'
#     elif value>1e9:
#         text=str(value/1e9)+' miljardia'
#     elif value==1e9:
#         text=' miljardia'
#     elif value>1e6:
#         text=str(value/1e6)+' miljoonaa'
#     elif value==1e6:
#         text='miljoonaa'
#     elif value>1e3:
#         text=str(value/1e3)+' tuhatta'
#     elif value==1e3:
#         text='tuhatta'

    offset.set_visible(False)
    ax.text(x_pos, y_pos, text, transform=ax.transAxes,
                   horizontalalignment=horizontalalignment,
                   verticalalignment=verticalalignment)    

