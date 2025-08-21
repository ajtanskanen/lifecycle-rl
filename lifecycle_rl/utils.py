'''
Utils
'''

import gym
import numpy as np
from gym import spaces, logger, utils, error
from gym.utils import seeding
from IPython.core.display import display,HTML
import matplotlib.pyplot as plt 
import json
#from stable_baselines.common import set_global_seeds


import seaborn as sns
import matplotlib.font_manager as font_manager

def add_source(source,**csfont):
    plt.annotate(source, xy=(0.88,-0.1), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', va='top', **csfont)

def get_palette_EK():
    colors1=['#003326','#05283d','#ff9d6a','#599956']
    colors2=['#295247','#2c495a','#fdaf89','#7cae79']
    colors3=['#88b2eb','#ffcb21','#e85c03']
    
    colors=['#295247','#7cae79','#ffcb21','#e85c03','#88b2eb','#2c495a','#fdaf89']
    
    return sns.color_palette(colors)
    
def setup_EK_fonts():
    pal=get_palette_EK()
    #csfont = {'fontname':'Comic Sans MS'}
    fontname='IBM Plex Sans'
    csfont = {'font':fontname,'family':fontname,'fontsize':15}
    #fontprop = font_manager.FontProperties(family=fontname,weight='normal',style='normal', size=12)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False, 'ytick.left': False}
    sns.set_theme(style="ticks", font=fontname,rc=custom_params)
    linecolors = {'color':'red'}
    
    return csfont,pal

def make_env(env_id, rank, kwargs, seed=None):
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


    
def print_q(a):
    '''
    pretty printer for dict
    '''
    for x in a.keys():
        if a[x]>0 or a[x]<0:
            print('{}:{:.2f} '.format(x,a[x]),end='')
            
    print('')
        
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NpEncoder, self).default(obj)



def get_palette_EK():
    colors1=['#003326','#05283d','#ff9d6a','#599956']
    colors2=['#295247','#2c495a','#fdaf89','#7cae79']
    colors3=['#88b2eb','#ffcb21','#e85c03']
    
    colors=['#295247','#7cae79','#ffcb21','#e85c03','#88b2eb','#2c495a','#fdaf89']
    
    return sns.color_palette(colors)
    
def get_style_EK():
    axes={'axes.facecolor': 'white',
     'axes.edgecolor': 'black',
     'axes.grid': False,
     'axes.axisbelow': 'line',
     'axes.labelcolor': 'black',
     'figure.facecolor': 'white',
     'grid.color': '#b0b0b0',
     'grid.linestyle': '-',
     'text.color': 'black',
     'xtick.color': 'black',
     'ytick.color': 'black',
     'xtick.direction': 'out',
     'ytick.direction': 'out',
     'lines.solid_capstyle': 'projecting',
     'patch.edgecolor': 'black',
     'patch.force_edgecolor': False,
     'image.cmap': 'viridis',
     'font.family': ['sans-serif'],
     'font.sans-serif': ['IBM Plex Sans',
      'DejaVu Sans',
      'Bitstream Vera Sans',
      'Computer Modern Sans Serif',
      'Lucida Grande',
      'Verdana',
      'Geneva',
      'Lucid',
      'Arial',
      'Helvetica',
      'Avant Garde',
      'sans-serif'],
     'xtick.bottom': True,
     'xtick.top': False,
     'ytick.left': True,
     'ytick.right': False,
     'axes.spines.left': False,
     'axes.spines.bottom': True,
     'axes.spines.right': False,
     'axes.spines.top': False}
     
    return axes
        
def lineplot(x,y,y2=None,y3=None,y4=None,y5=None,y6=None,
                label=None,label2=None,label3=None,label4=None,label5=None,label6=None,
                xlabel='',ylabel='',selite=False,source=None,xlim=None,ylim=None,figname=None,
                show=True,grayscale=False):
    csfont,pal=setup_EK_fonts()
    #if grayscale:
    #    pal = sns.light_palette("black", 8, reverse = True)

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

    if show:
        plt.show()

def add_label(ax, text='(a)', x0=None, y0=None, **kwargs):
    '''
    Adds labels to multi-axis figures.
    '''
    from string import ascii_lowercase

    if isinstance(ax, np.ndarray):
        abc = [f'({letter})' for letter in list(ascii_lowercase)]

        f = ax.flat[0].get_figure()
        if y0 is None:
            y0 = f.subplotpars.top
        if x0 is None:
            x0 = f.subplotpars.left
        for i, _ax in enumerate(ax.flat):
            _ax.annotate(r'{} '.format(abc[i]),
                         xy=(-x0, y0), annotation_clip=False,
                         xycoords='axes fraction', weight='bold',
                         fontsize=plt.rcParams['font.size'],
                         va='bottom', **kwargs
                         )
    else:
        fig = ax.get_figure()
        if x0 is None:
            x0 = fig.subplotpars.left
        if y0 is None:
            y0 = fig.subplotpars.top
        #if plt.rcParams['text.usetex'] is True:
        #    ax.annotate(r'\textbf {{{}}}'.format(text),
        #                xy=(-x0, y0), annotation_clip=False,
        #                xycoords='axes fraction', weight='bold',
        #                fontsize=plt.rcParams['font.size'],
        #                va='top', **kwargs,
        #                )
        #else:
        ax.annotate(r'{}'.format(text),
                    xy=(-x0, y0), annotation_clip=False,
                    xycoords='axes fraction', weight='bold',
                    fontsize=plt.rcParams['font.size'],
                    )        