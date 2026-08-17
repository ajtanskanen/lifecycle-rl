import fnmatch
from setuptools import find_packages, setup, Extension
from setuptools.command.build_py import build_py as build_py_orig

setup(name='lifecycle_rl',
	version='3.0.0',
	install_requires=['h5py','fin_benefits','numpy','gym_unemployment','numpy_financial','tabulate','pandas','tqdm','seaborn','matplotlib','ipython','celluloid'], 
	packages=find_packages(),	
    
    author="Antti J. Tanskanen",
    author_email="antti.tanskanen@gmail.com",
    description="Discrete choice life cycle model based on the Finnish social security",
    keywords="life cycle model",
)