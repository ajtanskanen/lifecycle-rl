from setuptools import setup

setup(name='lifecycle_rl',
	version='2.0.0',
	install_requires=['gym','fin_benefits','numpy','gym_unemployment','numpy_financial','tabulate','bayesian-optimization'], #And any other dependencies required
	packages=setuptools.find_packages(),	
    # metadata to display on PyPI
    author="Antti J. Tanskanen",
    author_email="antti.tanskanen@ek.fi",
    description="Discrete choice life cycle model based on the Finnish social security",
    keywords="social-security earnings-related",
    #url="http://example.com/HelloWorld/",   # project home page, if any
    #project_urls={
    #    "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #    "Documentation": "https://docs.example.com/HelloWorld/",
    #    "Source Code": "https://code.example.com/HelloWorld/",
    #},
    #classifiers=[
    #    'License :: OSI Approved :: Python Software Foundation License'
    #]      
)
