import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name= 'halo',
    version= '0.45.8',
    packages= find_packages(),
    include_package_data= True,
    description= 'Utility library for parsing, analysis, clustering, and classifying.',
    long_description= README,
    url= 'https://www.github.com/tannerburns/halo',
    author= 'Tanner Burns',
    author_email= 'tjburns102@gmail.com',
    install_requires=[
        'pandas',
        'pandas_profiling',
        'scikit-learn',
        'vast @ https://github.com/TannerBurns/vast/archive/master.zip'
    ],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
