import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name= 'halo',
    version= '0.18',
    packages= find_packages(),
    include_package_data= True,
    description= 'Library to dynamically train and test different classifiers in bulk',
    long_description= README,
    url= 'https://www.github.com/tannerburns/halo',
    author= 'Tanner Burns',
    author_email= 'tjburns102@gmail.com',
    install_requires=[
        "pandas",
        "scikit-learn"
    ],
    classifiers=[
        'Framework :: Django',
        'Framework :: Django-Rest-Framework',
        'Intended Audience :: Developers',  # example license
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
