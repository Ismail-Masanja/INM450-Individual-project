from os import path
from setuptools import find_packages, setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# read the contents of the README File
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='INM450 Individual Project',
      version='0.0.1',
      description='INM450 Individual Project - Detection, Classification and Defense of Malware in Network Traffic using Machine Learning techniques.',
      long_description=long_description,
      author='Ismail Masanja',
      author_email='ismail.masanja@city.ac.uk',
      packages=find_packages(where='src/'),
      package_dir={'': 'src/'},
      include_package_data=True,
      install_requires=requirements)
