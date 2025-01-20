from setuptools import setup, find_packages
from typing import List

HYPHEN_DOT = '-e .'

def get_requirements(file:str) -> List[str]:
    ''' Read requirements file and return list of requirements'''
    requirements = []
    with open(file) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
    
    return requirements

setup(
    name='MLProject',
    version='0.0.1',
    author='Agnat',
    author_email = 'aggy408@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    )