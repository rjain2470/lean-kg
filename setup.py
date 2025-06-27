
from setuptools import setup, find_packages

setup(
    name='lean-kg',
    version='0.1.3',
    packages=find_packages(),
    install_requires=[
        'pykeen',
        'networkx',
        'plotly',
        'matplotlib',
        'numpy',
        'pandas',
        'torch'
    ],
    author='Ritik Jain',
    author_email='rjain92682@gmail.com', 
    description='A library for generating and visualizing mathlib4 as a directed multigraph.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rjain2470/lean-kg', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
