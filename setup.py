
from setuptools import setup, find_packages

setup(
    name='lean-kg',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pykeen[all]',
        'networkx',
        'plotly',
        'matplotlib',
        'numpy',
        'pandas',
        'torch'
    ],
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    description='A library for generating and visualizing mathlib4 as a directed multigraph.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/lean-kg', # Replace with your GitHub repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
