from setuptools import setup

requirements = [
    'numpy',
    'pandas',
    'tensorflow',
    'matplotlib'
]

setup(
    name='unsupervised_nmt',
    version='0.1',
    url='https://github.com/acivgin1/M2-DS-internship',
    description='Testing standard classifiers on titanic dataset',
    author='Amar Civgin',
    author_email='amar.civgin@gmail.com',
    packages=['unsupervised_nmt'],
    install_requires=requirements
)