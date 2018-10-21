from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kulgap',
    version='0.1.dev0',
    packages=['kulgap', ],
    author='Elijah Tai, Janosch Ortmann',
    author_email='elijah.tai@outlook.com',
    url='https://github.com/itselijahtai/kulgap',
    long_description=long_description,
    license='MIT',
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'GPy',
        'matplotlib',
        'mypy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
