from distutils.core import setup

setup(
    name='pdxat',
    version='0.1dev',
    packages=['pdxat',],
    author='Elijah Tai and Janosch Ortmann',
    author_email='elijah.tai@outlook.com',
    url='https://github.com/itselijahtai/pdxat',
    long_description=open('README.md').read(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'GPy',
        'matplotlib',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
