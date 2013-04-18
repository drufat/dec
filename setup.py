from distutils.core import setup

setup(name='dec',
    packages=['dec'],
    package_dir={'dec': 'dec'},
    package_data={'dec': ['dec/data/*.dat']},
    author='Dzhelil Rufat',
    author_email='drufat@caltech.edu',
    license='GPLv3',
    url='http://drufat.github.com/dec',
    requires = ['numpy (>= 1.7.0)'],
)