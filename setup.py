from distutils.core import setup

setup(name='dec',
    packages=['dec'],
    package_dir={'dec': 'dec'},
    package_data={'dec': ['dec/data/*.dat']},
    description='Discrete Exterior Calculus package',
    author='Dzhelil Rufat',
    author_email='drufat@caltech.edu',
    license='GPLv3',
    license='GNU LGPL',
    url='http://dzhelil.info/dec',
    requires = ['numpy (>= 1.7.0)'],
)