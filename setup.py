from distutils.core import setup

setup(name='dec',
    packages=['dec'],
    package_dir={'dec': 'dec'},
    package_data={'dec': ['dec/data/*.dat']},
    version='0.1',
    description='Discrete Exterior Calculus package',
    author='Dzhelil Rufat',
    author_email='drufat@caltech.edu',
    license='GNU GPLv3',
    url='http://dzhelil.info/dec',
    requires = ['numpy (>= 1.7.0)'],
)