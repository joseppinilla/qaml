#!/usr/bin/env python

from setuptools import setup

packages = ['qaml']

install_requires = ['networkx>=2.0,<3.0',
                    'decorator>=4.1.0,<5.0.0',
                    'pulp>=1.6.0,<2.0.0',
                    'dimod>=0.8.2,<0.9.0',
                    'minorminer>=0.1.5,<0.2.0',
                    'dwave-networkx>=0.6.4,<0.8.0']


setup(name='qaml',
      version='0.0.1',
      description='Quantum Assisted Machine Learning Framework',
      long_description="",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/qaml',
      packages=packages,
      platforms='any',
      install_requires=install_requires,
      license='MIT'
     )
