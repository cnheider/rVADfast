#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 30-10-2020
           '''

import re

from setuptools import setup


def read_version():
  with open('rVADfast/__init__.py') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              f.read(), re.M)
  if version_match:
    return version_match.group(1)
  raise RuntimeError("Unable to find __version__ string")


__version__ = read_version()

if __name__ == '__main__':

  setup(
      name='rVADfast',
      version=__version__,
      description='fast robust voice activity detection',
      long_description=open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      license='MIT License',
      url='https://github.com/zhenghuatan/rVAD',
      author='Zheng Hua Tan',
      author_email='christian.heider@alexandra.dk',
      keywords='speech voice activity detection robust',
      packages=['rVADfast', ],
      classifiers=[
          # https://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          ],
      install_requires=[open('requirements.txt').read()],
      include_package_data=True,
      zip_safe=False,
      )
