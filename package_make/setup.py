# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sample',
    version='1.0',
    description='package make test',
    long_description=readme,
    author='kesheng wang',
    author_email='me@kesheng.com',
    url='https://github.com/kesheng/',
    license=license,
    packages=find_packages()
)






