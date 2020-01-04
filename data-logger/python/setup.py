from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

setup(
    name='deepracing',
    version='0.0.1',
    packages=['deepracing',],
    license='Apache License 2.0',
    long_description=open('README.txt').read(),
)