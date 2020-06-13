from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

setup(
    name='deepracing',
    version='0.0.1',
    packages=['deepracing',
              'deepracing/arma_utils',
              'deepracing/backend',
              'deepracing/controls',
              'deepracing/evaluation_utils',
              'deepracing/imutils',
              'deepracing/pose_utils',],
              'deepracing/protobuf_utils',],
    license='Apache License 2.0',
    long_description=open('README.txt').read(),
    install_requires=open("requirements.txt").readlines(),
)