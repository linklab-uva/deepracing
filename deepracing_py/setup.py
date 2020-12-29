from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob, os
package_name="deepracing"
setup(
    name=package_name,
    version='0.0.1',
    # packages=[
            #   'deepracing/**',
            #   'deepracing/arma_utils',
            #   'deepracing/backend',
            #   'deepracing/controls',
            #   'deepracing/exceptions',
            #   'deepracing/evaluation_utils',
            #   'deepracing/imutils',
            #   'deepracing/path_utils',
            #   'deepracing/pose_utils',
            #   'deepracing/protobuf_utils',
            #   'deepracing/raceline_utils',
            #   ],
    packages = [package_name] + [p for p in glob.glob(os.path.join(package_name,"*")) if os.path.isdir(p)],
    license='Apache License 2.0',
    long_description=open('README.txt').read(),
    install_requires=open("requirements.txt").readlines(),
)