from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob, os
package_name="deepracing_py"
install_folder="deepracing"
setup(
    name=package_name,
    version='0.0.1',
    packages = [install_folder] + [p for p in glob.glob(os.path.join(install_folder,"*")) if os.path.isdir(p)],
    license='Apache License 2.0',
    long_description=open('README.txt').read(),
    install_requires=open("requirements.txt").readlines(),
)