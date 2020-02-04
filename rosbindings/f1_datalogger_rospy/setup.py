from setuptools import find_packages
from setuptools import setup
import os
package_name = 'f1_datalogger_rospy'

setup(
    name=package_name,
    version='0.0.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Trent Weiss',
    author_email='ttw2xk@virginia.edu',
    maintainer='Trent Weiss',
    maintainer_email='ttw2xk@virginia.edu',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=(
        'A package for utilizing the F1 Datalogger in Python'
    ),
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    packages=list(set(find_packages(exclude=['test'])+[
                os.path.join(package_name,"controls"),
              ])),

    entry_points={
        'console_scripts': [
            'pose_publisher = %s.scripts.pose_publisher:main' % (package_name),
            'plot_recorder = %s.scripts.record_plots:main' % (package_name),
            'pure_pursuit_bezier = %s.scripts.admiralnet_bezier_script:main' % (package_name),
            'pure_pursuit_waypoint = %s.scripts.admiralnet_waypoint_script:main' % (package_name),
            'pure_pursuit_oracle = %s.scripts.oracle_pure_pursuit_script:main' % (package_name),
            'pilotnet = %s.scripts.pilotnet_script:main' % (package_name),
            'cnnlstm = %s.scripts.cnnlstm_script:main' % (package_name),
            
        ],
    },
)
