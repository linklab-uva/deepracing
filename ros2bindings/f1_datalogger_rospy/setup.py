from setuptools import find_packages
from setuptools import setup
import os
from glob import glob
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
                os.path.join(package_name,"convert"),
              ])),
   # data_files=[
       # ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Include our package.xml file
       # (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files.
      #  (os.path.join('share', package_name, 'launch'), glob('*.launch.py'))
    #],
    entry_points={
        'console_scripts': [
            'pose_publisher = %s.scripts.pose_publisher:main' % (package_name),
            'bezier_rviz = %s.scripts.bezier_rviz:main' % (package_name),
            'waypoint_plot_recorder = %s.scripts.record_plots_waypoint:main' % (package_name),
            'pure_pursuit_bezier = %s.scripts.admiralnet_bezier_script:main' % (package_name),
            'pure_pursuit_waypoint = %s.scripts.admiralnet_waypoint_script:main' % (package_name),
            'pure_pursuit_oracle = %s.scripts.oracle_pure_pursuit_script:main' % (package_name),
            'pilotnet = %s.scripts.pilotnet_script:main' % (package_name),
            'cnnlstm = %s.scripts.cnnlstm_script:main' % (package_name),
            'admiralnet_e2e = %s.scripts.admiralnet_e2e_script:main' % (package_name),
            'generate_steering_calibration = %s.scripts.generate_steering_calibration:main' % (package_name),
            'point_cloud_display = %s.scripts.point_cloud_display:main' % (package_name),
            
        ],
    },
)
