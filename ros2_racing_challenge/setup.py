import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'ros2_racing_challenge'


def existing_files(pattern):
    return [path for path in glob(pattern) if os.path.isfile(path)]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (os.path.join('share', package_name), ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            existing_files(os.path.join('launch', '*launch.[pxy][yma]*')),
        ),
        (
            os.path.join('share', package_name, 'rviz'),
            existing_files(os.path.join('rviz', '*.rviz')),
        ),
        (os.path.join('share', package_name), ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jungwon Park',
    maintainer_email='jungwonpark@seoultech.ac.kr',
    description='ROS 2 racing challenge simulator and controllers',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'racing_sim = ros2_racing_challenge.racing_sim:main',
            'racing_controller = '
            'ros2_racing_challenge.racing_controller:main',
            'keyboard_controller = '
            'ros2_racing_challenge.keyboard_controller:main',
        ],
    },
)
