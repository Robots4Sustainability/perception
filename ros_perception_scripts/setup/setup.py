from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohsin',
    maintainer_email='mohsinalimirxa@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolo_node = perception.yolo_object_detection:main",
            "classifier_node = perception.object_classifier_node:main",
            "pose_node = perception.pose_pca:main",
            "opencv_camera_node = perception.opencv_camera_feed:main",
            "opencv_yolo = perception.opencv_yolo_object_detection:main",
        ],
    },
)
