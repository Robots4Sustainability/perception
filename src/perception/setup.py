from setuptools import setup
from glob import glob
import os

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Install models
        (os.path.join('share', package_name, 'models'), glob('models/*')),

        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohsin',
    maintainer_email='mohsinalimirxa@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            "yolo_node = perception.yolo_object_detection:main",
            "classifier_node = perception.object_classifier_node:main",
            "pose_node = perception.pose_pca:main",
            "table_segmentation_node = perception.table_segmentation:main",
        ],
    },
)
