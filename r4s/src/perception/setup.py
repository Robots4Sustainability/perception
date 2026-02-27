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
            "action_pose_node = perception.action_pose_pca:main",
            "action_yolo_node = perception.action_yolo_object_detection:main",
            "vision_manager_node = perception.action_vision_manager:main",
            "brain_client = perception.brain_client:main",
            
            # 1
            "yolo_node = perception.yolo_object_detection:main",
            "pose_node = perception.yolo_pose:main",
            
            #2 Subdoor
            "subdoor_node = perception.subdoor_pose_estimator:main",

            #3 Table Height
            "table_height = perception.table_height_node:main",

            #4 Place Object
            "table_segmentation_node = perception.table_segmentation:main",

            #5 Screwdriver
            "obb_node = perception.action_obb_object_detection:main",
            "pose_obb_node = perception.action_obb_pose:main",
            # "obb_node = perception.obb_object_detection:main",
            # "pose_obb_node = perception.obb_pose_pca:main",

            # "segment_object = perception.segment_object:main",
            # "benchmark_node = perception.benchmark:main",
            # "detect_floor = perception.floor_detector_node:main",
            # "classifier_node = perception.object_classifier_node:main",
            
        ],
    },
)
