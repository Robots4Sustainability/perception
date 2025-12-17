from setuptools import find_packages, setup

package_name = 'table_height_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohsin',
    maintainer_email='mohsinalimirxa@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_floor = table_height_predictor.floor_detector_node:main',
            'table_height = table_height_predictor.table_height_node:main',
        ],
    },
)
