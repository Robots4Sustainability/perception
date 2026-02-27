import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EqualsSubstitution
from launch.conditions import IfCondition

def generate_launch_description():
    """
    Launches the correct camera based on the 'input_source' argument.
    - 'realsense': Starts the Realsense camera.
    - 'robot': Starts the Kinova camera.
    """

    # 1. Declare the launch argument to choose the camera
    input_source_arg = DeclareLaunchArgument(
        'input_source',
        default_value='realsense',
        description='Camera to launch. Can be "realsense" or "robot".'
    )

    # 2. Define the Realsense camera launch include
    # This will only be executed if input_source == 'realsense'
    realsense_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('input_source'), 'realsense')),
        launch_arguments={
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'pointcloud.ordered_pc': 'true'
        }.items()
    )

    # 3. Define the Kinova (robot) camera launch include
    # This will only be executed if input_source == 'robot'
    kinova_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('kinova_vision'), 'launch', 'kinova_vision.launch.py')
        ),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('input_source'), 'robot')),
        launch_arguments={
            'device': '192.168.1.12',
            'depth_registration': 'true',
            'color_camera_info_url': 'package://kinova_vision/launch/calibration/default_color_calib_1280x720.ini'
        }.items()
    )
    
    # Return the LaunchDescription with the argument and conditional includes
    return LaunchDescription([
        input_source_arg,
        realsense_camera_launch,
        kinova_camera_launch
    ])