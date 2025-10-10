# In your launch file, e.g., perception_launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EqualsSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    """
    Generates a launch description that conditionally starts the correct camera
    and configures the perception nodes based on the 'input_source' argument.
    """

    # 1. Declare the 'input_source' launch argument we'll use for switching
    input_source_arg = DeclareLaunchArgument(
        'input_source',
        default_value='realsense',
        description='Input source. Can be "realsense" or "robot".'
    )

    # --- Conditional Camera Launch ---

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

    # --- Perception Nodes (configured by the same launch argument) ---

    # 4. YOLO Node
    yolo_node = Node(
        package='perception',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        parameters=[{
            'input_mode': LaunchConfiguration('input_source'),
            'model_type': 'fine_tuned',
            'conf_threshold': 0.6,
            'device': 'cuda'
        }]
    )

    # 5. Pose Node
    pose_node = Node(
        package='perception',
        executable='pose_node',
        name='pose_node',
        output='screen',
        parameters=[{
            'input_mode': LaunchConfiguration('input_source')
        }]
    )

    # 6. Classifier Node
    classifier_node = Node(
        package='perception',
        executable='classifier_node',
        name='classifier_node',
        output='screen'
    )

    # Return the LaunchDescription with all components
    return LaunchDescription([
        input_source_arg,
        
        # Add the conditional camera launches
        realsense_camera_launch,
        kinova_camera_launch,
        
        # Add the perception nodes
        yolo_node,
        pose_node,
        classifier_node
    ])