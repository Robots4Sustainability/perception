from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='models/tools.pt',
            description='Path to the YOLO model file'
        ),
        DeclareLaunchArgument(
            'input_mode',
            default_value='realsense',
            description='Input mode: robot, realsense, or topic name'
        ),
        Node(
            package='perception',
            executable='yolo_node',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'input_mode': LaunchConfiguration('input_mode')
            }],
            remappings=[
                ('/detections', '/tool_detector/detections')
            ]
        ),
        Node(
            package='perception',
            executable='pose_node',
            name='pose_estimator',
            output='screen',
            parameters=[{
                'input_mode': LaunchConfiguration('input_mode')
            }],
            remappings=[
                ('/detections', '/tool_detector/detections')
            ]
        )
    ])
