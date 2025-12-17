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
            'device_id',
            default_value='0',
            description='Camera Device ID (0 for webcam, 1 for external, etc.)'
        ),
        
        # 1. Webcam Publisher
        Node(
            package='perception',
            executable='opencv_camera_node',
            name='webcam_publisher',
            output='screen',
            parameters=[{
                'topic_name': '/camera/color/image_raw',
                'device_id': LaunchConfiguration('device_id')
            }]
        ),

        # 2. Mock Depth Publisher
        Node(
            package='perception',
            executable='mock_depth_node',
            name='mock_depth_publisher',
            output='screen'
        ),

        # 3. YOLO Node
        Node(
            package='perception',
            executable='yolo_node',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'input_mode': 'robot' # Consumes standard topics
            }],
            remappings=[
                ('/detections', '/tool_detector/detections')
            ]
        ),

        # 4. Pose Node
        Node(
            package='perception',
            executable='pose_node',
            name='pose_estimator',
            output='screen',
            parameters=[{
                'input_mode': 'robot' # Consumes standard topics
            }],
            remappings=[
                ('/detections', '/tool_detector/detections')
            ]
        )
    ])
