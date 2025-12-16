from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Generates a launch description that can switch between robot and realsense inputs."""

    # 1. Declare a launch argument 'input_source'
    # This creates a variable that you can set from the command line.
    # We'll set 'realsense' as the default value.
    input_source_arg = DeclareLaunchArgument(
        'input_source',
        default_value='realsense',
        description='Input source for perception nodes. Can be "realsense" or "robot".'
    )

    # 2. YOLO Node
    yolo_node = Node(
        package='perception',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        parameters=[{
            # Use the value of the 'input_source' launch argument here
            'input_mode': LaunchConfiguration('input_source'),
            'model_type': 'fine_tuned',
            'conf_threshold': 0.6,
            'device': 'cuda'
        }]
    )

    # 3. Pose Node
    pose_node = Node(
        package='perception',
        executable='pose_node',
        name='pose_node',
        output='screen',
        parameters=[{
            # Use the same 'input_source' launch argument here
            'input_mode': LaunchConfiguration('input_source')
        }]
    )

    # 4. Classifier Node (this node is unaffected)
    classifier_node = Node(
        package='perception',
        executable='classifier_node',
        name='classifier_node',
        output='screen'
    )

    # Return the LaunchDescription, including the argument and all nodes
    return LaunchDescription([
        input_source_arg,
        yolo_node,
        pose_node,
        classifier_node
    ])