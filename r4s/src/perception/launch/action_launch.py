import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode

def generate_launch_description():
    # --- 1. Launch Arguments ---
    input_mode_arg = DeclareLaunchArgument(
        'input_mode',
        default_value='robot',
        description="Camera input mode: 'robot' or 'realsense'"
    )

    lazy_loading_arg = DeclareLaunchArgument(
        'lazy_loading',
        default_value='false',
        description="If true, models are loaded/unloaded per task to save VRAM. Slower per task."
    )

    input_mode = LaunchConfiguration('input_mode')
    lazy_loading = LaunchConfiguration('lazy_loading')
    pkg_name = 'perception'


    # --- 2. Lifecycle Nodes ---

    # ==========================================
    # TABLE HEIGHT
    # ==========================================
    table_height_node = LifecycleNode(
        package=pkg_name,
        executable='table_height',  
        name='table_height_estimator',
        namespace='',
        output='screen'
    )

    # ==========================================
    # SUBDOOR POSE
    # ==========================================
    subdoor_node = LifecycleNode(
        package=pkg_name,
        executable='subdoor_node',  
        name='subdoor_pose_estimator',
        namespace='',
        output='screen',
        parameters=[{'input_mode': input_mode}]
    )

    # ==========================================
    # Place Object (Table Segmentation)
    # ==========================================

    place_object_node = LifecycleNode(
        package=pkg_name,
        executable='place_object_node',  
        name='place_object_node',
        namespace='',
        output='screen',
        parameters=[{'mode': 'test', 'input_mode': input_mode}]
    )

    # ==========================================
    # PIPELINE 1: SCREWS (YOLO)
    # ==========================================
    yolo_screw_node = LifecycleNode(
        package=pkg_name,
        executable='action_yolo_node',   
        name='yolo_screw_node',
        namespace='',
        parameters=[{
            'model_type': 'screw',
            'input_mode': input_mode,
            'conf_threshold': 0.3
            }],
        remappings=[
            ('/yolo/detections', '/screw/detections') 
        ]
    )

    cropper_screw_node = LifecycleNode(
        package=pkg_name,
        executable='action_pose_node',   
        name='cropper_screw_node',
        namespace='',
        parameters=[{'input_mode': input_mode}],
        remappings=[
            ('/yolo/detections', '/screw/detections'), 
            ('/object_pose', '/screw/pose'),
            ('/perception/detected_object_radius', '/screw/radius')
        ]
    )

    # ==========================================
    # PIPELINE 2: CAR OBJECTS (YOLO)
    # ==========================================
    yolo_car_node = LifecycleNode(
        package=pkg_name,
        executable='action_yolo_node',   
        name='yolo_car_node',
        namespace='',
        parameters=[{
            'model_type': 'fine_tuned', 
            'input_mode': input_mode,
                     }],
        remappings=[
            ('/yolo/detections', '/car/detections') 
        ]
    )

    cropper_car_node = LifecycleNode(
        package=pkg_name,
        executable='action_pose_node',   
        name='cropper_car_node',
        namespace='',
        parameters=[{'input_mode': input_mode}],
        remappings=[
            ('/yolo/detections', '/car/detections'), 
            ('/object_pose', '/car/pose'),
            ('/perception/detected_object_radius', '/car/radius')
        ]
    )

    # ==========================================
    # PIPELINE 3: SCREWDRIVER (OBB)
    # ==========================================
    obb_detector_node = LifecycleNode(
        package=pkg_name,
        executable='obb_node',  
        name='obb_detector_node',
        namespace='',
        output='screen',
        parameters=[{
            'model_type': 'fine_tuned', 
            'input_mode': input_mode
        }]
    )

    obb_cropper_node = LifecycleNode(
        package=pkg_name,
        executable='pose_obb_node',  
        name='point_obb_cloud_cropper_node',
        namespace='',
        output='screen',
        parameters=[{'input_mode': input_mode}]
    )

    # --- 3. Standard Nodes ---
    dispatcher_node = Node(
        package=pkg_name,
        executable='vision_manager_node',  
        name='perception_pipeline_node',
        output='screen',
        parameters=[{'lazy_loading': lazy_loading}]
    )

    # --- 4. Build Launch Description ---
    
    ld = LaunchDescription()
    
    # Add Arguments
    ld.add_action(input_mode_arg)
    
    # Add Lifecycle Nodes
    ld.add_action(table_height_node)
    ld.add_action(subdoor_node)
    ld.add_action(place_object_node)
    
    ld.add_action(yolo_screw_node)
    ld.add_action(cropper_screw_node)
    
    ld.add_action(yolo_car_node)
    ld.add_action(cropper_car_node)
    
    ld.add_action(obb_detector_node)
    ld.add_action(obb_cropper_node)
    
    # Add Dispatcher last so the lifecycle nodes exist before it tries to manage them
    ld.add_action(dispatcher_node)

    return ld