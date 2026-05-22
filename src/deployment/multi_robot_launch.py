#!/usr/bin/env python3
"""
Multi TurtleBot3 ROS2 Launch 파일

Phase 4: 듀얼 로봇 시스템 실행
- Robot 1 (tb1): Jetson Orin Nano - RL Policy 기반
- Robot 2 (tb2): Raspberry Pi 5 - Nav2 기반

Usage:
    ros2 launch src/deployment/multi_robot_launch.py
"""

import os
import launch
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.actions import ComposableNodeContainer
from launch.actions import (
    GroupAction,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for dual TurtleBot3 system"""
    
    # ========== Launch Arguments ==========
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')
    formation_type = LaunchConfiguration('formation_type', default='column')
    
    # ========== Robot 1: Jetson Orin Nano (tb1) ==========
    robot1_group = GroupAction(
        actions=[
            PushRosNamespace('tb1'),
            
            LogInfo(msg='[tb1] Starting TurtleBot3 (Jetson Orin Nano)...'),
            
            # TurtleBot3 Hardware
            Node(
                package='turtlebot3_bringup',
                executable='turtlebot3_robot',
                name='turtlebot3_robot',
                namespace='tb1',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'port': '/dev/ttyACM0',
                    'baudrate': 115200,
                }],
            ),
            
            # RL Policy Node
            Node(
                package='turtlebot3_policy',
                executable='jetson_inference_node',
                name='policy_node',
                namespace='tb1',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'model_path': '/workspace/outputs/policy/turtlebot_policy.plan',
                    'model_type': 'tensorrt',
                    'policy_frequency': 10.0,
                }],
            ),
            
            # SLAM Toolbox
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                namespace='tb1',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'map_frame': 'tb1/map',
                    'odom_frame': 'tb1/odom',
                    'base_frame': 'tb1/base_footprint',
                    'scan_topic': '/tb1/scan',
                }],
            ),
            
            # EKF for odometry fusion
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                namespace='tb1',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'frequency': 30.0,
                    'sensor_timeout': 0.1,
                    'two_d_mode': True,
                    'odom_frame': 'tb1/odom',
                    'base_link_frame': 'tb1/base_footprint',
                    'world_frame': 'tb1/odom',
                    'imu0': '/tb1/imu',
                    'odom0': '/tb1/odom_raw',
                }],
            ),
        ]
    )
    
    # ========== Robot 2: Raspberry Pi 5 (tb2) ==========
    robot2_group = GroupAction(
        actions=[
            PushRosNamespace('tb2'),
            
            LogInfo(msg='[tb2] Starting TurtleBot3 (Raspberry Pi 5)...'),
            
            # TurtleBot3 Hardware
            Node(
                package='turtlebot3_bringup',
                executable='turtlebot3_robot',
                name='turtlebot3_robot',
                namespace='tb2',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'port': '/dev/ttyAMA0',
                    'baudrate': 115200,
                }],
            ),
            
            # Nav2 Controller (경량)
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                namespace='tb2',
                parameters=['config/nav2_light_params.yaml'],
            ),
            
            # Nav2 Planner
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                namespace='tb2',
                parameters=['config/nav2_light_params.yaml'],
            ),
            
            # Nav2 Lifecycle Manager
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager',
                namespace='tb2',
                parameters=[{
                    'autostart': True,
                    'node_names': [
                        'controller_server',
                        'planner_server',
                    ],
                }],
            ),
            
            # SLAM Toolbox
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                namespace='tb2',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'map_frame': 'tb2/map',
                    'odom_frame': 'tb2/odom',
                    'base_frame': 'tb2/base_footprint',
                    'scan_topic': '/tb2/scan',
                }],
            ),
        ]
    )
    
    # ========== Formation Controller ==========
    formation_node = Node(
        package='turtlebot3_multi',
        executable='formation_controller',
        name='formation_controller',
        parameters=[{
            'formation': formation_type,
            'separation': 0.8,
            'leader': 'tb1',
            'follower': 'tb2',
        }],
    )
    
    # ========== Multi Robot Manager ==========
    manager_node = Node(
        package='turtlebot3_multi',
        executable='multi_robot_manager',
        name='multi_robot_manager',
    )
    
    # ========== Static TF (Map to Robot Maps) ==========
    static_tf1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_tb1',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'tb1/map'],
    )
    
    static_tf2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_tb2',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'tb2/map'],
    )
    
    # ========== RViz ==========
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            os.path.dirname(__file__), '../../config/multi_robot.rviz'
        )],
        condition=IfCondition(launch_rviz),
    )
    
    # ========== Assemble Launch ==========
    return launch.LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'launch_rviz', default_value='true',
            description='Launch RViz for visualization'
        ),
        DeclareLaunchArgument(
            'formation_type', default_value='column',
            description='Formation type: column, line, diamond, staggered'
        ),
        
        # Log start
        LogInfo(msg='Launching Dual TurtleBot3 System...'),
        
        # Robot groups
        robot1_group,
        robot2_group,
        
        # Formation & Management
        formation_node,
        manager_node,
        
        # Static transforms
        static_tf1,
        static_tf2,
        
        # Visualization
        rviz_node,
        
        LogInfo(msg='✅ Dual TurtleBot3 System Launched!'),
    ])


if __name__ == '__main__':
    generate_launch_description()
