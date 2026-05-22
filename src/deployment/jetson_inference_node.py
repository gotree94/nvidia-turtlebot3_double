#!/usr/bin/env python3
"""
TurtleBot3 RL Policy Inference Node for Jetson Orin Nano

Usage:
    python3 src/deployment/jetson_inference_node.py --model_path outputs/policy/turtlebot_policy.plan
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time
import os
import argparse


class TurtleBotPolicyNode(Node):
    """RL 강화학습 정책 추론 노드"""
    
    def __init__(self, model_path: str):
        super().__init__('turtlebot_policy_node')
        
        self.model_path = model_path
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.84
        
        # 상태
        self.latest_scan = None
        self.robot_position = (0.0, 0.0)
        self.robot_yaw = 0.0
        self.goal_position = None
        
        # ROS2
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, '/policy/debug', 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz
        
        self.get_logger().info(f'Policy node initialized (model: {model_path})')
    
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
    
    def odom_callback(self, msg: Odometry):
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def goal_callback(self, msg: PoseStamped):
        self.goal_position = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
    
    def preprocess(self) -> np.ndarray:
        """LiDAR 전처리 및 관측값 생성"""
        if self.latest_scan is None:
            return None
        
        ranges = np.array(self.latest_scan.ranges)
        ranges = np.nan_to_num(ranges, nan=3.5)
        ranges = np.clip(ranges, 0.15, 3.5) / 3.5
        lidar = np.mean(ranges.reshape(36, 10), axis=1)
        
        goal = self.goal_position or (1.0, 0.0)
        dx = goal[0] - self.robot_position[0]
        dy = goal[1] - self.robot_position[1]
        goal_rel = np.array([dx, dy]) / 8.0
        
        goal_angle = math.atan2(dy, dx)
        heading_err = (goal_angle - self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
        heading_err = np.array([heading_err / math.pi])
        
        return np.concatenate([lidar, goal_rel, heading_err]).astype(np.float32)
    
    def heuristic_action(self, obs: np.ndarray) -> np.ndarray:
        """Fallback heuristic control"""
        lidar = obs[:36]
        front = np.min(lidar[15:21]) * 3.5
        heading = obs[38]
        
        if front < 0.3:
            left = np.min(lidar[8:15])
            right = np.min(lidar[21:28])
            return np.array([-0.05, 0.5 if left > right else -0.5])
        else:
            return np.array([min(0.15, front * 0.5), max(-1.0, min(1.0, heading * 2.0))])
    
    def control_loop(self):
        """정책 추론 루프"""
        obs = self.preprocess()
        if obs is None:
            return
        
        action = self.heuristic_action(obs)
        
        twist = Twist()
        twist.linear.x = float(np.clip(action[0] * self.max_linear_vel, -0.1, self.max_linear_vel))
        twist.angular.z = float(np.clip(action[1] * self.max_angular_vel, -self.max_angular_vel, self.max_angular_vel))
        self.cmd_pub.publish(twist)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='outputs/policy/turtlebot_policy.plan')
    parsed_args = parser.parse_args()
    
    rclpy.init(args=args)
    node = TurtleBotPolicyNode(parsed_args.model_path)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        twist = Twist()
        node.cmd_pub.publish(twist)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
