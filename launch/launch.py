'''
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare the path to the YAML file as a launch argument
        #DeclareLaunchArgument('config_file', default_value='config/params.yaml', description='Path to YAML config file'),

        # Launch the node and load parameters from the YAML file
        Node(
            package='road_segmentation',
            executable='inference',
            output='screen',
            name='inference_node',
            parameters=['$(find road_segmentation)/config/params.yaml']
        ),
    ])
'''

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
   # Define the path to your virtual environment's site-packages
   venv_python_path = '/home/eerik/henry_demo_env/lib/python3.10/site-packages'

    # Set the PYTHONPATH environment variable for the ROS 2 node
   os.environ['PYTHONPATH'] = f"{venv_python_path}:{os.environ.get('PYTHONPATH', '')}"

   config =  os.path.join(
      get_package_share_directory('road_segmentation'),
      'config',
      'params.yaml'
      )

   return LaunchDescription([
      Node(
         package='road_segmentation',
         executable='inference',
         name='inference_node',
         parameters=[config]
      )
   ])