## ROS2 road segmentation node

Here we present source code for ROS2 road segmentation node. The node reads images from ROS2, runs inference with Pytorch's Deeplabv3 implementation, and published the detections to ROS2. GPU is required for satisfactory results. See the video demo by clicking the thumbnail below

[![Video demo](https://img.youtube.com/vi/GzYVhFJsdts/hqdefault.jpg)](https://www.youtube.com/watch?v=GzYVhFJsdts)





## Usage
1. Clone this repo to your ROS2 workspace
```
cd <your ros2 workspace>/src
git clone https://github.com/eerik98/road_segmentation.git
```

2. create a new python venv and install dependencies with pip. Using venv avoids version conflicts with your system-level libraries. 
```
python3 -m venv <env name>
source <path to env>/bin/activate
pip3 install -r <path to requirements.txt>
```
3. In the launch file (`road_segmentation/launch/launch.py`) change the `venv_python_path` so that it points to the site packages of the created venv. 

4. Download Deeplabv3 weights for road segmentation from [this link](https://drive.google.com/file/d/1AhhItat4xGq1_fdx23CGU606nS17pMtl/view?usp=sharing). These weights have been trained with around
   10k winter driving samples labeled automatically as proposed in https://github.com/eerik98/road_segmentation. 

5. Set parameters in `config/params.yaml`. Most importantly you need to set the path to the model weights and the topic where images are published. Additionally, you can define crop boundaries and downscaling factor for the image to achieve higher fps. 

6. Build your ROS2 workspace with `colcon build`
   
7. Publish images to the topic defined in `config/params.yaml` either by playing a ros2 bag or running a ros2 camera driver.
    
8. Launch the node with `ros2 launch road_segmentation launch.py`. Detection are published to topics starting with `/road`:
     - `/mask` refers to binary detection: each pixel is either road or background
     - `/prob` refers to continuous detection: a higher value means a higher road probability
     - `/overlaid` overlays the detection with the image for visualization purposes
     - `/compressed` compressed the output for lower bitrate. If you want to record any output to a bagfile, use the compressed topics. The uncompressed images take up a lot of space.

9. Visualize the predictions with `rqt` or `ros2 run rviz2 rviz2`. 
