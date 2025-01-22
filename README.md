
## Usage
1. Download trained weights for road segmentation from [this link](https://drive.google.com/file/d/14lupCHV5pTs1rM5lTA0xl7trN7DdpVif/view?usp=drive_link)
1. Set parameters in `params.yaml`
2. Build your ROS2 workspace with `colcon build`
3. Launch the node with `ros2 launch road_segmentation launch.py`
4. Feed images to the topic defined in `params.yaml` by playing a rosbag or running the ros2 camera driver
5. Detection are published to topics starting with `/road`:
     - `/mask` refers to binary detection: each pixel is either road or background
     - `/prob` refers to continuous detection: a higher value means a higher road probability
     - `/overlaid` overlays the detection with the image for visualization purposes
     - `/compressed` compressed the output for lower bitrate. If you want to record any output to a bagfile, use the compressed topics. The uncompressed images take up a lot of space.   
