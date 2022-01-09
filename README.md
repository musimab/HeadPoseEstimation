Head pose estimation is one of the computer vision tasks that exist. In this task, we want to know the object’s pose from its translation and rotation.
As you know, we have only a two-dimensional image. How can we estimate the pose of an object by relying on the image itself? We can use a solution called the Perspective-n-Point (PnP).
if we want to solve PnP problem we need to know the 2D coordinates in the image and corrosponding 3D coordinates in the world space

## Installation
pip install -r requirements.txt

## Usage

#### Get Pose From RealSense Camera

For Realsense camera matrix 
run python RealSenseHeadPoseEstimation.py it will generate intrinsic parameters
python RealSenseHeadPoseEstimation.py -K_Matrix CalibrationMatrix.npy 
