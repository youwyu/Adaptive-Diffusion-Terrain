> This library has been integrated into [SAPIEN](https://github.com/haosulab/SAPIEN.git). SAPIEN is a state-of-the-art interactive simulation environment. Powered with SimSense and high-speed ray tracing engine, SAPIEN can simulate realistic depth in real time. Check https://github.com/haosulab/simsense for more details.

# SimSense: A Real-Time Depth Sensor Simulator
This is a GPU-accelerated depth sensor simulator for python. The goal of this project is to provide a real-time, adjustable, and off-the-shelf module that can simulate different kinds of real-world depth sensors. Building a reliable and fast depth sensor simulator is vital to closing the sim-to-real gap.

Based on stereo matching, SimSense encapsulates most of the common algorithms in real depth sensors. All you need is passing in left and right images used for stereo matching, and SimSense will directly output a computed depth map that almost looks like generated by a real depth sensor. Check `test/test1.py` and `test/test2.py` for examples of usage.

Depth by [RealSense D435](https://www.intelrealsense.com/depth-camera-d435/) (Invalid Marked as Black) | Depth by SimSense (Invalid Marked as Dark Blue)
:-:|:-:
<img src="doc/result_from_D435.png?" width="424" height="240" /> | <img src="doc/result_from_SimSense.png?" width="424" height="240" />

## Performance
Experiment settings:
- GPU: RTX 2080 Ti
- Input Image Size: 848x480

|                | 64 Disparity Level | 128 Disparity Level | 256 Disparity Level |
|----------------|--------------------|---------------------|---------------------|
| w/o Block Cost | 231.1 FPS          | 148.7 FPS           | 84.4 FPS            |
| w/ Block Cost  | 195.7 FPS          | 124.1 FPS           | 69.4 FPS            |

## Requirements
- CUDA ToolKit
- CMAKE 3.18 or later

CUDA Toolkit needs to be installed to compile .cu source files. Check https://developer.nvidia.com/cuda-downloads for instructions.

## Install
Run

    git clone git@github.com:angli66/simsense.git
    cd simsense
    git submodule update --init

to clone the repository and get the 3rd party library.

Before installing, check your NVIDIA GPU's compute capability at https://developer.nvidia.com/cuda-gpus. The default settings supports compute capability of 6.0, 6.1, 7.0, 7.5, 8.0 and 8.6. If yours is not listed above, add the value into line `19` of `CMakeLists.txt`.

Then, run

    pip install .

to build and install the package.

## Pipeline
![pipeline](doc/pipeline.png?)

### Infrared Noise Simulation
This part is optional. Applying infrared noise simulation to infrared input can sometimes enhance the sim-to-real performance. The noise model is derived from [Simulating Kinect Infrared and Depth Images](https://ieeexplore.ieee.org/document/7328728) by Landau et al.

### Rectification
For image pair that is not rectified, rectification needs to be performed before stereo matching to align the image planes of the two images.

### Center-Symmetric Census Transform
CSCT is a common pre-processing step performed in many real depth sensors before stereo matching. It transforms the pixels to contain information from their neighbors.

### Stereo Matching
The algorithm used here is **semi-global block matching**, which is a combination of semi-global matching and block matching algorithms. [Semi-global matching](https://en.wikipedia.org/wiki/Semi-global_matching) is a robust matching algorithm that can produce smooth disparity maps. The semi-global block matching algorithm implemented in this module differs from the original [H. Hirschmuller algorithm](https://core.ac.uk/download/pdf/11134866.pdf) as follows:
- The number of aggregation paths are four instead of eight. The four paths here are left-to-right, right-to-left, top-to-bottom and bottom-to-top.
- For two individual pixels in the left and right image, instead of directly matching them, the algorithm matches all of their neighboring pixels together, which is similar to block matching. The size of the neighborhood is defined by `block_width` and `block_height`. Setting both of them to 1 will reduce the blocks to single pixels.
- Mutual information cost is not implemented. Instead, the cost is the hamming distance between the CSCT features.

### Uniqueness Test
Uniqueness test filters out matches by checking its "uniqueness". For a pixel in the left image and its best match in the right image, uniqueness test will find the cost of second best match (excluding the left and right adjacent pixels of the best match), and compare it to the cost of the best one. If the best match's cost does not win the second best match's cost by a certain margin, the match will be marked as invalid.

### Left-Right Consistency Check
Left-Right consistency check is another common post-processing step for computing disparity. It filters out matches that are inconsistent between the computed left disparity map and right disparity map.

### Median Filter
A median filter is usually applied to smooth the computed disparity map.

### Disparity to Depth Conversion
Optional depth registration can be performed at this step, which will reproject the computed depth map from the left camera's frame to a specified RGB camera's frame. This allows easy integration with RGB data to get RGBD image. If RGB camera's frame is not specified, the output depth map will be in left camera's frame with the same size of the input images by default.

## Known Issues
- Doing numpy array slicing on column (e.g. `img = img[:, :-1]`) before feeding the image into the module will lead to incorrect output. This has something to do with the memory layout of the numpy array. To avoid this, use `numpy.delete()` instead of array slicing when necessary.

## Acknowledgement
Some acceleration techniques of the library are inspired by [Embedded real-time stereo estimation via Semi-Global Matching on the GPU](http://www.sciencedirect.com/science/article/pii/S1877050916306561) by [D. Hernandez-Juarez](http://danihernandez.eu) et al.