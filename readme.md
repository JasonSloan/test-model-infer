# Test Model Inference Code Repository

This repository provides test code for inference of **object detection** and **object tracking** using multiple inference backends with multiple input format sources supported(images, video, stream).

## Overview

- All files in the repository, except for those in the `tests` folder, are shared across all inference backends.

- The `tests` folder contains subfolders named after specific inference backends. Each subfolder includes the following files:

  - `CMakeLists.txt`
  - `easy-cmake.sh`

  The primary differences between inference backends lie in their respective `CMakeLists.txt` files.

- Currently supported inference backends:

  - **yolo-openvino**
  - **yolo-tensorrt**
  - **yolo-denglin**

------

## Usage Instructions

This repository is designed to work alongside other repositories that contain specific inference code (e.g., `yolo-openvino`). Follow the steps below to set up and run the tests:

### Step 1: Clone This Repository

```
git clone https://github.com/JasonSloan/test-model-infer.git
```

### Step 2: Clone the Target Inference Backend Repository

Build your OpenCV library with video support enabled, and place the compiled files into the 'test-model-infer' directory.

If you are using ubuntu, download the pre-built opencv4.2 library directly from [here](https://github.com/JasonSloan/DeepFusion/releases/download/v111/opencv4.2.tar), and put them into the  'test-model-infer' directory, then set

```bash
export LD_LIBRARY_PATH=/path/to/opencv4.2/lib:$LD_LIBRARY_PATH
```

### Step 3: Clone the Target Inference Backend Repository

Clone the repository for the inference backend you wish to test in the **same directory** as this repository. Ensure that the dependencies for the selected inference backend are correctly installed.

Example:

```
git clone https://github.com/JasonSloan/yolo-openvino.git
```

### Step 4: Configure the Inference Backend

Modify the `CMakeLists.txt` file in the subfolder corresponding to the target inference backend (located in the `tests` folder). Make sure the paths to the inference framework are configured correctly.

### Step 5: Run the Test Script

Navigate to the root directory of this repository and execute the `easy-cmake.sh` script for the specific inference backend.

Example:

```
cd test-model-infer
sh tests/openvino/easy-cmake.sh
```

------

This setup allows you to test various inference backends seamlessly with shared code while maintaining backend-specific configurations.