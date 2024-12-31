cmake_minimum_required(VERSION 3.0)
project(mainproject)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_STANDARD 17)

# 是否启用代码中的计时功能
if(WITH_CLOCKING STREQUAL "ON")
    add_definitions(-DWITH_CLOCKING)
endif()

# 寻找OpenVINO库
find_package(OpenVINO REQUIRED)

# 设置共享代码的路径
set(SHARED_CODE_HOME /path/to/test-model-infer)
# 设置模型推理代码路径
set(MODEL_HOME /path/to/yolo-openvino)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
set(CPP_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64 -fPIC -fopenmp -pthread -fpermissive)

# 设置OpenCV_HOME以及要链接的OpenCV库名
set(OpenCV_HOME ${SHARED_CODE_HOME}/opencv4.2)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)

# 头文件寻找路径
include_directories(${OpenCV_HOME}/include ${SHARED_CODE_HOME}/include ${SHARED_CODE_HOME}/src ${MODEL_HOME}/include)

# 包含库文件
link_directories(${OpenCV_HOME}/lib) 

# 添加源文件
file(GLOB_RECURSE SRC_TEST_SHARED "${SHARED_CODE_HOME}/src/*.cpp")
file(GLOB_RECURSE SRC_TEST_ALONE "${SHARED_CODE_HOME}/tests/main.cpp")
file(GLOB_RECURSE SRC_MODEL_CPP "${MODEL_HOME}/src/*.cpp")

# 设置编译输出
add_executable(${PROJECT_NAME} ${SRC_TEST_SHARED} ${SRC_TEST_ALONE} ${SRC_MODEL_CPP})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 链接一些库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} openvino fmt)
# 设置可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 打印调试信息
# message(STATUS "SRC: ${SRC}")
# message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")