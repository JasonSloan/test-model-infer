cmake_minimum_required(VERSION 3.0)
project(mainproject)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# 是否启用代码中的计时功能
if(WITH_CLOCKING STREQUAL "ON")
    add_definitions(-DWITH_CLOCKING)
endif()

# 设置共享代码的路径
set(SHARED_CODE_HOME /path/to/test-model-infer)
# 设置模型推理代码路径
set(MODEL_HOME /path/to/yolo-tensorrt)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
set(CPP_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64 -fPIC -fopenmp -pthread -fpermissive)
set(NVCC_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64)

# 设置OpenCV_HOME以及要链接的OpenCV库名
set(OpenCV_HOME ${SHARED_CODE_HOME}/opencv4.2)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)

# 设置CUDA_HOME以及要链接的CUDA的库名
set(CUDA_HOME /usr/local/cuda-12)
set(CUDA_LIBS cudart cudnn nvinfer nvinfer_plugin)

# 设置nvcc路径
set(NVCC_PATH ${CUDA_HOME}/bin)
# 设置cu文件编译出的静态库的名字
set(nvcc_target_lib nvtmp)       # nvtmp也就是静态库libnvtmp.a中间的名字
set(nvcc_target_lib_full_name libnvtmp.a)

# 头文件寻找路径
include_directories(${OpenCV_HOME}/include ${CUDA_HOME}/include ${SHARED_CODE_HOME}/include ${SHARED_CODE_HOME}/src ${MODEL_HOME}/include)
# 给cmake指定include_directories并不会让nvcc也去这里找, 所以还要再指定一遍
set(NVCC_INCLUDE_DIRS -I${OpenCV_HOME}/include -I${CUDA_HOME}/include)
set(NVCC_CXX_FLAGS ${NVCC_CXX_FLAGS} ${NVCC_INCLUDE_DIRS})

# 库文件寻找路径
# 指定库文件寻找路径包括${CMAKE_BINARY_DIR}是因为nvcc编译生成的临时静态库libnvtmp.a会产生在这里
link_directories(${OpenCV_HOME}/lib ${CUDA_HOME}/lib64 ${CMAKE_BINARY_DIR}) # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake ..命令的路径(也就是build文件夹)

# 添加源文件
file(GLOB_RECURSE SRC_TEST_SHARED "${SHARED_CODE_HOME}/src/*.cpp")
file(GLOB_RECURSE SRC_TEST_ALONE "${SHARED_CODE_HOME}/tests/tensorrt/main.cpp")
file(GLOB_RECURSE SRC_MODEL_CPP "${MODEL_HOME}/src/*.cpp")
list(REMOVE_ITEM SRC_MODEL_CPP "${MODEL_HOME}/src/compile-model.cpp")
file(GLOB_RECURSE SRC_MODEL_CU  "${MODEL_HOME}/src/*.cu")

# 设置编译输出
add_executable(${PROJECT_NAME} ${SRC_TEST_SHARED} ${SRC_TEST_ALONE} ${SRC_MODEL_CPP})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 链接一些库
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBS} ${nvcc_target_lib} fmt)

# 添加pre-build和post-build，使用dlcc并指定相应的编译选项编译出obj文件并在之后删除这些obj文件
foreach (cu_file ${SRC_MODEL_CU})
    set(tmp ${cu_file}.o)
    # CMAKE_SOURCE_DIR: cmake中的预置变量, 存储最顶层的CMakeLists.txt文件的目录的路径
    # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake命令的路径
    # 该行命令实现的是将tmp变量对应的string中的CMAKE_SOURCE_DIR的路径替换为CMAKE_BINARY_DIR
    string(REPLACE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} cu_obj "${tmp}")
    string(REGEX MATCH "/.*/" cu_dir ${cu_obj}) 
    # message("tmp: " ${tmp})             # /path/to/your/kernelfunction/dir/kernelfunction.cu.o
    # message("cu_obj: " ${cu_obj})       # /path/to/your/build/dir/kernelfunction.cu.o
    # message("cu_dir: " ${cu_dir})       # /path/to/your/build/dir
    set(cu_objs ${cu_objs} ${cu_obj})
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND mkdir -p ${cu_dir})
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ${NVCC_PATH}/nvcc ${NVCC_CXX_FLAGS} -o ${cu_obj} -c ${cu_file})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND rm ${cu_obj})
endforeach()

# 将nvcc编译出来的obj文件打包成临时静态库，编译出可执行程序后删除
# ar是编译静态库的命令
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ar cqs ${nvcc_target_lib_full_name} ${cu_objs})
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND rm ${nvcc_target_lib_full_name})

