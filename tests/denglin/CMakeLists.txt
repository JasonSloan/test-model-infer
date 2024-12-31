# 与普通的编译相比区别在于:
#   1. 遍历使用dlcc将.cu文件逐个编译成.cu.o文件
#   2. 将所有的.cu.o文件编译成静态库libdltmp.a
#   3. 将静态库链接到可执行文件
#   4. 删除静态库
# ------------------------------------------------
# 工程名称
project(mainproject)
# cmake最低版本要求
cmake_minimum_required(VERSION 3.5)
# 如果是想debug代码的话，必须加上这一行
set(CMAKE_BUILD_TYPE Debug)
# 是否启用代码中的计时功能
if(WITH_CLOCKING STREQUAL "ON")
    add_definitions(-DWITH_CLOCKING)
endif()
# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
# 设置cpp文件和cu文件的编译选项
set(CXX_FLAGS -fPIC -fpermissive)    # -fPIC: Position Independent Code  -fpermissive: 允许一些有争议的或者一些非标准的语法编译过去
set(DLCC_CXX_FLAGS --cuda-gpu-arch=dlgpuc64 -x cuda  -std=c++14 -fpermissive -fPIC)

# 设置共享代码的路径
set(SHARED_CODE_HOME /path/to/test-model-infer)
# 设置模型推理代码路径
set(MODEL_HOME /path/to/test-model-infer/yolo-denglin)

# 设置登临SDK路径
set(SDK_DIR $ENV{DLICC_PATH}/../)       # $ENV{DLICC_PATH}代表意思是获取环境变量中的DLICC_PATH的变量值
# 设置cu文件编译出的静态库的名字
set(DLCC_TARGET_SHORT_NAME dltmp)       # dltmp也就是静态库libdltmp.a中间的名字
set(DLCC_TARGET_FULL_NAME libdltmp.a)

# 设置OpenCV_HOME和CUDA_HOME以及要链接的OpenCV和CUDA的库名
set(OpenCV_HOME ${SHARED_CODE_HOME}/opencv4.2)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)
set(CUDA_HOME /dl/sdk)
set(CUDA_LIBS curt dlnne)

# 头文件寻找路径
include_directories(${OpenCV_HOME}/include ${CUDA_HOME}/include ${SHARED_CODE_HOME}/include ${SHARED_CODE_HOME}/src ${MODEL_HOME}/include)
# 库文件寻找路径, 指定库文件寻找路径包括${CMAKE_BINARY_DIR}是因为dlcc编译生成的临时静态库libdltmp.a会产生在这里
link_directories(${MODEL_HOME}/lib/dl-kernel-functions ${OpenCV_HOME}/lib ${CUDA_HOME}/lib ${CMAKE_BINARY_DIR})    # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake ..命令的路径(也就是build文件夹)

# 添加源文件
file(GLOB_RECURSE SRC_TEST_SHARED "${SHARED_CODE_HOME}/src/*.cpp")
set(SRC_TEST_MAIN "${SHARED_CODE_HOME}/tests/main.cpp")
file(GLOB_RECURSE SRC_MODEL_CPP "${MODEL_HOME}/src/*.cpp")
list(REMOVE_ITEM SRC_MODEL_CPP "${MODEL_HOME}/src/compile-model.cpp")
list(REMOVE_ITEM SRC_MODEL_CPP "${MODEL_HOME}/src/main.cpp")
file(GLOB_RECURSE SRC_MODEL_CU  "${MODEL_HOME}/src/*.cu")

# 设置编译输出
add_executable(${PROJECT_NAME} ${SRC_TEST_SHARED} ${SRC_TEST_MAIN} ${SRC_MODEL_CPP})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 链接一些库
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBS} ${DLCC_TARGET_SHORT_NAME} fmt dlpreprocess)

# 添加pre-build和post-build，使用dlcc并指定相应的编译选项编译出obj文件并在之后删除这些obj文件
foreach (cu_file ${SRC_MODEL_CU})
    set(tmp ${cu_file}.o)
    # CMAKE_SOURCE_DIR: cmake中的预置变量, 存储最顶层的CMakeLists.txt文件的目录的路径
    # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake命令的路径
    # 该行命令实现的是将tmp变量对应的string中的CMAKE_SOURCE_DIR的路径替换为CMAKE_BINARY_DIR
    string(REPLACE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} cu_obj "${tmp}")
    string(REGEX MATCH "/.*/" cu_dir ${cu_obj}) 
    # message("tmp: " ${tmp})             # /dl/sdk/samples/cuda/vectorAdd/vector_add_kernel.cu.o
    # message("cu_obj: " ${cu_obj})       # /dl/sdk/samples/cuda/vectorAdd/build/vector_add_kernel.cu.o
    # message("cu_dir: " ${cu_dir})       # /dl/sdk/samples/cuda/vectorAdd/build
    set(cu_objs ${cu_objs} ${cu_obj})
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND mkdir -p ${cu_dir})
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND dlcc ${DLCC_CXX_FLAGS}  -o ${cu_obj} -c ${cu_file})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND rm ${cu_obj})
endforeach()

# 将dlcc编译出来的obj文件打包成临时静态库，编译出可执行程序后删除
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ar cqs ${DLCC_TARGET_FULL_NAME} ${cu_objs})
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND rm ${DLCC_TARGET_FULL_NAME})

# 打印调试信息
# message("cu_files: " ${cu_files})
# message(STATUS "SRC: ${SRC}")
# message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
