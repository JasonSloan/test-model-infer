#! /bin/bash
# 是否启用代码中的计时
WITH_CLOCKING=$1 
mkdir -p tests/openvino/build  
mkdir -p tests/openvino/workspace
mkdir -p tests/openvino/outputs  
cd tests/openvino/build
cmake .. -D WITH_CLOCKING=${WITH_CLOCKING}
make -j12 
cd ..
./workspace/mainproject