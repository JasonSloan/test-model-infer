#! /bin/bash
# 是否启用代码中的计时
WITH_CLOCKING=$1 
mkdir -p tests/tensorrt/build  
mkdir -p tests/tensorrt/workspace
mkdir -p tests/tensorrt/outputs  
cd tests/tensorrt/build
cmake .. -D WITH_CLOCKING=${WITH_CLOCKING}
make -j12 
cd ..
./workspace/mainproject