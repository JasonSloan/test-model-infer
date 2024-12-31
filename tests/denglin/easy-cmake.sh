#! /bin/bash
# 是否启用代码中的计时
WITH_CLOCKING=$1 
mkdir -p tests/denglin/build  
mkdir -p tests/denglin/workspace
mkdir -p tests/denglin/outputs  
cd tests/denglin/build
cmake .. -D WITH_CLOCKING=${WITH_CLOCKING}
make -j12 
cd ..
./workspace/mainproject