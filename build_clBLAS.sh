#! /bin/bash
git submodule update --init
cd clBLAS
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DBUILD_KTEST=OFF
CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
make -j$CORES && make install
cd ../../
if [ "$(uname)" == "Darwin" ]; then
    cp clBLAS/build/package/lib64/libclBLAS.dylib sejits_caffe/layers
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    cp clBLAS/build/package/lib64/libclBLAS.so sejits_caffe/layers
fi
rm -rf clBLAS/build
