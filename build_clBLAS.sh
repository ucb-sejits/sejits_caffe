#! /bin/bash
cd clBLAS
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DBUILD_KTEST=OFF
CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
make -j$CORES && make install
cd ../../
cp clBLAS/build/package/lib64/libclBLAS.so sejits_caffe/layers
rm -rf clBLAS/build
