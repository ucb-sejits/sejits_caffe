# sejits_caffe
SEJITS port of the caffe framework

# Install
## OSX
```shell
conda install llvm llvmpy nose numpy
git clone https://github.com/ucb-sejits/ctree.git
cd ctree
python setup.py install
cd ..
git clone https://github.com/ucb-sejits/hindemith.git
cd hindemith
python setup.py install
cd ..
git clone https://github.com/ucb-sejits/pycl.git
cd pycl
python setup.py install
git clone git@github.com:ucb-sejits/sejits_caffe.git
cd sejits_caffe
nosetests
```
