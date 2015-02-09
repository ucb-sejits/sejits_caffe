# sejits_caffe
SEJITS port of the caffe framework

[![Build Status](https://travis-ci.org/ucb-sejits/sejits_caffe.svg)](https://travis-ci.org/ucb-sejits/sejits_caffe)

# Install
Install llvm 3.3
## For OSX
```shell
brew tap homebrew/versions
brew install llvm33
```
Update your `~/.ctree.cfg` to include

```
[c]
CC = clang-3.3
CFLAGS = -O3 -stdlib=libstdc++ -mmacosx-version-min=10.6

[opencl]
CC = clang-3.3
CFLAGS = -O3 -stdlib=libstdc++ -mmacosx-version-min=10.6
```

## Conda
Install requirements and try tests.
```shell
conda install llvm llvmpy nose numpy
pip install -r requirements.txt
git clone git@github.com:ucb-sejits/sejits_caffe.git
cd sejits_caffe
nosetests
```
