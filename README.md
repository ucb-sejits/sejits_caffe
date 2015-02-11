# sejits_caffe
SEJITS port of the caffe framework

[![Build Status](https://travis-ci.org/ucb-sejits/sejits_caffe.svg)](https://travis-ci.org/ucb-sejits/sejits_caffe)
[![Coverage Status](https://coveralls.io/repos/ucb-sejits/sejits_caffe/badge.svg?branch=master)](https://coveralls.io/r/ucb-sejits/sejits_caffe?branch=master)


# Install
Install llvm 3.5

```shell
$ git clone git@github.com:ucb-sejits/sejits_caffe.git
$ cd sejits_caffe
$ virtualenv venv
$ source venv/bin/activate
$ pip install nose
$ source venv/bin/activate  # have to do this twice to get nosetests in your path
$ export LLVM_CONFIG=llvm-config-3.5  # If your llvm-config for LLVM 3.5 has a special name
$ pip install -r requirements.txt
$ nosetests
.....
----------------------------------------------------------------------
Ran 5 tests in 14.763s

OK
```

To develop, remember to
```shell
$ source venv/bin/activate
```

when finished
```shell
$ deactivate
```
