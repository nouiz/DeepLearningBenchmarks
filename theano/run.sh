#!/bin/bash


# FOR MAGGIE I INSTALLED MKL SO DO LIKE THIS:
# LD_LIBRARY_PATH to include     /u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t 
# LIBRARY_PATH to include        /u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t
# THEANO_FLAGS="device=cpu,floatX=float64,blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def -lpthread" python mlp.py

BLAS='blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def'
mkdir -p outs
BLAS32='linker=c|py_nogc,device=cpu,floatX=float32',$BLAS
BLAS64='linker=c|py_nogc,device=cpu,floatX=float64',$BLAS
GPU32='linker=c|py_nogc,device=gpu0,floatX=float32,force_device=True'

#MLP tests need the cvm linker.
BLAS32_MLP='device=cpu,floatX=float32',$BLAS
BLAS64_MLP='device=cpu,floatX=float64',$BLAS
GPU32_MLP='device=gpu0,floatX=float32,force_device=True'

cat /proc/cpuinfo |grep "model name"|uniq > outs/${HOSTNAME}_config.conf
free >> outs/${HOSTNAME}_config.conf
uname -a >>  outs/${HOSTNAME}_config.conf
w >> outs/${HOSTNAME}_config.conf
python -c "import theano; print 'blas',theano.config.blas.ldflags; print 'amdlibm',theano.config.lib.amdlibm" >> outs/${HOSTNAME}_config.conf

(THEANO_FLAGS="$BLAS32_MLP" python control.py 2>>./outs/errors.log >> outs/${HOSTNAME}_control_cpu32.bmark) &&
(THEANO_FLAGS="$BLAS64_MLP" python control.py 2>>./outs/errors.log >> outs/${HOSTNAME}_control_cpu64.bmark) &&
(THEANO_FLAGS="$GPU32_MLP" python control.py 2>>./outs/errors.log >> outs/${HOSTNAME}_control_gpu32.bmark) ;

(THEANO_FLAGS="$BLAS32_MLP" python mlp.py 2>>./outs/errors.log >> outs/${HOSTNAME}_mlp_cpu32.bmark) &&
(THEANO_FLAGS="$BLAS64_MLP" python mlp.py 2>>./outs/errors.log >> outs/${HOSTNAME}_mlp_cpu64.bmark) &&
(THEANO_FLAGS="$GPU32_MLP" python mlp.py 2>>./outs/errors.log >> outs/${HOSTNAME}_mlp_gpu32.bmark) ;

(THEANO_FLAGS="$BLAS32" python convnet.py 2>>./outs/errors.log >> outs/${HOSTNAME}_convnet_cpu32.bmark) &&
(THEANO_FLAGS="$BLAS64" python convnet.py 2>>./outs/errors.log >> outs/${HOSTNAME}_convnet_cpu64.bmark) &&
(THEANO_FLAGS="$GPU32" python convnet.py 2>>./outs/errors.log >> outs/${HOSTNAME}_convnet_gpu32.bmark) ;

(THEANO_FLAGS="$BLAS32" python rbm.py 1024 1024 1 100 2>>./outs/errors.log >> outs/${HOSTNAME}_rbm_cpu32_b1.bmark) &&
(THEANO_FLAGS="$BLAS32" python rbm.py 1024 1024 60 20 2>>./outs/errors.log >> outs/${HOSTNAME}_rbm_cpu32_b60.bmark) ;

(THEANO_FLAGS="$BLAS64" python rbm.py 1024 1024 1 100 2>>./outs/errors.log >> outs/${HOSTNAME}_rbm_cpu64_b1.bmark) &&
(THEANO_FLAGS="$BLAS64" python rbm.py 1024 1024 60 20 2>>./outs/errors.log >> outs/${HOSTNAME}_rbm_cpu64_b60.bmark) ;

(THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 1 100 2>>./outs/errors.log >> outs/${HOSTNAME}_rbm_gpu32_b1.bmark) &&
(THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 60 20 2>>./outs/errors.log >> outs/${HOSTNAME}_rbm_gpu32_b60.bmark) ;
