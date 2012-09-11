#!/bin/bash
#THEANO_FLAGS=blas.ldflags="-lopenblas -L/u/bastienf/repos/OpenBLAS"

# FOR MAGGIE I INSTALLED MKL SO DO LIKE THIS:
# LD_LIBRARY_PATH to include     /u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t 
# LIBRARY_PATH to include        /u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t
# THEANO_FLAGS="device=cpu,floatX=float64,blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def -lpthread" python mlp.py

BLAS='blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def'
BLAS='blas.ldflags=-lopenblas'
mkdir -p outs
#TODO past linker was c|py_nogc. Do this make a big difference?
BLAS32='device=cpu,floatX=float32',$BLAS
BLAS64='device=cpu,floatX=float64',$BLAS
GPU32='device=gpu0,floatX=float32,force_device=True'

cat /proc/cpuinfo |grep "model name"|uniq > outs/${HOSTNAME}_config.conf
free >> outs/${HOSTNAME}_config.conf
uname -a >>  outs/${HOSTNAME}_config.conf
w >> outs/${HOSTNAME}_config.conf
THEANO_FLAGS="$BLAS32" python -c "import theano; print 'blas',theano.config.blas.ldflags; print 'amdlibm',theano.config.lib.amdlibm" >> outs/${HOSTNAME}_config.conf

if true ; then
    echo "Run control"
    export OMP_NUM_THREADS=1
    (THEANO_FLAGS="$BLAS32" python control.py 2>> outs/${HOSTNAME}_control_cpu32.err >> outs/${HOSTNAME}_control_cpu32.out) &&
    #(THEANO_FLAGS="$BLAS64" python control.py 2>> outs/${HOSTNAME}_control_cpu64.err >> outs/${HOSTNAME}_control_cpu64.out) &&
    (THEANO_FLAGS="$GPU32" python control.py 2>> outs/${HOSTNAME}_control_gpu32.err >> outs/${HOSTNAME}_control_gpu32.out) ;
    unset OMP_NUM_THREADS
    (THEANO_FLAGS="$BLAS32" python control.py 2>> outs/${HOSTNAME}_control_cpu32_openmp.err >> outs/${HOSTNAME}_control_cpu32_openmp.out)
    #(THEANO_FLAGS="$BLAS64" python control.py 2>> outs/${HOSTNAME}_control_cpu64_openmp.err >> outs/${HOSTNAME}_control_cpu64_openmp.out);
fi

for linker in cvm cvm_nogc;
do
  echo "Run $linker"
  for batch in 1 10 60;
  do
    if true ; then
        echo "batch $batch MLP"
        export OMP_NUM_THREADS=1
        (THEANO_FLAGS="$BLAS32",linker=$linker python mlp.py --batch $batch 2>> outs/${HOSTNAME}_mlp_cpu32.err >> outs/${HOSTNAME}_mlp_cpu32.out) &&
#        (THEANO_FLAGS="$BLAS64",linker=$linker python mlp.py --batch $batch 2>> outs/${HOSTNAME}_mlp_cpu64.err >> outs/${HOSTNAME}_mlp_cpu64.out) &&
        (THEANO_FLAGS="$GPU32",linker=$linker python mlp.py --batch $batch 2>> outs/${HOSTNAME}_mlp_gpu32.err >> outs/${HOSTNAME}_mlp_gpu32.out) ;
        unset OMP_NUM_THREADS
        (THEANO_FLAGS="$BLAS32",linker=$linker python mlp.py --batch $batch 2>> outs/${HOSTNAME}_mlp_cpu32_openmp.err >> outs/${HOSTNAME}_mlp_cpu32_openmp.out)
#        (THEANO_FLAGS="$BLAS64",linker=$linker python mlp.py --batch $batch 2>> outs/${HOSTNAME}_mlp_cpu64_openmp.err >> outs/${HOSTNAME}_mlp_cpu64_openmp.out)
    fi
    if true ; then
        echo "batch $batch CONV"
        export OMP_NUM_THREADS=1
        (THEANO_FLAGS="$BLAS32",linker=$linker python convnet.py --batch $batch 2>> outs/${HOSTNAME}_convnet_cpu32.err >> outs/${HOSTNAME}_convnet_cpu32.out) &&
#        (THEANO_FLAGS="$BLAS64",linker=$linker python convnet.py --batch $batch 2>> outs/${HOSTNAME}_convnet_cpu64.err >> outs/${HOSTNAME}_convnet_cpu64.out) &&
        (THEANO_FLAGS="$GPU32",linker=$linker python convnet.py --batch $batch 2>> outs/${HOSTNAME}_convnet_gpu32.err >> outs/${HOSTNAME}_convnet_gpu32.out) ;
        unset OMP_NUM_THREADS
        (THEANO_FLAGS="$BLAS32",linker=$linker python convnet.py --batch $batch 2>> outs/${HOSTNAME}_convnet_cpu32_openmp.err >> outs/${HOSTNAME}_convnet_cpu32_openmp.out)
#        (THEANO_FLAGS="$BLAS64",linker=$linker python convnet.py --batch $batch 2>> outs/${HOSTNAME}_convnet_cpu64_openmp.err >> outs/${HOSTNAME}_convnet_cpu64_openmp.out) &&
    fi
  done
done

#(THEANO_FLAGS="$BLAS32" python rbm.py 1024 1024 1 100 >> outs/${HOSTNAME}_rbm_cpu32_b1.out) &&
#(THEANO_FLAGS="$BLAS32" python rbm.py 1024 1024 60 20 >> outs/${HOSTNAME}_rbm_cpu32_b60.out) ;

#(THEANO_FLAGS="$BLAS64" python rbm.py 1024 1024 1 100 >> outs/${HOSTNAME}_rbm_cpu64_b1.out) &&
#(THEANO_FLAGS="$BLAS64" python rbm.py 1024 1024 60 20 >> outs/${HOSTNAME}_rbm_cpu64_b60.out) ;

#(THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 1 100 >> outs/${HOSTNAME}_rbm_gpu32_b1.out) &&
#(THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 60 20 >> outs/${HOSTNAME}_rbm_gpu32_b60.out) ;
