#!/bin/bash

# HACKS TO USE OPENBLAS
#export LIBRARY_PATH=./lib:~/.VENV/base/lib:$LIBRARY_PATH
#export LD_LIBRARY_PATH=./lib:~/.VENV/base/lib:$LD_LIBRARY_PATH

#-convfast use "fast" convolution code instead of standard [false]
#-openmp   use openmp *package* [false]
#-double   use doubles instead of floats [false]
#-cuda     use CUDA instead of floats [false]
#-batch    batch size [1]
#-gi       compute gradInput [false]
#-v        be verbose [false]

# this would use GEMM for convolution, Koray said this was not use
# and it makes a huge unrolled matrix for large problems.
#export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
#LUA=~/.local/bin/torch
export LD_LIBRARY_PATH=~/repos/intel/mkl/10.2.5.035/lib/em64t:$LD_LIBRARY_PATH

USE_CONVFAST=""
date
#Keep the 2 following variable in correspondance
LUAS=(~/.local.mkl.10.2.5.035.luagit/bin/torch ~/.local.mkl.10.2.5.035/bin/torch)
LUA_NOTES=(luajit lua)
for idx in 0 1 ; do
  LUA=${LUAS[idx]}
  LUA_NOTE=${LUA_NOTES[idx]}
  OUTPUT_DIR=outs_${LUA_NOTE}
  mkdir -p ${OUTPUT_DIR}
  echo $LUA_NOTE

  for PREC in 32 ; do #64 ; do
    if [ $PREC = 32 ] ; then
        USE_DOUBLE=""
    else
        USE_DOUBLE="-double"
    fi
    for batchsize in 1 10 60 ; do
        OUTPUT=${OUTPUT_DIR}/run.sh.results_${HOSTNAME}_b${batchsize}_p${PREC}
	if [ "$batchsize" == "1" -a "$LUA_NOTE" == "lua" ]; then
	    CONT=""
	else
	    CONT="-nocont"
	fi
	CONT="${CONT} -nexmlp 6000 -nexcnn 3000"
        if false ; then
            export OMP_NUM_THREADS=1
            echo "Running normal" ${OUTPUT}
            echo "host=$HOSTNAME" > "${OUTPUT}"
            echo "device=CPU" >> "${OUTPUT}"
	    echo "LUA_NOTE=$LUA_NOTE" >> "${OUTPUT}"
            echo "OMP_NUM_THREADS=1" >> "${OUTPUT}"
            echo "batch=$batchsize" >> "${OUTPUT}"
            echo "precision=$PREC" >> "${OUTPUT}"
            ${LUA} benchmark.lua -batch $batchsize $USE_DOUBLE $CONT $@ &>> "${OUTPUT}"
        fi

        if false ; then
            unset OMP_NUM_THREADS
            export OMP_NUM_THREADS=4
            echo "Running OpenMP " ${OUTPUT}_openmp
            echo "host=$HOSTNAME" > "${OUTPUT}_openmp"
            echo "device=CPU" >> "${OUTPUT}_openmp"
	    echo "LUA_NOTE=$LUA_NOTE" >> "${OUTPUT}_openmp"
            echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" >> "${OUTPUT}_openmp"
            echo "batch=$batchsize" >> "${OUTPUT}_openmp"
            echo "precision=$PREC" >> "${OUTPUT}_openmp"
            ${LUA} benchmark.lua -batch $batchsize $USE_DOUBLE $CONT $@ &>> "${OUTPUT}_openmp"
        fi

        if [ $PREC = 32 ] ; then
            export OMP_NUM_THREADS=1
            echo "Running CUDA " ${OUTPUT}_cuda
            echo "host=$HOSTNAME" > "${OUTPUT}_cuda"
            echo "device=GPU" >> "${OUTPUT}_cuda"
	    echo "LUA_NOTE=$LUA_NOTE" >> "${OUTPUT}_cuda"
            echo "OMP_NUM_THREADS=1" >> "${OUTPUT}_cuda"
            echo "OpenMP=0" >> "${OUTPUT}_cuda"
            echo "batch=$batchsize" >> "${OUTPUT}_cuda"
            echo "precision=32" >> "${OUTPUT}_cuda"
            nvidia-smi >> "${OUTPUT}_cuda"
            ${LUA} benchmark.lua -batch $batchsize $USE_DOUBLE -cuda $CONT $@ &>> "${OUTPUT}_cuda"
        fi
    done
  done
done
date