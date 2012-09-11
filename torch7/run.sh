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
USE_CONVFAST=""
LUA=~/.local/bin/torch
mkdir -p outs
date
for PREC in 32 ; do #64 ; do
    for batchsize in 1 10 60 ; do
	if [ "$batchsize" == "1" ]; then
	    CONT=""
	else
	    CONT="-nocont"
	fi
	CONT="${CONT} -nexmlp 6000 -nexcnn 3000"
        OUTPUT=outs/run.sh.results_${HOSTNAME}_b${batchsize}_p${PREC}
	print $OUTPUT
        if true ; then
            export OMP_NUM_THREADS=1
            echo "Running normal" $OUTPUT
            echo "host=$HOSTNAME" > "$OUTPUT"
            echo "device=CPU" >> "$OUTPUT"
            echo "OMP_NUM_THREADS=1" >> "$OUTPUT"
            echo "batch=$batchsize" >> "$OUTPUT"
            echo "precision=$PREC" >> "$OUTPUT"
            if [ $PREC = 32 ] ; then
                USE_DOUBLE=""
            else
                USE_DOUBLE="-double"
            fi

            ${LUA} benchmark.lua -batch $batchsize $USE_DOUBLE $CONT &>> "$OUTPUT"
        fi

        if true ; then
            unset OMP_NUM_THREADS
            echo "Running OpenMP " ${OUTPUT}_openmp
            echo "host=$HOSTNAME" > "${OUTPUT}_openmp"
            echo "device=CPU" >> "${OUTPUT}_openmp"
            echo "OMP_NUM_THREADS=unset" >> "${OUTPUT}_openmp"
            echo "batch=$batchsize" >> "${OUTPUT}_openmp"
            echo "precision=$PREC" >> "${OUTPUT}_openmp"
            if [ $PREC = 32 ] ; then
                USE_DOUBLE=""
            else
                USE_DOUBLE="-double"
            fi
            ${LUA} benchmark.lua -batch $batchsize $USE_DOUBLE $CONT &>> "${OUTPUT}_openmp"
        fi

        if [ $PREC = 32 ] ; then
            export OMP_NUM_THREADS=1
            echo "Running CUDA " ${OUTPUT}_cuda
            echo "host=$HOSTNAME" > "${OUTPUT}_cuda"
            echo "device=GPU" >> "${OUTPUT}_cuda"
            echo "OMP_NUM_THREADS=1" >> "${OUTPUT}_cuda"
            echo "OpenMP=0" >> "${OUTPUT}_cuda"
            echo "batch=$batchsize" >> "${OUTPUT}_cuda"
            echo "precision=32" >> "${OUTPUT}_cuda"
            nvidia-smi >> "${OUTPUT}_cuda"
            if [ $PREC = 32 ] ; then
                USE_DOUBLE=""
            else
                USE_DOUBLE="-double"
            fi
            ${LUA} benchmark.lua -batch $batchsize $USE_DOUBLE -cuda $CONT &>> "${OUTPUT}_cuda"
        fi
    done
done
date