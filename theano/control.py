import os

import numpy
from numpy import random

import theano
from theano.misc.check_blas import execute

from bench_reporter import *

random.seed(2344)
sizes = [500, 1000, 1500, 2000, 2500]
iters = 10

GlobalBenchReporter.__init__(algo=Algorithms.CONTROL)

for order in ['C']:  # , 'F']: they give mostly the same time.
    for size in sizes:
        a = theano.shared(random.rand(size, size).astype(theano.config.floatX))
        b = theano.shared(random.rand(size, size).astype(theano.config.floatX))
        c = theano.shared(random.rand(size, size).astype(theano.config.floatX))
        f = theano.function([], updates={c: theano.tensor.dot(a, b)},
                            mode=theano.compile.ProfileMode())

        GlobalBenchReporter.eval_model(f, "control_mm_%s" % size)

        f = theano.function([],
                            updates={c: 0.4 * c + .8 * theano.tensor.dot(a, b)},
                            mode=theano.compile.ProfileMode())

        GlobalBenchReporter.eval_model(f, "control_mm_gemm_%s" % size)

        f = theano.function([],
                            updates={c: theano.tensor.blas.gemm_no_inplace(c, 0.4, a, b, .8)},
                            mode=theano.compile.ProfileMode())

        GlobalBenchReporter.eval_model(f, "control_addmm_%s" % size)
