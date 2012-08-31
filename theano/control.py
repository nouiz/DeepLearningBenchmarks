import os

from numpy import random

import theano
from theano.misc.check_blas import execute

from bench_reporter import *

random.seed(2344)
sizes = [500, 1000, 1500, 2000, 2500]
iters = 10

GlobalBenchReporter.__init__(algo=Algorithms.CONTROL)

for order in ['C', 'F']:
    for size in sizes:
        def train():
            return execute(verbose=False, M=size, N=size, K=size,
                           iters=iters, order=order)[0]
        GlobalBenchReporter.eval_model(train, "control_%s" % size)
