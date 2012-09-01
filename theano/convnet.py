from optparse import OptionParser
import socket
import time

import numpy
from numpy import asarray, random

from theano.tensor import lscalar, tanh, dot, grad, log, arange
from theano.tensor.nnet import softmax
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano import shared, function, config

from bench_reporter import *

random.seed(2344)


def rand(*size):
    return asarray(random.rand(*size), dtype=config.floatX)


def randn(*size):
    return asarray(random.randn(*size), dtype=config.floatX)


def randint(size, high):
    return asarray(random.randint(size=size, low=0, high=high), dtype='int32')


def zeros(*size):
    return numpy.zeros(size, dtype=config.floatX)

n_examples = 6000
outputs = 10
lr = numpy.asarray(0.01, dtype=config.floatX)

data_x = shared(randn(n_examples, 1, 32, 32))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_x[si:si + nsi]
sy = data_y[si:si + nsi]


def bench_ConvSmall(batchsize):
    # Image shape 32x32
    GlobalBenchReporter.batch_size = batchsize
    data_x.set_value(randn(n_examples, 1, 32, 32))
    w0 = shared(rand(6, 1, 5, 5) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 5, 5) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16 * 5 * 5, 120) * numpy.sqrt(6.0 / 16. / 25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 32, 32),
                     filter_shape=(6, 1, 5, 5)) + b0.dimshuffle(0, 'x', 'x'))
    # this is not the correct leNet5 model, but it's closer to
    s0 = tanh(max_pool_2d(c0, (2, 2)))

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 14, 14),
                     filter_shape=(16, 6, 5, 5)) +
              b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool_2d(c1, (2, 2)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv) + cc), v) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p, p - lr * gp) for p, gp  in zip(params, gparams)])

    GlobalBenchReporter.eval_model(train, "ConvSmall")


def bench_ConvMed(batchsize):
    # Image shape 96x96
    GlobalBenchReporter.batch_size = batchsize
    data_x.set_value(randn(n_examples, 1, 96, 96))
    w0 = shared(rand(6, 1, 7, 7) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 7, 7) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16 * 8 * 8, 120) * numpy.sqrt(6.0 / 16. / 25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 96, 96),
                     filter_shape=(6, 1, 7, 7)) + b0.dimshuffle(0, 'x', 'x'))
    # this is not the correct leNet5 model, but it's closer to
    s0 = tanh(max_pool_2d(c0, (3, 3)))

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 30, 30),
                     filter_shape=(16, 6, 7, 7)) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool_2d(c1, (3, 3)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv) + cc), v) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p, p - lr * gp) for p, gp in zip(params, gparams)])
    GlobalBenchReporter.eval_model(train, "ConvMed")


def bench_ConvLarge(batchsize):
    # Image shape 256x256
    GlobalBenchReporter.batch_size = batchsize
    data_x.set_value(randn(n_examples, 1, 256, 256))
    w0 = shared(rand(6, 1, 7, 7) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 7, 7) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16 * 11 * 11, 120) * numpy.sqrt(6.0 / 16. / 25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 256, 256),
                     filter_shape=(6, 1, 7, 7)) + b0.dimshuffle(0, 'x', 'x'))
    # this is not the correct leNet5 model, but it's closer to
    s0 = tanh(max_pool_2d(c0, (5, 5)))

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 50, 50),
                     filter_shape=(16, 6, 7, 7)) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool_2d(c1, (4, 4)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv) + cc), v) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p, p - lr * gp) for p, gp in zip(params, gparams)])
    GlobalBenchReporter.eval_model(train, "ConvLarge")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--batch", default=60, type="int",
                      help="the batch size to use")
    (options, args) = parser.parse_args()

    GlobalBenchReporter.__init__(n_examples, options.batch,
                                 algo=Algorithms.CONVNET)
    bench_ConvSmall(options.batch)
    bench_ConvMed(options.batch)
    bench_ConvLarge(options.batch)
    #GlobalBenchReporter.report_speed_info()
