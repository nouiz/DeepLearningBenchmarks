from optparse import OptionParser


import numpy
from numpy import asarray, random

import theano
from theano.tensor import lscalar, tanh, dot, grad, log, arange
from theano.tensor.nnet import softmax
from theano.tensor.nnet import crossentropy_softmax_argmax_1hot_with_bias
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


n_examples = 60000
inputs = 784
outputs = 10
lr = numpy.asarray(0.01, dtype=config.floatX)

data_x = shared(randn(n_examples, inputs))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_x[si:si + nsi]
sy = data_y[si:si + nsi]

ssi = shared(0)
snsi = shared(0)
ssx = data_x[ssi:ssi + snsi]
ssy = data_y[ssi:ssi + snsi]


def online_mlp_784_10():
    assert False, "This is old stuff not up to date that you probably don't need"
    v = shared(zeros(outputs, inputs))
    c = shared(zeros(outputs))
    si = shared(0)    # current training example index
    sx = data_x[si]
    sy = data_y[si]

    nll, p_y_given_x, _argmax = crossentropy_softmax_argmax_1hot_with_bias(
            dot(sx, v.T).dimshuffle('x', 0),
            c,
            sy.dimshuffle('x'))
    cost = nll.mean()
    gv, gc = grad(cost, [v, c])
    train = function([], [],
            updates={
                v: v - lr * gv,
                c: c - lr * gc,
                si:  (si + 1) % n_examples})
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
    GlobalBenchReporter.simple_eval_model(train, 'mlp_784_10_hack')
    try:
        train.fn.update_profile(train.profile)
    except AttributeError:
        pass
    if 1:
        t = time.time()
        for i in xrange(n_examples):
            train()
        dt = time.time() - t
        reportmodel('mlp_784_10_hack2', 1, dt)
    if 1:
        t = time.time()
        fn = train.fn
        for i in xrange(n_examples):
            fn()
        dt = time.time() - t
        reportmodel('mlp_784_10_hack3', 1, dt)


def online_mlp_784_500_10():
    assert False, "This is old stuff not up to date that you probably don't need"
    HUs = 500
    w = shared(rand(HUs, inputs) * numpy.sqrt(6 / (inputs + HUs)))
    b = shared(zeros(HUs))
    v = shared(zeros(outputs, HUs))
    c = shared(zeros(outputs))
    si = shared(0)    # current training example index
    sx = data_x[si]
    sy = data_y[si]

    nll, p_y_given_x, _argmax = crossentropy_softmax_argmax_1hot_with_bias(
            dot(tanh(dot(sx, w.T) + b), v.T).dimshuffle('x', 0),
            c,
            sy.dimshuffle('x'))
    cost = nll.mean()
    gw, gb, gv, gc = grad(cost, [w, b, v, c])
    train = function([], [],
            updates={
                w: w - lr * gw,
                b: b - lr * gb,
                v: v - lr * gv,
                c: c - lr * gc,
                si: (si + 1) % n_examples})
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
    GlobalBenchReporter.simple_eval_model(train, "mlp_784_500_10_hack")
    try:
        train.fn.update_profile(train.profile)
    except AttributeError:
        pass


def online_mlp_784_1000_1000_1000_10():
    assert False, "This is old stuff not up to date that you probably don't need"
    w0 = shared(rand(inputs, 1000) * numpy.sqrt(6 / (inputs + 1000)))
    b0 = shared(zeros(1000))
    w1 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000 + 1000)))
    b1 = shared(zeros(1000))
    w2 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000 + 1000)))
    b2 = shared(zeros(1000))
    v = shared(zeros(1000, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, w2, b2, v, c]

    si = shared(0)    # current training example index
    sx = data_x[si]
    sy = data_y[si]
    h0 = tanh(dot(sx, w0) + b0)
    h1 = tanh(dot(h0, w1) + b1)
    h2 = tanh(dot(h1, w2) + b2)

    nll, p_y_given_x, _argmax = crossentropy_softmax_argmax_1hot_with_bias(
            dot(h2, v).dimshuffle('x', 0),
            c,
            sy.dimshuffle('x'))
    cost = nll.mean()
    gparams = grad(cost, params)
    updates = [(p, p - lr * gp) for p, gp in zip(params, gparams)]
    updates += [(si, (si + 1) % n_examples)]
    train = function([], [], updates=updates)
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
    GlobalBenchReporter.simple_eval_model(train,
                                          "mlp_784_1000_1000_1000_10_hack")
    try:
        train.fn.update_profile(train.profile)
    except AttributeError:
        pass


def bench_logreg():
    name = "mlp_784_10"
    v = shared(zeros(outputs, inputs))
    c = shared(zeros(outputs))
    #
    # Note on the transposed-ness of v for some reason, this data
    # layout is faster than the non-transposed orientation.
    # The change doesn't make much difference in the deeper models,
    # but in this case it was more than twice as fast.
    #
    p_y_given_x = softmax(dot(sx, v.T) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gv, gc = grad(cost, [v, c])

    theano.printing.debugprint(grad(cost, [v, c]), file=open('foo', 'wb'))
    train = function([si, nsi], [],
                     updates={v: v - lr * gv, c: c - lr * gc},
                     name=name)
#    theano.printing.debugprint(train, print_type=True)
    GlobalBenchReporter.eval_model(train, name)

    # Version with no inputs
    snsi.set_value(GlobalBenchReporter.batch_size)

    p_y_given_x = softmax(dot(ssx, v.T) + c)
    nll = -log(p_y_given_x)[arange(ssy.shape[0]), ssy]
    cost = nll.mean()

    gv, gc = grad(cost, [v, c])

    train2 = function([], [],
                      updates={v: v - lr * gv, c: c - lr * gc,
                               ssi: ssi + snsi},
                      name=name)
    GlobalBenchReporter.bypass_eval_model(train2, name, init_to_zero=ssi)


def bench_mlp_500():
    name = "mlp_784_500_10"
    HUs = 500
    w = shared(rand(HUs, inputs) * numpy.sqrt(6 / (inputs + HUs)))
    b = shared(zeros(HUs))
    v = shared(zeros(outputs, HUs))
    c = shared(zeros(outputs))

    p_y_given_x = softmax(dot(tanh(dot(sx, w.T) + b), v.T) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gw, gb, gv, gc = grad(cost, [w, b, v, c])

    train = function([si, nsi], cost,
                     updates={w: w - lr * gw,
                              b: b - lr * gb,
                              v: v - lr * gv,
                              c: c - lr * gc},
                     name=name)
    GlobalBenchReporter.eval_model(train, name)


    # Version with no inputs
    snsi.set_value(GlobalBenchReporter.batch_size)
    p_y_given_x = softmax(dot(tanh(dot(ssx, w.T) + b), v.T) + c)
    nll = -log(p_y_given_x)[arange(ssy.shape[0]), ssy]
    cost = nll.mean()

    gw, gb, gv, gc = grad(cost, [w, b, v, c])

    train2 = function([], cost,
                     updates={w: w - lr * gw,
                              b: b - lr * gb,
                              v: v - lr * gv,
                              c: c - lr * gc,
                              ssi: ssi + snsi},
                      name=name)
    GlobalBenchReporter.bypass_eval_model(train2, name, init_to_zero=ssi)


def bench_deep1000():
    name = "mlp_784_1000_1000_1000_10"
    w0 = shared(rand(inputs, 1000) * numpy.sqrt(6 / (inputs + 1000)))
    b0 = shared(zeros(1000))
    w1 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000 + 1000)))
    b1 = shared(zeros(1000))
    w2 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000 + 1000)))
    b2 = shared(zeros(1000))
    v = shared(zeros(1000, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, w2, b2, v, c]

    h0 = tanh(dot(sx, w0) + b0)
    h1 = tanh(dot(h0, w1) + b1)
    h2 = tanh(dot(h1, w2) + b2)

    p_y_given_x = softmax(dot(h2, v) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
                     updates=[(p, p - lr * gp)
                              for p, gp in zip(params, gparams)],
                     name=name)
    GlobalBenchReporter.eval_model(train, name)

    # Version with no inputs
    h0 = tanh(dot(ssx, w0) + b0)
    h1 = tanh(dot(h0, w1) + b1)
    h2 = tanh(dot(h1, w2) + b2)

    p_y_given_x = softmax(dot(h2, v) + c)
    nll = -log(p_y_given_x)[arange(ssy.shape[0]), ssy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train2 = function([], cost,
                      updates=[(p, p - lr * gp)
                               for p, gp in zip(params, gparams)] + [(ssi, ssi + snsi)],
                      name=name)
    snsi.set_value(GlobalBenchReporter.batch_size)
    GlobalBenchReporter.bypass_eval_model(train2, name, init_to_zero=ssi)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--batch", default=60, type="int",
                      help="the batch size to use")
    (options, args) = parser.parse_args()

    GlobalBenchReporter.__init__(n_examples, batch_size=options.batch,
                                 algo=Algorithms.MLP)
    #online_mlp_784_10()
    #online_mlp_784_500_10()
    #online_mlp_784_1000_1000_1000_10()
    bench_logreg()
    bench_mlp_500()
    bench_deep1000()
    #GlobalBenchReporter.report_speed_info()
