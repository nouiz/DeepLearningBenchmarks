import time, socket
from theano.tensor import lscalar, matrix, tanh, dot, grad, log, arange
from theano.tensor.nnet import softmax, crossentropy_softmax_argmax_1hot_with_bias
from theano import shared, function, config
import numpy, theano
from numpy import asarray, random

import bench_reporter
random.seed(2344)

import theano.tensor.blas_c

def rand(*size):
    return asarray(random.rand(*size), dtype=config.floatX)
def randn(*size):
    return asarray(random.randn(*size), dtype=config.floatX)
def randint(size, high):
    return asarray(random.randint(size=size, low=0, high=high), dtype='int32')
def zeros(*size):
    return numpy.zeros(size, dtype=config.floatX)

n_examples=6000
inputs=784
outputs=10
lr=numpy.asarray(0.01, dtype=config.floatX)

batchsize=60

data_x = shared(randn(n_examples, inputs))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_x[si:si + nsi]
sy = data_y[si:si + nsi]

<<<<<<< HEAD
bmark = open("%smlp_%s_%s.bmark" %(
    socket.gethostname(),
    config.device,
    config.floatX),
    'w')

=======
>>>>>>> 7e41484... Removed the unnecessary eval and report functions.
def reportmodel(model, batchsize, t):
    bmark.write("%s\t" % model)
    if config.floatX == 'float32':
        prec = 'float'
    else:
        prec = 'double'
    bmark.write("theano{%s/%s/%i}\t" % (
        config.device[0], prec, batchsize))
    bmark.write("%.2f\n"%(n_examples/t)) # report examples / second


def online_mlp_784_10():
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
                v:v - lr * gv,
                c:c - lr * gc,
                si: (si + 1) % n_examples})
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
<<<<<<< HEAD
    t = time.time()
    train.fn(n_calls=n_examples)
    dt = time.time() - t
=======
    GlobalBenchReporter.simple_eval_model(train, 'mlp_784_10_hack')
>>>>>>> 7e41484... Removed the unnecessary eval and report functions.
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
        for i in xrange(n_examples): fn()
        dt = time.time() - t
        reportmodel('mlp_784_10_hack3', 1, dt)

def online_mlp_784_500_10():
    HUs=500
    w = shared(rand(HUs, inputs) * numpy.sqrt(6 / (inputs + HUs)))
    b = shared(zeros(HUs))
    v = shared(zeros(outputs,HUs))
    c = shared(zeros(outputs))
    si = shared(0)    # current training example index
    sx = data_x[si]
    sy = data_y[si]

    nll, p_y_given_x, _argmax = crossentropy_softmax_argmax_1hot_with_bias(
            dot(tanh(dot(sx, w.T)+b), v.T).dimshuffle('x', 0),
            c,
            sy.dimshuffle('x'))
    cost = nll.mean()
    gw, gb, gv, gc = grad(cost, [w, b, v, c])
    train = function([], [],
            updates={
                w:w - lr * gw,
                b:b - lr * gb,
                v:v - lr * gv,
                c:c - lr * gc,
                si: (si + 1) % n_examples})
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
    GlobalBenchReporter.simple_eval_model(train, "mlp_784_500_10_hack")
    try:
        train.fn.update_profile(train.profile)
    except AttributeError:
        pass

def online_mlp_784_1000_1000_1000_10():
    w0 = shared(rand(inputs, 1000) * numpy.sqrt(6 / (inputs + 1000)))
    b0 = shared(zeros(1000))
    w1 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000+1000)))
    b1 = shared(zeros(1000))
    w2 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000+1000)))
    b2 = shared(zeros(1000))
    v = shared(zeros(1000, outputs))
    c = shared(zeros(outputs))
    params=[w0,b0,w1,b1,w2,b2,v,c]

    si = shared(0)    # current training example index
    sx = data_x[si]
    sy = data_y[si]
    h0 = tanh(dot(sx, w0)+b0)
    h1 = tanh(dot(h0, w1)+b1)
    h2 = tanh(dot(h1, w2)+b2)

    nll, p_y_given_x, _argmax = crossentropy_softmax_argmax_1hot_with_bias(
            dot(h2, v).dimshuffle('x', 0),
            c,
            sy.dimshuffle('x'))
    cost = nll.mean()
    gparams = grad(cost, params)
    updates = [(p,p-lr*gp) for p,gp in zip(params, gparams)]
    updates += [(si, (si + 1) % n_examples)]
    train = function([], [], updates=updates)
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
    GlobalBenchReporter.simple_eval_model(train, "mlp_784_1000_1000_1000_10_hack")
    try:
        train.fn.update_profile(train.profile)
    except AttributeError:
        pass

def bench_logreg():
    v = shared(zeros(outputs, inputs))
    c = shared(zeros(outputs))
    #
    # Note on the transposed-ness of v for some reason, this data layout is faster than the
    # non-transposed orientation.
    # The change doesn't make much difference in the deeper models, 
    # but in this case it was more than twice as fast.
    #

    p_y_given_x = softmax(dot(sx, v.T) + c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gv, gc = grad(cost, [v, c])

    theano.printing.debugprint(grad(cost, [v, c]), file=open('foo', 'wb'))
    train = function([si, nsi], [],
            updates={ v:v - lr * gv, c:c - lr * gc })
    theano.printing.debugprint(train, file=open('foo_train', 'wb'))
<<<<<<< HEAD

    eval_and_report(train, "mlp_784_10")
=======
    
    GlobalBenchReporter.eval_model(train, "mlp_784_10")
>>>>>>> 7e41484... Removed the unnecessary eval and report functions.
    print v.get_value().mean()
    print v.get_value()[:5,:5]

def bench_mlp_500():
    HUs=500
    w = shared(rand(HUs, inputs) * numpy.sqrt(6 / (inputs + HUs)))
    b = shared(zeros(HUs))
    v = shared(zeros(outputs,HUs))
    c = shared(zeros(outputs))

    p_y_given_x = softmax(dot(tanh(dot(sx, w.T)+b), v.T)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gw,gb,gv,gc = grad(cost, [w,b,v,c])

    train = function([si, nsi], cost,
            updates={ w:w-lr*gw,
                      b:b-lr*gb,
                      v:v-lr*gv,
                      c:c-lr*gc })
<<<<<<< HEAD
    eval_and_report(train, "mlp_784_500_10")
=======
    
    GlobalBenchReporter.eval_model(train, "mlp_784_500_10")
>>>>>>> 7e41484... Removed the unnecessary eval and report functions.

def bench_deep1000():
    w0 = shared(rand(inputs, 1000) * numpy.sqrt(6 / (inputs + 1000)))
    b0 = shared(zeros(1000))
    w1 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000+1000)))
    b1 = shared(zeros(1000))
    w2 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000+1000)))
    b2 = shared(zeros(1000))
    v = shared(zeros(1000, outputs))
    c = shared(zeros(outputs))
    params=[w0,b0,w1,b1,w2,b2,v,c]

    h0 = tanh(dot(sx, w0)+b0)
    h1 = tanh(dot(h0, w1)+b1)
    h2 = tanh(dot(h1, w2)+b2)

    p_y_given_x = softmax(dot(h2, v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])
    GlobalBenchReporter.eval_model(train, "mlp_784_1000_1000_1000_10")

if __name__ == '__main__':
    online_mlp_784_10()
    online_mlp_784_500_10()
    bench_logreg()
    bench_mlp_500()
<<<<<<< HEAD
    #online_mlp_784_1000_1000_1000_10()
    #bench_deep1000()
=======
    online_mlp_784_1000_1000_1000_10()
    bench_deep1000()
    GlobalBenchReporter.report_speed_info()
>>>>>>> 7e41484... Removed the unnecessary eval and report functions.
