#!/usr/bin/env python
import os
import sys


def build_results(path='.'):
    results = {}  # map task -> impl -> time

    for root, dirs, files in os.walk(path):
        for bmark in [f for f in files if f.endswith('.bmark')]:
            for line in open(os.path.join(root, bmark)):
                if (not line or line == "\n" or
                    line.startswith("Using gpu device")):
                    continue
                try:
                    task, impl, t = line[:-1].split('\t')[:3]
                except:
                    print >> sys.stderr, "PARSE ERR:", line
                    continue

                if task.startswith('#'):
                    print >> sys.stderr, "Skipping", task, impl, t
                else:
                    results.setdefault(task, {})[impl] = float(t)
    return results

def build_db(paths='.'):
    results = []  # list of dict
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    def list_files(p):
        if not os.path.isdir(p):
            return [(os.path.split(p)[0], [], [os.path.split(p)[1]])]
        else:
            return os.walk(path)
    for path in paths:
        for root, dirs, files in list_files(path):
            for bmark in [f for f in files if f.endswith('.bmark')]:
                for line in open(os.path.join(root, bmark)):
                    if (not line or line == "\n" or
                        line.startswith("Using gpu device")):
                        continue
                    try:
                        task, impl, t = line[:-1].split('\t')[:3]
                    except:
                        print >> sys.stderr, "PARSE ERR:", line
                        continue

                    if task.startswith('#'):
                        print >> sys.stderr, "Skipping", task, impl, t
                        continue
                    d = {}
                    if "openmp=1" in impl:
                        d['OMP_NUM_THREADS'] = 1
                    elif "openmp=}" in impl:
                        d['OMP_NUM_THREADS'] = -1
                    else:
                        d['OMP_NUM_THREADS'] = "unknow"
                    if 'cpu' in impl:
                        d['device'] = 'CPU'
                    elif 'gpu' in impl:
                        d['device'] = 'GPU'
                    else:
                        d['device'] = 'unknow'
                    sp = impl.split('/')
                    if "linker=cvm_nogc}" in impl:
                        d['linker'] = "cvm_nogc"
                    elif "linker=cvm}" in impl:
                        d['linker'] = "cvm"
                    if 'batch_size=' in impl:
                        d['batch'] = [int(part[11:]) for part in sp
                                      if part.startswith('batch_size=')][0]
                    if 'float' in impl:
                        d['precision'] = 32
                    elif 'double' in impl:
                        d['precision'] = 64
                    else:
                        d['precision'] = -1
                    d['host'] = bmark[:bmark.index('.ca')+3]
                    if task == "ConvLarge":
                        task = "cnn_256x256"
                    elif task == "ConvMed":
                        task = "cnn_96x96"
                    elif task == "ConvSmall":
                        task = "cnn_32x32"
                    if task.startswith("control_"):
                        task = "control_addmm_" + task[8:]
                    d['problem'] = task
                    d['speed'] = float(t)
                    results.append(d)

    return results

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise Exception("Need a path as the first argument")

    r = build_results(sys.argv[1])

    for k in r:
        for i in r[k]:
            print '%s\t%s\t%f' % (k, i, r[k][i])

    if 0:

        keys = r.keys()
        keys.sort()

        for k in keys:
            v = r[k]
            print k
            r_k = [(v[i], i) for i in v]
            r_k.sort()
            r_k.reverse()
            for t, i in r_k:
                print "   %10.2f - %s" % (t, i)
            print ''
