#!/usr/bin/env python
import os
import sys


def build_results():
    results = {} # map task -> impl -> time

    for root, dirs, files in os.walk('.'):
        for bmark in [f for f in files if f.endswith('.bmark')]:
            for line in open(os.path.join(root,bmark)):
                if not line or line == "\n":
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

if __name__ == '__main__':
    r = build_results()

    for k,v in r.items():
        print k
        r_k = [(v[i],i) for i in v]
        r_k.sort()
        for t, i in r_k:
            print "   %8.2f - %s" %(t, i)
        print ''