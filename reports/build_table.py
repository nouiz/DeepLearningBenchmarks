#!/usr/bin/env python
import cPickle
import os
import sys

from build_csv import build_db


if __name__ == '__main__':
    assert sys.argv[1] == '--db'
    db = cPickle.load(open(sys.argv[2]))

    if len(sys.argv) == 3:
        raise Exception(
            "build_table.py --db torch_db.pkl theano_results_paths ...")

    # get Torch7 results
    # list of dict: {'OMP_NUM_THREADS': True, 'batch': 100,
    #    'precision': 32, 'host': 'assam.iro.umontreal.ca',
    #    'device': 'CPU', 'problem': 'control_mm_500_',
    #    'speed': 0.15}
    torch7_db = cPickle.load(open(sys.argv[2]))

    # get Theano results
    # map task -> impl -> time
    # ex task: 'mlp_784_10'
    # ex impl: 'theano{gpu/float/batch_size=100/openmp=1}'
    theano_db = build_db(sys.argv[3:])

    batch_sizes = [ent.get('batch', 1) for ent in theano_db]
    batch_sizes = list(set(batch_sizes))
    batch_sizes.sort()
    problems = [ent['problem'] for ent in theano_db]
    problems = list(set(problems))
    problems.sort()
    problems_to = [ent['problem'] for ent in torch7_db]
    problems_to = set(problems_to)
    hosts = [ent['host'] for ent in torch7_db if 'host' in ent]
    hosts += [ent['host'] for ent in theano_db]
    hosts = list(set(hosts))
    hosts.sort()
    print problems_to.difference(problems)
    # Convert Torch7 db to the dict of dict format
#    torch7_res = {}
#    for entry in torch7_db:
#        torch7_res.setdefault(entry['problem'], {})
#        torch7_res[entry['problem']]
    batch_sizes = list(batch_sizes)
    batch_sizes.sort()
    for problem in problems:
        # float CPU implementations
        print problem, 'float CPU 1 threads'
        for host in hosts:
            print host
            for batch in batch_sizes:
                res = []
                for db in [theano_db, torch7_db]:
                    ent = [entry['speed'] for entry in db
                           if entry['problem'] == problem and
                           entry['precision'] == 32 and
                           entry.get('OMP_NUM_THREADS', False) in [1, '1'] and
                           entry.get('batch', 1) == batch and
                           entry['device'] == 'CPU' and
                           entry['host'] == host]
                #            assert len(res) == 1
                    ent = list(set(ent))
                    ent.sort()
                    res.append(ent)
                if res[0] and res[1]:
                    print "\tbatch", batch, res, res[0][0] / res[1][0]
                elif res[0] or res[1]:
                    print "\tbatch", batch, res