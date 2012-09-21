#!/usr/bin/env python
import cPickle
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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
    all_suffixes = ["_bypass_noinputs",
                    "_trust_input",
                    "_bypass_inputs_parsing",
                    "_bypass_c_loop",]
    variants = [('cvm', ''),
                ('cvm_nogc', ''),
                ('cvm_nogc', "_trust_input"),
                ('cvm_nogc', "_bypass_c_loop")]
    luas = ['lua', 'luajit']
    for problem in problems:
        if any(problem.endswith(suf) for suf in all_suffixes):
            continue

        if problem.startswith("control"):
            suffixes = []
            plot_gpu = False
        else:
            suffixes = all_suffixes[:]
            plot_gpu = True

        for suf in suffixes[:]:
            ent = [entry for entry in theano_db
                   if entry['problem'] == problem + suf and
                   entry['precision'] == 32]
            if len(ent) == 0:
                suffixes.remove(suf)

        # float CPU implementations
        print
        print problem, 'float 1 threads'
        for host in hosts:
            print host
            plot_res = []
            for batch in batch_sizes:
                for device, openmp in [('CPU', 1),
                                       ('CPU', "4"),
                                       ('GPU', 1)]:
                    if device == 'GPU' and not plot_gpu:
                        continue
                    res = []
                    for note in luas:
                        ent = [entry['speed'] for entry in torch7_db
                               if entry['problem'] == problem and
                               entry['precision'] == 32 and
                               str(entry['OMP_NUM_THREADS']) == str(openmp) and
                               entry.get('batch', 1) == batch and
                               entry['device'] == device and
                               entry['host'] == host and
                               entry.get('LUA_NOTE', 'lua') == note]
#                        import pdb;pdb.set_trace()
                        ent = list(set(ent))
                        ent.sort()
                        res.append(ent)
                    for linker, suffix in variants:
                        ent = [entry['speed'] for entry in theano_db
                               if entry['problem'] == problem + suffix and
                               entry['precision'] == 32 and
                               str(entry['OMP_NUM_THREADS']) == str(openmp) and
                               entry.get('batch', 1) == batch and
                               entry['device'] == device and
                               entry['host'] == host and
                               entry.get('linker', linker) == linker]
                        res.append(ent)

                    if len(res) > len(luas) and res[0] and res[len(luas)]:
                        print "%8s batch %s %s omp=%.5s" % (
                            linker, batch, device, openmp),
                        print  res, res[0][0] / res[len(luas)][0]
                    elif res[0] or res[len(luas)]:
                        print "%8s batch %s %s omp=%.5s" % (
                            linker, batch, device, openmp),
                        print res

                    plot_res.extend(res)

            if (len(reduce(list.__add__, plot_res)) > 2 and
                    (problem.startswith('mlp')
                     or problem.startswith('control_addmm_2000')
#                     or problem.startswith('cnn')
                 )):
                fig = plt.figure()
                plot_res_batches = plot_res
                res_per_batch = len(plot_res_batches) // len(batch_sizes)
                n_plot = len(batch_sizes)
                if problem.startswith("control"):
                    n_plot = 1
                for idx, batch in enumerate(batch_sizes):
                    if idx >= n_plot:
                        break
                    plot_res = plot_res_batches[idx * res_per_batch:
                                                (1 + idx) * res_per_batch]
                    ax = fig.add_subplot(n_plot, 1, idx + 1)
                    ind = np.arange(len(plot_res))  # the x locations for the groups
                    width = 0.333       # the width of the bars
                    rec = []
                    legends = ['Torch7',
                               'Theano']
                    for linker, suf in variants:
                        s = "Theano " + linker
                        if suf == "_bypass_noinputs":
                            s +=  "no input"
                        elif suf == "_trust_input":
                            s += "trust inputs"
                        elif suf == "_bypass_inputs_parsing":
                            s += "f.fn()"
                        elif suf == "_bypass_c_loop":
                            s += "f.fn(n_calls=N)"
                        legends.append(s)
                    tmp = legends[:]
                    tests_per_categ = len(legends)
                    for l in tmp:
                        legends.append(l + " OpemMP")
                    if plot_gpu:
                        for l in tmp:
                            legends.append(l + " GPU")
                    assert len(plot_res) == len(legends)
                    leg = []
#                    import pdb;pdb.set_trace()
                    skipped = 0
                    for i in ind:
                        if len(plot_res[i]) == 0:
                            skipped += 1
                            continue
                        leg.append(legends[i])
                        ii = i % len(tmp)
                        if ii < len(luas):
                            if ii == 0:
                                c = "#FFAAAA"
                            else:
                                c = "#FF0000"
                        else:
                            ii -= len(luas)
                            if ii % len(tmp) == 3:
                                c = '#0000FF'
                            elif ii % len(tmp) == 2:
                                c = '#4444FF'
                            elif ii % len(tmp) == 1:
                                c = '#8888FF'
                            else:
                                c = '#AAAAFF'
#                        if problem.startswith("control"):
#                            sep = 0
#                        else:
                        sep = i // tests_per_categ
#                        print i, len(suffixes), sep
#                        import pdb;pdb.set_trace()
                        pos = i * width + sep
                        if problem.startswith("control"):
                            pos = (i - skipped) * width + sep
                        rec.append(ax.bar(pos,
                                          plot_res[i][0],
                                          width, color=c))
                    host_data = host
                    if host.startswith("assam."):
                        host_data = "Core i7 930"
                        gpu_data = "GTX 480"
                    elif host.startswith("eos"):
                        host_data = "Core i7-2600K"
                        if host.startswith("eos3"):
                            gpu_data = "GTX 680"
                        if host.startswith("eos4") or host.startswith("eos5") or host.startswith("eos6"):
                            gpu_data = "GTX 580"
                    elif host.startswith("oolong"):
                        host_data = "Core2 Duo E8500"
                        gpu_data = "GTX 470"

                    if problem.startswith("control"):
                        ax.set_ylabel('time for 10 calls(s)')
                        ax.set_xticks(ind[:2] * 2 * width + width + ind[:2])
                        ax.set_xticklabels(('CPU', 'OpenMP'))#', 'GPU'))
                        #ax.legend(rec[:2], leg[:2], loc=1)
                    else:
                        if idx == 1 or len(batch_sizes) == 1:
                            ax.set_ylabel('examples/second')
                        ax.set_xticks(ind[:3] * tests_per_categ * width +
                                      ind[:3] +
                                      tests_per_categ * width / 2)
                        if idx == n_plot - 1:
                            ax.set_xticklabels(('CPU', 'OpenMP', 'GPU'))
                        else:
                            ax.set_xticklabels(('', '', ''))
                        size = tests_per_categ - len(suffixes) - 1
                        #ax.legend(rec[:size], leg[:size], loc=2)
                        ax.set_title("batch " + str(batch))
                plt.savefig(problem + ".pdf")
                plt.show()
