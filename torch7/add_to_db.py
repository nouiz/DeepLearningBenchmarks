"""
Write the results of run.sh to the pickled database of timing results

add_to_db.py --db DB_FILE.pkl results_files...

"""
import os
import sys
import cPickle

def main():
    assert sys.argv[1] == '--db'
    try:
        db = cPickle.load(open(sys.argv[2]))
    except IOError:
        db = []

    for results_file in sys.argv[3:]:
        template = dict()
        if results_file.endswith('_cuda'):
            template['device'] = 'GPU'
            template['OMP_NUM_THREADS'] = 1
        elif results_file.endswith('_openmp'):
            template['device'] = 'CPU'
        for lineno, line in enumerate(open(results_file)):
            if (line.startswith('|============') or
                line.startswith('+---------------------') or
                line.startswith('|---------------------') or
                line.startswith('| ') or
                line.startswith("Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU")):
                continue
            if '=' in line:
                key = line[:line.index('=')]
                val = line[line.index('=') + 1:]
                if key in ('host', 'device', 'OMP_NUM_THREADS'):
                    template[key] = val.strip()
                elif key in ('OpenMP') :
                    template[key] = bool(int(val))
                elif key in ('batch', 'precision'):
                    template[key] = int(val)
                else:
                    raise ValueError(results_file, line, key)
            elif line.startswith('unset OMP_NUM_THREADS'):
                template['OMP_NUM_THREADS'] = -1

            elif line.startswith('mlp'):
                problem, speed_str = line.split('\t')
                entry = dict(template)
                entry['problem'] = problem
                entry['speed'] = float(speed_str)
                db.append(entry)
            elif line.startswith('cnn'):
                problem, speed_str = line.split('\t')
                entry = dict(template)
                entry['problem'] = problem
                entry['speed'] = float(speed_str)
                db.append(entry)
            elif line.startswith('control'):
                problem, speed_str = line.split('\t')
                entry = dict(template)
                entry['problem'] = problem
                entry['speed'] = float(speed_str)
                db.append(entry)
            else:
                print "ERROR: ", results_file, line

    if 1:
        print "Writing database to", sys.argv[2]
        cPickle.dump(db, open(sys.argv[2], 'wb'))
    else:
        print "DEBUG FINAL DB:"
        for entry in db:
            print entry

if __name__ == '__main__':
    sys.exit(main())
