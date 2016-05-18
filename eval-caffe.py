#!/usr/bin/env python
import sys 
import os
import re
import glob
import shutil
import logging
import argparse
import subprocess
import simplejson as json
from jinja2 import Environment, FileSystemLoader

base_dir = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("ws", nargs=1)  # workspace, must not exist
parser.add_argument("--channels", default=3, type=int)
parser.add_argument("--split", default=10, type=int)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--annotate", default="none")
parser.add_argument("--batch", default=16, type=int)
args = parser.parse_args()

#db = os.path.abspath(args.db[0])
ws = args.ws[0]

assert os.path.isdir(ws)
os.chdir(ws)

# evaluation
shots = []
# find all saved snapshots
for x in glob.glob('snapshots/*.caffemodel'):
    it = int(x.split('_')[-1].split('.')[0])
    shots.append((it, x))
    pass

# sort by iteration
shots = sorted(shots, key = lambda x: x[0])

if not os.path.exists("eval"):
    os.mkdir("eval")
    pass

hist = []
best = None
best_score = 100
best_path = None
for it, path in shots:
    print it, path
    out = os.path.join('eval', str(it))
    if os.path.exists(out):
        #print "%d already done, skipping..." % it
        subprocess.check_call('cat %s' % out, shell=True)
    else:
        if os.path.islink('model/caffe.params'):
            os.remove('model/caffe.params')
        os.symlink(os.path.abspath(path), 'model/caffe.params')
        cmd = '%s model db --batch %s --mode 1 --split %d --fold %d --annotate %s | tee %s' % (os.path.join(base_dir, 'xnn-roc'), args.batch, args.split, args.fold, args.annotate, out)
        print cmd
        subprocess.check_call(cmd, shell=True)
        os.remove('model/caffe.params')
    continue
    cc = []
    with open(out, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            if (len(l) != 2):
                continue
            x, y = l
            cc.append((float(x), float(y)))
            #if float(y) > 0.5:
            #    hist.append((it, float(x)))
            #    break
            #pass
            pass
    for x, y in cc:
        if y > 0.5:
            hist.append((it, x))
            if x <= best_score:
                best = it
                best_score = x
                best_path = path
            break
        pass
    print hist[-1]
    pass

sys.exit(0)

print "Best iteration is %s, with score %g" % (best, best_score)
shutil.copy(best_path, 'model/caffe.params')

