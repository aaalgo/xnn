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
TMPL_ROOT = os.path.join(base_dir, "templates")

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("db", nargs=1)
parser.add_argument("ws", nargs=1)
parser.add_argument("--it", default=100000, type=int)
parser.add_argument("--channels", default=3, type=int)
parser.add_argument("--split", default=5, type=int)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--snapshot", default=1000, type=int)

args = parser.parse_args()

db = os.path.abspath(args.db[0])
ws = args.ws[0]

MAX_R = 100000

if os.path.exists(ws):
    logging.error("%s already exists" % ws)
    sys.exit(1)

params = {
        "template": "fcn",
        "channels": args.channels,
        "db_path": db,
        "split": args.split,
        "split_fold": args.fold,
        "num_output": 2,
        "display_interval": 1000,
        "snapshot_interval": args.snapshot,
        "max_iter": args.it,
        "device": "GPU",
}

os.mkdir(ws)
os.chdir(ws)

TMPL_DIR = os.path.join(TMPL_ROOT, params['template'])

TO_BE_REPLACED = ["train.prototxt", "model/caffe.model", "solver.prototxt"]

cmd = "cp -r %s/* ./" % TMPL_DIR
subprocess.check_call(cmd, shell=True)

templateLoader = FileSystemLoader(searchpath='.')
templateEnv = Environment(loader=templateLoader)

for path in TO_BE_REPLACED:
    template = templateEnv.get_template(path)
    out = template.render(params)
    with open(path, 'w') as f:
        f.write(out)

subprocess.check_call("./train.sh 2>&1 | tee train.log", shell=True)

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
        print "%d already done, skipping..." % it
    else:
        os.symlink(os.path.abspath(path), 'model/caffe.params')
        subprocess.check_call('%s model %s --mode 1 --split %d --fold %d --channels %s > %s' % (os.path.join(base_dir, 'xnn-roc'), db, args.split, args.fold, args.channels, out), shell=True)
        os.remove('model/caffe.params')
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

print "Best iteration is %s, with score %g" % (best, best_score)
shutil.copy(best_path, 'model/caffe.params')

