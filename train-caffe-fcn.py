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

logging.basicConfig(level=logging.DEBUG)

base_dir = os.path.abspath(os.path.dirname(__file__))
TMPL_ROOT = os.path.join(base_dir, "templates")

# templates are named as "xxx.tmpl", find all such files
def find_templates (root):
    paths = []
    for base, dirs, files in os.walk(root):
        for f in files:
            if '.tmpl' in f:
                paths.append(os.path.join(base, f))
    return paths

# enumerate the variables in templates
def print_template_variables (env, paths):
    from jinja2 import meta
    vars = set()
    for path in paths:
        src = env.loader.get_source(env, path)[0]
        ast = env.parse(src)
        vars |= meta.find_undeclared_variables(ast)
        pass
    print "Templates variables:", vars
    pass

def render_templates (env, paths, params):
    for path in paths:
        template = env.get_template(path)
        fname = os.path.splitext(path)[0]
        with open(fname, 'w') as f:
            f.write(template.render(params))
            pass
        pass
    pass

# prepare workspace
def prepare_ws (ws, params):
    # tmpl dir to copy from
    TMPL_DIR = os.path.join(TMPL_ROOT, params['template'])
    assert os.path.isdir(TMPL_DIR)
    assert not os.path.exists(ws)
    os.mkdir(ws)
    os.chdir(ws)
    # copy data
    cmd = "cp -r %s/* ./" % TMPL_DIR
    subprocess.check_call(cmd, shell=True)

    TO_BE_REPLACED = find_templates('.')

    env = Environment(loader=FileSystemLoader(searchpath='.'))
    tmpls = find_templates('.')

    print_template_variables(env, tmpls)
    render_templates(env, tmpls, params)
    pass

parser = argparse.ArgumentParser()
parser.add_argument("template", nargs=1)
parser.add_argument("db", nargs=1)  # database
parser.add_argument("ws", nargs=1)  # workspace, must not exist
parser.add_argument("--it", default=100000, type=int)
parser.add_argument("--channels", default=3, type=int)
parser.add_argument("--split", default=5, type=int)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--snapshot", default=1000, type=int)

args = parser.parse_args()

db = os.path.abspath(args.db[0])
ws = args.ws[0]

if os.path.exists(ws):
    logging.error("%s already exists" % ws)
    sys.exit(1)

params = {
        "template": args.template[0],
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

prepare_ws(ws, params)
sys.exit(0)

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

