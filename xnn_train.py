import sys 
import os
import glob
import shutil
import logging
import argparse
import subprocess
import argparse
import pickle
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
# - copy templates
# - replace params
# - chdir into the working directory
def prepare_ws_chdir (ws, params):
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
    with open('params.pickle', 'w') as f:
        pickle.dump(params, f)
    pass

def basic_args (snapshot = 4000, channels=3, iteration=400000):
    parser = argparse.ArgumentParser()
    parser.add_argument("template", nargs=1)
    parser.add_argument("db", nargs=1)  # database
    parser.add_argument("ws", nargs=1)  # workspace, must not exist
    parser.add_argument("--it", default=iteration, type=int)
    parser.add_argument("--channels", default=channels, type=int)
    parser.add_argument("--split", default=5, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--snapshot", default=snapshot, type=int)
    parser.add_argument("--mixin")
    parser.add_argument("--anno_min_ratio", default=0.05, type=float)
    return parser

def caffe_scan_snapshots ():
    shots = []
# find all saved snapshots
    for x in glob.glob('snapshots/*.caffemodel'):
        it = int(x.split('_')[-1].split('.')[0])
        shots.append((it, x))
        pass
# sort by iteration
    shots = sorted(shots, key = lambda x: x[0])
    return shots

# must work within ws directory
def caffe_eval_fcn ():
    if not os.path.exists("eval"):
        os.mkdir("eval")
        pass
    with open('params.pickle', 'r') as f:
        params = pickle.load(f)
    hist = []
    best = None
    best_score = 100
    best_path = None
    shots = caffe_scan_snapshots()

    params_path = 'model/caffe.params'
    for it, path in shots:
        print it, path
        out = os.path.join('eval', str(it))
        if os.path.exists(params_path):
            os.remove(params_path)
        os.symlink(os.path.abspath(path), params_path)
        if not os.path.exists(out):
            subprocess.check_call('%s model %s --mode 1 --anno_min_ratio %g --split %d --split_fold %d --annotate json --channels %s > %s' % (os.path.join(base_dir, 'xnn-roc'), params['db_path'], params['anno_min_ratio'], params['split'], params['split_fold'], params['channels'], out), shell=True)
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
