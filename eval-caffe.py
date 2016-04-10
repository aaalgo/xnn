#!/usr/bin/env python
import sys
import glob
import os
import subprocess
#import matplotlib.pyplot as plt

def check_exist (path, msg):
    if not os.path.exists(path):
        print msg
        sys.exit(1)
        pass
    pass

check_exist("snapshots", "no snapshots directory.")
check_exist("caffe.model", "no caffe.model (copy deploy.prototxt)")
check_exist("blobs", "no blobs (usually contains a single line 'prob'")
check_exist("db", "no db")

if not os.path.exists("eval"):
    os.mkdir("eval")

all = []
# find all saved snapshots
for x in glob.glob('snapshots/*.caffemodel'):
    it = int(x.split('_')[-1].split('.')[0])
    all.append((it, x))
    pass

# sort by iteration
all = sorted(all, key = lambda x: x[0])

hist = []
for it, path in all:
    print it, path
    out = os.path.join('eval', str(it))
    if os.path.exists(out):
        print "%d already done, skipping..." % it
    else:
        try:
            os.remove('caffe.params')
        except:
            pass
        os.symlink(os.path.abspath(path), 'caffe.params')
        subprocess.check_call('xnn-roc . db --mode 1 > %s' % out, shell=True)
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
            break
        pass
    print hist[-1]
    pass

import pickle
pickle.dump(hist, open('eval.curve', 'wb'))

