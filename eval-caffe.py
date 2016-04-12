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

check_exist("eval", "no eval directory.")

all = []
# find all saved snapshots
for x in glob.glob('eval/*'):
    it = int(os.path.basename(x))
    all.append((it, x))
    pass

# sort by iteration
all = sorted(all, key = lambda x: x[0])

hist = []
for it, out in all:
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

