#!/usr/bin/env python
import glob
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import pickle

def load (dir):
    all = []
    for x in glob.glob('%s/*' % dir):
        it = int(os.path.basename(x))
        all.append((it, x))
        pass
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
    return hist

fig, ax = plt.subplots(nrows=1, ncols=1)

l_handles = []
l_labels = []
for name in sys.argv[1:]:
    hist = load(os.path.join(name, 'eval'))
    x, y = zip(*hist)
    l, = ax.plot(x, y)
    l_handles.append(l)
    l_labels.append(name)
    pass

fig.legend(l_handles, l_labels)
fig.savefig('eval.png')
plt.close(fig)

