#!/usr/bin/env python
import glob
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import pickle

fig, ax = plt.subplots(nrows=1, ncols=1)

l_handles = []
l_labels = []
for p in sys.argv[1:]:
    name = os.path.basename(os.path.dirname(p))
    print name
    with open(p, 'rb') as f:
        hist = pickle.load(f)
    x, y = zip(*hist)
    l, = ax.plot(x, y)
    l_handles.append(l)
    l_labels.append(name)
    pass

fig.legend(l_handles, l_labels)
fig.savefig('eval.png')
plt.close(fig)

