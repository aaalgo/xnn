#!/usr/bin/env python
import glob
import os
import sys
import subprocess
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1)

for p in sys.argv[1:]:
    name = os.path.basename(os.path.dirname(p))
    print name
    with open(p, 'rb') as f:
        hist = pickle.load(f)
    x, y = zip(*hist)
    ax.plot(x, y, label=name)
    pass

fig.savefig('eval.png')
plt.close(fig)

