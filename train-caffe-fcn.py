#!/usr/bin/env python
import sys 
import os
import re
import glob
import shutil
import logging
import subprocess
import argparse
import xnn_train

logging.basicConfig(level=logging.DEBUG)

parser = xnn_train.basic_args()

args = parser.parse_args()

db = os.path.abspath(args.db[0])
mixin = args.mixin
if mixin:
    mixin = os.path.abspath(mixin)
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
        "annotate": "json",
        "mixin": mixin,
        "mixin_group_delta": 1,
        "anno_min_ratio": args.anno_min_ratio
}

xnn_train.prepare_ws_chdir(ws, params)

subprocess.check_call("./train.sh 2>&1 | tee train.log", shell=True)

# evaluation
xnn_train.caffe_eval_fcn()


