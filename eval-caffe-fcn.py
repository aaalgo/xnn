#!/usr/bin/env python
import sys 
import os
import logging
import argparse
import xnn_train

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("ws", nargs=1)  # workspace, must not exist

args = parser.parse_args()
ws = args.ws[0]

if not os.path.exists(ws):
    logging.error("%s does not exists" % ws)
    sys.exit(1)

os.chdir(ws)
xnn_train.caffe_eval_fcn()

