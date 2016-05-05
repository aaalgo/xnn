#!/usr/bin/env sh

export GLOG_log_dir=log
export GLOG_logtostderr=1

CAFFE=caffe

mkdir -p log snapshots

SNAP=$1
if [ -z "$SNAP" ]
then
    $CAFFE train --solver quick_solver.prototxt $*
else
    shift
    $CAFFE train -solver quick_solver.prototxt -snapshot $SNAP $*
fi

