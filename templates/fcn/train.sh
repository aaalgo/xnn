#!/usr/bin/env sh

export GLOG_log_dir=log
export GLOG_logtostderr=1

CAFFE=caffe-fcn

mkdir -p log snapshots

SNAP=$1
if [ -z "$SNAP" ]
then
    $CAFFE train --solver solver.prototxt $*
else
    shift
    $CAFFE train -solver solver.prototxt -snapshot $SNAP $*
fi

