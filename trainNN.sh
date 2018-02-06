#!/bin/bash

#$ -cwd
#$ -pe mcore 3

#$ -l container=True
#...$ -v CONTAINER=CENTOS7
#$ -v CONTAINER=UBUNTU16

#...$ -v SGEIN=script.py
#...$ -v SGEIN=pima-indians-diabetes.data

#...$ -v SGEOUT=accuracy.pickle
#...$ -v SGEOUT=loss.pickle

#$ -l gpu,release=el7

cd /exper-sw/cmst3/cmssw/users/dbastos/StopNN/

module load root-6.10.02

python trainNN.py -bt -l 1 -n 200 -e 1000 -bs 30000 -r 0.003 -dc 0 -dp 0
