#!/bin/bash

cd ../..

DATA='data/data'


DATASET=$1
NLAB=$2 # total number of labels
DEVICE=$3

exp_name=$4

echo "DATASET: ${DATASET}"
echo "NLAB: ${NLAB}"
echo "DEVICE: ${DEVICE}"
echo "exp_name: ${exp_name}"
echo "program started"

sleep 1m
echo "program ended at gpu ${DEVICE}"