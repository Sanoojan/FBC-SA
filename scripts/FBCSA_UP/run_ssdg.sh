#!/bin/bash

cd ../..

DATA='/share/data/drive_1/Sanoojan/data'


DATASET=$1
NLAB=$2 # total number of labels
DEVICE=$3
exp_name=$4
exp_config=$5
echo exp_config: ${exp_config}

if [ ${DATASET} == ssdg_pacs ]; then
    # NLAB: 210 or 105
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == ssdg_officehome ]; then
    # NLAB: 1950 or 975
    D1=art
    D2=clipart
    D3=product
    D4=real_world
fi

TRAINER=FBCSA_UP
NET=resnet18

for SEED in 1 2 3 4 5
do
    for SETUP in 1 2 3 4
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --exp-name ${exp_name} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
        --output-dir output/${TRAINER}_${exp_name}/${DATASET}/nlab_${NLAB}/${NET}/${T}/seed${SEED} \
        --exp-config configs/trainers/${TRAINER}/${exp_config} \
        MODEL.BACKBONE.NAME ${NET} \
        DATASET.NUM_LABELED ${NLAB}
    done
done
