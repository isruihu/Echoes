#!/bin/bash

biasA_id=20
biasB_id=39

target_id=23
python create_dataset/celeba/gen_celeba.py \
    --target_id ${target_id} \
    --biasA_id ${biasA_id} \
    --biasB_id ${biasB_id} \

target_id=31
python create_dataset/celeba/gen_celeba.py \
    --target_id ${target_id} \
    --biasA_id ${biasA_id} \
    --biasB_id ${biasB_id} \

target_id=1
python create_dataset/celeba/gen_celeba.py \
    --target_id ${target_id} \
    --biasA_id ${biasA_id} \
    --biasB_id ${biasB_id} \