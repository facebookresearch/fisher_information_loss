#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

REPO_ROOT=`pwd`
RESULT_FOLDER=$REPO_ROOT/uai2021

### MNIST AND CIFAR EXPERIMENTS ###

# Linear regression for MNIST and CIFAR-10
for DATASET in "mnist" "cifar10"
do
  RESULTS_FILE="${RESULT_FOLDER}/${DATASET}_linear_pca20.json"
  python $REPO_ROOT/fisher_experiment.py \
    --dataset $DATASET \
    --model least_squares \
    --trials 1 \
    --results_file $RESULTS_FILE
done

# Logistic regression for MNIST and CIFAR-10
RESULTS_FILE="${RESULT_FOLDER}/mnist_logistic_pca20.json"
python $REPO_ROOT/fisher_experiment.py \
  --dataset mnist \
  --model logistic \
  --l2 8e-4 \
  --trials 1 \
  --results_file $RESULTS_FILE

RESULTS_FILE="${RESULT_FOLDER}/cifar10_logistic_pca20.json"
python $REPO_ROOT/fisher_experiment.py \
  --dataset cifar10 \
  --model logistic \
  --l2 8e-5 \
  --trials 1 \
  --results_file $RESULTS_FILE

# Iteratively reweighted FP for linear regression
for DATASET in "mnist" "cifar10"
do
  RESULTS_FILE="${RESULT_FOLDER}/${DATASET}_linear_reweighted"
  python $REPO_ROOT/reweighted.py \
    --dataset $DATASET \
    --model least_squares \
    --weight_method sample \
    --iters 15 \
    --results_file $RESULTS_FILE
done

# Iteratively reweighted FP for logistic regression
RESULTS_FILE="${RESULT_FOLDER}/mnist_logistic_reweighted"
python $REPO_ROOT/reweighted.py \
  --dataset mnist \
  --model logistic \
  --weight_method sample \
  --iters 15 \
  --l2 8e-4 \
  --results_file $RESULTS_FILE

RESULTS_FILE="${RESULT_FOLDER}/cifar10_logistic_reweighted"
python $REPO_ROOT/reweighted.py \
  --dataset cifar10 \
  --model logistic \
  --weight_method sample \
  --iters 15 \
  --l2 8e-5 \
  --results_file $RESULTS_FILE


### IWPC EXPERIMENTS ###

DATASET="iwpc"
MODEL="least_squares"

# For L2 and sigma inversion plots:
for L2 in "1e-5" "1e-3" "1e-1" "1"
do
  FIL_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_fil_l2_${L2}"
  python $REPO_ROOT/reweighted.py \
    --dataset $DATASET \
    --model $MODEL \
    --pca_dims 0 \
    --no_norm \
    --l2 $L2 \
    --attribute 11 13 \
    --results_file $FIL_RESULTS

  INVERSION_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_inversion_l2_${L2}.json"
  python $REPO_ROOT/model_inversion.py \
    --inverter all \
    --dataset $DATASET \
    --model $MODEL \
    --l2 $L2 \
    --results_file $INVERSION_RESULTS
  
  for INVERTER in 'fredrikson14' 'whitebox'
  do
    INVERSION_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_${INVERTER}_private_inversion_l2_${L2}.json"
    python $REPO_ROOT/private_model_inversion.py \
      --dataset $DATASET \
      --trials 100 \
      --noise_scales 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1 \
      --inverter $INVERTER \
      --model $MODEL \
      --l2 $L2 \
      --results_file $INVERSION_RESULTS
  done
done

# For IRFIL inversion plots:
L2=1e-2
IRFIL_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_irfil"
python $REPO_ROOT/reweighted.py \
  --dataset $DATASET \
  --model $MODEL \
  --pca_dims 0 \
  --iters 10 \
  --no_norm \
  --l2 $L2 \
  --attribute 11 13 \
  --results_file $IRFIL_RESULTS

for INVERTER in 'fredrikson14' 'whitebox'
do
  INVERSION_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_${INVERTER}_private_inversion_irfil.json"
  python $REPO_ROOT/private_model_inversion.py \
    --dataset $DATASET \
    --trials 100 \
    --noise_scales 1e-4 1e-3 1e-2 \
    --inverter $INVERTER \
    --model $MODEL \
    --l2 $L2 \
    --weights_file ${IRFIL_RESULTS}.pth \
    --results_file $INVERSION_RESULTS
done

### UCI ADULT EXPERIMENTS ###
DATASET="uciadult"
MODEL="least_squares"
L2=1e-3

RESULTS_FILE="${RESULT_FOLDER}/${DATASET}_${MODEL}_inversion.json"
python $REPO_ROOT/model_inversion.py \
  --dataset $DATASET \
  --model $MODEL \
  --l2 $L2 \
  --inverter all \
  --results_file $RESULTS_FILE

IRFIL_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_irfil"
python $REPO_ROOT/reweighted.py \
  --dataset $DATASET \
  --model $MODEL \
  --iters 10 \
  --l2 $L2 \
  --pca_dims 0 \
  --no_norm \
  --attribute 24 25 \
  --results_file $IRFIL_RESULTS

RESULTS_FILE="${RESULT_FOLDER}/${DATASET}_${MODEL}_whitebox_private_inversion_irfil.json"
python $REPO_ROOT/private_model_inversion.py \
  --dataset $DATASET \
  --model $MODEL \
  --l2 $L2 \
  --inverter whitebox \
  --trials 100 \
  --noise_scales 1e-4 1e-3 1e-2 1e-1 1 \
  --weights_file ${IRFIL_RESULTS}.pth \
  --results_file $RESULTS_FILE
