#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import math
import time
import torch

import models
import dataloading
from model_inversion import fredrikson14_inverter, WhiteboxInverter, compute_metrics, features_to_category

# set up logger:
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def eval_model(model, data, regression):
    predictions = model.predict(data["features"], regression=regression)
    if regression:
        acc = (predictions - data["targets"]).pow(2).mean()
    else:
        acc = ((predictions == data["targets"]).float()).mean()
    return acc.item()


def run_inversion(args, data, test_data, weights):
    regression = (args.dataset == "iwpc" or args.dataset == "synth")

    # Train model:
    model = models.get_model(args.model)
    logging.info(f"Training model {args.model}")
    model.train(data, l2=args.l2, weights=weights)

    if args.dataset == "uciadult":
        target_attribute = (24, 25) # [not married, married]
    elif args.dataset == "iwpc":
        #target_attribute = (2, 7) # CYP2C9 genotype
        target_attribute = (11, 13) # VKORC1 genotype
    elif args.dataset == "synth":
        target_attribute = (0, 2)
    else:
        raise NotImplementedError("Dataset not yet implemented.")

    if args.inverter == "fredrikson14":
        def invert(private_model, noise_scale=None):
            return fredrikson14_inverter(
                data, target_attribute, private_model, weights)
        invert_fn = invert
    elif args.inverter == "whitebox":
        inverter = WhiteboxInverter(
            data, target_attribute, type(model), weights, args.l2)
        def invert(private_model, noise_scale=None):
            return inverter.predict(private_model, gamma=noise_scale)
        invert_fn = invert

    results = {}
    theta = model.get_params()
    for noise_scale in args.noise_scales:
        logging.info(f"Running inversion for noise scale {noise_scale}.")
        all_predictions = []
        train_accs = []
        test_accs = []
        for trial in range(args.trials):
            # Add noise:
            theta_priv = theta + torch.randn_like(theta) * noise_scale
            model.set_params(theta_priv)
            # Check train and test predictions:
            train_acc = eval_model(model, data, regression)
            test_acc = eval_model(model, test_data, regression)
            if regression:
                logging.info(f"MSE Train {train_acc:.3f}, MSE Test {test_acc:.3f}.")
            else:
                logging.info(f"Acc Train {train_acc:.3f}, Acc Test {test_acc:.3f}.")
            predictions = invert_fn(model, noise_scale=noise_scale)
            acc = compute_metrics(data, predictions, target_attribute)
            logging.info(f"Private inversion accuracy {acc:.4f}")
            all_predictions.append(predictions.tolist())
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        results[noise_scale] = {
                "predictions" : all_predictions,
                "train_acc" : train_accs,
                "test_acc" : test_accs,
            }

    results["target"] = features_to_category(
            data["features"][:, range(*target_attribute)]).tolist()
    return results


def main(args):
    regression = (args.dataset == "iwpc" or args.dataset == "synth")
    data = dataloading.load_dataset(
        name=args.dataset, split="train", normalize=False,
        num_classes=2, root=args.data_folder, regression=regression)
    test_data = dataloading.load_dataset(
        name=args.dataset, split="test", normalize=False,
        num_classes=2, root=args.data_folder, regression=regression)

    if args.subsample > 0:
        data = dataloading.subsample(data, args.subsample)

    if args.weights_file is not None:
        all_weights = torch.load(args.weights_file)
    else:
        all_weights = [torch.ones(len(data["targets"]))]

    results = []
    for it, weights in enumerate(all_weights):
        if len(all_weights) > 1:
            logging.info(f"Iteration {it} weights for model inversion.")
        results.append(run_inversion(args, data, test_data, weights))

    if args.results_file is not None:
        with open(args.results_file, 'w') as fid:
            json.dump(results, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inversion.")
    parser.add_argument("--data_folder", default="/tmp", type=str,
        help="folder in which to store data (default: '/tmp')")
    parser.add_argument("--dataset", default="uciadult", type=str,
        choices=["uciadult", "iwpc", "synth"],
        help="dataset to use.")
    parser.add_argument("--model", default="least_squares", type=str,
        choices=["least_squares", "logistic"],
        help="type of model (default: least_squares)")
    parser.add_argument("--l2", default=0, type=float,
        help="l2 regularization parameter")
    parser.add_argument("--inverter", default="whitebox", type=str,
        choices=["fredrikson14", "whitebox"],
        help="inversion method to use (default: whitebox)")
    parser.add_argument("--noise_scales", metavar='N', type=float,
        nargs='+', default=[0],
        help="Gaussian noise scales for output perturbation")
    parser.add_argument("--trials", default=1, type=int,
        help="number of noise vectors to test")
    parser.add_argument("--subsample", default=0, type=int,
        help="number of training examples")
    parser.add_argument("--weights_file", default=None, type=str,
        help="(optional) file to load IRFIL weights from")
    parser.add_argument("--results_file", default=None, type=str,
        help="(optional) path to save results")
    args = parser.parse_args()
    main(args)
