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

# set up logger:
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def compute_accuracy(model, data, regression=False):
    X, y = data["features"], data["targets"]
    if regression:
        acc = model.loss(data).mean().item()
    else:
        predictions = model.predict(X)
        acc = ((predictions == y).float()).mean().item()
    return acc


def get_weights(method, prev_weights, data):
    weights = torch.ones(len(data["targets"]))
    if method == "sample":
        weights[:] = prev_weights.data
    elif method == "class":
        n_class = data["targets"].max() + 1
        for c in range(n_class):
            mask = data["targets"] == c
            weights[mask] = prev_weights[mask].mean()
    elif method == "hybrid":
        n_class = data["targets"].max() + 1
        for c in range(n_class):
            mask = data["targets"] == c
            weights[mask] = prev_weights[mask].mean()
        weights *= prev_weights
    else:
        raise ValueError(f"Invalid weight method {method}.")
    return weights


def main(args):
    regression = args.dataset == "iwpc" or args.dataset == "synth"
    data = dataloading.load_dataset(
        name=args.dataset, split="train", normalize=not args.no_norm,
        num_classes=2, root=args.data_folder, regression=regression)
    test_data = dataloading.load_dataset(
        name=args.dataset, split="test", normalize=not args.no_norm,
        num_classes=2, root=args.data_folder, regression=regression)
    if args.pca_dims > 0:
        data, pca = dataloading.pca(data, num_dims=args.pca_dims)
        test_data, _ = dataloading.pca(test_data, mapping=pca)

    model = models.get_model(args.model)

    # Find the optimal parameters for the model:
    logging.info(f"Training {args.model} model.")
    model.train(data, l2=args.l2)

    train_accuracy = compute_accuracy(model, data, regression=regression)
    test_accuracy = compute_accuracy(model, test_data, regression=regression)
    if regression:
        logging.info(f"MSE train {train_accuracy:.3f},"
            f" test: {test_accuracy:.3f}.")
    else:
        logging.info(f"Accuracy train {train_accuracy:.3f},"
            f" test: {test_accuracy:.3f}.")

    # Compute the Fisher information loss, eta, for each example in the
    # training set:
    logging.info("Computing unweighted etas on training set...")
    J = model.influence_jacobian(data)
    etas = models.compute_information_loss(J, target_attribute=args.attribute,
                                           constrained=args.constrained)
    logging.info(f"etas max: {etas.max().item():.4f},"
        f" mean: {etas.mean().item():.4f}, std: {etas.std().item():.4f}.")

    # Reweight using the fisher information loss:
    updated_fi = etas.reciprocal().detach()
    maxs = [etas.max().item()]
    means = [etas.mean().item()]
    stds = [etas.std().item()]
    train_accs = [train_accuracy]
    test_accs = [test_accuracy]
    all_weights = [torch.ones(len(updated_fi))]
    for i in range(args.iters):
        logging.info(f"Iter {i}: Training weighted model...")
        updated_fi *= (len(updated_fi) / updated_fi.sum())
        # TODO does it make sense to renormalize after clamping?
        updated_fi.clamp_(min=args.min_weight, max=args.max_weight)
        weights = get_weights(args.weight_method, updated_fi, data)
        model.train(data, l2=args.l2, weights=weights.detach())

        # Check predictions of weighted model:
        train_accuracy = compute_accuracy(model, data, regression=regression)
        test_accuracy = compute_accuracy(model, test_data, regression=regression)
        if regression:
            logging.info(f"Weighted model MSE train {train_accuracy:.3f},"
                f" test: {test_accuracy:.3f}.")
        else:
            logging.info(f"Weighted model accuracy train {train_accuracy:.3f},"
                f" test: {test_accuracy:.3f}.")

        J = model.influence_jacobian(data)
        weighted_etas = models.compute_information_loss(J, target_attribute=args.attribute,
                                                        constrained=args.constrained)
        updated_fi /= weighted_etas
        maxs.append(weighted_etas.max().item())
        means.append(weighted_etas.mean().item())
        stds.append(weighted_etas.std().item())
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        all_weights.append(weights)
        logging.info(f"Weighted etas max: {maxs[-1]:.4f},"
            f" mean: {means[-1]:.4f},"
            f" std: {stds[-1]:.4f}.")

    results = {
        "weights" : weights.tolist(),
        "etas" : etas.tolist(),
        "weighted_etas" : weighted_etas.tolist(),
        "eta_maxes" : maxs,
        "eta_means" : means,
        "eta_stds" : stds,
        "train_accs" : train_accs,
        "test_accs" : test_accs,
    }

    with open(args.results_file + ".json", 'w') as fid:
        json.dump(results, fid)
    torch.save(torch.stack(all_weights), args.results_file + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Information loss.")
    parser.add_argument("--data_folder", default="/tmp", type=str,
        help="folder in which to store data (default: '/tmp')")
    parser.add_argument("--dataset", default="mnist", type=str,
        choices=["mnist", "cifar10", "cifar100", "uciadult", "iwpc", "synth"],
        help="dataset to use.")
    parser.add_argument("--model", default="least_squares", type=str,
        choices=["least_squares", "logistic"],
        help="type of model (default: least_squares)")
    parser.add_argument("--weight_method", default="sample", type=str,
        choices=["sample", "class", "hybrid"],
        help="Method to weight the loss by (default: sample)")
    parser.add_argument("--min_weight", default=0, type=float,
        help="Minimum per-sample weight (default: 0)")
    parser.add_argument("--max_weight", default=float("inf"), type=float,
        help="Maximum per-sample weight (default: inf)")
    parser.add_argument("--attribute", default=None, nargs="+", type=int,
        help="Which attributes to reweight for privacy (None for full)")
    parser.add_argument("--iters", default=1, type=int,
        help="Number of iterations.")
    parser.add_argument("--pca_dims", default=20, type=int,
        help="Number of PCA dimensions (if 0, uses raw features)")
    parser.add_argument("--no_norm", default=False, action="store_true",
        help="Don't normalize examples to lie in unit ball")
    parser.add_argument("--constrained", default=False, action="store_true",
        help="Use constrained Fisher information matrix")
    parser.add_argument("--l2", default=0, type=float,
        help="l2 regularization parameter")
    parser.add_argument("--results_file",
        default="/tmp/private_model_results", type=str,
        help="file in which to save the results")
    args = parser.parse_args()
    main(args)
