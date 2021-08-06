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


def compute_accuracy(model, data, noise_scale=0, trials=100, regression=False):
    accuracies = []
    X, y = data["features"], data["targets"]
    theta = model.get_params()
    for _ in range(trials):
        theta_priv = theta + torch.randn_like(theta) * noise_scale
        model.set_params(theta_priv)
        if regression:
            acc = model.loss(data).mean().item()
        else:
            predictions = model.predict(X)
            acc = ((predictions == y).float()).mean().item()
        accuracies.append(acc)
    model.set_params(theta)
    accuracies = torch.tensor(accuracies)
    return torch.mean(accuracies).item(), torch.std(accuracies).item()


def clip_data(data, etas, clip):
    keep_ids = etas < clip
    return {"features" : data["features"][keep_ids, ...],
            "targets" : data["targets"][keep_ids]}


def eval_comparison_stats(model, data):
    theta = model.get_params()
    theta.requires_grad = True

    # compute per sample losses:
    losses = model.loss(data)

    # compute the norm of the gradient of the loss at each sample
    # w.r.t. the model weights:
    def func(theta):
        model.theta = theta
        return model.loss(data)
    ind_grads = torch.autograd.functional.jacobian(func, theta)
    grad_norms = ind_grads.norm(dim=1)

    theta.requires_grad = False

    # compute per sample inner product with weights
    weight_dots = data["features"] @ theta

    return losses.tolist(), weight_dots.tolist(), grad_norms.tolist()

def main(args):
    regression = (args.dataset == "iwpc")
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
    logging.info(f"Training model {args.model}")
    model.train(data, l2=args.l2)

    # Check predictions for sanity:
    accuracy, _ = compute_accuracy(model, data, regression=regression)
    if regression:
        logging.info("Training MSE of classifier {:.3f}".format(accuracy))
    else:
        logging.info("Training accuracy of classifier {:.3f}".format(accuracy))

    # Compute the Jacobian of the influence of each example on the optimal
    # parameters:
    logging.info(f"Computing influence Jacobian on training set...")
    start = time.time()
    J = model.influence_jacobian(data)
    time_per_sample = 1e3 * (time.time() - start) / len(data["targets"])
    logging.info("Time taken per example {:.3f} (ms)".format(time_per_sample))

    # Compute the Fisher information loss from the FIM (J^T J) for each example
    # in the training set (J^T J is the Fisher information with Gaussian output
    # perturbation on the parameters at a scale of 1):
    start = time.time()
    logging.info(f"Computing Fisher information loss...")
    etas = models.compute_information_loss(J)
    time_per_sample = 1e3 * (time.time() - start) / len(etas)
    logging.info(
        "Computed {} examples, maximum eta: {:.3f}, "
        "time per sample {:.3f} (ms).".format(
            len(etas), max(etas), time_per_sample))

    # Compute some comparison points:
    losses, weight_dots, grad_norms = eval_comparison_stats(model, data)

    # Retrain the model and measure the new etas if removing most lossy
    # examples:
    if args.clip > 0:
        clipped_data = clip_data(data, etas, args.clip)
        logging.info(
            "Kept {}/{} samples, retrain and compute eta..".format(
                len(clipped_data["targets"]), len(data["targets"])))
        model.train(clipped_data, l2=args.l2)
        J = model.influence_jacobian(clipped_data)
        etas = models.compute_information_loss(J)
        etamax = max(etas)
    else:
        etamax = max(etas)

    # Measure the test accuracy as a function of the noise needed to attain a
    # desired eta:
    accuracies = []
    stds = []
    for eta in args.etas:
        # Compute the Gaussian noise scale needed for eta:
        scale = etamax / eta
        # Measure test accuracy:
        accuracy, std = compute_accuracy(
            model, test_data, noise_scale=scale, trials=args.trials,
            regression=regression)
        accuracies.append(accuracy)
        stds.append(std)

    results = {
        "clip" : args.clip,
        "accuracies" : accuracies,
        "stds" : stds,
        "etas" : etas.tolist(),
        "train_losses" : losses,
        "train_dot_weights" : weight_dots,
        "train_grad_norms" : grad_norms,
    }
    with open(args.results_file, 'w') as fid:
        json.dump(results, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fisher information loss.")
    parser.add_argument("--data_folder", default="/tmp", type=str,
        help="folder in which to store data (default: '/tmp')")
    parser.add_argument("--dataset", default="mnist", type=str,
        choices=["mnist", "cifar10", "cifar100", "uciadult", "iwpc"],
        help="dataset to use.")
    parser.add_argument("--model", default="least_squares", type=str,
        choices=["least_squares", "logistic"],
        help="type of model (default: least_squares)")
    parser.add_argument("--pca_dims", default=20, type=int,
        help="Number of PCA dimensions (if 0, uses raw features)")
    parser.add_argument("--no_norm", default=False, action="store_true",
        help="Don't normalize examples to lie in unit ball")
    parser.add_argument("--l2", default=0, type=float,
        help="l2 regularization parameter")
    parser.add_argument("--results_file",
        default="/tmp/private_model_results.json", type=str,
        help="file in which to save the results")
    parser.add_argument('--etas', metavar='N', type=float,
        nargs='+', default=[1.0],
        help='Fisher information loss levels (eta) to evaluate accuracy')
    parser.add_argument('--clip', type=float, default=0.0,
        help='eta removal threshold for data')
    parser.add_argument('--trials', type=int, default=100,
        help='number of trials to evaluate an output perturbed model')
    args = parser.parse_args()
    main(args)
