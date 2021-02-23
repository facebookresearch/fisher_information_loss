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


INVERTERS = ["baseline", "ideal", "fredrikson14", "whitebox", "all"]


def features_to_category(values, one_hot=False):
    if not one_hot:
        last_value = (values==0).all(axis=1, keepdim=True).to(values.dtype)
        values = torch.cat([values, last_value], dim=1)
    return torch.argmax(values, dim=1)


def compute_log_marginal(train_data, target_attribute, one_hot=False, weights=None):
    if weights is None:
        weights = torch.ones(train_data["features"].size(0))
    target_values = train_data["features"][:, range(*target_attribute)]
    if not one_hot:
        last_value = (target_values==0).all(
            axis=1, keepdim=True).to(target_values.dtype)
        target_values = torch.cat([target_values, last_value], dim=1)
    return (weights[:, None] * target_values).sum(axis=0).log()


def baseline_inverter(
        train_data, target_attribute, weights=None, one_hot=False, **kwargs):
    """
    The baseline inverter simply measures the prior for the target attribute
    (using the training data) and predicts the mode of the prior for every
    target example.
    """
    # NB A stronger baseline might have access to the joint prior and/or the
    # target label
    log_marginal = compute_log_marginal(
            train_data, target_attribute, weights=weights, one_hot=one_hot)
    prediction = torch.argmax(log_marginal)
    return torch.full(
        size=(train_data["features"].shape[0],),
        fill_value=prediction,
        dtype=torch.long)


def ideal_inverter(train_data, target_attribute, **kwargs):
    """
    The ideal inverter uses the training data to learn a model to predict the
    target attribute given all other features including the label.
    """
    def swap_feature_target(data):
        features = data["features"]
        new_features = torch.cat([
            features[:, :target_attribute[0]],
            features[:, target_attribute[1]:],
            data["targets"][:, None].float()
        ], axis=1)
        new_targets = features_to_category(
            features[:, range(*target_attribute)])
        return {
            "features" : new_features,
            "targets" : new_targets,
        }
    tmp_train = swap_feature_target(train_data)
    model = models.MultinomialLogistic()
    model.train(tmp_train)
    return model.predict(tmp_train["features"])


def fredrikson14_inverter(
        train_data, target_attribute, model, weights=None, one_hot=False, **kwargs):
    """
    Implements the model inversion attack of:
        Fredrikson, 2014, Privacy in Pharmacogenetics: An End-to-End Case Study
        of Personalized Warfarin Dosing
    """
    if weights is None:
        weights = torch.ones(train_data["features"].size(0))

    log_marginal = compute_log_marginal(
        train_data, target_attribute, weights=weights, one_hot=one_hot)

    if type(model) == models.LeastSquares:
        n, d = train_data["features"].shape
        std_var = (weights * model.loss(train_data)).sum().true_divide(n - d)
        score_fn = lambda data : -0.5 * (weights * model.loss(data)) / std_var
    elif type(model) == models.Logistic:
        preds = model.predict(train_data["features"])
        y = train_data["targets"]
        matched = preds == y
        confusions = torch.tensor([
            [matched[y == 0].sum(), (~matched)[y == 0].sum()],
            [(~matched)[y == 1].sum(), matched[y == 1].sum()]
        ])
        pi = confusions.true_divide(confusions.sum(axis=0, keepdim=True))
        def score_fn(data):
            preds = model.predict(data["features"])
            y = data["targets"]
            return pi[y, preds.long()].log()
    else:
        raise ValueError("Unknown model type.")

    # For each possible value of the target attribute compute score for the
    # attribute which should be proportional to log pi(y, y') + log p(x), where
    # pi(y, y') is a model dependent performance measure.
    tgt_features = train_data["features"].clone().detach()
    tgt_features[:, range(*target_attribute)] = 0.

    scores = []
    for c in range(*target_attribute):
        tgt_features[:, c] = 1.
        score = score_fn(
            {"features": tgt_features, "targets": train_data["targets"]})
        score += log_marginal[c - target_attribute[0]]
        scores.append(score)
        tgt_features[:, c] = 0.
    # Try all 0s
    if not one_hot:
        score = score_fn(
            {"features": tgt_features, "targets": train_data["targets"]})
        score += log_marginal[-1]
        scores.append(score)

    # Make the prediction:
    return torch.argmax(torch.stack(scores, axis=1), axis=1)


class WhiteboxInverter:

    def __init__(self, train_data, target_attribute, model_type, weights, l2, one_hot=False):
        # Compute marginal counts (proportional to log marginal):
        self.log_marginal = compute_log_marginal(
            train_data, target_attribute, weights=weights, one_hot=one_hot)
        self.one_hot = one_hot
        # Store learned theta for each possible attribute
        self.thetas = torch.zeros(
                train_data["features"].size(0),
                target_attribute[1] - target_attribute[0] + (not one_hot),
                train_data["features"].size(1))
        for i in range(train_data["features"].size(0)):
            tmp = train_data["features"][i, range(*target_attribute)].clone()
            train_data["features"][i, range(*target_attribute)] = 0.
            for c in range(*target_attribute):
                j = c - target_attribute[0]
                train_data["features"][i, c] = 1.
                model = models.get_model(model_type)
                model.train(train_data, l2=l2, weights=weights)
                self.thetas[i, j] = model.theta
                train_data["features"][i, c] = 0.
            if not one_hot:
                # Try all 0s (last attribute)
                model = models.get_model(model_type)
                model.train(train_data, l2=l2, weights=weights)
                self.thetas[i, -1] = model.theta
            train_data["features"][i, range(*target_attribute)] = tmp

    # prior_lam controls strength of prior
    def predict(self, model, gamma=0, prior_lam=0):
        scores = -(self.thetas - model.theta.view(1, 1, -1)).pow(2).sum(2)
        if gamma > 0 and prior_lam > 0:
            scores = 0.5 * scores / gamma**2 + prior_lam * self.log_marginal.unsqueeze(0)
        return torch.argmax(scores, axis=1)


def whitebox_inverter(train_data, target_attribute, model, weights=None, l2=0.0, **kwargs):
    """
    Whitebox inverter has access to the trained model and all of the trained
    data less the one example's target attribute value. Predictions are made
    from solving:
        argmax_{attribute} p(model | available data, attribute) p(attribute)
    """
    inverter = WhiteboxInverter(train_data, target_attribute, type(model), weights, l2)
    return inverter.predict(model)


def compute_metrics(data, predictions, attribute):
    """
    Computes the accuracy for the `predictions` of
    `attribute` on `data`.
    """
    reference = features_to_category(data["features"][:, range(*attribute)])
    return (predictions == reference).float().mean().item()


def run_inversion(args, data, weights):
    regression = (args.dataset == "iwpc" or args.dataset == "synth")

    # Train model:
    model = models.get_model(args.model)
    logging.info(f"Training model {args.model}")
    model.train(data, l2=args.l2, weights=weights)
    # Check predictions for sanity:
    predictions = model.predict(data["features"], regression=regression)
    if regression:
        acc = (predictions - data["targets"]).pow(2).mean()
        logging.info(f"Training MSE of regressor {acc.item():.3f}")
    else:
        acc = ((predictions == data["targets"]).float()).mean()
        logging.info(f"Training accuracy of classifier {acc.item():.3f}")

    # The target attribute can be specified as a range, e.g. `(4, 8)` means the
    # 4th through the 7th feature are the values of the encoded target attribute.
    if args.dataset == "uciadult":
        target_attribute = (24, 25) # [not married, married]
    elif args.dataset == "iwpc":
        #target_attribute = (2, 7) # CYP2C9 genotype
        target_attribute = (11, 13) # VKORC1 genotype
    else:
        raise NotImplementedError("Dataset not yet implemented.")

    if args.inverter == "all":
        inverters = INVERTERS[:-1]
    else:
        inverters = [args.inverter]

    target = features_to_category(data["features"][:, range(*target_attribute)])
    results = { "target" : target.tolist() }

    for inverter in inverters:
        invert_fn = globals()[f"{inverter}_inverter"]
        predictions = invert_fn(
            data, target_attribute, model=model, weights=weights, l2=args.l2)
        acc = compute_metrics(data, predictions, target_attribute)
        logging.info(f"{inverter} inverter Accuracy {acc:.4f}")
        results[inverter] = predictions.tolist()

    return results


def main(args):
    regression = (args.dataset == "iwpc" or args.dataset == "synth")
    data = dataloading.load_dataset(
        name=args.dataset, split="train", normalize=False,
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
        results.append(run_inversion(args, data, weights))

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
    parser.add_argument("--inverter", default="fredrikson14", type=str,
        choices=INVERTERS,
        help="inversion method to use (default: fredrikson14)")
    parser.add_argument("--l2", default=0, type=float,
        help="l2 regularization parameter")
    parser.add_argument("--subsample", default=0, type=int,
        help="number of training examples")
    parser.add_argument("--weights_file", default=None, type=str,
        help="(optional) file to load IRFIL weights from")
    parser.add_argument("--results_file", default=None, type=str,
        help="(optional) path to save results")
    args = parser.parse_args()
    main(args)
