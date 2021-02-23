#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import os
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import download_url

import time
import sys
import git


class Synth:
    """
    Synthetic dataset for testing.
    """
    def __init__(self, root, train=True, download=True):
        regression = True
        drop = True
        num_examples = 1000
        num_attributes = 3
        categories_per_attribute = 3

        # For repeatability
        torch.manual_seed(int(train))
        X = torch.randint(
            0, categories_per_attribute, (num_examples, num_attributes))
        X = torch.nn.functional.one_hot(X, categories_per_attribute)
        if drop:
            # Remove the last attribute to get rid of perfect collinearity
            X = X[:, :, :-1]
        X = X.reshape(num_examples, -1)
        self.data = X.float()

        if regression:
            self.targets = torch.randn(num_examples) * 50
        else:
            self.targets = torch.randint(0, 2, (num_examples,))
        torch.manual_seed(time.time())


class UCIAdult:
    """
    UCI Adult dataset:
        http://archive.ics.uci.edu/ml/datasets/Adult

    The task is to classify individuals as making above or below $50k
    based on certain demographic attributes.

    Data key:
        0 age: continuous.
        1 workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov,
            State-gov, Without-pay, (NB removed with ?: Never-worked).
        2 fnlwgt: continuous.
        3 education: Bachelors, Some-college, 11th, HS-grad, Prof-school,
          Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th,
          10th, Doctorate, 5th-6th, Preschool.
        4 education-num: continuous.
        5 marital-status: Married-civ-spouse, Divorced, Never-married, Separated,
            Widowed, Married-spouse-absent, Married-AF-spouse.
        6 occupation: Tech-support, Craft-repair, Other-service, Sales,
            Exec-managerial, Prof-specialty, Handlers-cleaners,
            Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving,
            Priv-house-serv, Protective-serv, Armed-Forces.
        7 relationship: Wife, Own-child, Husband, Not-in-family, Other-relative,
            Unmarried.
        8 race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        9 sex: Female, Male.
        10 capital-gain: continuous.
        11 capital-loss: continuous.
        12 hours-per-week: continuous.
        13 native-country: United-States, Cambodia, England, Puerto-Rico, Canada,
            Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South,
            China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica,
            Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos,
            Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
            Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru,
            Hong, Holand-Netherlands.

        14 label: >50K, <=50K.
    """

    def __init__(self, root, train=True, download=False, mehnaz20=True, drop=True):
        """
        mehnaz20: If True, then pre-process the data according to:
              Black-box Model Inversion Attribute Inference Attacks on
              Classification Models, Mehnaz 2020, https://arxiv.org/abs/2012.03404.
            Namely, they combine the marital status features (5) into a single
            binary feature {Married-civ-spouse, Married-spouse-absent,
            Married-Af-spouse} vs {Divorced, Never-married, Separated, Widowed} and
            remove the relationship features (7).
        drop: If True, remove the last feature for a one-hot encoded
            attribute. This helps alleviate perfect colinearity amongst
            the features.
        """
        self.root = root
        if download:
            self.download()
        continuous_ids = set([0, 2, 4, 10, 11, 12])
        feature_keys = [set() for _ in range(15)]

        def load(dataset):
            with open(os.path.join(root, dataset)) as fid:
                # load data ignoring rows with missing values
                lines = (l.strip() for l in fid)
                lines = (l.split(",") for l in lines if "?" not in l)
                lines = [l for l in lines if len(l) == 15]
            return lines

        for line in load("adult.data"):
            for e, k in enumerate(line):
                if e in continuous_ids:
                    continue
                k = k.strip()
                feature_keys[e].add(k)
        feature_keys = [{k: i for i, k in enumerate(sorted(fk))}
            for fk in feature_keys]
        self.feature_keys = feature_keys
        if mehnaz20:
            # Remap marital status to binary feature:
            marital_status = feature_keys[5]
            for ms in ["Divorced", "Never-married", "Separated", "Widowed"]:
                marital_status[ms] = 0
            for ms in ["Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent"]:
                marital_status[ms] = 1

        def process(dataset, mean_stds=None):
            features = []
            targets = []
            for line in load(dataset):
                example = []
                for e, k in enumerate(line):
                    k = k.strip().strip(".")
                    example.append(int(feature_keys[e].get(k, k)))
                features.append(example[:-1])
                targets.append(example[-1])
            features = torch.tensor(features, dtype=torch.float)
            features = list(features.split(1, dim=1))
            targets = torch.tensor(targets)

            if mean_stds is None:
                mean_stds = {}
            for e, feat in enumerate(features):
                keys = feature_keys[e]
                # Normalize continuous features:
                if len(keys) == 0:
                    if e not in mean_stds:
                        mean_stds[e] = (torch.mean(feat), torch.std(feat))
                    mean, std = mean_stds[e]
                    features[e] = (feat - mean) / std
                # One-hot encode non-continuous features:
                else:
                    num_feats = max(keys.values()) + 1
                    features[e] = torch.nn.functional.one_hot(
                        feat.squeeze().to(torch.long), num_feats)
                    if drop:
                        features[e] = features[e][:, :-1]
                    features[e] = features[e].to(torch.float)
            if mehnaz20:
                # Remove relationship status:
                features.pop(7)
            features = torch.cat(features, dim=1)
            return features, targets, mean_stds

        features, targets, mean_stds  = process("adult.data")
        if not train:
            features, targets, _ = process("adult.test", mean_stds)
        self.data = features
        self.targets = targets

    def download(self):
        logging.info("Maybe downloading UCI Adult dataset.")
        base_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
        for f in ["adult.data", "adult.names", "adult.test"]:
            download_url(os.path.join(base_url, f), root=self.root)


class IWPC:
    """
    International Warfarin Pharmacogenetics Consortium (IWPC) dataset:
        https://www.pharmgkb.org/downloads
    Pre-processed version from https://github.com/samuel-yeom/ml-privacy-csf18

    The task is to predict stable Warfarin dosage given demographic, medical, and genetic attributes.

    Processed data has the following attributes:
        0 race: asian/white/black
        1 age: rounded down to 10s and whitened
        2 height: continuous; whitened
        3 weight: continuous; whitened
        4 amiodarone: binary
        5 CYP2C9 inducer: binary
        6 CYP2C9 genotype: 11/12/13/22/23/33
        7 VKORC1 genotype: CC/CT/TT
        8 label: Warfarin dosage; whitened
    """
    def __init__(self, root, train=True, download=False, drop=True):
        """
        drop: If True, remove the last feature for a one-hot encoded
            attribute. This helps alleviate perfect colinearity amongst
            the features.
        """
        if download:
            # download dataloading code and data from repo
            try:
                git.Repo.clone_from("git@github.com:samuel-yeom/ml-privacy-csf18.git", root)
            except git.GitCommandError:
                print("Directory exists and is non-empty, skipping download")
        sys.path.append(os.path.join(root, "code"))
        from main import load_iwpc
        X, y, featnames = load_iwpc(os.path.join(root, "data"))
        X = X.todense()
        if drop:
            attrs = [f.split("=")[0] for f in featnames]
            drop_keys = ["cyp2c9", "race", "vkorc1"]
            drop_idx = [attrs.index(k) for k in drop_keys]
            X = np.delete(X, drop_idx, axis=1)
            featnames = np.delete(featnames, drop_idx)
        print("Attributes: " + str(featnames))
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        # fix a random 80:20 train-val split
        torch.manual_seed(0)
        perm = torch.randperm(X.size(0))
        n_train = int(0.8 * X.size(0))
        if train:
            self.data = X[perm[:n_train], :]
            self.targets = y[perm[:n_train]]
        else:
            self.data = X[perm[n_train:], :]
            self.targets = y[perm[n_train:]]
        torch.manual_seed(time.time())


class MNIST1M(MNIST):
    """
    MNIST1M dataset that can be generated using InfiMNIST.

    Note: This dataset cannot be downloaded automatically.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST1M, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def download(self):
        """
        Process MNIST1M data if it does not exist in processed_folder already.
        """

        # check if processed data does not exist:
        if self._check_exists():
            return

        # process and save as torch files:
        logging.info("Processing MNIST1M data...")
        os.makedirs(self.processed_folder, exist_ok=True)
        training_set = (
            read_image_file(os.path.join(self.raw_folder, "mnist1m-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "mnist1m-labels-idx1-ubyte"))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte"))
        )
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)
        logging.info("Done!")


def load_dataset(
    name="mnist",
    split="train",
    normalize=True,
    reshape=True,
    num_classes=None,
    regression=False,
    root="/tmp",
):
    """
    Loads train or test `split` from the dataset with the specified `name`.
    Setting `normalize` to `True` (default) normalizes each feature vector to
    lie on the unit ball. Setting `reshape` to `True` (default) flattens n-D
    arrays into vectors. Specifying `num_classes` selects only the first
    `num_classes` of the classification problem (default: all classes).
    """

    # assertions:
    assert split in ["train", "test"], f"unknown split: {split}"
    image_sets = {
        "mnist": MNIST,
        "mnist1m": MNIST1M,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }
    datasets = {
        "uciadult": UCIAdult, "iwpc": IWPC, "synth": Synth}
    datasets.update(image_sets)
    assert name in datasets, f"unknown dataset: {name}"

    # download the image dataset:
    dataset = datasets[name](
        f"{root}/{name}_original",
        download=True,
        train=(split == "train"),
    )

    # preprocess the image dataset:
    features, targets = dataset.data, dataset.targets
    if not torch.is_tensor(features):
        features = torch.from_numpy(features)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    if name in image_sets:
        features = features.float().div_(255.)
    if not regression:
        targets = targets.long()

    # flatten images or convert to NCHW:
    if reshape:
        features = features.reshape(features.size(0), -1)
    else:
        if len(features.shape) == 3:
            features = features.unsqueeze(3)
        features = features.permute(0, 3, 1, 2)

    # select only subset of classes:
    if not regression and num_classes is not None:
        assert num_classes >= 2, "number of classes must be >= 2"
        mask = targets.lt(num_classes)  # assumes classes are 0, 1, ..., C - 1
        features = features[mask, :]
        targets = targets[mask]

    # normalize all samples to lie within unit ball:
    if normalize:
        assert reshape, "normalization without reshaping unsupported"
        features.div_(features.norm(dim=1).max())
    # return dataset:
    return {"features": features, "targets": targets}


def load_datasampler(dataset, batch_size=1, shuffle=True, transform=None):
    """
    Returns a data sampler that yields samples of the specified `dataset` with the
    given `batch_size`. An optional `transform` for samples can also be given.
    If `shuffle` is `True` (default), samples are shuffled.
    """
    assert dataset["features"].size(0) == dataset["targets"].size(0), \
        "number of feature vectors and targets must match"
    if transform is not None:
        assert callable(transform), "transform must be callable if specified"
    N = dataset["features"].size(0)

    # define simple dataset sampler:
    def sampler():
        idx = 0
        perm = torch.randperm(N) if shuffle else torch.range(0, N).long()
        while idx < N:

            # get batch:
            start = idx
            end = min(idx + batch_size, N)
            batch = dataset["features"][perm[start:end], :]

            # apply transform:
            if transform is not None:
                transformed_batch = [
                    transform(batch[n, :]) for n in range(batch.size(0))
                ]
                batch = torch.stack(transformed_batch, dim=0)

            # return sample:
            yield {"features": batch, "targets": dataset["targets"][perm[start:end]]}
            idx += batch_size

    # return sampler:
    return sampler


def subsample(data, num_samples, random=True):
    """
    Subsamples the specified `data` to contain `num_samples` samples. Set
    `random` to `False` to not select samples randomly but only pick top ones.
    """

    # assertions:
    assert isinstance(data, dict), "data must be a dict"
    assert "targets" in data, "data dict does not have targets field"
    dataset_size = data["targets"].nelement()
    assert num_samples > 0, "num_samples must be positive integer value"
    assert num_samples <= dataset_size, "num_samples cannot exceed data size"

    # subsample data:
    if random:
        permutation = torch.randperm(dataset_size)
    for key, value in data.items():
        if random:
            data[key] = value.index_select(0, permutation[:num_samples])
        else:
            data[key] = value.narrow(0, 0, num_samples).contiguous()
    return data


def pca(data, num_dims=None, mapping=None):
    """
    Applies PCA on the specified `data` to reduce its dimensionality to
    `num_dims` dimensions, and returns the reduced data and `mapping`.

    If a `mapping` is specified as input, `num_dims` is ignored and that mapping
    is applied on the input `data`.
    """

    # work on both data tensor and data dict:
    data_dict = False
    if isinstance(data, dict):
        assert "features" in data, "data dict does not have features field"
        data_dict = True
        original_data = data
        data = original_data["features"]
    assert data.dim() == 2, "data tensor must be two-dimensional matrix"

    # compute PCA mapping:
    if mapping is None:
        assert num_dims is not None, "must specify num_dims or mapping"
        mean = torch.mean(data, 0, keepdim=True)
        zero_mean_data = data.sub(mean)
        covariance = torch.matmul(zero_mean_data.t(), zero_mean_data)
        _, projection = torch.symeig(covariance, eigenvectors=True)
        projection = projection[:, -min(num_dims, projection.size(1)):]
        mapping = {"mean": mean, "projection": projection}
    else:
        assert isinstance(mapping, dict), "mapping must be a dict"
        assert "mean" in mapping and "projection" in mapping, "mapping missing keys"
        if num_dims is not None:
            logging.warning("Value of num_dims is ignored when mapping is specified.")

    # apply PCA mapping:
    reduced_data = data.sub(mapping["mean"]).matmul(mapping["projection"])

    # return results:
    if data_dict:
        original_data["features"] = reduced_data
        reduced_data = original_data
    return reduced_data, mapping
