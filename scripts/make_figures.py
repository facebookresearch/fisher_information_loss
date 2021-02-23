#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as scs
import seaborn as sns

import sys
sys.path.append("..")

import dataloading
import plotting


COLOR = sns.cubehelix_palette(1, start=2, rot=0, dark=0, light=.5)[0]


def load_results(results_path, file_name):
    with open(os.path.join(results_path, file_name), 'r') as fid:
        return json.load(fid)


def eta_overlap(results_path, prefix):
    etas_li = np.array(load_results(
        results_path, f"{prefix}_linear_pca20.json")["etas"])
    etas_lo = np.array(load_results(
        results_path, f"{prefix}_logistic_pca20.json")["etas"])
    idx_li = etas_li.argsort()
    idx_lo = etas_lo.argsort()
    n_overlap = len(set(idx_li[-500:]).intersection(idx_lo[-500:]))
    print("="*20)
    print(prefix)
    print(f"Num overlap {n_overlap}/100")


def eta_histogram(results_path, save_path, prefix, train, labels):
    """
    Histogram of samples by individual sample eta.
    """
    plt.clf()

    n_class = len(labels)
    colors = sns.cubehelix_palette(n_class, start=2, rot=0, dark=0, light=.5)

    plt.figure(figsize=(6, 6))
    etas = np.array(load_results(
        results_path, f"{prefix}_pca20.json")["etas"])
    targets = train["targets"].numpy()
    etas = [etas[targets == c] for c in range(n_class)]

    for c in range(n_class):
        plt.hist(
            etas[c], bins=80, color=colors[c], alpha=0.5, label=f"{labels[c]}")
        plt.axvline(
            etas[c].mean(), color=colors[c], linestyle='dashed', linewidth=2)
    plt.xlabel("Per sample $\eta$", fontsize=30)
    plt.xticks(fontsize=26)
    plt.ylabel("Number of samples", fontsize=30)
    plt.yticks(fontsize=26)
    plt.legend()
    plotting.savefig(os.path.join(save_path, f"{prefix}_eta_hist"))


def view_images(train, results_path, save_path, prefix):
    """
    View the most and least leaked images.
    """
    # sort etas by index
    etas = load_results(
        results_path, f"{prefix}_pca20.json")["etas"]
    sorted_etas = sorted(
        zip(etas, range(len(etas))), key=lambda x: x[0], reverse=True)

    ims = train["features"].squeeze()
    n_ims = 8
    f, axarr = plt.subplots(2, n_ims, figsize=(7, 2.2))
    f.subplots_adjust(wspace=0.05)
    for priv in [False, True]:
        for i in range(n_ims):
            ax = axarr[int(priv), i]
            idx = -(i + 1) if priv else i
            im = sorted_etas[idx][1]
            image = ims[im, ...]
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
            ax.imshow(image, cmap='gray')
            ax.axis("off")
            title = "{:.1e}".format(sorted_etas[idx][0])
            ax.set_title(title, fontsize=14)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plotting.savefig(os.path.join(save_path, f"{prefix}_images"))
    plt.close(f)


def correlations(results_path, save_path, prefix):
    results = load_results(results_path, f"{prefix}_pca20.json")
    etas = np.array(results["etas"])
    n_samples = 2000
    np.random.seed(n_samples)
    samples = np.random.permutation(len(etas))[:n_samples]
    losses = np.array(results["train_losses"])
    grad_norms = np.array(results["train_grad_norms"])
    alternatives = [
        ("loss", "(a) Loss $\ell({\\bf w^*}^\\top {\\bf x}, y)$", losses),
        ("gradnorm", "(b) Gradient norm $\|\\nabla_{\\bf w^*} \ell\|_2$", grad_norms)]
    f, axarr = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    f.subplots_adjust(wspace=0.1)
    for e, (method, xlabel, values) in enumerate(alternatives):
        ax = axarr[e]
        ax.scatter(values[samples], etas[samples], s=2.5, color=COLOR)
        ax.set_xlabel(xlabel)
    axarr[0].set_ylabel("FIL $\eta$")

    plotting.savefig(os.path.join(save_path, f"{prefix}_scatter_alternatives_eta"))
    plt.clf()


def iterated_reweighted_etas(results_path, save_path, prefix):
    """
    Line plot of variance of Fisher information loss eta with iterated
    reweighting.
    """
    linear = load_results(results_path, f"{prefix}_linear_reweighted.json")
    logistic = load_results(results_path, f"{prefix}_logistic_reweighted.json")
    stds = np.array([linear["eta_stds"], logistic["eta_stds"]])
    iterations = np.arange(stds.shape[1])
    plotting.line_plot(
        stds, iterations, legend=["Linear", "Logistic"],
        xlabel="Iteration", ylabel="Standard deviation of $\eta$",
        ylog=True,
        size=(5, 5),
        filename=os.path.join(save_path, f"{prefix}_eta_std_vs_iterations"))
    for results, model in [(linear, "linear"), (logistic, "logistic")]:
        em = results["eta_means"]
        std = results["eta_stds"]
        acc = results["test_accs"]
        print("="*20)
        print(f"{prefix} {model}")
        print(f"Pre IRFP eta {em[0]:.3f}, std {std[0]:.3f}, test accuracy {acc[0]:.3f}")
        print(f"Post IRFP eta {em[-1]:.3f}, std {std[-1]:.3f}, test accuracy {acc[-1]:.3f}")


def private_mse_and_fil(results_path, save_path):
    L2s = ['1e-5', '1e-3', '1e-1', '1']
    noise_scales = [
        '1e-05', '2e-05', '5e-05', '0.0001', '0.0002',
        '0.0005', '0.001', '0.002', '0.005', '0.01',
        '0.02', '0.05', '0.1', '0.2', '0.5', '1.0',
    ]

    fils = []
    mean_etas = []
    mses = []
    for l2 in L2s:
        etas = load_results(
            results_path, f"iwpc_least_squares_fil_l2_{l2}.json")["etas"]
        fils.append(etas)
        inversion_results = load_results(
            results_path,
            f"iwpc_least_squares_whitebox_private_inversion_l2_{l2}.json")
        mses.append([inversion_results[0][noise_scale]['test_acc']
            for noise_scale in noise_scales])
        mean_etas.append(np.mean(etas))
    sigmas = np.array([float(ns) for ns in noise_scales])
    mean_etas = np.array(mean_etas)[:, None] / sigmas

    l2s = np.array([float(l2) for l2 in L2s])
    legend = ["$\lambda=10^{%d}$"%int(math.log10(float(l2))) for l2 in L2s]

    # Plot FILs:
    num_bins = 100
    fil_counts = []
    fil_centers = []
    for fil in fils:
        lower = math.log10(np.min(fil))
        upper = math.log10(np.max(fil) + 1e-4)
        bins = np.logspace(lower, upper, num_bins + 1)
        counts, edges = np.histogram(fil, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        fil_counts.append(counts)
        fil_centers.append(centers)
    plotting.line_plot(
        np.array(fil_counts),
        np.array(fil_centers),
        xlabel="FIL $\eta$ (at $\sigma=1$)",
        ylabel="Number of examples",
        legend=legend,
        marker=None,
        size=(5, 5),
        xlog=True,
        filename=os.path.join(
            args.save_path,
            f"iwpc_fil_counts_varying_l2"),
    )

    # PLot MSEs
    mses = np.array(mses) # [L2, noise_scale, trials]
    mse_means = mses.mean(axis=2)
    mse_stds = mses.std(axis=2)
    plotting.line_plot(
        mse_means, mean_etas, legend=legend,
        xlabel="Mean $\\bar{\eta}$",
        ylabel="Test MSE",
        ylog=True,
        xlog=True,
        size=(5, 5),
        errors=mse_stds,
        filename=os.path.join(args.save_path, f"iwpc_mse_vs_eta_varying_l2"),
    )


def private_inversion_accuracy(results_path, save_path):
    L2s = ['1e-5', '1e-3', '1e-1', '1']
    noise_scales = [
        '1e-05', '2e-05', '5e-05', '0.0001', '0.0002',
        '0.0005', '0.001', '0.002', '0.005', '0.01',
        '0.02', '0.05', '0.1', '0.2', '0.5', '1.0',
    ]

    inversion_results = load_results(
        results_path, f"iwpc_least_squares_inversion_l2_1e-3.json")[0]
    target = np.array(inversion_results["target"])
    baseline = (target == np.array(inversion_results["baseline"])).mean()

    # Load the max eta for each L2.
    mean_etas = []
    for l2 in L2s:
        etas = load_results(
            results_path, f"iwpc_least_squares_fil_l2_{l2}.json")["etas"]
        mean_etas.append(np.mean(etas))
    sigmas = np.array([float(ns) for ns in noise_scales])
    mean_etas = np.array(mean_etas)[:, None] / sigmas

    results = {"whitebox": {}, "fredrikson14": {}}
    for inverter in ["whitebox", "fredrikson14"]:
        results = []
        for l2 in L2s:
            # inversion results are in a list ordered by ieration or IRFIL,
            # each dictionary is the results at a given noise scale along with
            # the target values
            inversion_results = load_results(
                results_path,
                f"iwpc_least_squares_{inverter}_private_inversion_l2_{l2}.json")
            all_accs = []
            for noise_scale in noise_scales:
                accs = []
                for prediction in inversion_results[0][noise_scale]['predictions']:
                    accs.append((np.array(prediction) == target).mean())
                all_accs.append([np.mean(accs), np.std(accs)])
            results.append(all_accs)

        results = np.array(results) # [L2, noise scale, mean/std]
        means = results[:, :, 0]
        stds = results[:, :, 1]

        legend = ["$\lambda=10^{%d}$"%int(math.log10(float(l2))) for l2 in L2s]

        plotting.line_plot(
            means, mean_etas, legend=legend,
            xlabel="Mean $\\bar{\eta}$",
            ylabel="Attribute inversion accuracy",
            errors=stds,
            ymax=1.02,
            ymin=0.2,
            xlog=True,
            size=(5, 5))
        plt.semilogx([0, 1e3], [baseline]*2, 'k--', label="Prior")
        plt.legend()
        plt.xlim(right=1e3)
        plotting.savefig(os.path.join(
            args.save_path,
            f"iwpc_{inverter}_vs_eta_varying_l2"))


def irfil_inversion(results_path, dataset, save_path):
    noise_scale = "0.001"
    its = 10

    irfil_results = load_results(results_path, f"{dataset}_least_squares_irfil.json")
    etas = np.array(irfil_results["etas"])
    eta_means = np.array(irfil_results["eta_means"])[:its]
    eta_stds = np.array(irfil_results["eta_stds"])[:its]
    plotting.line_plot(
        eta_means[None, :], np.arange(its),
        xlabel="Steps of IRFIL",
        ylabel="Mean $\\bar{\eta}$",
        errors=eta_stds[None, :],
        size=(4.85, 5.05),
        filename=os.path.join(args.save_path, f"{dataset}_mean_fil"),
    )

    def compute_correct_ratio(etas, num_bins, predictions, target):
        order = etas.argsort()
        bin_size = len(target) // num_bins + 1
        bin_accs = []
        for prediction in predictions:
            prediction = np.array(prediction)
            correct = (prediction == target)
            bin_accs.append([correct[order[lower:lower + bin_size]].mean()
                for lower in range(0, len(correct), bin_size)])
        return np.array(bin_accs)

    inversion_results = load_results(
        results_path,
        f"{dataset}_least_squares_whitebox_private_inversion_irfil.json")
    target = np.array(inversion_results[0]["target"])
    num_bins = 10
    ratio_means = []
    ratio_stds = []
    its = [0, 2, 10]
    for it in its:
        predictions = inversion_results[it][noise_scale]['predictions']
        ratios = compute_correct_ratio(etas, num_bins, predictions, target)
        ratio_means.append(ratios.mean(axis=0))
        ratio_stds.append(ratios.std(axis=0))
    ratio_means = np.array(ratio_means)
    ratio_stds = np.array(ratio_stds)

    plotting.line_plot(
        ratio_means, np.arange(num_bins),
        legend=["Iteration {}".format(it) for it in its],
        xlabel="FIL ($\eta$) percentile",
        ylabel="Attribute inversion accuracy",
        errors=ratio_stds,
        size=(5, 5),
        filename=os.path.join(
            args.save_path,
            f"{dataset}_whitebox_eta_percentile"),
    )



def main(args):
    labels = {"mnist" : ["0", "1"], "cifar10": ["Plane", "Car"]}
    for dataset in ["mnist", "cifar10"]:
        train = dataloading.load_dataset(
            name=dataset, split="train", normalize=False,
            num_classes=2, reshape=False, root=args.data_folder)

        for model in ["linear", "logistic"]:
            prefix = f"{dataset}_{model}"

            # Histogram of etas:
            eta_histogram(
                args.results_path, args.save_path,
                prefix, train, labels[dataset])

            # Most and least leaked images:
            view_images(train, args.results_path, args.save_path, prefix)

        eta_overlap(args.results_path, f"{dataset}")

        # Plot of eta stds vs iterations of reweighting
        iterated_reweighted_etas(
            args.results_path, args.save_path, f"{dataset}")

    # Plot correlations of eta with other metrics
    correlations(args.results_path, args.save_path, "mnist_linear")

    # IWPC MSE and FIL with output pertubration
    private_mse_and_fil(args.results_path, args.save_path)

    # IWPC Fredrikson and whitebox attribute inversion results.
    private_inversion_accuracy(args.results_path, args.save_path)

    # IWPC and UCI Adult attribute inversion results as a function of
    # iterations of IRFIL
    for dataset in ["iwpc", "uciadult"]:
        irfil_inversion(args.results_path, dataset, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to make all the figures.")
    parser.add_argument("--results_path", type=str, default=".",
        help="Path of saved results")
    parser.add_argument("--data_folder", default="/tmp", type=str,
        help="folder in which to store data (default: '/tmp')")
    parser.add_argument("--save_path", default=".", type=str,
        help="folder in which to store figures (default: '.')")
    parser.add_argument("--format", default=None, type=str,
        help="format to save figures (default: \"pdf\")")
    args = parser.parse_args()
    if args.format is not None:
        plotting.FORMAT = args.format

    plt.rcParams.update({
        "axes.titlesize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize" : 18,
        "axes.labelsize": 24
    })

    main(args)
