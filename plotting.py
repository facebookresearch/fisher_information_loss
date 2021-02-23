#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
A simple example of creating a figure with text rendered in LaTeX.

https://jwalton.info/Embed-Publication-Matplotlib-Latex/
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Using seaborn's style
plt.style.use('seaborn-white')

WIDTH = 345
GR = (5**.5 - 1) / 2
FORMAT = "pdf"

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 14,
    "font.size": 14,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
}

plt.rcParams.update(tex_fonts)
plt.rcParams.update({"legend.handlelength": 1})

def savefig(filename):
    plt.savefig(
        filename + "." + FORMAT, format=FORMAT, dpi=1200, bbox_inches="tight")

def line_plot(
        Y, X, xlabel=None, ylabel=None, ymax=None, ymin=None,
        xmax=None, xmin=None, filename=None, legend=None, errors=None,
        xlog=False, ylog=False, size=None, marker="s"):
    colors = sns.cubehelix_palette(Y.shape[0], start=2, rot=0, dark=0, light=.5)
    plt.clf()
    if legend is None:
        legend = [None] * Y.shape[0]

    if size is not None:
        plt.figure(figsize=size)

    for n in range(Y.shape[0]):
        x = X[n, :] if X.ndim == 2 else X
        plt.plot(x, Y[n, :], label=legend[n], color=colors[n],
                marker=marker, markersize=5)
        if errors is not None:
            plt.fill_between(
                x, Y[n, :] - errors[n, :], Y[n, :] + errors[n, :],
                alpha=0.1, color=colors[n])

    if ymax is not None:
        plt.ylim(top=ymax)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if xmax is not None:
        plt.xlim(right=xmax)
    if xmin is not None:
        plt.xlim(left=xmin)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend[0] is not None:
        plt.legend()

    axes = plt.gca()
    if xlog:
        axes.semilogx(10.)
    if ylog:
        axes.semilogy(10.)

    if filename is not None:
        savefig(filename)
