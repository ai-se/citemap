from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from utils.lib import O

_dend_7_9 = O(
    fig_size=(8, 8),
    col_axes=[0.25, 0.6, 0.51, 0.18],
    row_axes=[0.0, 0.15, 0.21, 0.4],
    plot_axes=[0.25, 0.05, 0.63, 0.6]
)

_dend_7_8 = O(
    fig_size=(8, 8),
    col_axes=[0.3, 0.6, 0.5, 0.18],
    row_axes=[0.0, 0.13, 0.21, 0.45],
    plot_axes=[0.3, 0.05, 0.63, 0.6]
)

_dend_11_26 = O(
    fig_size=(8, 8),
    col_axes=[0.3, 0.53, 0.63, 0.18],
    row_axes=[0.0, 0.23, 0.19, 0.27],
    plot_axes=[0.3, 0.05, 0.63, 0.6]
)

_dend_11_24 = O(
    fig_size=(8, 8),
    col_axes=[0.3, 0.53, 0.63, 0.18],
    row_axes=[0.0, 0.23, 0.19, 0.29],
    plot_axes=[0.3, 0.05, 0.63, 0.6]
)

_dend_11_18 = O(
    fig_size=(8, 8),
    col_axes=[0.3, 0.63, 0.63, 0.18],
    row_axes=[0.0, 0.23, 0.19, 0.39],
    plot_axes=[0.3, 0.05, 0.63, 0.6]
)

dend = {
    "dend_7_9": _dend_7_9,
    "dend_7_8": _dend_7_8,
    "dend_11_26": _dend_11_26,
    "dend_11_24": _dend_11_24,
    "dend_11_18": _dend_11_18
}
