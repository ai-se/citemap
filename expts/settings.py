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


dend = {
    "dend_7_9": _dend_7_9
}
