"""rmbench_lila — run LiLa-WAM on RMBench without modifying the LiLa-WAM main project.

Layering (lower layers never import higher ones):

    schema, codec, tasks, config, utils      pure definitions / helpers (no torch)
    episodes                                 HDF5 reading (h5py + numpy)
    data.*                                   torch Dataset built from feature builders
    lila_api                                 THE ONLY module that imports main-project code
    stats, vtt, trainer, policy              experiment logic

Rule: nothing in this package edits files of the main project. If the main
project's interface changes, only ``lila_api.py`` needs to adapt.
"""

__version__ = "0.1.0"
