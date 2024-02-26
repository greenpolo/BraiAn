# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import platform

match platform.system():
    case "Windows":
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        def resolve_symlink(path: str):
            return shell.CreateShortCut(path).Targetpath if path.endswith(".lnk") else path
    case _:
        resolve_symlink = lambda path: os.path.realpath(path)

from .brain_hierarchy import AllenBrainHierarchy, MAJOR_DIVISIONS, UPPER_REGIONS
from .brain_slice import BrainSlice
from .sliced_brain import SlicedBrain
from .brain_data import BrainData
from .brain_metrics import BrainMetrics
from .animal_brain import AnimalBrain
from .animal_group import AnimalGroup, PLS
from .statistics import as_prism_data
from .utils import cache, save_csv, regions_to_plot, remote_dirs
from .plot import plot_groups, plot_pie, plot_cv_above_threshold, plot_region_density, plot_permutation, plot_groups_salience, plot_latent_variables, plot_salient_regions, plot_gridgroups

from .connectome.connectome_adjacency import ConnectomeAdjacency
from .connectome.cross_correlation import CrossCorrelation
from .connectome.connectome import Connectome
from .connectome.functional import FunctionalConnectome
from .connectome.structural import StructuralConnectome
from .connectome.pruned import PrunedConnectomics
from .connectome.plot import draw_network_plot
from .connectome.plot_chord import draw_chord_plot

from .config import BraiAnConfig