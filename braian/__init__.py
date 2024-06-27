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

from .brain_hierarchy import *
from .brain_slice import *
from .sliced_brain import *
from .brain_data import *
from .brain_metrics import *
from .animal_brain import *
from .animal_group import *
from .statistics import *
from .utils import cache, save_csv, regions_to_plot, remote_dirs
from .plot import *

# from braian.connectome import *

from .config import BraiAnConfig