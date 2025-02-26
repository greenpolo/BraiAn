<!--
SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>

SPDX-License-Identifier: CC-BY-4.0
-->

# ![braian logo](docs/assets/logo/network.svg) BraiAn
[![PyPI - Version](https://img.shields.io/pypi/v/braian)](https://pypi.org/project/braian)
[![status-badge](https://ci.codeberg.org/api/badges/13585/status.svg)](https://ci.codeberg.org/repos/13585)
<!--mkdocs-start-->
<!--install-start-->
## Installation
Once you are in an active `python>=3.11,<3.14` environment, you can run:
```bash
python3 -m pip install braian
```
<!--install-end-->

## Citing
If you use BraiAn in your work, please cite the paper below, currently in pre-print:

> Chiaruttini, N., Castoldi, C. et al. **ABBA, a novel tool for whole-brain mapping, reveals brain-wide differences in immediate early genes induction following learning**. _bioRxiv_ (2024).\
> [https://doi.org/10.1101/2024.09.06.611625](https://doi.org/10.1101/2024.09.06.611625)

<!--build-start-->
## Building
### Prerequisites
- [git](https://git-scm.com/downloads)
- [Poetry](https://python-poetry.org/docs/#installation) or [venv](https://docs.python.org/3/library/venv.html)/[conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)/[pyenv](https://github.com/pyenv/pyenv)/[pyenv-win](https://pyenv-win.github.io/pyenv-win/#installation) to manage dependencies.

Windows instructions assume that you configured the `PATH` and `PATHEXT` variables with its command-line program location (e.g. `git` or `pip`). If you can't/didn't, you can juxtapose the path-to-exectuable to the respective commands (e.g., `C:\Python311\python` instead of `python`).\
If you don't know how, we recommend using [Scoop](https://scoop.sh/).

### Step 1: clone the repository
```bash
git clone https://codeberg.org/SilvaLab/BraiAn.git /path/to/BraiAn
```

### Step 2: install
#### with Poetry
```bash
cd /path/to/BraiAn
poetry install # --with docs, to install documentation dependencies
               # --with dev, to install basic dependencies to work on ipython
```
Poetry will automatically create a [virtual environment](https://docs.python.org/3/library/venv.html#how-venvs-work) in which it installs all the dependencies.\
If, instead, you want to manage the environment yourself, Poetry uses the one active during the installation.

#### with pip
Requires [python](https://www.python.org/downloads/)>=3.11,<3.14.


```bash
pip install -e /path/to/BraiAn
```
**Note**: installing with pip doesn't assure to install the same version of the dependencies used by developers to run and test `braian`.
<!--build-end-->
<!--mkdocs-end-->