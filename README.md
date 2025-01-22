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
* [python>=3.11<3.14](https://www.python.org/downloads/).
  If needed, you can use [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pyenv](https://github.com/pyenv/pyenv)/[pyenv-win](https://pyenv-win.github.io/pyenv-win/#installation) to manage the correct version;

* [Poetry](https://python-poetry.org/docs/#installation): for dependency management.

### Step 1: clone the repository
```bash
git clone https://codeberg.org/SilvaLab/BraiAn.git /path/to/BraiAn
```

### Step 2: install with Poetry
```bash
cd /path/to/BraiAn
poetry install # --with docs, if building the documentation is of your interest
```
Poetry will automatically create a [virtual environment](https://docs.python.org/3/library/venv.html#how-venvs-work) in which it installs all the dependencies.
If, instead, you want to manage the environment yourself, Poetry use the one active during the installation.
<!--build-end-->
<!--mkdocs-end-->