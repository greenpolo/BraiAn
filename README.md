<!--
SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>

SPDX-License-Identifier: CC0-1.0
-->

# BraiAn

## Prerequisites
* [python=>3.11](https://www.python.org/downloads/):
  you can use [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pyenv](https://github.com/pyenv/pyenv)/[pyenv-win](https://pyenv-win.github.io/pyenv-win/#installation) to manage the correct version

* [Poetry](https://python-poetry.org/docs/#installation): for dependency management


## Installation Guide
### Step 1: Set-up an environment for BraiAn
Create a new python=>3.11 environment.
* if you intend to use conda, you can create a new environment:

  `conda create --name braian_env python=3.11`

  and activate it with `conda activate braian_env`
* if you intend to use python's virtual environments:
  
  `python3.11 -m venv /path/to/new/braian_env`

  and [activate it](https://docs.python.org/3/library/venv.html#how-venvs-work)

### Step 2: clone the repository
`git clone https://codeberg.org/SilvaLab/BraiAn.git /path/to/BraiAn`

### Step 3: install BraiAn inside the environment with Poetry
With the environment set and Poetry installed:
```bash
cd /path/to/BraiAn
poetry install
```