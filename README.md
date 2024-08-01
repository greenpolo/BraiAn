<!--
SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>

SPDX-License-Identifier: CC0-1.0
-->

# ![braian logo](docs/assets/logo/network.svg) BraiAn
<!--mkdocs-start-->
## Prerequisites
* [python=>3.11](https://www.python.org/downloads/):
  you can use [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pyenv](https://github.com/pyenv/pyenv)/[pyenv-win](https://pyenv-win.github.io/pyenv-win/#installation) to manage the correct version

* [Poetry](https://python-poetry.org/docs/#installation): for dependency management


## Installation Guide
### Step 1: Set-up an environment for BraiAn
Create and activate a new `python=>3.11` environment.

<table border="0">
 <tr>
    <td>
      Using <a href="https://docs.continuum.io/anaconda/"><code>conda</code></a>:</p>
      <pre><code class="language-bash">$ conda create --name braian_env python=3.11
$ conda activate braian_env</code></pre>
    </td>
    <td>
      Using python venv:</p>
      <code class="language-bash">$ python3.11 -m venv /path/to/new/braian_env</code>
      </p>
      and <a href="https://docs.python.org/3/library/venv.html#how-venvs-work">activate it</a>
    </td>
 </tr>
</table>

### Step 2: clone the repository
```bash
$ git clone https://codeberg.org/SilvaLab/BraiAn.git /path/to/BraiAn
```

### Step 3: install BraiAn inside the environment with Poetry
With the environment set and Poetry installed:
```bash
(braian_env) cd /path/to/BraiAn
(braian_env) poetry install
```
<!--mkdocs-end-->