<!--
SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>

SPDX-License-Identifier: CC-BY-4.0
-->
# BraiAn documentation

BraiAn is a versatile toolkit for whole-**brai**n quantitative **an**alysis of large datasets.
Specifically, it was first designed for studying brain activity in behavioral groups through immediate early genes and imaging.

This site contains the project documentation for the `braian`, the Python library.
Its aim is to give you a framework to build your own analysis, comparing activity in different brain regions,
finding the which ones' activity are the most representative of a group and visualising  data in interactive and informative plots.

!!! warning "Alpha Software"

    BraiAn, while already being a capable and extensible library, is still under heavy development.
    Some functionalities might expose non-intuitive interfaces, others could be outright buggy.
    For this reason, we consider it still in alpha stage. \
    If you encounter any problem, we strongly encourage to [let us know](https://codeberg.org/SilvaLab/BraiAn/issues)
    so that we can work together on improving BraiAn!

{%
    include-markdown "../README.md"
    heading-offset=1
    start="<!--mkdocs-start-->"
    end="<!--mkdocs-end-->"
%}

{% include "resources/allen_ontology.html" %}

{% include-markdown "resources/gridplot_cfos_vs_Arc1_density_summary_structures.html" %}
