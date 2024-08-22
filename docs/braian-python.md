# BraiAn python library

BraiAn offer a python library for whole-brain quantitative analysis and visualisation of per-region cell counts. The data it can work with can either come from QuPath, with little-to-no work thanks to [_BraiAn for QuPath_](braian-qupath.md#output) or any other software of brain cell segmentation (e.g. [Clearmap](https://clearanatomics.github.io/ClearMapDocumentation/)). 

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
    start="<!--install-start-->"
    end="<!--install-end-->"
%}

## Getting started
### Project arrangement
### Package organization

{%
    include-markdown "../README.md"
    start="<!--build-start-->"
    end="<!--build-end-->"
%}

<!--
{% include "resources/allen_ontology.html" recursive=false %}

{% include "resources/gridplot_cfos_vs_Arc1_density_summary_structures.html" recursive=false %}
-->