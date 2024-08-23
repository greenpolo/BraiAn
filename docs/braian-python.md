# BraiAn python library

BraiAn offer a python library for whole-brain quantitative analysis and visualisation of per-region cell counts. The input data can either be from QuPath, with little-to-no work thanks to [_BraiAn for QuPath_](braian-qupath.md#output) or any other software of brain cell segmentation (e.g. [Clearmap](https://clearanatomics.github.io/ClearMapDocumentation/)).

Its aim is to give you a framework to build your own analysis, comparing activity in different brain regions, finding the which ones' activity are the most representative of a group and visualising data in interactive and informative plots.

!!! warning "Alpha Software"

    `braian`, while already being a capable and extensible library, is still under development. Some functionalities might expose non-intuitive interfaces, others could be outright buggy. For this reason, we still consider it in alpha stage.\
    If you encounter any problem, we strongly encourage to [let us know](https://codeberg.org/SilvaLab/BraiAn/issues) so that we can work together on improving BraiAn!

{%
    include-markdown "../README.md"
    start="<!--install-start-->"
    end="<!--install-end-->"
%}

## Getting started
### Package organization

The `braian` package wants to help neuroscientists in three main tasks. For each one of them, there is a dedicated submodule:

* [`braian`](api-braian.md): intended for importing brain data from different sources (e.g. QuPath) and checking its integrity against a brain ontology. It can also prepare the data in cohorts/groups and projects;
* [`braian.stats`](api-stats.md): intended for brain data normalization, group and marker analysis;
* [`braian.plot`](api-plot.md): intended for brain data visualisation, in interactive.

BraiAn overall aim is to expose non-trivial operation with an intuitive interface. Nonetheless, the user is required to have basic programming skills in order to perform the analysis accordingly to the parameters that better suits their needs.

### Configuration file

Just like [BraiAn for QuPath](braian-qupath.md#configuration-file), `braian` can use a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file to store metadata about a project along with some additional parameters. Such file can be read in `braian` and used to apply a consistent behaviour across all animals and runs.

For a detailed explanation on how to format such file and understanding what each parameters does we suggest to look at [this example YAML file](https://codeberg.org/SilvaLab/BraiAn/raw/branch/master/config_example.yml).

{%
    include-markdown "../README.md"
    start="<!--build-start-->"
    end="<!--build-end-->"
%}

<!--
{% include "resources/allen_ontology.html" recursive=false %}

{% include "resources/gridplot_cfos_vs_Arc1_density_summary_structures.html" recursive=false %}
-->