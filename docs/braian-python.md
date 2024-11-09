# BraiAn python library (BraiAnalyse)

BraiAnalyse a modular Python library for the easy navigation, visualisation, and analysis of whole-brain quantification outputs. The input data can either be from QuPath, with little-to-no work thanks to [_BraiAn for QuPath_](braian-qupath.md#output) or any other software of brain cell segmentation (e.g. [Clearmap](https://clearanatomics.github.io/ClearMapDocumentation/)).

Its aim is to give you a framework to build your own analysis, typically comparing multiple markers across groups of animals across the entire brain. It provides statistical tools optimized for whole-brain analysis of multiple markers allowing the identification  of the whole-brain sets of regions that differ between groups and visualising data in interactive and informative plots.

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

The `braian` package wants to help neuroscientists in three main tasks. For each of them, we designed a dedicated sub-module:

* [`braian`](api-braian.md): intended for importing brain data from different sources (e.g. QuPath) and checking data integrity against a brain atlas ontology. It can also prepare the data in cohorts/groups and experiments;
* [`braian.stats`](api-stats.md): intended for brain data normalization, aggregation and group and markers statistical analysis;
* [`braian.plot`](api-plot.md): intended for brain data visualisation, in an interactive intuitive manner.

Overall, BraiAn aims at exposing non-trivial operations with an intuitive interface. Nonetheless, the user is required to have basic programming skills in order ensure the statistical analyses are performed accordingly to the parameters that better suits their needs.

### Configuration file

Just like [BraiAn for QuPath](braian-qupath.md#configuration-file), `braian` can use a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file to store metadata about an experiment along with some additional parameters. Such file can be read in `braian` and used to apply a consistent analysis parameters across all animals and runs.

For a detailed explanation on how to format such file and understanding what each parameter does, we suggest to look at [this example YAML file](https://codeberg.org/SilvaLab/BraiAn/raw/branch/master/config_example.yml) and [`BraiAnConfig` API][braian.config.BraiAnConfig].

### Prebaked analysis

Examples of available whole-brain statistical analysis provided by braian are:

* [braian.stats.PLS][]
* [braian.stats.density][]
* [braian.stats.percentage][]
* [braian.stats.relative_density][]
* [braian.stats.fold_change][]
* [braian.stats.diff_change][]
* [braian.stats.markers_overlap][]
* [braian.stats.markers_jaccard_index][]
* [braian.stats.markers_similarity_index][]
* [braian.stats.markers_overlap_coefficient][]
* [braian.stats.markers_difference][]
* [braian.stats.markers_correlation][]

{%
    include-markdown "../README.md"
    start="<!--build-start-->"
    end="<!--build-end-->"
%}
