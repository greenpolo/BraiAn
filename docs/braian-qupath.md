# BraiAn for QuPath

The first module of BraiAn, also known as _BraiAnDetect_, consists of a [QuPath](https://qupath.github.io/) extension for image analysis of serial brain sections across many animals. It is designed for multichannel cell segmentation across large and variable datasets and ensures consistency in image analysis across large datasets. This module leverages QuPath's built-in algorithms to provide a multi-channel, whole-brain optimised object detection pipeline. BraiAnDetect features options for refining signal quantification, including machine learning-based object classification, region specific cell segmentation, multiple marker co-expression algorithms and an interface for selective exclusion of damaged tissue portions.

It works best if coupled with [`qupath-extension-abba`](https://github.com/biop/qupath-extension-abba) for importing whole-brain atlas registrations from [ABBA](https://go.epfl.ch/abba) as annotations.

## Installation

### Using QuPath _Catalogs_
This is only possible in newer QuPath versions (0.6.0, or newer):

+ Open QuPath extension manager by clicking on `Extensions > Manage extensions`;
+ Click on `Manage extension catalogs`;
+ Paste the following URL and click the `Add` button: `https://github.com/carlocastoldi/qupath-extension-braian-catalog`;
+ In the extension manager, click on the `+` symbol next to BraiAn extension.

This new installation method for third-party extensions was introduced in QuPath to help users keep them up to date.

### Manual
You can download the [latest](https://github.com/carlocastoldi/qupath-extension-braian/releases/latest) release from the the official GitHub page of the project named `qupath-extension-braian-<VERSION>.jar`. Generally, there is no need to download `-javadoc.jar` and `-sources.jar`.\
Later you can drag the downloaded file onto QuPath, restart and be good to go!

Up until QuPath 0.5.1 (included), new extension releases are notified by QuPath on startups. You'll then be able to update them through QuPath's extension manager with one click.\
From QuPath 0.6.+, extensions installed manually no longer receive updates.

## Features

BraiAnDetect helps you manage image analysis across multiple QuPath projects ensuring consistency. In particular, it is designed to perform batch analysis across many QuPath projects in an automated manner. Typically, in whole-brain datasets, one brain = one QuPath project and BraiAn makes sure the exact same analysis parameters are consistently applied across different projects.
It was first developed to work with [ABBA](https://go.epfl.ch/abba), but can be used for other purposes as well.
Its core idea is to move the input image analysis parameters used to analyse multiple QuPath projects of the same cohort/experiment _outside_ of scripts' code (in a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file, see below). This allows having a reproducible configuration that can be shared, backed up and ran after long periods of time.

The extensions exposes a proper library [API](https://carlocastoldi.github.io/qupath-extension-braian/docs/). Here are some examples. It allows you to:

- multi-channel automatic object segmentation (e.g. [cell detections](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AbstractDetections.html))
- machine-learning-based [object classification](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/PartialClassifier.html) (e.g. apply custom classifiers on different detection types).
- co-localization analysis (i.e. _quickly_ find all detections that are double—or triple/multiple—positive, through [`BoundingBoxHierarchy`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/BoundingBoxHierarchy.html))
- fine tune image analysis using [channel histograms](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/ChannelHistogram.html)
- tag certain brain regions to be excluded from further analysis due to tissue, imaging or alignment problems
- export to file the quantification results (number of detections/double+ found in each brain region)
- export to file a list of regions flagged to be excluded

Where to start from, though? Reading [this script](https://github.com/carlocastoldi/qupath-extension-braian/blob/v{{braian_qupath.latest}}/src/main/resources/scripts/compute_classify_overlap_export_exclude_detections.groovy) and the associated [config file](https://github.com/carlocastoldi/qupath-extension-braian/blob/v{{braian_qupath.latest}}/BraiAn.yml) is a good start!

## Getting started

This extension does not currently expose any user-friendly interface, but it mostly offers a set of functions and [prebaked scripts](#prebaked-scripts) that should be easily modifiable from someone with little programming knowledge.

!!! abstract "Citation"

    Please, if you use any module of BraiAn or code here shown, follow our [citation guide](index.md#how-to-cite-braian)!

### Prebaked scripts

Once you installed the extension, you can load prebaked scripts by clicking on the top menu: `Extensions ‣ BraiAn ‣ scripts`.

The very same ones can also be checked out from the [official repository](https://github.com/carlocastoldi/qupath-extension-braian/tree/v{{braian_qupath.latest}}/src/main/resources/scripts):

* [compute_classify_overlap_export_exclude_detections.groovy](https://github.com/carlocastoldi/qupath-extension-braian/blob/v{{braian_qupath.latest}}/src/main/resources/scripts/compute_classify_overlap_export_exclude_detections.groovy): this script reads the YAML configuration file and applies all of its parameters. <!-- SAY MORE HERE! -->
* [find_threshold.groovy](https://github.com/carlocastoldi/qupath-extension-braian/blob/v{{braian_qupath.latest}}/src/main/resources/scripts/find_threshold.groovy): suggests a threshold to apply with WatershedCellDetection algorithm by choosing a local maximum from the image's histogram.
* [run_script_for_multiple_projects.groovy](https://github.com/carlocastoldi/qupath-extension-braian/blob/v{{braian_qupath.latest}}/src/main/resources/scripts/run_script_for_multiple_projects.groovy): helps running a script for multiple project at once. It is compatible with the [LightScriptRunner](light-script-runner.md).


However, before running any script using this extension, we need to describe how BraiAn works and its assumptions.

### Project arrangement

All the brain section images of the same animal should be collected into a single QuPath project. If there are multiple animals, one should create a project for each one of them and save the respective project folders in the same directory (e.g. `QuPath_projects/`).

<!--
### Cell Detections
RUN POSITIVE DETECTIONS FROM QUPATH https://qupath.readthedocs.io/en/stable/docs/tutorials/cell_detection.html#run-positive-cell-detection

extracts parameters from quPath's interface and reads them from the configuration file for reproducibility. <-- link to paragraph

Per image-channel fine tuning of such parameters is required and  BraiAn can help.

------
The first thing with BraiAn detect is segmenting positive objects (typically cells but can be nuclei, spines, axon terminals,...) for each channel. EXPLAIN THIS. ADD HERE THE SCRIPT RUNNING THE DETECTIONS
-->

### Classifiers

When you can't find parameters to automatically segment your markers that are satisfying for all sections and animals of the groups, you might want to try applying an object classifier that filters out the detections that are not correct. In this case, the detection should first be computed with highly permissive parameters that don't miss ~any positive cell; even at the cost of having a high number of false detection: it will be the classifier job to filter them out.

If you don't know how to create an object classifier in QuPath, we suggest you to read [the official tutorial](https://qupath.readthedocs.io/en/stable/docs/tutorials/cell_classification.html#train-a-cell-classifier-based-on-annotations). Once the classifier is sufficiently accurate, you can save it and QuPath will create a `.json` file in `<QUPATH_PROJECT>/classifiers/object_classifiers/`.\
For it to be read by BraiAn, copy it in the same location where the YAML configuration file is.

### Configuration file

BraiAn for QuPath can use a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file. This configuration file contains all the customizable image analysis parameters in one place and thus allows to apply the exact same signal quantification settings across all projects (i.e. animals) and runs. In addition, this is the place image analysis details are stored and is great to keep track of how the analysis was done even after a long time.  The file has to be positioned either in a QuPath project folder or in its parent directory (e.g. `QuPath_projects/`), depending on whether the analysis is meant for a specific project or multiple ones.

For a detailed explanation on how to format such file and understanding what each parameter does, we suggest check out the [template file](https://raw.githubusercontent.com/carlocastoldi/qupath-extension-braian/refs/tags/v{{braian_qupath.latest}}/BraiAn.yml) of the config file expected by BraiAnDetect.

### Detection containers

Managing large numbers of detections that have been computed across many brain regions and channels can be challenging. BraiAnDetect is designed to facilitate this step and prevent classical errors (such as detections computed over overlapping annotations or detections of different image channels colliding in the same area).\
To handle annotations derived from multiple channels, BraiAnDetect uses what we call **detection containers** (or just _containers_). Containers are QuPath annotations that _contain_ detections computed on each specific channel and belong (in QuPath annotation hierarchy) to the parent annotation containing all detections. If two containers of the same channel overlap, only the detection from one of the two will be kept to avoid having double the cells.

In addition, BraiAnDetect allows you, for each channel, to select a specific subset of annotations in which any image analysis algorithm is run. This allows you, for example, to create [`AbstractDetections`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AbstractDetections.html) using different parameters for different subsets of brain regions. This is done setting different [object classification](https://qupath.readthedocs.io/en/stable/docs/concepts/objects.html#classification) to each annotations subset.

![**_Figure 1_**: Example of annotations in QuPath using BraiAn](resources/braian-qupath/qupath_annotations.png "Example of annotations in QuPath using BraiAn")



!!! warning

    Containers are kept after the BraiAn scripts have run. This means that the user could tinker with them, effectively breaking any possible future run of the scripts. For this reason we highly suggest you not to do so, apart from removing them (and their contents) all together.

### Region exclusions

Sometimes we would like to discard some parts of a section—be it for problems with the tissue, imaging or registration to an atlas—but we would still like to keep some portions!

In this case you can draw an annotation over the desired portion to exclude from the analysis, making sure that it completely covers the atlas annotations below. This can be done using the Polygon tool or duplicating an atlas annotation (Shift+D) and [classifying them](https://qupath.readthedocs.io/en/stable/docs/concepts/objects.html#classification) with the ad-hoc QuPath classificat·on for the _exclusion_ annotations: "Exclude" (see _Fig. 1_).

!!! warning

    Under no circumstances you have to modify (i.e. deleting, re-classifying, renaming or moving) the atlas annotations imported from ABBA.

If you want to make sure that you excluded all brain regions you wanted from an image, you can click on `Extensions ‣ BraiAn ‣ Show regions currently excluded` and see which regions from the atlas have been selected.

### Output

[`AtlasManager.saveResults()`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AtlasManager.html#saveResults(java.util.List,java.io.File)) and [`AtlasManager.saveExcludedRegions()`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AtlasManager.html#saveExcludedRegions(java.io.File)) will export the data from cell counts and exclusions to a file for each image.

## Building

You can build the QuPath BraiAn extension from source with:

```bash
./gradlew clean build
```

The built `.jar` extension file will be under `build/libs`.
