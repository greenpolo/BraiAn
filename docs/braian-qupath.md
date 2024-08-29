# BraiAn for QuPath (BraiAnDetect)

The first module, named BraiAnDetect, consists of a [QuPath](https://qupath.github.io/) extension for image analysis of serial brain sections across many animals. If is designed for multichannel cell segmentation across large and variable datasets and ensures consistency in image analysis across large datasets. This module leverages QuPath’s built-in algorithms to provide a multi-channel, whole-brain optimised object detection pipeline. BraiAnDetect features options for refining signal quantification, including machine learning-based object classification, region specific cell segmentation, multiple marker co-expression analysis and an interface for selective exclusion of damaged tissue portions. 

It works best if coupled with [`qupath-extension-abba`](https://github.com/biop/qupath-extension-abba) for importing whole-brain atlas registrations from [ABBA](https://go.epfl.ch/abba) as annotations.

## Features

BraiAnDetect helps you manage image analysis across multiple QuPath projects ensuring consistency. In particular, it is designed to perform batch analysis across many QuPath projects in an automated manner. Typically, in whole-brain datasets, one brain = one QuPath project and BraiAn makes sure the exact same analysis parameters are consistently applied across different projects.
It was first developed to work with [ABBA](https://go.epfl.ch/abba), but can be used for other purposes as well.
Its core idea is to move the input image analysis parameters used to analyse multiple QuPath projects of the same cohort/experiment _outside_ of scripts' code (in a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file, see below). This allows having a reproducible configuration that can be shared, backed up and ran after long periods of time.

The extensions exposes a proper library [API](https://carlocastoldi.github.io/qupath-extension-braian/docs/). Here are some examples. It allows you to:

- work with image [channel histograms](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/ChannelHistogram.html)
- compute and manage [detections](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AbstractDetections.html) separately for each image channel
- apply different [classifiers](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/PartialClassifier.html) on different subsets of detections
- _quickly_ find all detections that are double—or triple/multiple—positive, thanks to [`BoundingBoxHierarchy`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/BoundingBoxHierarchy.html)
- tag certain brain regions to be excluded from further analysis due to tissue, imaging or alignment problems
- export to file the quantification results (number of detections/double+ found in each brain region)
- export to file a list of regions flagged to be excluded

Where to start from, though? Reading [this script](https://github.com/carlocastoldi/qupath-extension-braian/blob/master/src/main/resources/scripts/compute_classify_overlap_export_exclude_detections.groovy) and the associated [config file](https://github.com/carlocastoldi/qupath-extension-braian/blob/master/BraiAn.yml) is a good start!

## Installation

You can download the [latest](https://github.com/carlocastoldi/qupath-extension-braian/releases/latest) release from the the official GitHub page of the project named `qupath-extension-braian-<VERSION>.jar`. Generally, there is no need to download `-javadoc.jar` and `-sources.jar`.\
Later you can simply drag the downloaded file onto QuPath, restart and be good to go!

Any new available release of the extension should be notified on QuPath startup, and its update can be handled in QuPath itself.

## Getting started

This extension does not currently expose any user-friendly interface, but it mostly offers a set of functions and [example scripts](#prebaked-scripts) that should be easily modifiable from someone with little programming knowledge.

!!! abstract "Citation"

    Please, if you use any module of BraiAn or code here shown, follow our [citation guide](index.md#how-to-cite-braian)!

However, before running any script using this extension, we need to describe how BraiAn works and its assumptions.

### Project arrangement

All the brain section images of the same animal should be collected into a single QuPath project. If there are multiple animals, one should create a project for each one of them and save the respective project folders in the same directory (e.g. `QuPath_projects/`).

### Cell Detections
The first thing with BraiAn detect is segmenting positive objects (typically cells but can be nuclei, spines, axon terminals,...) for each channel. EXPLAIN THIS. ADD HERE THE SCRIPT RUNNING THE DETECTIONS

### Classifiers

When you can't find parameters that are satisfying for all sections and animals of the groups, you might want to try applying an object classifier that filters out the detections that are not satisfying. In such a case, the detection should first be computed permissive parameters that don't miss ~any positive cell; even at the cost of having a high number of false detection: it will be the classifier job to filter them out.

If you don't know how to create an object classifier in QuPath, we suggest you to read [the official tutorial](https://qupath.readthedocs.io/en/stable/docs/tutorials/cell_classification.html#train-a-cell-classifier-based-on-annotations). Once the classifier is sufficiently accurate, you can save it and QuPath will create a `.json` file in `<QUPATH_PROJECT>/classifiers/object_classifiers/`.\
For it to be read by BraiAn, copy it in the same location where the YAML configuration file is.

### Configuration file

BraiAn for QuPath can use a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file. This configuration file contains all the customizable image analysis parameters in one place and thus allows to apply the exact same signal quantification settings across all projects (i.e. animals) and runs. In addition, this is the place image analysis details are stored and is great to keep track of how the analysis was done even after a long time.  The file has to be positioned either in a QuPath project folder or in its parent directory (e.g. `QuPath_projects/`), depending on whether the analysis is meant for a specific project or multiple ones.

For a detailed explanation on how to format such file and understanding what each parameter does, we suggest to look at [this example YAML file](https://raw.githubusercontent.com/carlocastoldi/qupath-extension-braian/master/BraiAn.yml).

### Detection containers

The extension lets you select the annotations in which to compute the cell detection on the desired image channels. This selection can be controlled passing the name of the annotations' classification.\
Effectively, this may lead to regions that overlap or detections of different image channels colliding in the same area.

![**_Figure 1_**: Example of annotations in QuPath using BraiAn](img/qupath_annotations.png "Example of annotations in QuPath using BraiAn")

In order to handle this complexity, BraiAn uses what we call "_detection containers_". They are annotations specific to an image channel placed, in QuPath hierarchy, under a selected annotation and they _contain_ all detection computed on a specific channel. If two containers of the same channel overlap, only the detection from one of the two will be kept to avoid having double the cells.

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

### Prebaked scripts

Once you installed the extension, you can load example scripts by clicking on the top menu: `Extensions ‣ BraiAn ‣ scripts`.

The very same ones can also be checked out from the [official repository](https://github.com/carlocastoldi/qupath-extension-braian/tree/master/src/main/resources/scripts):

* [compute_classify_overlap_export_exclude_detections.groovy](https://github.com/carlocastoldi/qupath-extension-braian/blob/master/src/main/resources/scripts/compute_classify_overlap_export_exclude_detections.groovy): this script reads the YAML configuration file and applies all of its parameters. SAY MORE HERE!
* [find_threshold.groovy](https://github.com/carlocastoldi/qupath-extension-braian/blob/master/src/main/resources/scripts/find_threshold.groovy): suggests a threshold to apply with WatershedCellDetection algorithm by choosing a local maximum from the image's histogram.
* [run_script_for_multiple_projects.groovy](https://github.com/carlocastoldi/qupath-extension-braian/blob/master/src/main/resources/scripts/run_script_for_multiple_projects.groovy): helps running a script for multiple project at once. It is compatible with the [LightScriptRunner](light-script-runner.md).

## Building

You can build the QuPath BraiAn extension from source with:

```bash
./gradlew clean build
```

The built `.jar` extension file will be under `build/libs`.
