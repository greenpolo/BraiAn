import yaml
from pathlib import Path

from braian import AllenBrainOntology, AnimalBrain, Experiment, SlicedGroup, SlicedExperiment
import braian.utils

class BraiAnConfig:
    def __init__(self,
                 config_file: Path|str,
                 cache_path: Path|str, # for now used only to load the ontology from. If it doesn't find it it also downloads it there (for allen ontologies).
                 ) -> None:
        """
        Reads a [YAML](https://en.wikipedia.org/wiki/YAML) configuration file for
        managing a whole-brain experiment, made of multiple cohorts, with `braian`.

        An example of a valid configuration file is the following:
        ```yaml
        # SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
        #
        # SPDX-License-Identifier: CC0-1.0

        experiment:
        name: "example"
        output_dir: "data/BraiAn_output"

        groups:
        HC: ["287HC", "342HC", "343HC", "346HC", "371HC"]
        CTX: ["329CTX", "331CTX", "355CTX", "400CTX", "401CTX", "402CTX"]
        FC: ["367FC", "368FC", "369FC", "426FC", "427FC", "428FC"]

        atlas:
        version: "v3"
        excluded-branches: ["retina", "VS", "grv", "fiber tracts", "CB"]

        brains:
        raw-metric: "sum"

        qupath:
        files:
            dirs:
            output: "data/QuPath_output"
            results_subdir: "results"
            exclusions_subdir: "regions_to_exclude"
            suffix:
            results: "_regions.txt"
            exclusions: "_regions_to_exclude.txt"
            markers:
            AF568: "cFos"
            AF647: "Arc"
        exclude-parents: true
        min-slices: 0
        ```

        Parameters
        ----------
        config_file
            The path to a valid YAML configuration file.
        cache_path
            A path to a folder used to store files downloaded from the web and used for computations.\\
            At the moment this is used only to store the brain ontology to which the whole-brain data was aligned to.
        """
        if not isinstance(config_file, Path):
            config_file = Path(config_file)
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.cache_path = Path(cache_path)
        self.experiment_name = self.config["experiment"]["name"]
        self.output_dir = _resolve_dir(self.config["experiment"]["output_dir"], relative=self.config_file.absolute().parent)
        self._brain_ontology: AllenBrainOntology = None

    def read_atlas_ontology(self) -> AllenBrainOntology:
        """
        Reads the brain ontology specified in the configuration file, and, if necessary, it dowloads it from the web.

        Returns
        -------
        :
            The brain ontology associated with the whole-brain data of the experiment.
        """
        cached_allen_json = self.cache_path/"AllenMouseBrainOntology.json"
        braian.utils.cache(cached_allen_json, "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        self._brain_ontology = AllenBrainOntology(cached_allen_json,
                                                 self.config["atlas"]["excluded-branches"],
                                                 version=self.config["atlas"]["version"])
        return self._brain_ontology

    def experiment_from_csv(self, sep: str=",", from_brains: bool=False, fill_nan: bool=True) -> Experiment:
        metric = self.config["brains"]["raw-metric"]
        assert AnimalBrain.is_raw(metric), f"Configuration files should specify raw metrics only, not '{metric}'"
        group2brains: dict[str,str] = self.config["groups"]
        if not from_brains:
            return Experiment.from_group_csv(self.experiment_name, group2brains.keys(), metric, self.output_dir, sep)
        if self._brain_ontology is None:
            self.read_atlas_ontology()
        return Experiment.from_brain_csv(self.experiment_name, group2brains, metric,
                                      self.output_dir, sep, brain_ontology=self._brain_ontology,
                                      fill_nan=fill_nan)

    def experiment_from_qupath(self, sliced: bool=False, validate: bool=True) -> Experiment|SlicedExperiment:
        """
        Reads all the slice data exported to files with BraiAn's QuPath extension,
        and organises them into braian data structure used to identify an experiment.

        If [`read_atlas_ontology()`][braian.config.BraiAnConfig.read_atlas_ontology]
        was not called previously, it reads the ontology.

        Parameters
        ----------
        sliced
            If False, after reading all the files about each section of the experiment,
            it reduces, for each brain, the data of every brain region into a single value
            accordingly to the method specified in the configuration file.\\
            Otherwise, it keeps the raw data.
        validate
            If True, it validates each region in each brain, checking that they are present
            in the brain region ontology against which the brains were alligned.

        See also
        --------
        [`SlicedGroup.from_qupath`][braian.SlicedGroup.from_qupath]
        [`SlicedExperiment.to_experiment`][braian.SlicedExperiment.to_experiment]

        Returns
        -------
        :
            _description_
        """
        qupath = self.config["qupath"]
        qupath_dir = _resolve_dir(qupath["files"]["dirs"]["output"], relative=self.config_file.absolute().parent)
        results_subir = qupath["files"]["dirs"].get("results_subdir", ".")
        if results_subir is None:
            results_subir = "." 
        results_suffix = qupath["files"]["suffix"]["results"]
        exclusions_subdir = qupath["files"]["dirs"].get("exclusions_subdir", ".")
        if exclusions_subdir is None:
            exclusions_subdir = "."
        exclusions_suffix = qupath["files"]["suffix"]["exclusions"]
        markers = qupath["files"]["markers"]
        
        exclude_parents = qupath["exclude-parents"]
        group2brains: dict[str,str] = self.config["groups"]
        groups = []
        if self._brain_ontology is None:
            self.read_atlas_ontology()
        for g_name, brain_names in group2brains.items():
            group = SlicedGroup.from_qupath(g_name, brain_names, qupath_dir,
                                            self._brain_ontology, markers, exclude_parents,
                                            results_subir, results_suffix,
                                            exclusions_subdir, exclusions_suffix)
            groups.append(group)

        sliced_exp = SlicedExperiment(self.experiment_name, *groups)
        return sliced_exp if sliced else self.experiment_from_sliced(sliced_exp, validate=validate)

    def experiment_from_sliced(self, sliced_exp: SlicedExperiment, validate: bool=True) -> Experiment:
        """
        It reduces, for each brain of a sliced experiment, the data of every brain region into a single value
        accordingly to the method specified in the configuration file.

        Parameters
        ----------
        sliced_exp
            The experiment sliced into sections to reduce into one value per-region per-brain.
        validate
            If True, it validates each region in each brain, checking that they are present
            in the brain region ontology against which the brains were alligned.

        Returns
        -------
        :
            _description_
        """
        return sliced_exp.to_experiment(self.config["brains"]["raw-metric"],
                                    self.config["qupath"]["min-slices"],
                                    densities=False, # raw matrics will never be a density
                                    hemisphere_distinction=True,
                                    validate=validate)
        

def _resolve_dir(path: Path|str, relative: Path|str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_absolute():
        return path
    if not isinstance(relative, Path):
        relative = Path(relative)
    return relative/path
