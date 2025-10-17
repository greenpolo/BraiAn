import yaml
import warnings
from pathlib import Path

from braian import from_csv, AtlasOntology, BrainData, Experiment, AnimalGroup, AnimalBrain, SlicedBrain, SlicedGroup, SlicedExperiment
from braian.utils import deprecated

__all__ = ["BraiAnConfig"]

class BraiAnConfig:
    @deprecated(since="1.1.0", params=["cache_path"])
    def __init__(self,
                 config_file: Path|str,
                 cache_path: Path|str=None, # for now used only to load the ontology from. If it doesn't find it it also downloads it there (for allen ontologies).
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
            output-dir: "data/BraiAn"

        groups:
            HC: ["287HC", "342HC", "343HC", "346HC", "371HC"]
            CTX: ["329CTX", "331CTX", "355CTX", "400CTX", "401CTX", "402CTX"]
            FC: ["367FC", "368FC", "369FC", "426FC", "427FC", "428FC"]

        atlas:
            name: "allen_mouse_10um"
            excluded-branches: ["VS", "fiber tracts", "CB"]

        brains:
            raw-metric: "sum"

        qupath:
            files:
                dirs:
                    output: "data/BraiAnDetect"
                    results-subdir: "results"
                    exclusions-subdir: "regions_to_exclude"
                suffix:
                    results: "_regions.txt"
                    exclusions: "_regions_to_exclude.txt"
                markers:
                    AF568: "cFos"
                    AF647: "Arc"
            exclude-ancestors-layer1: false
            min-slices: 0
        ```

        Parameters
        ----------
        config_file
            The path to a valid YAML configuration file.
        cache_path
            A path to a folder used to store files downloaded from the web and used for computations.\\
            Since '1.1.0' it's useless.
        """
        if not isinstance(config_file, Path):
            config_file = Path(config_file)
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self._conf = yaml.safe_load(f)

        # self.cache_path = Path(cache_path)
        self.experiment_name = self._conf["experiment"]["name"]
        if "output_dir" in self._conf["experiment"]:
            warnings.warn("Option 'experiment|output_dir' is deprecated since '1.1.0' and support may be removed in future versions. Use 'experiment|output-dir' instead.", DeprecationWarning)
            output_dir = self._conf["experiment"]["output_dir"]
        else:
            output_dir = self._conf["experiment"]["output-dir"]
        self.output_dir = _resolve_dir(output_dir, relative=self.config_file.absolute().parent)
        self._ontology: AtlasOntology = self._read_ontology()

    @property
    def ontology(self) -> AtlasOntology:
        """The atlas ontology as defined in the configuration file."""
        return self._ontology

    def _read_ontology(self) -> AtlasOntology:
        """
        Reads the brain ontology specified in the configuration file, and, if necessary, it dowloads it from the web.

        Returns
        -------
        :
            The brain ontology associated with the whole-brain data of the experiment.
        """
        if "version" in self._conf["atlas"] is not None:
            warnings.warn("Option 'atlas|version' is deprecated since '1.1.0' and support may be removed in future versions. Use 'atlas|name' instead.", DeprecationWarning)
            version = self._conf["atlas"]["version"]
            if version != 3 and version.lower() not in {"2017", "ccfv3", "v3"}:
                raise ValueError(f"Unsupported version of the Allen Mouse Brain ontology: '{self._conf['atlas']['version']}'. "+\
                                 "If you think this version of the Common Coordinate Framework should be supported, please open an issue on https://github.com/brainglobe/brainglobe-atlasapi")
                # import braian.utils
                # cached_allen_json = self.cache_path/"AllenMouseBrainOntology.json"
                # braian.utils.cache(cached_allen_json, "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
                # self._brain_ontology = AllenBrainOntology(cached_allen_json,
                #                                         self.config["atlas"]["excluded-branches"],
                #                                         version=self.config["atlas"]["version"])
            atlas_name = "allen_mouse_10um"
        else:
            atlas_name = self._conf["atlas"]["name"]
        if "excluded-branches" in self._conf["atlas"]:
            warnings.warn("Option 'atlas|excluded-branches' is deprecated since '1.1.0' and support may be removed in future versions. Use 'atlas|blacklisted' instead.", DeprecationWarning)
            blacklisted = self._conf["atlas"]["excluded-branches"]
        else:
            blacklisted = self._conf["atlas"]["blacklisted"]
        return AtlasOntology(atlas_name,
                             blacklisted=blacklisted,
                             unreferenced=False)

    @deprecated(since="1.1.0", alternatives=["braian.config.BraiAnConfig.from_csv"])
    def experiment_from_csv(self, sep: str=",", from_brains: bool=False, fill_nan: bool=True, legacy: bool=True) -> Experiment:
        return self.from_csv(sep=sep, legacy=legacy)

    def from_csv(self, name: str=None,
                 *, sep: str=",",
                 legacy: bool=False,
                 remove_unknown: bool=False) -> Experiment|AnimalGroup|AnimalBrain:
        """
        Reads some brain data with metric=`brains > raw-metric` from a comma-separated
        value (CSV) file saved in `experiment > output-dir`.

        Parameters
        ----------
        name
            The name of the group or of the animal to read.\\
            By default, it reads the whole experiment.
        sep
            Character or regex pattern to treat as the delimiter.
        remove_unknown
            If True and `filepath` contains data for regions not in `ontology`, it removes them
            instead of raising `UnknownBrainRegionsError`.

        Returns
        -------
        :
            The brain data that corresponds to `name`.

        Raises
        ------
        UnknownBrainRegionsError
            If the data the CSV contains regions not present in the ontology defined
            by `BraiAnConfig`.

        See also
        --------
        [`from_csv`][braian.from_csv]
        [`AnimalBrain.from_csv`][braian.AnimalBrain.from_csv]
        [`AnimalGroup.from_csv`][braian.AnimalGroup.from_csv]
        [`Experiment.from_csv`][braian.Experiment.from_csv]
        """
        metric = self._conf["brains"]["raw-metric"]
        assert BrainData.is_raw(metric), f"Configuration files should specify raw metrics only, not '{metric}'"
        if name is not None:
            group2brains: dict[str,list[str]] = self._conf["groups"]
            t = "group"
            if name not in group2brains:
                brains = {a for g in group2brains.values() for a in g}
                t = "brain"
                if name not in brains:
                    raise KeyError(f"No groups or brains named '{name}'")
        else:
            name = self.experiment_name
            t = "experiment"
        if legacy:
            group2brains: dict[str,list[str]] = self._conf["groups"]
            groups = [AnimalGroup.from_csv(self.output_dir/f"{name_}_{metric}.csv", name=name_,
                                             ontology=self._ontology, sep=sep, remove_unknown=remove_unknown, legacy=True)
                      for name_ in group2brains.keys()]
            
            return Experiment(name, *groups)
        return from_csv(self.output_dir/f"{name}_{metric}.csv", t, ontology=self._ontology, sep=sep, remove_unknown=remove_unknown)

    @deprecated(since="1.1.0", alternatives=["braian.config.BraiAnConfig.from_qupath"])
    def experiment_from_qupath(self, sliced: bool=False, validate: bool=True) -> Experiment|SlicedExperiment:
        return self.from_qupath(sliced=sliced)

    def from_qupath(self, sliced: bool=False, force_exclusion_files: bool=True) -> Experiment|SlicedExperiment:
        """
        Reads all the slice data exported to files with BraiAn's QuPath extension,
        and organises them into braian data structure used to identify an experiment.

        Parameters
        ----------
        sliced
            If False, after reading all the files about each section of the experiment,
            it reduces, for each brain, the data of every brain region into a single value
            accordingly to the method specified in the configuration file.\\
            Otherwise, it keeps the raw data.
        force_exclusion_files
            If True, it will skip any section with no exclusion file associated.

            Otherwise, it will keep them, along with every regional data from the section.

        See also
        --------
        [`SlicedGroup.from_qupath`][braian.SlicedGroup.from_qupath]
        [`SlicedExperiment.reduce`][braian.SlicedExperiment.reduce]

        Returns
        -------
        :
            An Experiment object, with all animals' and groups' data from QuPath.\\
            If sliced=True, it returns a SlicedExperiment.
        """
        qupath = self._conf["qupath"]
        qupath_dir = _resolve_dir(qupath["files"]["dirs"]["output"], relative=self.config_file.absolute().parent)
        if "results_subdir" in qupath["files"]["dirs"]:
            warnings.warn("Option 'qupath|files|dirs|results_subdir' is deprecated since '1.1.0' and support may be removed in future versions. Use 'qupath|files|dirs|results-subdir' instead.", DeprecationWarning)
            results_subdir = "results_subdir"
        else:
            results_subdir = "results-subdir"
        results_subir = qupath["files"]["dirs"].get(results_subdir, ".")
        if results_subir is None:
            results_subir = "."
        results_suffix = qupath["files"]["suffix"]["results"]
        if "exclusions_subdir" in qupath["files"]["dirs"]:
            warnings.warn("Option 'qupath|files|dirs|exclusions_subdir' is deprecated since '1.1.0' and support may be removed in future versions. Use 'qupath|files|dirs|exclusions-subdir' instead.", DeprecationWarning)
            exclusions_subdir = "exclusions_subdir"
        else:
            exclusions_subdir = "exclusions-subdir"
        exclusions_subdir = qupath["files"]["dirs"].get(exclusions_subdir, ".")
        if exclusions_subdir is None:
            exclusions_subdir = "."
        exclusions_suffix = qupath["files"]["suffix"]["exclusions"]
        markers = qupath["files"]["markers"]

        if "exclude-parents" in qupath:
            warnings.warn("Option 'exclude-parents' is now ignored. "+\
                          "Quantifications in ancestor regions are now always completely removed too.")
        exclude_ancestors_layer1 = qupath.get("exclude-ancestors-layer1", True)
        group2brains: dict[str,str] = self._conf["groups"]
        groups = []
        for g_name, brain_names in group2brains.items():
            group = SlicedGroup.from_qupath(name=g_name, brain_names=brain_names,
                                            qupath_dir=qupath_dir,
                                            brain_ontology=self._ontology,
                                            ch2marker=markers, #exclude_parents,
                                            exclude_ancestors_layer1=exclude_ancestors_layer1,
                                            results_subdir=results_subir, results_suffix=results_suffix,
                                            exclusions_subdir=exclusions_subdir, exclusions_suffix=exclusions_suffix,
                                            force_exclusion_files=force_exclusion_files)
            groups.append(group)

        sliced_exp = SlicedExperiment(self.experiment_name, *groups)
        return sliced_exp if sliced else self.reduce(sliced_exp)

    @deprecated(since="1.1.0", alternatives=["braian.config.BraiAnConfig.reduce"])
    def experiment_from_sliced(self,
                               sliced_exp: SlicedExperiment,
                               hemisphere_distinction: bool=True,
                               validate: bool=True) -> Experiment:
        return self.reduce(sliced_exp=sliced_exp)

    def reduce(self, sliced: SlicedBrain|SlicedGroup|SlicedExperiment) -> Experiment:
        """
        It [reduces][braian.reduce] sliced brain data, using `brains > raw-metric` and `qupath > min-slices`
        from the configuration file.

        Parameters
        ----------
        sliced
            Sliced brain data, composed of multiple [`BrainSlice`][braian.BrainSlice].

        Returns
        -------
        : AnimalBrain
            If `d` is a `SlicedBrain`
        : AnimalGroup
            If `d` is a `SlicedGroup`
        : Experiment
            If `d` is a `SlicedExperiment`

        See also
        --------
        [`braian.reduce`][braian.reduce]
        [`SlicedBrain.reduce`][braian.SlicedBrain.reduce]
        [`SlicedGroup.reduce`][braian.SlicedGroup.reduce]
        [`SlicedExperiment.reduce`][braian.SlicedExperiment.reduce]
        """
        return sliced.reduce(metric=self._conf["brains"]["raw-metric"],
                             min_slices=self._conf["qupath"]["min-slices"],
                             densities=False) # raw matrics will never be a density

def _resolve_dir(path: Path|str, relative: Path|str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_absolute():
        return path
    if not isinstance(relative, Path):
        relative = Path(relative)
    return relative/path
