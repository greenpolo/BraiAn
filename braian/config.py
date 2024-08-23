import yaml
from pathlib import Path

from braian import AllenBrainOntology, AnimalBrain, Project, SlicedGroup, SlicedProject
import braian.utils

class ProjectDir:
    def __init__(self) -> None:
        pass

class BraiAnConfig:
    def __init__(self,
                 config_file: Path|str,
                 cache_path: Path|str, # for now used only to load the ontology from. If it doesn't find it it also downloads it there (for allen ontologies).
                 ) -> None:
        if not isinstance(config_file, Path):
            config_file = Path(config_file)
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.cache_path = Path(cache_path)
        self.project_name = self.config["project"]["name"]
        self.output_dir = _resolve_dir(self.config["project"]["output_dir"], relative=self.config_file.absolute().parent)
        self._brain_ontology: AllenBrainOntology = None

    def read_atlas_ontology(self):
        cached_allen_json = self.cache_path/"AllenMouseBrainOntology.json"
        braian.utils.cache(cached_allen_json, "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        self._brain_ontology = AllenBrainOntology(cached_allen_json,
                                                 self.config["atlas"]["excluded-branches"],
                                                 version=self.config["atlas"]["version"])
        return self._brain_ontology

    def project_from_csv(self, sep=",", from_brains: bool=False, fill_nan: bool=True) -> Project:
        metric = self.config["brains"]["raw-metric"]
        assert AnimalBrain.is_raw(metric), f"Configuration files should specify raw metrics only, not '{metric}'"
        group2brains: dict[str,str] = self.config["groups"]
        if not from_brains:
            return Project.from_group_csv(self.project_name, group2brains.keys(), metric, self.output_dir, sep)
        if self._brain_ontology is None:
            self.read_atlas_ontology()
        return Project.from_brain_csv(self.project_name, group2brains, metric,
                                      self.output_dir, sep, brain_ontology=self._brain_ontology,
                                      fill_nan=fill_nan)

    def project_from_qupath(self, sliced: bool=False, fill_nan: bool=True) -> Project|SlicedProject:
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
        min_slices = qupath["min-slices"]
        group2brains: dict[str,str] = self.config["groups"]
        groups = []
        if self._brain_ontology is None:
            self.read_atlas_ontology()
        for g_name, brain_names in group2brains.items():
            group = SlicedGroup.from_qupath(g_name, brain_names, markers,
                                            qupath_dir, self._brain_ontology, exclude_parents,
                                            results_subir, results_suffix,
                                            exclusions_subdir, exclusions_suffix)
            groups.append(group)

        sliced_pj = SlicedProject(self.project_name, *groups)
        return sliced_pj if sliced else sliced_pj.to_project(self.config["brains"]["raw-metric"],
                                                             min_slices,
                                                             densities=False, # raw matrics will never be a density
                                                             fill_nan=fill_nan)

    def project_from_sliced(self, sliced_pj: SlicedProject, fill_nan: bool) -> Project:
        return sliced_pj.to_project(self.config["brains"]["raw-metric"],
                                    self.config["qupath"]["min-slices"],
                                    densities=False, # raw matrics will never be a density
                                    fill_nan=fill_nan)
        

def _resolve_dir(path: Path|str, relative: Path|str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_absolute():
        return path
    if not isinstance(relative, Path):
        relative = Path(relative)
    return relative/path
