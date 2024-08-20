import os
import pandas as pd
import yaml
from pathlib import Path

from braian import AllenBrainOntology, AnimalBrain, Project, SlicedBrain, SlicedGroup, SlicedProject
from braian.utils import cache

class ProjectDir:
    def __init__(self) -> None:
        pass

class BraiAnConfig:
    def __init__(self,
                 data_path: Path|str,
                 config_file: Path|str,
                 ) -> None:
        if not isinstance(config_file, Path):
            config_file = Path(config_file)
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_path = data_path
        path_to_allen_json = os.path.join(self.data_path, "AllenMouseBrainOntology.json")
        cache(path_to_allen_json, "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        self.brain_ontology = AllenBrainOntology(path_to_allen_json,
                                                   self.config["atlas"]["excluded-branches"],
                                                   version=self.config["atlas"]["version"])
        self.project_name = self.config["project"]["name"]
        self.output_dir = _resolve_dir(self.config["project"]["output_dir"], relative=self.config_file.absolute().parent)

    def project_from_csv(self, sep=",", from_brains: bool=False, fill_nan: bool=True) -> Project:
        metric = self.config["brains"]["raw-metric"]
        assert AnimalBrain.is_raw(metric), f"Configuration files should specify raw metrics only, not '{metric}'"
        group2brains: dict[str,str] = self.config["groups"]
        if not from_brains:
            return Project.from_group_csv(self.project_name, group2brains.keys(), metric, self.output_dir, sep)
        return Project.from_brain_csv(self.project_name, group2brains, metric,
                                      self.output_dir, sep, ontology=self.brain_ontology,
                                      fill_nan=fill_nan)

    def project_from_qupath(self, sliced: bool=False, fill_nan: bool=True) -> Project|SlicedProject:
        qupath_dir = _resolve_dir(self.config["qupath"]["dir"], relative=self.config_file.absolute().parent)
        markers = self.config["qupath"]["markers"]
        exclude_parents = self.config["qupath"]["exclude-parents"]
        group2brains: dict[str,str] = self.config["groups"]
        groups = []
        for g_name, brain_names in group2brains.items():
            group = SlicedGroup.from_qupath(g_name, brain_names, markers, qupath_dir, self.brain_ontology, exclude_parents)
            groups.append(group)

        sliced_pj = SlicedProject(self.project_name, *groups)
        return sliced_pj if sliced else sliced_pj.to_project(self.config["brains"]["raw-metric"],
                                                             self.config["qupath"]["min-slices"],
                                                             fill_nan)

    def project_from_sliced(self, sliced_pj: SlicedProject, fill_nan: bool) -> Project:
        return sliced_pj.to_project(self.config["brains"]["raw-metric"],
                                    self.config["qupath"]["min-slices"],
                                    fill_nan)
        

def _resolve_dir(path: Path|str, relative: Path|str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_absolute():
        return path
    if not isinstance(relative, Path):
        relative = Path(relative)
    return relative/path