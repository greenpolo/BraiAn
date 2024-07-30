# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import itertools
import os
import pandas as pd
import re
import toml
from collections import OrderedDict, namedtuple
from typing import Self

from braian.animal_brain import AnimalBrain
from braian.animal_group import AnimalGroup
from braian.brain_data import BrainData
from braian.brain_hierarchy import AllenBrainHierarchy, MAJOR_DIVISIONS
from braian.brain_metrics import BrainMetrics
from braian.brain_slice import BrainSlice
from braian.sliced_brain import SlicedBrain
from braian.utils import cache


# # ######################################### LOAD CONFIG #########################################
# EXPERIMENT_NAME = config["experiment"]["name"]

class BraiAnConfig:
    class GroupDirectory:
        def __init__(self, id: int, name: str, animal_directories: list[str]) -> None:
            self.id = id
            self.name = name
            self.animal_names = animal_directories
            self.sliced_brains: list[SlicedBrain] = []  # filled in read()
            self.brains: list[AnimalBrain] = []         # filled in reduce_slices()
        
        def read(self, path_to_group, threshold: float, remove_singles: bool,
                 *args, **kwargs) -> list[SlicedBrain]:
            for name in self.animal_names:
                animal_dir = os.path.join(path_to_group, name)
                if not os.path.isdir(animal_dir):
                    print(f"WARNING: could not find the directory '{animal_dir}'. Skipping animal '{name}'.")
                    continue
                sliced_brain = SlicedBrain.from_qupath(name,
                                                       animal_dir,
                                                       *args,
                                                       **kwargs)
                # self._fix_overlap_detection_if_old_qpscript(sliced_brain)
                if threshold > 0:
                    # TODO: consider to use this option *only* for the coefficient of variation plot!
                    self._remove_small_regions(sliced_brain, threshold)
                if remove_singles:
                    self._remove_singles(sliced_brain)
                self.sliced_brains.append(sliced_brain)
            return self.sliced_brains
        
        def reduce_slices(self, metric: str) -> list[AnimalBrain]:
            self.brains = [
                AnimalBrain.from_slices(sliced_brain, mode=metric, hemisphere_distinction=False)
                for sliced_brain in self.sliced_brains]
            return self.brains
        
        def to_group(self, metric: str, brain_ontology: AllenBrainHierarchy, *args, **kwargs):
            return AnimalGroup(self.name, self.brains, metric=metric, brain_ontology=brain_ontology, merge_hemispheres=True, *args, **kwargs)
        
        def _remove_small_regions(self, animal: SlicedBrain, threshold: float) -> None:
            for s in animal.slices:
                # s._data = s.data
                # TODO: currently there is no differentiation between real markers and overlapping markers.
                # This bad workaround excludes all those markers having a '+' in the name.
                s.data = s.data[s.data.area > threshold].copy(deep=True)
        
        def _remove_singles(self, animal: SlicedBrain) -> None:
            for s in animal.slices:
                # TODO: currently there is no differentiation between real markers and overlapping markers.
                # This bad workaround excludes all those markers having a '+' in the name.
                real_markers = [m for m in animal.markers if "+" not in m]
                s.data = s.data[(s.data[real_markers] != 1).any(axis=1)].copy(deep=True)
        
        # COMMENTED OUT because mkdocs complains and we don't need it anymore
        # @staticmethod
        # def _fix_overlap_detection_if_old_qpscript(sliced_brain: SlicedBrain):
        #     # if sliced_brain has was computed on data collected from an old QuPath script,
        #     # then the number of detection of the first marker is ~wrong. It must be summed to the overlaps between marker1 and marker2
        #     for i in range(len(sliced_brain.markers)): # e.g. header="GABA-(cFos+GABA)"
        #         marker1_diff = sliced_brain.markers[i]
        #         # see https://regex101.com/r/LLwGIl/1
        #         markers = re.compile("(?P<m1>\w+)-\((?:(\w+)\+(?P=m1)|(?P=m1)\+(\w+))\)").findall(marker1_diff)
        #         # e.g markers=[('GABA', 'cFos', '')]
        #         if len(markers) == 0:
        #             continue
        #         markers = [m for m in markers[0] if len(m) != 0]
        #         marker1, marker2 = markers  # e.g. marker1="GABA" and marker2="cFos"
        #         for brain_slice in sliced_brain.slices:
        #             brain_slice.data[marker1_diff] += brain_slice.data[f"{marker1_diff}+{marker2}"]
        #             brain_slice.data.rename(columns={marker1_diff: marker1, f"{marker1_diff}+{marker2}": f"{marker1}+{marker2}"}, inplace=True)
        #         sliced_brain.markers[i] = marker1
        #         overlap_i = next(i for i in range(len(sliced_brain.markers)) if sliced_brain.markers[i] == f"{marker1_diff}+{marker2}")
        #         sliced_brain.markers[overlap_i] = f"{marker1}+{marker2}"
    
    class Comparison:
        def __init__(self, id, group_reduction: str, metric: str,
                     min_area: float, regions_to_plot: list[str],
                     type: str, selected_groups, selected_markers, # list[GroupDirectory]
                     brain_ontology: AllenBrainHierarchy, dir_name: str, **kwargs) -> None:
            self.id = id
            self.metric = metric
            self.min_area = min_area # TODO
            self.regions_to_plot = regions_to_plot
            self.type = type
            self.groups = selected_groups
            self.markers = selected_markers
            self.brain_ontology = brain_ontology
            self.dir = dir_name
            self.group_reduction = group_reduction.lower()

            match self.group_reduction:
                case "mean" | "avg":
                    self.group_redux = lambda animal_group, marker: animal_group.mean[marker]
                # case "corr" | "correlation":
                #     self.group_redux = lambda animal_group, marker: animal_group.markers_corr(*markers)
                case "pls":
                    assert len(self.groups) == 2, "PLS from config file supports only two groups!"
                    n_permutations = kwargs["n_permutations"]
                    n_bootstrap = kwargs["n_bootstrap"]
                    self.group_redux = lambda groups, marker: \
                        groups[0].pls_regions(groups[1],
                                              self.regions_to_plot,
                                              marker,
                                              n_bootstrap,
                                              fill_nan=True)
                     # remove the kwargs that will be used in apply()
                    del kwargs["n_permutations"]
                    del kwargs["n_bootstrap"]
                case _:
                    raise ValueError(f"[comparison.{self.id}] - Unknown '{group_reduction}' group reduction")
            
            self.kwargs = kwargs
        
        def apply(self):
            if "result" not in self.__dict__:
                self.result = [AnimalGroup(group.name, group.brains, self.metric, brain_ontology=self.brain_ontology,
                                merge_hemispheres=True, **self.kwargs) for group in self.groups]
                if self.markers is None:
                    self.markers = {m for group in self.result for m in group.markers}
                markers = {m for m in self.markers if all(m in g.markers for g in self.result)}
                if len(missing_markers:=set(self.markers)^markers) != 0:
                    print(f"[comparison.{self.id}] - The following markers are missing from some of the selected groups: {missing_markers}")
                    self.markers = markers
            return self.result

        def is_commutative(self):
            # NOTE: also SIMILARITY_INDEX, DENSITY_DIFFERENCE and "correlation" are on markers
            return self.type == "groups" and self.group_reduction == "pls"
        
        def to_braindata(self) -> dict[str, tuple[BrainData, ...]]:
            # group1.is_comparable(group2) should always be true
            if "braindata" in self.__dict__:
                return self.braindata
            if self.type == "groups":
                if not self.is_commutative():
                    self.braindata = {marker: tuple(self.group_redux(g, marker) for g in self.result) for marker in self.markers}
                else:
                    self.braindata = {marker:      (self.group_redux(self.result, marker),)           for marker in self.markers}
            elif self.type == "markers":
                if not self.is_commutative():
                    self.braindata = {group.name: tuple(self.group_redux(group, m) for m in self.markers) for group in self.result}
                else:
                    self.braindata = {group.name:      (self.group_redux(group, self.markers),)           for group in self.result}
                for markers_data in self.braindata.values():
                    for data, marker in zip(markers_data, self.markers):
                        data.data_name = marker
            else:
                raise ValueError(f"Unknown '{self.type}' type")
            return self.braindata

        def plot_heatmaps(self, plots_output_dir: str,
                          brain_data: dict[str, tuple[BrainData, BrainData]]=None,
                          **kwargs):
            if brain_data is None:
                brain_data = self.to_braindata()
            compatible_data = ((common_str, data if len(data) == 2 else (data[0], None))
                               for common_str, data in brain_data.items()
                               if len(data) <= 2 and len(data) >= 1)
            output_dir = os.path.join(plots_output_dir, self.dir)
            os.makedirs(output_dir, exist_ok=True)
            print("comparison", self.id, "---", str(self.metric))
            for common_str, (right_data, left_data) in compatible_data:
                metric = self._get_metric(right_data, left_data)
                centered_cmap, cmin, cmax = self._cmap_range(metric)
                if left_data is not None and right_data.data_name != left_data.data_name:
                    comparison_str = f"{right_data.data_name}+{left_data.data_name}"
                else:
                    comparison_str = right_data.data_name
                filename = self.make_filename(metric, common_str, comparison_str)
                print(f"\t{filename}: ", end="")
                right_data.plot(self.regions_to_plot,
                            output_dir, filename, other=left_data,
                            cmin=cmin, cmax=cmax, centered_cmap=centered_cmap,
                            **kwargs)
        
        def make_filename(self, *ss: str):
            return "_".join((s.replace(' ', '_') for s in ss if s != ""))

        def _get_metric(self, data: BrainData, *other_data: BrainData) -> str:
            metrics = list({d.metric.lower().split(" ")[0] for d in itertools.chain((data,), other_data) if d is not None})
            assert len(metrics) == 1, "You can't plot multiple BrainData of different metrics!"
            return metrics[0]
        
        def _cmap_range(self, metric: str) -> tuple[bool, int, int]:
            if metric.endswith("-corr") or metric.startswith(str(BrainMetrics.DENSITY_DIFFERENCE)) or metric.startswith("pls_"):
                centered_cmap = True
            else:
                centered_cmap = False
            if metric.endswith("-corr"):
                cmin, cmax = -1, 1
            elif metric.startswith(str(BrainMetrics.SIMILARITY_INDEX)) or \
                metric.startswith(str(BrainMetrics.OVERLAPPING)):
                cmin, cmax = 0, 1
            else:
                cmin, cmax = None, None
            return centered_cmap, cmin, cmax

    def __init__(self,
                 data_path: str,
                 config_file: str,
                 ) -> None:
        with open(config_file, "r") as f:
            self.config = toml.load(f, _dict=OrderedDict)
        
        self.data_path = data_path
        path_to_allen_json = os.path.join(self.data_path, "AllenMouseBrainOntology.json")
        cache(path_to_allen_json, "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
        self.brain_ontology = AllenBrainHierarchy(path_to_allen_json,
                                                   self.config["atlas"]["excluded-branches"],
                                                   version=self.config["atlas"]["version"])
        self.groups = [BraiAnConfig.GroupDirectory(
                            id=int(group[len("group"):]),
                            name=self.config["experiment"][group]["name"],
                            animal_directories=self.config["experiment"][group]["dirs"]
                        ) for group in self.config["experiment"]
                          if group.startswith("group") and group[len("group"):].isdigit()
                      ]
        self.comparisons = self._read_comparisons()
    
    def read_groups(self, path_to_groups) -> list[list[SlicedBrain]]:
        overlapping_tracers = [[v-1 for k,v in comp["parameters"].items() if k.startswith("marker")]
                                    for comp in self.config["comparison"].values()
                                    if isinstance(comp, dict) and comp["metric"] == "Overlapping"] # and comp["type"] == "markers"
        # overlapping_tracers = [self._imarker(m) if isinstance(m,int) else m for m in overlapping_tracers]
        assert len(overlapping_tracers) <= 1, "There are multiple 'overlapping' comparisons in the config file. "+\
                                              "Currently it BraiAnConfig supports only one overlapping comparison between two markers."
        if isinstance(path_to_groups, str):
            path_to_groups = [path_to_groups]*len(self.groups)
        elif len(path_to_groups) != len(self.groups):
            raise ValueError("'path_to_groups' must be a string or a list of strings of the same length as the number of groups")
        for group_dir, path_to_group in zip(self.groups, path_to_groups):
            group_dir.read(
                path_to_group,
                self.config["brains"]["slices-min-area"],
                self.config["brains"]["slices-remove-singles"],
                ch2marker=self.config["brains"]["markers"],
                brain_ontology=self.brain_ontology,
                overlapping_markers=overlapping_tracers[0],
                area_units="µm2",
                exclude_parent_regions=True
            )
            print(f"Imported all brain slices from {len(group_dir.animal_names)} animals of '{group_dir.name}' group.")
        return [g.sliced_brains for g in self.groups]
    
    def reduce_slices(self):
        for group in self.groups:
            group.reduce_slices(self.config["brains"]["slices-aggregation-mode"])
    
    def to_groups(self, metric: str, *args, **kwargs):
        return [group_dir.to_group(metric, self.brain_ontology, *args, **kwargs) for group_dir in self.groups]

    def _read_comparisons(self) -> list[Comparison]:
        # if "comparison" in config:
        _group_reduction = self.config["comparison"]["group-reduction"]
        _min_area        = self.config["comparison"]["min-area"]
        _regions_to_plot = self.config["comparison"]["regions-to-plot"]
        _regions_to_plot = self.brain_ontology.get_regions(_regions_to_plot)
        result = []
        for id, comp in self.config["comparison"].items():
            if not isinstance(comp, dict):
                continue
            group_reduction = comp["group-reduction"] if "group-reduction" in comp else _group_reduction
            min_area        = comp["min-area"]        if "min-area"        in comp else _min_area
            metric          = comp["metric"]
            kwargs          = comp["parameters"]      if "parameters"      in comp else dict()
            kwargs = {k: self._imarker(i) if k.startswith("marker") and isinstance(i, int) else i for k,i in kwargs.items()}
            if "regions-to-plot" in comp:
                regions_to_plot = self.brain_ontology.get_regions(comp["regions-to-plot"])
            else:
                regions_to_plot = _regions_to_plot
            if "groups" in comp:
                selected_groups = [group for group in self.groups if group.id in comp["groups"]]
            else:
                selected_groups = self.groups
            if "markers" in comp:
                selected_markers = [m if isinstance(m, str) else self._imarker(m) for m in comp["markers"]]
            else:
                selected_markers = None
            if comp["type"] in ("groups", "markers"):
                result.append(
                    BraiAnConfig.Comparison(id, group_reduction, metric, min_area, regions_to_plot, comp["type"],
                                            selected_groups, selected_markers, self.brain_ontology, comp["dir"], **kwargs))
            else:
                print(f"WARNING: comparison '{id}' has an unknown '{comp['type']}' type. Valid types are 'groups' and 'markers'")
        return result
    
    def _imarker(self, i):
        return list(self.config["brains"]["markers"].values())[i-1]
    
    # def remove_high_variation_regions(self, threshold: float):
    #     for group in self.groups:
    #         for animal_brain, slices in zip(groups_sum_brains[i], group_slices):
    #             cvars = AnimalBrain.from_slices(slices, mode="cvar", hemisphere_distinction=animal_brain.is_split, min_slices=0)
    #             # TODO: currently there is no differentiation between real markers and overlapping markers.
    #             # This bad workaround excludes all those markers having a '+' in the name.
    #             real_markers = [m for m in cvars.markers if "+" not in m]
    #             cvars_data = cvars.to_pandas()
    #             disperse_regions = cvars_data.index[(cvars_data > CVAR_THRESHOLD)[real_markers].any(axis=1)]
    #             print(f"removing {len(disperse_regions)}/{len(cvars_data)} dispersive regions from '{slices.name}'")
    #             animal_brain.remove_region(*disperse_regions)
    
    def check_animal_region(self, animal_name: str, region_acronym: str, marker=None):
        try:
            sliced_brain: SlicedBrain = next(animal for group in self.groups for animal in group.sliced_brains if animal.name == animal_name)
        except StopIteration:
            print(f"Can't find region '{region_acronym}' for animal '{animal_name}'")
            return
        sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)
        all_slices_df = sliced_brain.concat_slices()
        slices_per_area = all_slices_df.groupby(all_slices_df.index).count().iloc[:,0]
        if region_acronym not in slices_per_area.index:
            print(f"Can't find region '{region_acronym}' for animal '{animal_name}'")
            return
        markers = sliced_brain.markers if marker is None else [marker]
        brain_avg = AnimalBrain.from_slices(sliced_brain, mode="avg", hemisphere_distinction=False)
        brain_std = AnimalBrain.from_slices(sliced_brain, mode="std", hemisphere_distinction=False)
        for m in markers:
            marker_avg = brain_avg[m]
            marker_std = brain_std[m]
            print(f"""Summary for brain region '{region_acronym}' of marker '{m}':
                - N slices: {slices_per_area[region_acronym]}
                - Mean: {marker_avg[region_acronym]:.2f} {m}/mm²),
                - S.D.: {marker_std[region_acronym]:.2f} {m}/mm²,
                - Coefficient of Variation: {marker_avg[region_acronym]}
            """)
    
    def check_animal_region_slices(self, animal_name: str, region_acronym: str):
        slices = []
        try:
            sliced_brain: SlicedBrain = next(animal for group in self.groups for animal in group.sliced_brains if animal.name == animal_name)
            sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)
            for slice in sliced_brain.slices:
                if region_acronym not in slice.markers_density.index:
                    continue
                region_densities = slice.markers_density.loc[region_acronym].copy()
                region_densities.index += " density"
                region_densities.name = slice.name
                slices.append(region_densities)
        except StopIteration:
            print(f"Can't find region '{region_acronym}' for animal '{animal_name}'")
        return pd.concat(slices, axis=1) if len(slices) != 0 else None
        