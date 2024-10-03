from collections.abc import Iterable, Callable
from braian import AllenBrainOntology, AnimalBrain, AnimalGroup, SlicedBrain, SlicedGroup, SliceMetrics
from pathlib import Path
from typing import Any, Self

__all__ = ["Experiment", "SlicedExperiment"]

class Experiment:
    def __init__(self, name: str, group1: AnimalGroup, group2: AnimalGroup,
                 *groups: AnimalGroup) -> None:
        self._name = str(name)
        self._groups = (group1, group2, *groups)

    @property
    def name(self) -> str:
        return self._name

    @property
    def groups(self) -> tuple[AnimalGroup]:
        return self._groups

    def __getattr__(self, name: str) -> Any:
        for g in self._groups:
            if g.name.lower() == name.lower():
                return g
        raise AttributeError(f"Uknown group named '{name.lower()}'")

    def apply(self, f: Callable[[AnimalBrain], AnimalBrain],
              brain_ontology: AllenBrainOntology=None, fill_nan: bool=False) -> Self:
        return Experiment(self._name, *[g.apply(f, brain_ontology, fill_nan) for g in self._groups])

    @staticmethod
    def from_group_csv(name: str, group_names: Iterable[str],
                       metric: str, basedir: Path|str, sep=",") -> Self:
        if not isinstance(basedir, Path):
            basedir = Path(basedir)
        groups = []
        for name in group_names:
            group = AnimalGroup.from_csv(basedir/f"{name}_{metric}.csv", name, sep)
            groups.append(group)
        return Experiment(name, *groups)

    @staticmethod
    def from_brain_csv(name: str, group2brains: dict[str,Iterable[str]],
                       metric: str, basedir: Path|str, sep=",",
                       **kwargs) -> Self:
        if not isinstance(basedir, Path):
            basedir = Path(basedir)
        groups = []
        for name, brain_names in group2brains.items():
            brains = []
            for brain_name in brain_names:
                brain = AnimalBrain.from_csv(basedir/f"{brain_name}_{metric}.csv", brain_name, sep)
                brains.append(brain)
            group = AnimalGroup(name, brains, **kwargs)
            groups.append(group)
        return Experiment(name, *groups)

class SlicedExperiment:
    def __init__(self, name: str, group1: SlicedGroup, group2: SlicedGroup,
                 *groups: Iterable[SlicedGroup]) -> None:
        self._name: str = str(name)
        self._groups: tuple[SlicedGroup] = (group1, group2, *groups)

    @property
    def name(self) -> str:
        return self._name

    @property
    def groups(self) -> tuple[SlicedGroup]:
        return self._groups

    def __getattr__(self, name: str) -> Any:
        for g in self._groups:
            if g.name.lower() == name.lower():
                return g
        raise AttributeError(f"Uknown group named '{name.lower()}'")

    def to_experiment(self, metric: SliceMetrics,
                      min_slices: int, densities: bool,
                      hemisphere_distinction: bool, validate: bool) -> Experiment:
        groups = [g.to_group(metric, min_slices, densities, hemisphere_distinction, validate) for g in self._groups]
        return Experiment(self.name, *groups)

    def __contains__(self, animal_name: str) -> bool:
        """
        Parameters
        ----------
        animal_name
            The name of an animal

        Returns
        -------
        :
            True, if current experiment contains an animal with `animal_name`. False otherwise.
        """
        return any(brain.name == animal_name for group in self.groups for brain in group.animals)

    def __getitem__(self, animal_name: str) -> SlicedBrain:
        """

        Parameters
        ----------
        animal_name
            The name of an animal part of the experiment

        Returns
        -------
        :
            The corresponding `AnimalBrain` in the current experiment.

        Raises
        ------
        TypeError
            If `animal_name` is not a string.
        KeyError
            If no brain with `animal_name` was found in the experiment.
        """
        if not isinstance(animal_name, str):
            raise TypeError("SlicedExperiment animals are identified by strings")
        try:
            return next(brain for group in self.groups for brain in group.animals if brain.name == animal_name)
        except StopIteration:
            raise KeyError(f"'{animal_name}'")