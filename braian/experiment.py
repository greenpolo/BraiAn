from collections.abc import Iterable, Callable
from braian import AllenBrainOntology, AnimalBrain, AnimalGroup, SlicedBrain, SlicedGroup, SliceMetrics
from pathlib import Path
from typing import Any, Self

__all__ = ["Experiment", "SlicedExperiment"]

class Experiment:
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

    def __init__(self, name: str, group1: AnimalGroup, group2: AnimalGroup,
                 *groups: AnimalGroup) -> None:
        """
        Creates an experiment from the data of two or more [`AnimalGroups`][braian.AnimalGroup].

        Parameters
        ----------
        name
            The name of the experiment.
        group1
            The first group of the experiment.
        group2
            The second group of the experiment.
        *groups
            Any other group of the experiment.
        """
        self._name = str(name)
        self._groups = (group1, group2, *groups)

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._name

    @property
    def groups(self) -> tuple[AnimalGroup]:
        """The groups making up the current experiment."""
        return self._groups

    def __getattr__(self, name: str) -> Any:
        """
        Get a specific group in the current experiment by accessing it with an attribute named like the name of the group.

        Parameters
        ----------
        name
            The name of the group

        Returns
        -------
        :
            The group in the experiment having the same name as `name`.

        Raises
        ------
        AttributeError
            If no group with `name` was found in the current experiment.
        """
        for g in self._groups:
            if g.name.lower() == name.lower():
                return g
        raise AttributeError(f"Uknown group named '{name.lower()}'")

    def apply(self, f: Callable[[AnimalBrain], AnimalBrain],
              brain_ontology: AllenBrainOntology=None, fill_nan: bool=False) -> Self:
        """
        Applies a function to each animal of the groups of the experiment and creates a new `Experiment`.
        Especially useful when applying some sort of metric to the brain data.

        Parameters
        ----------
        f
            A function that maps an `AnimalBrain` into another `AnimalBrain`.
        brain_ontology
            The ontology to which the brains' data was registered against.\\
            If specified, it sorts the data in depth-first search order with respect to brain_ontology's hierarchy.
        fill_nan
            If True, it sets the value to [`NA`][pandas.NA] for all the regions missing from the data but present in `brain_ontology`.

        Returns
        -------
        :
            An experiment with the data of each animal changed accordingly to `f`.
        """
        return Experiment(self._name, *[g.apply(f, brain_ontology, fill_nan) for g in self._groups])

class SlicedExperiment:
    def __init__(self, name: str, group1: SlicedGroup, group2: SlicedGroup,
                 *groups: Iterable[SlicedGroup]) -> None:
        """
        Creates an experiment from the data of two or more [`SlicedGroups`][braian.SlicedGroup].

        Parameters
        ----------
        name
            The name of the sliced experiment.
        group1
            The first group of the sliced experiment.
        group2
            The second group of the sliced experiment.
        *groups
            Any other group of the sliced experiment.
        """
        self._name: str = str(name)
        self._groups: tuple[SlicedGroup] = (group1, group2, *groups)

    @property
    def name(self) -> str:
        """The name of the sliced experiment."""
        return self._name

    @property
    def groups(self) -> tuple[SlicedGroup]:
        """The groups making up the current sliced experiment."""
        return self._groups

    def __getattr__(self, name: str) -> Any:
        """
        Get a specific group in the current sliced experiment by accessing it with an attribute named like the name of the group.

        Parameters
        ----------
        name
            The name of the group

        Returns
        -------
        :
            The group in the sliced experiment having the same name as `name`.

        Raises
        ------
        AttributeError
            If no group with `name` was found in the current sliced experiment.
        """
        for g in self._groups:
            if g.name.lower() == name.lower():
                return g
        raise AttributeError(f"Uknown group named '{name.lower()}'")

    def to_experiment(self, metric: SliceMetrics,
                      min_slices: int, densities: bool,
                      hemisphere_distinction: bool, validate: bool) -> Experiment:
        """
        Aggrecates the data from all sections of each [`SlicedBrain`][braian.SlicedBrain]
        into [`SlicedGroup`][braian.SlicedGroup] and organises them into the corresponding
        [`Experiment`][braian.Experiment].

        Parameters
        ----------
        metric
            The metric used to reduce sections data from the same region into a single value.
        min_slices
            The minimum number of sections for a reduction to be valid. If a region has not enough sections, it will disappear from the dataset.
        densities
            If True, it computes the reduction on the section density (i.e., marker/area) instead of doing it on the raw cell counts.
        hemisphere_distinction
            If False, it merges, for each region, the data from left/right hemispheres into a single value.
        validate
            If True, it validates each region in each brain, checking that they are
            present in the brain region ontology against which the brains were alligned.

        Returns
        -------
        :
            An experiment with the values from sections of the same animals aggregated.

        See also
        --------
        [`SlicedGroup.to_group`][braian.SlicedGroup.to_group]
        """
        groups = [g.to_group(metric, min_slices, densities, hemisphere_distinction, validate) for g in self._groups]
        return Experiment(self.name, *groups)

    def __contains__(self, animal_name: str) -> bool:
        """
        Parameters
        ----------
        animal_name
            The name of an animal.

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
            The name of an animal part of the experiment.

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
