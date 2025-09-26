from braian import AtlasOntology
from braian.utils import deprecated
from collections.abc import Iterable, Container
from pathlib import Path
from typing import Literal

__all__ = ["AllenBrainOntology"]

class AllenBrainOntology(AtlasOntology):
    @deprecated(since="1.1.0",
                params=["path_to_allen_json", "blacklisted_acronyms", "version"],
                alternatives=dict(blacklisted_acronyms="blacklisted"))
    def __init__(self,
                 path_to_allen_json: Path|str,
                 blacklisted_acronyms: Iterable=[],
                 blacklisted: Iterable=[],
                 version: str|None=None,
                 resolution: Literal[10,25,50]=50,
                 unreferenced: bool=False,
                 ):
        """
        Crates an ontology of brain regions based on Allen Institute's structure graphs.
        To know more where to get the structure graphs, read the
        [official guide](https://community.brain-map.org/t/downloading-an-ontologys-structure-graph/2880)
        from Allen Institute.
        """
        if blacklisted_acronyms:
            blacklisted = blacklisted_acronyms
        super().__init__(
            atlas=f"allen_mouse_{resolution}um",
            blacklisted=blacklisted,
            unreferenced=unreferenced
        )

    @deprecated(since="1.1.0")
    def contains_all_children(self, parent: str, regions: Container[str]) -> bool:
        """
        Check whether a brain region contains all the given regions

        Parameters
        ----------
        parent
            An acronym of a brain region
        regions
            The regions to check as subregions of `parent`

        Returns
        -------
        :
            True if all `regions` are direct subregions of `parent`
        """
        return set(self.direct_subregions[parent]) == set(regions)

    def select_at_structural_level(self, level: int):
        """Select all non-overlapping brain regions at the same structural level in the ontology."""
        raise NotImplementedError