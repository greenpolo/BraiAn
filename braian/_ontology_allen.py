import warnings

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
                 resolution: Literal[10,25,50]=None,
                 unreferenced: bool=False,
                 ):
        """
        Crates an ontology of brain regions based on [Allen Brain Reference Atlas of the Adult Mouse](https://atlas.brain-map.org/).

        Parameters
        ----------
        path_to_allen_json
        blacklisted_acronyms
            Acronyms of branches from the onthology to exclude completely from the analysis
        blacklisted
            Acronyms of branches from the onthology to exclude completely from the analysis
        version
            The version of the Common Coordinate Framework, as defined by Allen Institute.
            It defaults to CCFv3.
        resolution
            The resolution, in µm, of the annotation atlas associated to this ontology.
            It is required because, at low resolutions (e.g. 50µm), some tiny brain structures may disappear from the atlas.
        unreferenced
            If True, it considers as part of the ontology all those brain regions that have no references in the atlas annotations.
            Otherwise, it removes them from the ontology.
            On Allen's website, unreferenced brain regions are identified in grey italic text: [](https://atlas.brain-map.org).

        Raises
        ------
        ValueError
            If there is at least one structure that appears twice in `blacklisted`.
        KeyError
            If any of the `blacklisted` structures is not found in the ontology.
        """
        if blacklisted_acronyms:
            blacklisted = blacklisted_acronyms
        if resolution is None:
            msg = "No 'resolution' specified. Since '1.1.0' this behaviour is deprecated, and future versions may require for it to be explicitly specified."
            warnings.warn(msg, FutureWarning)
            resolution = 50
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

    @deprecated(since="1.1.0", message="In BrainGlobe's atlases, structural level information is missing. Use braian.legacy.AllenBrainOntology if you really need it.")
    def select_at_structural_level(self, level: int):
        raise NotImplementedError