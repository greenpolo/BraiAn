import json
import warnings

from braian import AtlasOntology
from braian.legacy import _visit_dict
from braian.utils import deprecated
from braian._ontology import first_subtrees
from collections.abc import Iterable
from pathlib import Path
from treelib import Tree, Node
from treelib.exceptions import NodeIDAbsentError
from typing import Literal

__all__ = ["AllenBrainOntology"]

def _add_st_level(st_node: dict, tree: Tree):
    try:
        region: Node = tree[st_node["id"]]
    except NodeIDAbsentError:
        return
    region.data = dict(st_level=st_node["st_level"])

class AllenBrainOntology(AtlasOntology):
    @deprecated(since="1.1.0",
                params=["path_to_allen_json", "blacklisted_acronyms", "version"],
                alternatives=dict(path_to_allen_json="allen_json", blacklisted_acronyms="blacklisted"))
    def __init__(self,
                 allen_json: Path|str|dict=None,
                 blacklisted: Iterable=[],
                 version: str|None=None,
                 resolution: Literal[10,25,50,100]=None,
                 unreferenced: bool=False,
                 path_to_allen_json: Path|str=None,
                 blacklisted_acronyms: Iterable=[],
                 ):
        """
        Creates an ontology of the brain structures defined in the
        [Allen Brain Reference Atlas of the Adult Mouse](https://atlas.brain-map.org/).\\
        Compared to [`AltasOntology`][braian.AtlasOntology] with `name="allen_mouse_{resolution}um"` it
        has information of the _"[structural level][braian.AllenBrainOntology.select_at_structural_level]"_
        (or `st_level`) for each brain structure, a metadata that [BrainGlobe](https://brainglobe.info/)
        doesn't capture.

        `AllenBrainOntology` retrieves this information from Allen Institute's official _structure graphs_.
        To know more what they are and where to get them, read the
        [official guide](https://community.brain-map.org/t/downloading-an-ontologys-structure-graph/2880)
        from Allen Institute.

        Parameters
        ----------
        allen_json
            The path to an Allen structural graph json.
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
        path_to_allen_json
            The path to an Allen structural graph json.
        blacklisted_acronyms
            Acronyms of branches from the onthology to exclude completely from the analysis.

        Raises
        ------
        ValueError
            If there is at least one structure that appears twice in `blacklisted`.
        KeyError
            If any of the `blacklisted` structures is not found in the ontology.

        Examples
        --------
        >>> import braian
        >>> import braian.plot as bap
        >>> import braian.utils as bau
        >>> from tempfile import TemporaryDirectory
        >>> from pathlib import Path
        >>> tmp = TemporaryDirectory("_braian")
        >>> st_graph = Path(tmp.name)/"allen_mouse_st_graph.json"
        >>> bau.cache(st_graph, "https://api.brain-map.org/api/v2/structure_graph_download/1.json")
        >>> ontology = braian.AllenBrainOntology(path_to_allen_json=st_graph, resolution=10)
        >>> tmp.cleanup()
        >>> ontology.partition("st level 5")
        ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CBX', 'CBN']
        >>> ontology.partitioned(["CA1", "RE", "SSp-un6a"], partition="st level 6")
        OrderedDict({'CA1': 'HIP', 'RE': 'DORpm', 'SSp-un6a': 'SS'})
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
        if path_to_allen_json is not None:
            allen_json = path_to_allen_json
        if isinstance(allen_json, (str, Path)):
            with open(allen_json, "r") as file:
                allen_data = json.load(file)
            if "msg" in allen_data:
                st_tree = allen_data["msg"][0]
            else:
                st_tree = allen_data
        else:
            assert isinstance(allen_json, st_tree)
            st_tree = allen_json
        _visit_dict.visit_dfs(st_tree, "children", lambda n,d: _add_st_level(n,self._tree_full))

    def select_at_structural_level(self, level: int):
        """
        Select all non-overlapping brain structure at the same structural level in the ontology.
        The _"structural level"_ is an attribute defined by the Allen Institute for each structure,
        that suggests a level of granularity in the ontology that may be specific to particular studies.

        If a region [is a leaf][braian.AtlasOntology.select_leaves] but has a _"structural level"_ above
        `level`, it is not selected. This means that, differently from
        [`select_at_depth`][braian.AtlasOntology.select_at_depth], some structural level selections
        _might_ not cover the whole brain.

        Parameters
        ----------
        level
            The structural level in the ontology to select

        See also
        --------
        [`has_selection`][braian.legacy.AllenBrainOntology.has_selection]
        [`get_selected_regions`][braian.legacy.AllenBrainOntology.get_selected_regions]
        [`unselect_all`][braian.legacy.AllenBrainOntology.unselect_all]
        [`add_to_selection`][braian.legacy.AllenBrainOntology.add_to_selection]
        [`select_at_depth`][braian.legacy.AllenBrainOntology.select_at_depth]
        [`select_leaves`][braian.legacy.AllenBrainOntology.select_leaves]
        [`select_summary_structures`][braian.legacy.AllenBrainOntology.select_summary_structures]
        [`select_regions`][braian.legacy.AllenBrainOntology.select_regions]
        [`get_regions`][braian.legacy.AllenBrainOntology.get_regions]
        """
        def select(n: Node) -> bool:
            st_level = n.data["st_level"]
            return st_level == level
        self._select(first_subtrees(self._tree, select), add=False)

    def partition(self,
                  selection_method: Literal["summary structures","major divisions",
                                            "smallest","leaves","depth <n>",
                                            "structural level <n>","st level <n>"],
                  *,
                  blacklisted: bool=False,
                  unreferenced: bool=False,
                  key: Literal["id","acronym"]="acronym") -> list:
        """
        Returns a list of non-overlapping, brain structures representing
        the whole non-blacklisted brain.

        Parameters
        ----------
        selection_method
            In order to retrieve the partition, it uses some of the available section methods.
            Must be "summary structure", "major divisions", "leaves", "depth <d>" or "st level <n>".
        blacklisted
            If True, it includes also structures blacklisted in the ontology.
        unreferenced
            If True, it includes also structures with no reference in the atlas annotations.

        Returns
        -------
        :
            A list of brain structures uniquely identified by `key`.

        Raises
        ------
        MissingWholeBrainPartitionError
            If `selection_method` is `"summary structures"` or `"major divisions"`,
            but no corresponding list of regions is not known for this atlas's ontology.
        ValueError
            If `selection_method` is not recognised
        ValueError
            If `key` is not known as an unique identifier for brain structures.

        See also
        --------
        [`has_selection`][braian.AtlasOntology.has_selection]
        [`selected`][braian.AtlasOntology.selected]
        [`unselect_all`][braian.AtlasOntology.unselect_all]
        [`select`][braian.AtlasOntology.select]
        [`add_to_selection`][braian.AtlasOntology.add_to_selection]
        [`select_at_depth`][braian.AtlasOntology.select_at_depth]
        [`select_at_structural_level`][braian.AllenBrainOntology.select_at_structural_level]
        [`select_leaves`][braian.AtlasOntology.select_leaves]
        [`select_major_divisions`][braian.AtlasOntology.select_major_divisions]
        [`select_summary_structures`][braian.AtlasOntology.select_summary_structures]
        """
        match selection_method:
            case s if s.startswith("structural level") or s.startswith("st level"):
                n = selection_method.split(" ")[-1]
                try:
                    level = int(n)
                except Exception:
                    raise ValueError("Could not retrieve the <n> parameter of the 'structural level' method for 'selection_method'")
                select = lambda: self.select_at_structural_level(level)  # noqa: E731
                return self._partition(selection=select,
                                       blacklisted=blacklisted,
                                       unreferenced=unreferenced,
                                       key=key)
            case _:
                return super().partition(selection_method=selection_method,
                                         blacklisted=blacklisted,
                                         unreferenced=unreferenced,
                                         key=key)

    def to_igraph(self, unreferenced: bool=False, blacklisted: bool=True):
        g = super().to_igraph(unreferenced=unreferenced, blacklisted=blacklisted)
        ids_full = set(self._tree_full.expand_tree())
        for v in g.vs.select(id_in=ids_full):
            region: Node = self._tree_full[v["id"]]
            v["structural_level"] = region.data["st_level"]
        return g