import brainglobe_atlasapi as bga
import numpy as np
import numpy.testing as npt
import pytest
from braian import AtlasOntology, utils
from braian.legacy import AllenBrainOntology
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

@pytest.fixture
def allen_ontology_complete():
    return AtlasOntology("allen_mouse_10um", blacklisted=[], unreferenced=True)

@pytest.fixture
def allen_ontology_complete_blacklisted_hpf(allen_ontology_complete: AtlasOntology):
    allen_ontology_complete.blacklist_regions(["HPF"], has_reference=True)
    return allen_ontology_complete

@pytest.fixture
def allen_ontology_complete_unreferenced_hpf(allen_ontology_complete: AtlasOntology):
    allen_ontology_complete.blacklist_regions(["HPF"], has_reference=False)
    return allen_ontology_complete

@pytest.fixture
def allen_ontology():
    return AtlasOntology("allen_mouse_10um", blacklisted=[], unreferenced=False)

@pytest.fixture
def allen_ontology_blacklisted_all(allen_ontology: AtlasOntology):
    allen_ontology.blacklist_regions(["root"], has_reference=True)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_all_no_reference(allen_ontology: AtlasOntology):
    allen_ontology.blacklist_regions(["root"], has_reference=False)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_depth3(allen_ontology: AtlasOntology):
    # NOTE: differently from AllenBrainOntology, AtlasOntology has no trace of "retina"
    allen_ontology.blacklist_regions(["CTX", "CNU", "IB", "MB", "HB", "CBX", "CBN", "fiber tracts", "VS"], has_reference=True)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_depth3_no_reference(allen_ontology: AtlasOntology):
    # NOTE: differently from AllenBrainOntology, AtlasOntology has no trace of "retina"
    allen_ontology.blacklist_regions(["CTX", "CNU", "IB", "MB", "HB", "CBX", "CBN", "fiber tracts", "VS"], has_reference=False)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_hpf(allen_ontology: AtlasOntology):
    allen_ontology.blacklist_regions(["HPF"], has_reference=True)
    return allen_ontology

@pytest.fixture
def allen_ontology_unreferenced_hpf(allen_ontology: AtlasOntology):
    allen_ontology.blacklist_regions(["HPF"], has_reference=False)
    return allen_ontology

def test_direct_adjacent_regions(allen_ontology_complete):
    o: AtlasOntology = allen_ontology_complete
    with pytest.raises(KeyError):
        _ = o.parent_region["root"]
    assert o.parent_region["CA1"] == "CA"
    with pytest.raises(KeyError):
        _ = o.direct_subregions["CA1"] # leaf region
    assert o.parent_region["HPF"] == "CTXpl"
    assert o.direct_subregions["CTXpl"] == ["Isocortex", "OLF", "HPF"]
    o.blacklist_regions(["HPF"], has_reference=True)
    assert o.parent_region["HPF"] == "CTXpl"
    assert o.parent_region["CA1"] == "CA"
    assert o.direct_subregions["CTXpl"] == ["Isocortex", "OLF", "HPF"]

def test_direct_adjacent_regions_unreferenced(allen_ontology_complete: AtlasOntology):
    o: AtlasOntology = allen_ontology_complete
    assert o.parent_region["CA1"] == "CA"
    assert o.direct_subregions["CA"] == ["CA1", "CA2", "CA3"]
    o.blacklist_regions(["CA2", "CA3"], has_reference=False) # unreference SOME subregions of CA1
    assert o.direct_subregions["CA"] == ["CA1"]
    o.blacklist_regions(["CA1", "CA2", "CA3"], has_reference=False) # unreference all subregions of CA1
    with pytest.raises(KeyError):
        o.direct_subregions["CA"]
    with pytest.raises(KeyError):
        assert o.parent_region["CA1"]
    assert o.parent_region["HPF"] == "CTXpl"
    assert o.direct_subregions["CTXpl"] == ["Isocortex", "OLF", "HPF"]
    o.blacklist_regions(["HPF"], has_reference=False)
    with pytest.raises(KeyError):
        _ = o.parent_region["HPF"]
    with pytest.raises(KeyError):
        _ = o.parent_region["CA1"]
    with pytest.raises(KeyError):
        _ = o.direct_subregions["HPF"]
    with pytest.raises(KeyError):
        _ = o.direct_subregions["CA1"]
    assert o.direct_subregions["CTXpl"] == ["Isocortex", "OLF"]

@pytest.mark.parametrize(
    "ontology, acronym, expected",
    [
        ("allen_ontology", "root", True),
        ("allen_ontology_blacklisted_all", "root", True),
        ("allen_ontology_blacklisted_all_no_reference", "root", False),
        ("allen_ontology_blacklisted_depth3", "root", True),
        ("allen_ontology", "HPF", True),
        ("allen_ontology_blacklisted_hpf", "HPF", True),
        ("allen_ontology_unreferenced_hpf", "HPF", False),
        ("allen_ontology_blacklisted_depth3", "HPF", True),
        ("allen_ontology", "NOT_A_REGION", False)
    ]
)
def test_is_region(ontology, acronym, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert o.is_region(acronym, unreferenced=False) == expected
    if ontology.endswith("_no_reference"):
        assert o.is_region(acronym, unreferenced=True)

@pytest.mark.parametrize(
    "ontology, acronyms, unreferenced, expected",
    [
        (
            "allen_ontology",
            ["Isocortex", "FRP", "MOp"],
            False,
            np.array([True, True, True])
        ),
        (
            "allen_ontology",
            ["NOT_A_REGION", "FAKE"],
            False,
            np.array([False, False])
        ),
        (
            "allen_ontology",
            ["root", "HPF", "NOT_A_REGION"],
            False,
            np.array([True, True, False])
        ),
        (
            "allen_ontology_blacklisted_hpf",
            ["root", "HPF", "NOT_A_REGION"],
            False,
            np.array([True, True, False])
        ),
        (
            "allen_ontology_unreferenced_hpf",
            ["root", "HPF", "NOT_A_REGION"],
            False,
            np.array([True, False, False])
        ),
        (
            "allen_ontology_unreferenced_hpf",
            ["root", "HPF", "NOT_A_REGION"],
            True,
            np.array([True, True, False])
        ),
    ]
)
def test_are_regions(ontology, acronyms, unreferenced, expected, request):
    ontology: AtlasOntology = request.getfixturevalue(ontology)
    result = ontology.are_regions(acronyms, unreferenced=unreferenced)
    npt.assert_array_equal(result, expected)

def test_are_regions_should_raise(allen_ontology: AtlasOntology):
    with pytest.raises(ValueError, match=".*Duplicates.*"):
        allen_ontology.are_regions(["Isocortex", "FRP", "Isocortex", "MOp"], duplicated=False)

@pytest.mark.parametrize(
    "ontology, parent, regions, expected",
    [
        ("allen_ontology",
         "CTXpl", ["Isocortex", "OLF", "HPF"], True),
        ("allen_ontology_blacklisted_hpf",
         "CTXpl", ["Isocortex", "OLF", "HPF"], True),
        ("allen_ontology_unreferenced_hpf",
         "CTXpl", ["Isocortex", "OLF", "HPF"], False),
        ("allen_ontology",
         "CTXpl", ["CTXpl"], False),
        ("allen_ontology",
         "CTXpl", ["CA1", "CA2", "CA3"], False),
        ("allen_ontology",
         "CTXpl", ["Isocortex", "OLF", "PAL", "TH", "HY"], False),
        ("allen_ontology",
         "CA", ["CA1", "NOT_A_REGION"], False),
    ]
)
def test_contains_all_children(ontology, parent, regions, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert o.contains_all_children(parent, regions) == expected

@pytest.mark.parametrize(
    "ontology, acronyms, include_blacklisted, include_unreferenced, expected",
    [
        ("allen_ontology_complete",
         [], True, True, set()),
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "HPF", "TH", "HY"], True, True, {"CTXpl", "IB"}),
        ("allen_ontology_unreferenced_hpf",
         ["RE", "Xi", "PVT", "PT", "TH"], True, True, {"MTN", "TH"}),
        ("allen_ontology_unreferenced_hpf",
         ["P", "MB", "TH", "MY", "CB", "HY"], True, True, {"BS", "CB"}),
        ("allen_ontology_blacklisted_hpf",
         ["Isocortex", "OLF", "TH", "HY"], True, False, {"Isocortex", "OLF", "IB"}), # 'HPF' is blacklisted, so 'CTXpl' is an ancestor region that covers the input acronyms
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "TH", "HY"], True, False, {"CTXpl", "IB"}), # 'HPF' is blacklisted, so 'CTXpl' is an ancestor region that covers the input acronyms
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "TH", "HY"], True, True, {"Isocortex", "OLF", "IB"}), # 'HPF' is missing for the input acronyms to cover 'CTXpl' completely
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "TH", "HY"], False, True, {"Isocortex", "OLF", "IB"}), # 'HPF' is missing for the input acronyms to cover 'CTXpl' completely
        ("allen_ontology_unreferenced_hpf",
         ["CA1", "CA2", "CA3", "NOT_A_REGION"], True, True, {"CA"}),
    ]
)
def test_minimum_treecover(ontology, acronyms, include_blacklisted, include_unreferenced, expected, request):
    ontology: AtlasOntology = request.getfixturevalue(ontology)
    result = ontology.minimum_treecover(acronyms, blacklisted=include_blacklisted, unreferenced=include_unreferenced)
    assert set(result) == expected

@pytest.mark.parametrize(
    "ontology, unreferenced, expected",
    [
        ("allen_ontology_complete", True, []),
        ("allen_ontology_complete", False, []),
        ("allen_ontology_complete_blacklisted_hpf", True, ["HPF"]),
        ("allen_ontology_complete_blacklisted_hpf", False, ["HPF"]),
        ("allen_ontology_complete_unreferenced_hpf", True, ["HPF"]),
        ("allen_ontology_complete_unreferenced_hpf", False, []),
    ]
)
def test_get_blacklisted_trees(ontology, unreferenced, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    if ontology.endswith("_hpf"):
        o.blacklist_regions(("CA",)) # blacklisting something below HPF should not affect the result
        o.blacklist_regions(("ENT",), has_reference=True)
    result = o.get_blacklisted_trees(unreferenced=unreferenced)
    assert result == expected

@pytest.mark.parametrize(
    "ontology, blacklisted_branches, should_raise",
    [
        ("allen_ontology_complete", ["Isocortex", "HPF"], False),
        ("allen_ontology_complete", ["NOT_A_REGION"], True),
        ("allen_ontology_complete_blacklisted_hpf", ["Isocortex", "HPF"], False),
        ("allen_ontology_complete_blacklisted_hpf", ["NOT_A_REGION"], True),
        ("allen_ontology_complete_unreferenced_hpf", ["Isocortex", "HPF"], False),
    ]
)
def test_blacklist_regions(ontology, blacklisted_branches, should_raise, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    if should_raise:
        with pytest.raises(KeyError, match=".*not found.*"):
            o.blacklist_regions(blacklisted_branches, has_reference=True)
    else:
        o.blacklist_regions(blacklisted_branches, has_reference=True)
        assert o.get_blacklisted_trees(unreferenced=True) == blacklisted_branches
        if ontology.endswith("_unreferenced_hpf"):
            assert o.get_blacklisted_trees(unreferenced=False) == ["Isocortex"]

def test_blacklist_regions_duplicate(allen_ontology_complete: AtlasOntology):
    with pytest.raises(ValueError, match=".*Duplicates.*"):
        allen_ontology_complete.blacklist_regions(["HPF", "HPF"], has_reference=True)

@pytest.mark.parametrize(
    "ontology, regions",
    [
        ("allen_ontology_complete", ["Isocortex", "HPF"]),
        ("allen_ontology_complete_blacklisted_hpf", ["Isocortex", "HPF"]),
        ("allen_ontology_complete_unreferenced_hpf", ["Isocortex", "HPF"]),
    ]
)
def test_blacklist_regions_unreferenced(ontology, regions, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    o.blacklist_regions(regions, has_reference=False)
    assert o.get_blacklisted_trees(unreferenced=True) == regions
    assert len(o.get_blacklisted_trees(unreferenced=False)) == 0

@pytest.mark.parametrize(
    "ontology, depth, expected",
    [
        ("allen_ontology_complete", 0, ["root"]),
        ("allen_ontology_complete", 1, ["grey", "fiber tracts", "VS"]),
        ("allen_ontology_blacklisted_depth3", 3, []),
        ("allen_ontology_blacklisted_depth3_no_reference", 3, ["CH", "BS", "CB"]), # depth 2 regions, excluding "fiber tracts", "VS"
    ]
)
def test_select_at_depth(ontology, depth, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert not o.has_selection()
    o.select_at_depth(depth)
    assert o.get_selected_regions() == expected
    o.unselect_all()
    assert not o.has_selection()
    assert o.get_selected_regions() == []

def test_select_at_depth3_blacklisted(allen_ontology_blacklisted_depth3: AtlasOntology, request):
    o: AtlasOntology = allen_ontology_blacklisted_depth3
    o.select_at_depth(3)
    assert o.get_selected_regions() == []

def test_select_at_depth5_hpf_blacklisted(allen_ontology_complete: AtlasOntology):
    allen_ontology_complete.select_at_depth(5)
    selected = allen_ontology_complete.get_selected_regions()
    allen_ontology_complete.blacklist_regions(["HPF"], has_reference=True)
    selected_hpf_blacklisted = allen_ontology_complete.get_selected_regions()
    assert set(selected) == set(selected_hpf_blacklisted) | {"HPF"}
    allen_ontology_complete.blacklist_regions(["HPF"], has_reference=False)
    selected_hpf_unreferenced = allen_ontology_complete.get_selected_regions()
    assert selected_hpf_blacklisted == selected_hpf_unreferenced

def test_select_at_depth3_unreferenced(allen_ontology_blacklisted_depth3_no_reference: AtlasOntology, request):
    o: AtlasOntology = allen_ontology_blacklisted_depth3_no_reference
    o.select_at_depth(3)
    selected_d3 = o.get_selected_regions()
    o.unselect_all()
    assert o.get_selected_regions() == []
    o.select_at_depth(2)
    selected_d2 = o.get_selected_regions()
    assert selected_d3 == selected_d2
    o.unselect_all()
    assert o.get_selected_regions() == []
    o.select_leaves()
    selected_l = o.get_selected_regions()
    assert selected_d3 == selected_l

def test_select_at_depth_beyond_leaves(allen_ontology_complete: AtlasOntology):
    allen_ontology_complete.select_at_depth(9999)
    selected_d = allen_ontology_complete.get_selected_regions()
    allen_ontology_complete.unselect_all()
    assert allen_ontology_complete.get_selected_regions() == []
    allen_ontology_complete.select_leaves()
    selected_l = allen_ontology_complete.get_selected_regions()
    assert selected_d == selected_l

@pytest.mark.parametrize(
    "selection, args",
    [
        ("select_at_depth", [0]),
        # ("select_at_structural_level", [0]),
        ("select_regions", [["root"]]),
    ]
)
def test_select_with_blacklist_removed(selection, args, allen_ontology: AtlasOntology):
    allen_ontology.__getattribute__(selection)(*args)
    selected = allen_ontology.get_selected_regions()
    assert selected == ["root"]

def test_select_leaves(allen_ontology_complete: AtlasOntology):
    atlas = bga.BrainGlobeAtlas("allen_mouse_10um")
    expected = {n.identifier for n in atlas.structures.tree.leaves()}
    allen_ontology_complete.select_leaves()
    selected = allen_ontology_complete.get_selected_regions(key="id")
    # All selected regions should be leaves (no children)
    assert set(selected) == expected
    allen_ontology_complete.unselect_all()

@pytest.mark.parametrize(
    "blacklisted, referenced, included_leaves, excludeed_leaves",
    [
        (["HIP", "RHP"], None, ["CA1", "CA2", "CA3"], []),
        (["HIP", "RHP"], True, [], ["CA1", "CA2", "CA3"]),
        (["HIP", "RHP"], False, ["HPF"], []),
    ]
)
def test_select_leaves_blacklisted(blacklisted, referenced, included_leaves, excludeed_leaves, allen_ontology_complete: AtlasOntology):
    assert not allen_ontology_complete.has_selection()
    if referenced is not None:
        allen_ontology_complete.blacklist_regions(blacklisted, has_reference=referenced)
    allen_ontology_complete.select_leaves()
    result = allen_ontology_complete.get_selected_regions()
    for included_leaf in included_leaves:
        assert included_leaf in result
    for excluded_leaf in excludeed_leaves:
        assert excluded_leaf not in result

def test_first_tree():
    from braian._ontology import first_subtrees
    from treelib import Tree
    tree = Tree()
    tree.create_node("Root", "root")
    tree.create_node("Child A", "a", parent="root")
    tree.create_node("Child B", "b", parent="root")
    tree.create_node("Grandchild", "a1", parent="a")
    assert [n.identifier for n in first_subtrees(tree, func=lambda n: tree.depth(n) >= 0)] == ["root"]
    assert [n.identifier for n in first_subtrees(tree, func=lambda n: tree.depth(n) >= 1)] == ["a", "b"]
    assert [n.identifier for n in first_subtrees(tree, func=lambda n: tree.depth(n) >= 2)] == ["a1"]
    assert [n.identifier for n in first_subtrees(tree, func=lambda n: tree.depth(n) >= 3)] == []

def test_select_regions_and_add_to_selection(allen_ontology: AtlasOntology):
    with pytest.raises(KeyError, match=".*not found.*"):
        allen_ontology.select_regions(["HPF", "HIP", "CA", "NOT_A_REGION"])
    assert not allen_ontology.has_selection()
    allen_ontology.select_regions(["HPF", "HIP", "CA"])
    assert allen_ontology.has_selection()
    assert allen_ontology.get_selected_regions() == ["HPF"]
    allen_ontology.select_regions(["TH", "HY"])
    assert allen_ontology.get_selected_regions() == ["TH", "HY"]
    with pytest.raises(KeyError, match=".*not found.*"):
        allen_ontology.select_regions(["CTXpl", "BS", "NOT_A_REGION"])
    allen_ontology.add_to_selection(["CTXpl", "BS"])
    assert allen_ontology.has_selection()
    assert allen_ontology.get_selected_regions() == ["CTXpl", "BS"]
    allen_ontology.unselect_all()
    assert not allen_ontology.has_selection()
    assert allen_ontology.get_selected_regions() == []

def test_ids_acronym_conversions(allen_ontology_blacklisted_all_no_reference: AtlasOntology):
    o = allen_ontology_blacklisted_all_no_reference
    ids = [997, 8, 343]
    acronyms = ["root", "grey", "BS"]
    assert o.acronyms_to_id(acronyms) == ids
    assert o.ids_to_acronym(ids) == acronyms

def test_ids_to_acronym_modes(allen_ontology_blacklisted_all_no_reference: AtlasOntology):
    o = allen_ontology_blacklisted_all_no_reference
    ids = [567, 343, 512, 997, 8, 1009, 73] # order: CH, BS, CB, root, grey, fiber tracts, VS
    assert o.ids_to_acronym(ids, mode="depth") == ["root", "grey", "CH", "BS", "CB", "fiber tracts", "VS"]
    assert o.ids_to_acronym(ids, mode="breadth") == ["root", "grey", "fiber tracts", "VS", "CH", "BS", "CB"]
    assert o.ids_to_acronym(ids, mode=None) == ["CH", "BS", "CB", "root", "grey", "fiber tracts", "VS"]
    with pytest.raises(ValueError, match=".*Unsupported.*mode.*"):
        o.ids_to_acronym(ids, mode="INVALID")
    with pytest.raises(ValueError, match=".*Duplicates.*"):
        o.ids_to_acronym([997, 997]) # duplicate IDs

def test_acronym_to_ids_modes(allen_ontology_blacklisted_all_no_reference: AtlasOntology):
    o = allen_ontology_blacklisted_all_no_reference
    acronyms = ["CH", "BS", "CB", "root", "grey", "fiber tracts", "VS"] # order: 567, 343, 512, 997, 8, 1009, 73
    assert o.acronyms_to_id(acronyms, mode="depth") == [997, 8, 567, 343, 512, 1009, 73]
    assert o.acronyms_to_id(acronyms, mode="breadth") == [997, 8, 1009, 73, 567, 343, 512]
    assert o.acronyms_to_id(acronyms, mode=None) == [567, 343, 512, 997, 8, 1009, 73]
    with pytest.raises(ValueError, match=".*Unsupported.*mode.*"):
        o.acronyms_to_id(acronyms, mode="INVALID")
    with pytest.raises(ValueError, match=".*Duplicates.*"):
        o.acronyms_to_id(["root", "root"]) # duplicate acronymss

@pytest.mark.parametrize("ontology,region,expected", [
    ("allen_ontology_complete", "root", ["root"]),
    ("allen_ontology_complete", "OLF", ["Isocortex", "OLF", "HPF"]),
    ("allen_ontology_complete_blacklisted_hpf", "OLF", ["Isocortex", "OLF", "HPF"]),
    ("allen_ontology_complete_blacklisted_hpf", "CA1", ["CA1", "CA2", "CA3"]),
    ("allen_ontology_complete_unreferenced_hpf", "OLF", ["Isocortex", "OLF"]),
])
def test_get_sibiling_regions(ontology, region, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert expected == o.get_sibiling_regions(region, key="acronym")
    expected.remove(region)
    assert expected == o.siblings(region, key="acronym")

@pytest.mark.parametrize("ontology,region", [
    ("allen_ontology_complete", "NOT_A_REGION"),
    ("allen_ontology_complete_unreferenced_hpf", "HPF"),
    ("allen_ontology_complete_unreferenced_hpf", "CA"),
])
def test_get_sibiling_regions_raises(ontology, region, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    with pytest.raises(KeyError, match=".*not found.*"):
        o.get_sibiling_regions(region)

@pytest.mark.parametrize("ontology,regions,expected", [
    ("allen_ontology_complete", ["root", "CTX", "BS", "HPF", "CA1"], {"root": None, "CTX": "CH", "BS": "grey", "HPF": "CTXpl", "CA1": "CA"}),
    ("allen_ontology_complete_blacklisted_hpf", ["root", "CTX", "BS", "HPF", "CA1"], {"root": None, "CTX": "CH", "BS": "grey", "HPF": "CTXpl", "CA1": "CA"}),
    ("allen_ontology_complete_unreferenced_hpf", ["root", "CTX", "BS", "HPF", "CA1"], {"root": None, "CTX": "CH", "BS": "grey", "HPF": "CTXpl", "CA1": "CA"}),
])
def test_get_parent_regions(ontology, regions, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    parents = o.get_parent_regions(regions, key="acronym")
    assert expected == parents

def test_get_parent_regions_raises(allen_ontology_complete: AtlasOntology):
    regions = ["root", "CTX", "NOT_A_REGION"]
    with pytest.raises(KeyError, match=".*not found.*"):
        allen_ontology_complete.get_parent_regions(regions)

@pytest.mark.parametrize("acronym,mode,blacklisted,unreferenced,expected", [
    ("CA", "breadth", False, False, []),
    ("CA", "breadth", False, True, []),
    ("CA", "breadth", True, False, ["CA"]),
    ("CA", "breadth", True, True, ["CA", "CA1", "CA2", "CA3"]),
    ("ATN", "breadth", True, True, ["ATN", "AV", "AM", "AD", "IAM", "IAD", "LD", "AMd", "AMv"]),
    ("ATN", "depth", True, True, ["ATN", "AV", "AM", "AMd", "AMv", "AD", "IAM", "IAD", "LD"]),
])
def test_list_all_subregions(allen_ontology_complete,
                             acronym, mode, blacklisted, unreferenced, expected):
    o: AtlasOntology = allen_ontology_complete
    o.blacklist(["HPF"], unreferenced=False)
    o.blacklist(["CA1", "CA2", "CA3"], unreferenced=True)
    assert o.list_all_subregions(
        acronym, mode=mode,
        blacklisted=blacklisted,
        unreferenced=unreferenced
    ) == expected

@pytest.mark.parametrize("acronym", [
    "NOT_A_REGION",
    "CA1slm"
])
def test_list_all_subregions_raises(acronym, allen_ontology_unreferenced_hpf):
    o: AtlasOntology = allen_ontology_unreferenced_hpf
    with pytest.raises(KeyError, match=".*not found.*"):
        o.list_all_subregions(acronym, unreferenced=False)

@pytest.mark.parametrize("ontology,acronym,expected", [
    ("allen_ontology_complete", "root", []),
    ("allen_ontology_complete", "HPF", ["CTXpl", "CTX", "CH", "grey", "root"]),
    ("allen_ontology_complete", "CA1", ["CA", "HIP", "HPF", "CTXpl", "CTX", "CH", "grey", "root"]),
    ("allen_ontology_complete_blacklisted_hpf", "HPF", ["CTXpl", "CTX", "CH", "grey", "root"]),
    ("allen_ontology_complete_blacklisted_hpf", "CA1", ["CA", "HIP", "HPF", "CTXpl", "CTX", "CH", "grey", "root"]),
])
def test_get_regions_above(ontology, acronym, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert expected == o.get_regions_above(acronym)

@pytest.mark.parametrize("acronym", [
    "HPF",
    "CA1",
    "NOT_A_REGION",
])
def test_get_regions_above_raises(acronym, allen_ontology_complete_unreferenced_hpf):
    o: AtlasOntology = allen_ontology_complete_unreferenced_hpf
    with pytest.raises(KeyError, match=".*not found.*"):
        o.get_regions_above(acronym)

@pytest.mark.parametrize("ontology, acronyms, expected", [
    ("allen_ontology_complete",
        ["root", "CTX", "BS", "HPF", "CA1"],
        OrderedDict([
            ("root", None),
            ("CTX", None),
            ("BS", None),
            ("HPF", "HPF"),
            ("CA1", "HPF")
    ])),
    ("allen_ontology_complete_blacklisted_hpf",
        ["CA1"],
        OrderedDict([
            ("CA1", "HPF")
    ])),
])
def test_get_corresponding_md(ontology, acronyms, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert o.get_corresponding_md(*acronyms) == expected

@pytest.mark.parametrize("acronym", [
    "HPF",
    "CA1",
    "NOT_A_REGION",
])
def test_get_corresponding_md_raises(acronym, allen_ontology_complete_unreferenced_hpf):
    o: AtlasOntology = allen_ontology_complete_unreferenced_hpf
    with pytest.raises(KeyError, match=".*not found.*"):
        o.get_corresponding_md(acronym)
    if acronym != "NOT_A_REGION":
        with pytest.raises(ValueError, match="Duplicate.*"):
            o.get_corresponding_md(acronym, acronym)

@pytest.mark.parametrize("ontology, acronym, full_name, colour", [
    ("allen_ontology_complete", "HPF", "Hippocampal formation", "#7ED04B"),
    ("allen_ontology_complete", "CA1", "Field CA1", "#7ED04B"),
    ("allen_ontology_complete_blacklisted_hpf", "HPF", "Hippocampal formation", "#7ED04B"),
    ("allen_ontology_complete_blacklisted_hpf", "CA1", "Field CA1", "#7ED04B"),
    ("allen_ontology_complete_unreferenced_hpf", "HPF", "Hippocampal formation", "#7ED04B"),
    ("allen_ontology_complete_unreferenced_hpf", "CA1", "Field CA1", "#7ED04B"),
])
def test_full_names_and_get_region_colors(ontology, acronym, full_name, colour, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    assert o.full_name[acronym] == full_name
    colors = o.get_region_colors()
    assert colors[acronym] == colour

@pytest.mark.parametrize("ontology, partition, regions, expected", [
    ("allen_ontology_complete",
     "major divisions",
     ["RE", "CA1", "CA", "HPF", "CTXpl", "root"],
     {"RE": "TH", "CA1": "HPF", "CA": "HPF", "HPF": "HPF", "CTXpl": None, "root": None}),
    ("allen_ontology_complete_blacklisted_hpf",
     "major divisions",
     ["RE", "CA1", "CA", "HPF", "CTXpl", "root"],
     {"RE": "TH", "CA1": "HPF", "CA": "HPF", "HPF": "HPF", "CTXpl": None, "root": None}),
    ("allen_ontology_complete",
     "summary structures",
     ["RE", "CA1", "CA", "HPF", "CTXpl", "root"],
     {"RE": "RE", "CA1": "CA1", "CA": None, "HPF": None, "CTXpl": None, "root": None}),
    ("allen_ontology_complete",
     ["grey", "fiber tracts"],
     ["RE", "CA1", "CA", "HPF", "CTXpl", "root", "cm", "fp"],
     {"RE": "grey", "CA1": "grey", "CA": "grey", "HPF": "grey", "CTXpl": "grey", "root": None, "cm": "fiber tracts", "fp": "fiber tracts"}),
    ("allen_ontology_complete",
     "major divisions",
     [],
     {}),
])
def test_partitioned(ontology, partition, regions, expected, request):
    o: AtlasOntology = request.getfixturevalue(ontology)
    o.partitioned(regions, partition=partition, key="acronym") == expected

def test_partitioned_raises(allen_ontology_complete_unreferenced_hpf: AtlasOntology):
    with pytest.raises(KeyError):
        allen_ontology_complete_unreferenced_hpf.partitioned(
            ["RE", "CA1", "CA", "HPF", "CTXpl", "root"], partition="major divisions")

@pytest.mark.parametrize("unreferenced, blacklisted", [
    (False, True),
    (False, False)
    # unreferenced=True can't be tested because BrainGlobe removes many of the unrefenced structures
])
def test_igraph(unreferenced: bool, blacklisted: bool):
    atlas = AtlasOntology("allen_mouse_10um",
                          blacklisted=["Isocortex", "fiber tracts", "VS"],
                          unreferenced=False)
    allen_ontology_json = Path(TemporaryDirectory(prefix="braian").name)/"allen_ontology_ccfv3.json"
    utils.cache(allen_ontology_json, "http://api.brain-map.org/api/v2/structure_graph_download/1.json")
    atlas_legacy = AllenBrainOntology(allen_ontology_json,
                                      version="CCFv3",
                                      blacklisted_acronyms=["Isocortex", "fiber tracts", "VS"],
                                      unreferenced=False)
    g = atlas.to_igraph(unreferenced=unreferenced, blacklisted=blacklisted)
    g_legacy = atlas_legacy.to_igraph(unreferenced=unreferenced, blacklisted=blacklisted)
    vs = set([v["id"] for v in g.vs])
    vs_legacy = set([v["id"] for v in g_legacy.vs])
    es = {(g.vs[e.source]["id"], g.vs[e.target]["id"]) for e in g.es}
    es_legacy = {(g_legacy.vs[e.source]["id"], g_legacy.vs[e.target]["id"]) for e in g_legacy.es}
    # BrainGlobe thinks that RSPd4 (id=545) does not exists in 'allen_mouse_10um'
    # see: https://github.com/brainglobe/brainglobe-atlasapi/issues/647
    if blacklisted: # The whole Isocortex is in the graph, RSPd4 included
        assert (vs|{545}) == vs_legacy
        assert (es|{(879,545)}) == es_legacy
    else:
        assert vs == vs_legacy
        assert es == es_legacy

def test_constructor_variants():
    # Default: no blacklist, unreferenced False
    o1 = AtlasOntology("allen_mouse_10um")
    assert o1.get_blacklisted_trees(unreferenced=False) == []
    assert o1.name == "allen_mouse_10um"
    o2 = AtlasOntology("allen_mouse_10um", blacklisted=["root"])
    assert o2.get_blacklisted_trees() == ["root"]
    o3 = AtlasOntology("allen_mouse_10um", blacklisted=["CH", "BS", "grey", "fiber tracts"])
    assert set(o3.get_blacklisted_trees()) == {"grey", "fiber tracts"}
    o4 = AtlasOntology("allen_mouse_50um", unreferenced=True) # 50um version has 'RSPd4 (545)' missing
    assert len(o4.get_blacklisted_trees(unreferenced=True)) == 0
    o5 = AtlasOntology("allen_mouse_50um", unreferenced=False)
    assert o5.get_blacklisted_trees(unreferenced=True) == ["RSPd4"]
    with pytest.raises(ValueError):
        AtlasOntology("other_allen_atlas", unreferenced=False)