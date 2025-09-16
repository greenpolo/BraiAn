import brainglobe_atlasapi as bga
import numpy as np
import numpy.testing as npt
import pytest
from braian.ontology_bg import AtlasOntology
from braian.ontology import AllenBrainOntology

# @pytest.fixture(scope="module")
# def allen_mouse_ontology():
#     return AtlasOntology("allen_mouse_10um")

# @pytest.fixture(scope="module")
# def allen_brain_ontology():
#     import json
#     with open("/home/castoldi/Projects/BraiAn/data/allen_ontology_ccfv3.json") as f:
#         data = json.load(f)
#     allen_dict = data["msg"][0] if "msg" in data else data
#     return AllenBrainOntology(allen_dict, name="allen_mouse_10um_java", version="CCFv3")

# def test_root_node(allen_mouse_ontology, allen_brain_ontology):
#     assert allen_mouse_ontology.get_root() == allen_brain_ontology.dict["acronym"]

# def test_all_acronyms(allen_mouse_ontology, allen_brain_ontology):
#     acronyms_bg = set(allen_mouse_ontology.get_all_acronyms())
#     acronyms_ab = set([n["acronym"] for n in allen_brain_ontology._get_all_nodes()])
#     assert acronyms_bg == acronyms_ab

# def test_is_region(allen_mouse_ontology, allen_brain_ontology):
#     for acronym in ["Isocortex", "FRP", "MOp", "NOT_A_REGION"]:
#         assert allen_mouse_ontology.is_region(acronym) == allen_brain_ontology.is_region(acronym)

# def test_get_children(allen_mouse_ontology, allen_brain_ontology):
#     for acronym in ["root", "grey", "CH", "BS"]:
#         assert set(allen_mouse_ontology.get_children(acronym)) == set(allen_brain_ontology.direct_subregions.get(acronym, []))

# def test_get_parent(allen_mouse_ontology, allen_brain_ontology):
#     for acronym in ["CH", "BS", "CB", "CTX", "CNU"]:
#         assert allen_mouse_ontology.get_parent(acronym) == allen_brain_ontology.parent_region.get(acronym)

# def test_get_leaves(allen_mouse_ontology, allen_brain_ontology):
#     leaves_bg = set(allen_mouse_ontology.get_leaves())
#     leaves_ab = set([n["acronym"] for n in allen_brain_ontology._get_all_nodes() if n.get("children") == []])
#     assert leaves_bg == leaves_ab

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
    assert o.parent_region["CA1slm"] == "CA1"
    with pytest.raises(KeyError):
        _ = o.direct_subregions["CA1slm"] # leaf region
    assert o.parent_region["HPF"] == "CTXpl"
    assert o.direct_subregions["CTXpl"] == ["Isocortex", "OLF", "HPF"]
    o.blacklist_regions(["HPF"], has_reference=True)
    assert o.parent_region["HPF"] == "CTXpl"
    assert o.parent_region["CA1"] == "CA"
    assert o.direct_subregions["CTXpl"] == ["Isocortex", "OLF", "HPF"]

def test_direct_adjacent_regions_unreferenced(allen_ontology_complete: AtlasOntology):
    o: AtlasOntology = allen_ontology_complete
    assert o.parent_region["CA1slm"] == "CA1"
    assert o.direct_subregions["CA1"] == ["CA1slm", "CA1so", "CA1sp", "CA1sr"]
    o.blacklist_regions(["CA1sp", "CA1sr"], has_reference=False) # unreference SOME subregions of CA1
    assert o.direct_subregions["CA1"] == ["CA1slm", "CA1so"]
    o.blacklist_regions(["CA1slm", "CA1so", "CA1sp", "CA1sr"], has_reference=False) # unreference all subregions of CA1
    with pytest.raises(KeyError):
        o.direct_subregions["CA1"]
    with pytest.raises(KeyError):
        assert o.parent_region["CA1slm"]
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


def test_select_regions_and_add_to_selection(allen_ontology: AtlasOntology):
    with pytest.raises(KeyError, match=".*not found.*"):
        allen_ontology.select_regions(["HPF", "HIP", "CA", "NOT_A_REGION"])
    assert not allen_ontology.has_selection()
    allen_ontology.select_regions(["HPF", "HIP", "CA"])
    assert allen_ontology.has_selection()
    assert allen_ontology.get_selected_regions() == ["HPF"]
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