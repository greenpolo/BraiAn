import json
import numpy as np
import numpy.testing as npt
import pytest
from braian import AllenBrainOntology
from braian import visit_dict
from braian.ontology import MAJOR_DIVISIONS
from collections import OrderedDict

@pytest.fixture(params=[
    {},
    {"blacklisted_acronyms": ["root"]},
    {"blacklisted_acronyms": ["CH", "BS"]},
    {"blacklisted_acronyms": ["grey", "fiber tracts", "VS", "grv", "retina"]},
    {"blacklisted_acronyms": []},
])
def allen_ontology_param(request):
    with open("/home/castoldi/Projects/BraiAn/data/allen_ontology_ccfv3.json") as f:
        data = json.load(f)
    allen_dict = data["msg"][0] if "msg" in data else data
    kwargs = request.param
    return AllenBrainOntology(allen_dict, version="CCFv3", **kwargs)

@pytest.fixture
def allen_ontology():
    with open("/home/castoldi/Projects/BraiAn/data/allen_ontology_ccfv3.json") as f:
        data = json.load(f)
    allen_dict = data["msg"][0] if "msg" in data else data
    return AllenBrainOntology(allen_dict,
                              blacklisted_acronyms=[],
                              name="allen_mouse_10um_java",
                              version="CCFv3",
                              unreferenced=False)

@pytest.fixture
def allen_ontology_complete():
    with open("/home/castoldi/Projects/BraiAn/data/allen_ontology_ccfv3.json") as f:
        data = json.load(f)
    allen_dict = data["msg"][0] if "msg" in data else data
    return AllenBrainOntology(allen_dict,
                              blacklisted_acronyms=[],
                              name="allen_mouse_10um_java",
                              version="CCFv3",
                              unreferenced=True)

@pytest.fixture
def allen_ontology_complete_blacklisted_hpf(allen_ontology_complete: AllenBrainOntology):
    allen_ontology_complete.blacklist_regions(["HPF"], has_reference=True)
    return allen_ontology_complete

@pytest.fixture
def allen_ontology_complete_unreferenced_hpf(allen_ontology_complete: AllenBrainOntology):
    allen_ontology_complete.blacklist_regions(["HPF"], has_reference=False)
    return allen_ontology_complete

@pytest.fixture
def allen_ontology_blacklisted_all(allen_ontology: AllenBrainOntology):
    allen_ontology.blacklist_regions(["root"], has_reference=True)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_all_no_reference(allen_ontology: AllenBrainOntology):
    allen_ontology.blacklist_regions(["root"], has_reference=False)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_depth3(allen_ontology: AllenBrainOntology):
    allen_ontology.blacklist_regions(["CTX", "CNU", "IB", "MB", "HB", "CBX", "CBN", "fiber tracts", "VS", "retina"], has_reference=True)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_depth3_no_reference(allen_ontology: AllenBrainOntology):
    allen_ontology.blacklist_regions(["CTX", "CNU", "IB", "MB", "HB", "CBX", "CBN", "fiber tracts", "VS", "retina"], has_reference=False)
    return allen_ontology

@pytest.fixture
def allen_ontology_blacklisted_hpf(allen_ontology: AllenBrainOntology):
    allen_ontology.blacklist_regions(["HPF"], has_reference=True)
    return allen_ontology

@pytest.fixture
def allen_ontology_unreferenced_hpf(allen_ontology: AllenBrainOntology):
    allen_ontology.blacklist_regions(["HPF"], has_reference=False)
    return allen_ontology

def test_direct_adjacent_regions(allen_ontology: AllenBrainOntology):
    # assert allen_ontology.parent_region["CH"] == "grey"
    assert len(allen_ontology.direct_subregions["CA1"]) == 0 # Allen ontology has unreferenced subregions of CA1
    assert set(allen_ontology.direct_subregions["CTXpl"]) == set(["Isocortex", "OLF", "HPF"])
    allen_ontology.blacklist_regions(["HPF"], has_reference=True)
    assert set(allen_ontology.direct_subregions["CTXpl"]) == set(["Isocortex", "OLF", "HPF"])

def test_direct_adjacent_regions_unreferenced(allen_ontology: AllenBrainOntology):
    assert set(allen_ontology.direct_subregions["CTXpl"]) == set(["Isocortex", "OLF", "HPF"])
    allen_ontology.blacklist_regions(["HPF"], has_reference=False)
    assert set(allen_ontology.direct_subregions["CTXpl"]) == set(["Isocortex", "OLF"])

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
    o: AllenBrainOntology = request.getfixturevalue(ontology)
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
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    result = ontology.are_regions(acronyms, unreferenced=unreferenced)
    npt.assert_array_equal(result, expected)

def test_are_regions_should_raise(allen_ontology: AllenBrainOntology):
    with pytest.raises(ValueError):
        allen_ontology.are_regions(["Isocortex", "FRP", "Isocortex", "MOp"])

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
    o: AllenBrainOntology = request.getfixturevalue(ontology)
    assert o.contains_all_children(parent, regions) == expected

@pytest.mark.parametrize(
    "ontology, acronyms, include_blacklisted, include_unreferenced, expected",
    [
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "HPF", "TH", "HY"], True, True, {"CTXpl", "IB"}),
        ("allen_ontology_unreferenced_hpf",
         ["RE", "Xi", "PVT", "PT", "TH"], True, True, {"MTN", "TH"}),
        ("allen_ontology_unreferenced_hpf",
         ["P", "MB", "TH", "MY", "CB", "HY"], True, True, {"BS", "CB"}),
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "TH", "HY"], True, False, {"CTXpl", "IB"}), # 'HPF' is blacklisted, so 'CTXpl' is an ancestor region that covers the input acronyms
        ("allen_ontology_unreferenced_hpf",
         ["Isocortex", "OLF", "TH", "HY"], True, True, {"Isocortex", "OLF", "IB"}), # 'HPF' is missing for the input acronyms to cover 'CTXpl' completely
        ("allen_ontology_unreferenced_hpf",
         ["CA1", "CA2", "CA3", "NOT_A_REGION"], True, True, {"CA"}),
    ]
)
def test_minimum_treecover(ontology, acronyms, include_blacklisted, include_unreferenced, expected, request):
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    result = set(ontology.minimum_treecover(acronyms, blacklisted=include_blacklisted, unreferenced=include_unreferenced))
    assert result == expected

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
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    result = ontology.get_blacklisted_trees(unreferenced=unreferenced)
    assert result == expected

@pytest.mark.parametrize(
    "ontology, regions, should_raise",
    [
        ("allen_ontology_complete", ["HPF", "Isocortex"], False),
        ("allen_ontology_complete", ["NOT_A_REGION"], True),
        ("allen_ontology_complete_blacklisted_hpf", ["HPF", "Isocortex"], False),
        ("allen_ontology_complete_blacklisted_hpf", ["NOT_A_REGION"], True),
        ("allen_ontology_complete_unreferenced_hpf", ["HPF", "Isocortex"], False),
    ]
)
def test_blacklist_regions(ontology, regions, should_raise, request):
    o: AllenBrainOntology = request.getfixturevalue(ontology)
    if should_raise:
        with pytest.raises(ValueError, match="Some given regions are not recognised as part of the ontology"):
            o.blacklist_regions(regions, has_reference=True)
    else:
        o.blacklist_regions(regions, has_reference=True)
        assert set(o.get_blacklisted_trees(unreferenced=True)) == set(regions)
        if ontology.endswith("_unreferenced_hpf"):
            assert set(o.get_blacklisted_trees(unreferenced=False)) == {"Isocortex"}

@pytest.mark.parametrize(
    "ontology, regions",
    [
        ("allen_ontology_complete", ["HPF", "Isocortex"]),
        ("allen_ontology_complete_blacklisted_hpf", ["HPF", "Isocortex"]),
        ("allen_ontology_complete_unreferenced_hpf", ["HPF", "Isocortex"]),
    ]
)
def test_blacklist_regions_unreferenced(ontology, regions, request):
    o: AllenBrainOntology = request.getfixturevalue(ontology)
    o.blacklist_regions(regions, has_reference=False)
    assert set(o.get_blacklisted_trees(unreferenced=True)) == set(regions)
    assert len(o.get_blacklisted_trees(unreferenced=False)) == 0

@pytest.mark.parametrize(
    "ontology, depth, expected",
    [
        ("allen_ontology", 0, ["root"]),
        ("allen_ontology", 1, ["grey", "fiber tracts", "VS", "grv", "retina"]),
        ("allen_ontology_param", 0, ["root"]),
        ("allen_ontology_param", 1, ["grey", "fiber tracts", "VS", "grv", "retina"]),
    ]
)
def test_select_at_depth(ontology, depth, expected, request):
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    ontology.select_at_depth(depth)
    selected = set(ontology.get_selected_regions())
    for e in expected:
        assert e in selected or e not in selected  # allow for blacklisting
    ontology.unselect_all()

@pytest.mark.parametrize(
    "ontology, level, expected",
    [
        ("allen_ontology", 1, ["grey", "fiber tracts", "VS", "grv", "retina"]),
        ("allen_ontology", 2, ["CH", "BS", "CB"]),
        ("allen_ontology_param", 1, ["grey", "fiber tracts", "VS", "grv", "retina"]),
        ("allen_ontology_param", 2, ["CH", "BS", "CB"]),
    ]
)
def test_select_at_structural_level(ontology, level, expected, request):
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    ontology.select_at_structural_level(level)
    selected = set(ontology.get_selected_regions())
    for e in expected:
        assert e in selected or e not in selected  # allow for blacklisting
    ontology.unselect_all()

def test_select_leaves(allen_ontology):
    allen_ontology.select_leaves()
    selected = allen_ontology.get_selected_regions()
    # All selected regions should be leaves (no children)
    for acronym in selected:
        node = visit_dict.find_subtree(allen_ontology.dict, "acronym", acronym, "children")
        assert node is not None
        assert node["children"] == []
    allen_ontology.unselect_all()

def test_select_regions_and_add_to_selection(allen_ontology):
    allen_ontology.select_regions(["CH", "CB"])
    selected = set(allen_ontology.get_selected_regions())
    assert "CH" in selected and "CB" in selected
    allen_ontology.add_to_selection(["BS"])
    selected = set(allen_ontology.get_selected_regions())
    assert "BS" in selected
    allen_ontology.unselect_all()

def test_has_selection_and_unselect_all(allen_ontology):
    allen_ontology.select_regions(["CH"])
    assert allen_ontology.has_selection()
    allen_ontology.unselect_all()
    assert not allen_ontology.has_selection()

def test_get_regions(allen_ontology):
    regions = allen_ontology.get_regions("major divisions")
    for acronym in MAJOR_DIVISIONS:
        assert acronym in regions
    regions = allen_ontology.get_regions("leaves")
    # All returned regions should be leaves
    for acronym in regions:
        node = visit_dict.find_subtree(allen_ontology.dict, "acronym", acronym, "children")
        assert node is not None
        assert node["children"] == []

def test_ids_to_acronym_and_acronyms_to_id(allen_ontology):
    ids = [997, 8, 343]
    acronyms = allen_ontology.ids_to_acronym(ids)
    assert acronyms == ["root", "grey", "BS"]
    back_ids = allen_ontology.acronyms_to_id(["root", "grey", "BS"])
    assert back_ids == ids

@pytest.mark.parametrize("region,key,expected", [
    ("CTX", "acronym", ["CTX", "CNU"]),
    ("BS", "acronym", ["IB", "MB", "HB"]),
])
def test_get_sibiling_regions(allen_ontology, region, key, expected):
    siblings = allen_ontology.get_sibiling_regions(region, key=key)
    for e in expected:
        assert e in siblings

@pytest.mark.parametrize("regions,key,expected", [
    (["CTX", "BS"], "acronym", {"CTX": "CH", "BS": "grey"}),
])
def test_get_parent_regions(allen_ontology, regions, key, expected):
    parents = allen_ontology.get_parent_regions(regions, key=key)
    for k, v in expected.items():
        assert parents[k] == v

@pytest.mark.parametrize("acronym,mode,expected", [
    ("CH", "breadth", ["CH", "CTX", "CNU"]),
    ("BS", "depth", ["BS", "IB", "MB", "HB"]),
])
def test_list_all_subregions(allen_ontology, acronym, mode, expected):
    subregions = allen_ontology.list_all_subregions(acronym, mode=mode)
    for e in expected:
        assert e in subregions

@pytest.mark.parametrize("acronym,expected", [
    ("CTX", ["CH", "grey", "root"]),
    ("BS", ["grey", "root"]),
])
def test_get_regions_above(allen_ontology, acronym, expected):
    path = allen_ontology.get_regions_above(acronym)
    for e in expected:
        assert e in path

@pytest.mark.parametrize("acronym,acronyms,expected", [
    ("CTX", ["BS"], OrderedDict([("CTX", "CH"), ("BS", "grey")])),
])
def test_get_corresponding_md(allen_ontology, acronym, acronyms, expected):
    result = allen_ontology.get_corresponding_md(acronym, *acronyms)
    for k, v in expected.items():
        assert result[k] == v

def test_full_names_and_get_region_colors(allen_ontology):
    assert allen_ontology.full_name["CH"] == "Cerebrum"
    colors = allen_ontology.get_region_colors()
    assert colors["CH"] == "#B0F0FF"

def test_constructor_variants():
    with open("/home/castoldi/Projects/BraiAn/data/allen_ontology_ccfv3.json") as f:
        data = json.load(f)
    allen_dict = data["msg"][0] if "msg" in data else data
    # Default: name and version deduced, no blacklist, unreferenced False
    o1 = AllenBrainOntology(allen_dict)
    assert o1.get_blacklisted_trees(unreferenced=False) == []
    assert o1.name == "allen_mouse_10um_java"
    assert o1.annotation_version == "ccf_2017"
    o2 = AllenBrainOntology(allen_dict, blacklisted_acronyms=["root"])
    assert o2.get_blacklisted_trees() == ["root"]
    o3 = AllenBrainOntology(allen_dict, blacklisted_acronyms=["CH", "BS", "grey", "fiber tracts"])
    assert set(o3.get_blacklisted_trees()) == {"grey", "fiber tracts"}
    o4 = AllenBrainOntology(allen_dict, unreferenced=True, version="CCFv3")
    assert len(o4.get_blacklisted_trees(unreferenced=True)) == 0 # CCFv3 has some unreferenced regions in its ontology
    o5 = AllenBrainOntology(allen_dict, unreferenced=False, version="CCFv3")
    assert len(o5.get_blacklisted_trees(unreferenced=True)) > 0
    o6 = AllenBrainOntology(allen_dict, name="other_allen_atlas", version="CCFv4", unreferenced=False)
    assert o6.annotation_version == "ccf_2022"
    with pytest.raises(ValueError):
        AllenBrainOntology(allen_dict, name="other_allen_atlas")
    # Name as BrainGlobe alias (should not raise if atlas is available)
    try:
        o9 = AllenBrainOntology(allen_dict, name="allen_mouse_10um", version="CCFv3")
        assert o9.name == "allen_mouse_10um"
    except ValueError:
        pass  # Acceptable if atlas is not available locally
    # Version deduction from name
    o10 = AllenBrainOntology(allen_dict, name="allen_mouse_10um_java")
    assert o10.annotation_version == "ccf_2017"
    o11 = AllenBrainOntology(allen_dict, name="allen_mouse_10um")
    assert o11.annotation_version == "ccf_2017"