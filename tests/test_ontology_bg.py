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
    return AtlasOntology("allen_mouse_10um", blacklisted_acronyms=[], unreferenced=True)

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
    return AtlasOntology("allen_mouse_10um", blacklisted_acronyms=[], unreferenced=False)

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
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    result = ontology.are_regions(acronyms, unreferenced=unreferenced)
    npt.assert_array_equal(result, expected)

def test_are_regions_should_raise(allen_ontology: AllenBrainOntology):
    with pytest.raises(ValueError, match=".*Duplicates.*"):
        allen_ontology.are_regions(["Isocortex", "FRP", "Isocortex", "MOp"])

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
    ontology: AllenBrainOntology = request.getfixturevalue(ontology)
    result = ontology.minimum_treecover(acronyms, blacklisted=include_blacklisted, unreferenced=include_unreferenced)
    assert set(result) == expected

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