import pytest
import json
from braian import AllenBrainOntology

@pytest.fixture
def allen_ontology():
    with open("/home/castoldi/Projects/BraiAn/data/allen_ontology_ccfv3.json") as f:
        data = json.load(f)
    allen_dict = data["msg"][0] if "msg" in data else data
    return AllenBrainOntology(allen_dict, version="CCFv3")

def test_is_region_true(allen_ontology):
    # Example acronyms known to exist in CCFv3
    assert allen_ontology.is_region("Isocortex")
    assert allen_ontology.is_region("FRP")
    assert allen_ontology.is_region("MOp")

def test_is_region_false(allen_ontology):
    # Example acronyms not present
    assert not allen_ontology.is_region("NOT_A_REGION")
    assert not allen_ontology.is_region(999999)
