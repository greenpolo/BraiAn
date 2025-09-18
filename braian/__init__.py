from braian._ontology import *
from braian._ontology_bg import *
from braian._brain_data import *
from braian._brain_slice import *
from braian._sliced_brain import *
from braian._animal_brain import *
from braian._animal_group import *
from braian._experiment import *

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("braian")
except PackageNotFoundError:
    # package is not installed
    pass