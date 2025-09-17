import platform

from pathlib import Path, WindowsPath
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("braian")
except PackageNotFoundError:
    # package is not installed
    pass

match platform.system():
    case "Windows":
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        def resolve_symlink(path: str|WindowsPath):
            if isinstance(path, WindowsPath):
                return shell.CreateShortCut(path).Targetpath if path.suffix == ".lnk" else path
            return shell.CreateShortCut(path).Targetpath if path.endswith(".lnk") else path
    case _:
        def resolve_symlink(path: Path):
            return path.resolve(strict=True)

from braian._ontology import *
from braian._ontology_bg import *
from braian._brain_slice import *
from braian._sliced_brain import *
from braian._brain_data import *
from braian._animal_brain import *
from braian._animal_group import *
from braian._experiment import *